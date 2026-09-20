// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    /// ODERHSFunction is nonthrowing. A run-local error slot lets generated
    /// operators fail without indexing invalid storage; the driver checks it
    /// immediately after each attempted step, before delivering any output.
    internal final class Failure {
        var error: SolverError?
        func check() throws { if let error { throw error } }
    }

    internal struct Operator: ~Copyable {
        let source: TimeDependentOperator
        let isConstant: Bool
        var matrix: UniqueMatrix<Complex<Double>>
        var loss: UniqueMatrix<Complex<Double>>
        let needsLoss: Bool

        init(_ source: TimeDependentOperator, dimension: Int, needsLoss: Bool = false) throws {
            if source.firstDimensionMismatch(expected: dimension) != nil {
                throw SolverError.operatorDimensionMismatch
            }
            self.source = source
            self.isConstant = source.isConstant
            self.needsLoss = needsLoss
            matrix = .zeros(rows: dimension, columns: dimension)
            loss = .zeros(rows: needsLoss ? dimension : 0, columns: needsLoss ? dimension : 0)
            if isConstant {
                source.insert(t: 0, into: &matrix)
                if needsLoss {
                    MatrixAction.product(
                        matrix.elements, matrix.elements, dimension: dimension,
                        adjointA: true, adding: false, into: loss.elements)
                }
            }
        }

        mutating func update(at time: Double, dimension: Int) -> Bool {
            if !isConstant {
                source.insert(t: time, into: &matrix)
                guard matrix.rows == dimension && matrix.columns == dimension else { return false }
                if needsLoss {
                    MatrixAction.product(
                        matrix.elements, matrix.elements, dimension: dimension,
                        adjointA: true, adding: false, into: loss.elements)
                }
            }
            return true
        }
    }

    /// Equations (11.5), (11.8) and (11.9) of the latent-basis notes.
    /// All buffers are allocated at construction. The Hamiltonian, bath and
    /// collapse operators are evaluated once per RK stage, shared by all ADOs.
    internal struct RightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable, ODERHSFunction {
        let hamiltonian: Hamiltonian
        let hierarchy: HEOM.Hierarchy
        let dimension: Int
        let centered: Bool
        let failure: Failure
        let coefficients: _LatentBathCoefficients
        let bathIsConstant: Bool
        let rates: [ScalarTimeFunction]
        var bathOperators: UniqueArray<Operator>
        var collapseOperators: UniqueArray<Operator>
        var rateValues: UniqueArray<Double>
        var h: UniqueMatrix<Complex<Double>>
        var lambda: UniqueMatrix<Complex<Double>>
        var memory: UniqueMatrix<Complex<Double>>
        var means: UniqueVector<Complex<Double>>
        var temporary: UniqueMatrix<Complex<Double>>

        init(
            problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
            failure: Failure
        ) throws {
            let d = problem.system.dimension
            precondition(d > 0, "The system dimension must be positive.")
            // Complete throwing preparation before initializing the
            // noncopyable RHS (Swift cannot unwind a partially initialized one).
            var preparedBath = UniqueArray<Operator>()
            for op in configuration.hierarchy.environment.couplingOperators {
                preparedBath.append(try Operator(op, dimension: d))
            }
            var preparedCollapse = UniqueArray<Operator>()
            for channel in problem.markovianChannels {
                preparedCollapse.append(try Operator(channel.collapseOperator, dimension: d, needsLoss: true))
            }
            hamiltonian = problem.system.hamiltonian
            hierarchy = configuration.hierarchy
            dimension = d
            centered = configuration.shiftType == .meanField
            self.failure = failure
            coefficients = _LatentBathCoefficients(hierarchy.environment.bath)
            bathIsConstant = hierarchy.environment.couplingOperators.allSatisfy(\.isConstant)
            bathOperators = preparedBath
            collapseOperators = preparedCollapse
            rates = problem.markovianChannels.map(\.rate)
            rateValues = .init(repeating: 0, count: rates.count)
            h = .zeros(rows: d, columns: d)
            lambda = .zeros(rows: coefficients.poles.count, columns: d * d)
            memory = .zeros(rows: coefficients.poles.count, columns: d * d)
            means = coefficients.poles.isEmpty ? .init() : .zero(coefficients.poles.count)
            temporary = .zeros(rows: d, columns: d)
            if bathIsConstant { buildBathOperators() }
        }

        mutating func buildBathOperators() {
            lambda.zeroElements()
            memory.zeroElements()
            let size = dimension * dimension
            for p in coefficients.poles.indices {
                for i in 0..<bathOperators.count {
                    let up = coefficients.upward[i, p].conjugate
                    let down = coefficients.downward[i, p]
                    for j in 0..<size {
                        let value = bathOperators[i].matrix.elements[j]
                        lambda.elements[p * size + j] += up * value
                        memory.elements[p * size + j] += down * value
                    }
                }
            }
        }

        mutating func evaluate(t: Double, y: borrowing State, dy: inout State) {
            dy.zero()
            if failure.error != nil { return }
            let d = dimension
            let size = d * d
            let poles = coefficients.poles
            let pCount = poles.count
            let hamiltonian = self.hamiltonian
            hamiltonian.hamiltonian(t: t, into: &h)
            guard h.rows == d && h.columns == d else {
                failure.error = .operatorDimensionMismatch
                return
            }
            if !bathIsConstant {
                for i in 0..<bathOperators.count {
                    guard bathOperators[i].update(at: t, dimension: d) else {
                        failure.error = .operatorDimensionMismatch
                        return
                    }
                }
                buildBathOperators()
            }
            for i in 0..<collapseOperators.count {
                let rate = rates[i](t)
                guard rate.isFinite && rate >= 0 else {
                    failure.error = .invalidMarkovianRate(time: t)
                    return
                }
                rateValues[i] = rate
                guard collapseOperators[i].update(at: t, dimension: d) else {
                    failure.error = .operatorDimensionMismatch
                    return
                }
            }

            if centered && pCount > 0 {
                // The first block is always the unconditioned physical guide.
                var trace = Complex<Double>.zero
                for i in 0..<d { trace += y.ados.elements[i * d + i] }
                guard trace.real.isFinite && trace.imaginary.isFinite && trace.length > 0 else {
                    failure.error = .invalidGuideTrace(time: t)
                    return
                }
                for p in 0..<pCount {
                    var mean = Complex<Double>.zero
                    for i in 0..<d {
                        for j in 0..<d {
                            mean += memory.elements[p * size + i * d + j] * y.ados.elements[j * d + i]
                        }
                    }
                    mean /= trace
                    means[p] = mean
                    dy.shifts[p] = -poles[p] * y.shifts[p] + mean
                    // H_mu = H + i sum_p (mu_p* Lambda_p - mu_p Lambda_p^dagger).
                    for i in 0..<d {
                        for j in 0..<d {
                            h[i, j] +=
                                .i
                                * (y.shifts[p].conjugate * lambda.elements[p * size + i * d + j]
                                    - y.shifts[p] * lambda.elements[p * size + j * d + i].conjugate)
                        }
                    }
                }
            }

            let count = hierarchy.count
            for block in 0..<(y.ados.rows / count) {
                let base = block * count * size
                for ado in 0..<count {
                    let input = y.ados.elements + base + ado * size
                    let output = dy.ados.elements + base + ado * size
                    let damping = hierarchy.kWArray[ado]
                    for j in 0..<size { output[j] = damping * input[j] }
                    MatrixAction.commutator(h.elements, input, dimension: d, scale: -.i, into: output)
                    for i in 0..<collapseOperators.count where rateValues[i] != 0 {
                        let rate = Complex<Double>(rateValues[i])
                        let op = collapseOperators[i].matrix.elements
                        let loss = collapseOperators[i].loss.elements
                        MatrixAction.product(op, input, dimension: d, adding: false, into: temporary.elements)
                        MatrixAction.product(
                            temporary.elements, op, dimension: d, adjointB: true,
                            scale: rate, into: output)
                        MatrixAction.product(loss, input, dimension: d, scale: -0.5 * rate, into: output)
                        MatrixAction.product(input, loss, dimension: d, scale: -0.5 * rate, into: output)
                    }
                    for p in 0..<pCount {
                        let op = lambda.elements + p * size
                        let down = memory.elements + p * size
                        let ket = ado * (2 * pCount) + p
                        let bra = ket + pCount
                        let ketChild = hierarchy.childIndices[ket]
                        let braChild = hierarchy.childIndices[bra]
                        if ketChild >= 0 {
                            MatrixAction.commutator(
                                op, y.ados.elements + base + ketChild * size, dimension: d,
                                adjoint: true, scale: Complex(-hierarchy.childWeights[ket]), into: output)
                        }
                        if braChild >= 0 {
                            MatrixAction.commutator(
                                op, y.ados.elements + base + braChild * size, dimension: d,
                                scale: Complex(hierarchy.childWeights[bra]), into: output)
                        }
                        let ketParent = hierarchy.parentIndices[ket]
                        let braParent = hierarchy.parentIndices[bra]
                        if ketParent >= 0 {
                            let parent = y.ados.elements + base + ketParent * size
                            let weight = hierarchy.parentWeights[ket]
                            MatrixAction.product(
                                down, parent, dimension: d, scale: Complex(weight), into: output)
                            if centered {
                                let value = weight * means[p]
                                for j in 0..<size { output[j] -= value * parent[j] }
                            }
                        }
                        if braParent >= 0 {
                            let parent = y.ados.elements + base + braParent * size
                            let weight = hierarchy.parentWeights[bra]
                            MatrixAction.product(
                                parent, down, dimension: d, adjointB: true,
                                scale: Complex(weight), into: output)
                            if centered {
                                let value = weight * means[p].conjugate
                                for j in 0..<size { output[j] -= value * parent[j] }
                            }
                        }
                    }
                }
            }
        }
    }
}
