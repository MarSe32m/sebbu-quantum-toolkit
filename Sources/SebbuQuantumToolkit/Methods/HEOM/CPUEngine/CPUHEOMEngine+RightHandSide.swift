// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    /// ODERHSFunction is nonthrowing. A run-local error slot lets generated
    /// operators fail without indexing invalid storage; the driver checks it
    /// immediately after each attempted step, before delivering any output.
    @usableFromInline
    internal final class Failure {
        @usableFromInline
        var error: SolverError?
        
        @inlinable
        init(error: SolverError? = nil) {
            self.error = error
        }
        
        @inlinable
        func check() throws { if let error { throw error } }
    }

    @usableFromInline
    internal struct Operator: ~Copyable {
        @usableFromInline
        let source: TimeDependentOperator
        @usableFromInline
        let isConstant: Bool
        @usableFromInline
        var matrix: UniqueMatrix<Complex<Double>>
        @usableFromInline
        var loss: UniqueMatrix<Complex<Double>>
        @usableFromInline
        let needsLoss: Bool

        @inlinable
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

        @inlinable
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
    @usableFromInline
    internal struct RightHandSide: ~Copyable, ODERHSFunction {
        @usableFromInline
        let hamiltonian: TimeDependentOperator
        @usableFromInline
        let hierarchy: HEOM.Hierarchy
        @usableFromInline
        let dimension: Int
        @usableFromInline
        let centered: Bool
        @usableFromInline
        let failure: Failure
        @usableFromInline
        let coefficients: _LatentBathCoefficients
        @usableFromInline
        let bathIsConstant: Bool
        @usableFromInline
        let rates: [ScalarTimeFunction]
        @usableFromInline
        var bathOperators: UniqueArray<Operator>
        @usableFromInline
        var collapseOperators: UniqueArray<Operator>
        @usableFromInline
        var collapseViews: UniqueVector<CollapseView>
        @usableFromInline
        var h: UniqueMatrix<Complex<Double>>
        @usableFromInline
        var hRight: UniqueMatrix<Complex<Double>>
        @usableFromInline
        var lambda: UniqueMatrix<Complex<Double>>
        @usableFromInline
        var memory: UniqueMatrix<Complex<Double>>
        @usableFromInline
        var means: UniqueVector<Complex<Double>>
        @usableFromInline
        var temporary: UniqueMatrix<Complex<Double>>
        @usableFromInline
        let pool: PersistentWorkerPool<ADOBatch>?

        @usableFromInline
        internal var workerCount: Int { pool?.workerCount ?? 1 }

        @inlinable
        init(
            problem: DensityMatrixProblem,
            configuration: HEOM.Configuration,
            failure: Failure,
            copies: Int = 1
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
            var views: [CollapseView] = []
            for channel in problem.markovianChannels {
                let op = try Operator(channel.collapseOperator, dimension: d, needsLoss: true)
                views.append(CollapseView(matrix: UnsafePointer(op.matrix.elements), rate: 0))
                preparedCollapse.append(op)
            }
            let workers = configuration.parallelism.workerCount(
                dimension: d, adoCount: configuration.hierarchy.count * copies,
                poleCount: configuration.hierarchy.multiIndexCount / 2,
                collapseCount: problem.markovianChannels.count)
            let preparedPool: PersistentWorkerPool<ADOBatch>? = workers > 1
                ? .init(workers: workers) { batch, worker in batch.evaluate(worker: worker) } : nil
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
            collapseViews = .init(views)
            h = .zeros(rows: d, columns: d)
            hRight = .zeros(rows: d, columns: d)
            lambda = .zeros(rows: coefficients.poles.count, columns: d * d)
            memory = .zeros(rows: coefficients.poles.count, columns: d * d)
            means = coefficients.poles.isEmpty ? .init() : .zero(coefficients.poles.count)
            temporary = .zeros(rows: preparedPool?.workerCount ?? 1, columns: d * d)
            pool = preparedPool
            if bathIsConstant { buildBathOperators() }
        }

        @inlinable
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

        @inlinable
        mutating func evaluate(t: Double, y: borrowing State, dy: inout State) {
            // Successful stages overwrite every ADO in parallel. Avoid a serial
            // pass over the entire hierarchy merely to initialize the outputs.
            if failure.error != nil { dy.zero(); return }
            let d = dimension
            let size = d * d
            let poles = coefficients.poles
            let pCount = poles.count
            let hamiltonian = self.hamiltonian
            //TODO: Take advantage of potentially sparse Hamiltonian etc.
            hamiltonian.insert(t: t, into: &h)
            guard h.rows == d && h.columns == d else {
                failure.error = .operatorDimensionMismatch
                dy.zero()
                return
            }
            if !bathIsConstant {
                for i in 0..<bathOperators.count {
                    guard bathOperators[i].update(at: t, dimension: d) else {
                        failure.error = .operatorDimensionMismatch
                        dy.zero()
                        return
                    }
                }
                buildBathOperators()
            }
            for i in 0..<collapseOperators.count {
                let rate = rates[i](t)
                guard rate.isFinite && rate >= 0 else {
                    failure.error = .invalidMarkovianRate(time: t)
                    dy.zero()
                    return
                }
                guard collapseOperators[i].update(at: t, dimension: d) else {
                    failure.error = .operatorDimensionMismatch
                    dy.zero()
                    return
                }
                // Generated operators may replace their storage on each call.
                collapseViews[i] = CollapseView(
                    matrix: UnsafePointer(collapseOperators[i].matrix.elements), rate: rate)
            }

            if centered && pCount > 0 {
                // The first block is always the unconditioned physical guide.
                var trace = Complex<Double>.zero
                for i in 0..<d { trace += y.ados.elements[i * d + i] }
                guard trace.real.isFinite && trace.imaginary.isFinite && trace.length > 0 else {
                    failure.error = .invalidGuideTrace(time: t)
                    dy.zero()
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

            // Prepare H_eff and its right-action partner once per stage.
            // Keeping both also preserves the commutator for a supplied H that
            // is not exactly Hermitian, without conjugating the companion ADO.
            hRight.copyElements(from: h)
            for i in 0..<collapseOperators.count where collapseViews[i].rate != 0 {
                let factor = Complex<Double>(0, 0.5 * collapseViews[i].rate)
                let loss = collapseOperators[i].loss.elements
                for j in 0..<size {
                    h.elements[j] -= factor * loss[j]
                    hRight.elements[j] += factor * loss[j]
                }
            }
            let batch = ADOBatch(
                hierarchy: hierarchy, dimension: d, centered: centered,
                workerCount: workerCount, adoCount: y.ados.rows,
                input: UnsafePointer(y.ados.elements), output: dy.ados.elements,
                hLeft: UnsafePointer(h.elements), hRight: UnsafePointer(hRight.elements),
                lambda: UnsafePointer(lambda.elements), memory: UnsafePointer(memory.elements),
                means: UnsafePointer(means.components), collapse: UnsafePointer(collapseViews.components),
                collapseCount: collapseViews.count, temporaries: temporary.elements)
            if let pool {
                pool.run(batch)
            } else {
                batch.evaluate(worker: 0)
            }
        }
    }
}
