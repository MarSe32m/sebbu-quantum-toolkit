// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

/// Constant CSR, constant expansion, changing CSR expansion, mixed expansion,
/// and a CSR expansion alongside a generated dense channel, respectively.
private func heomBathFixture(
    _ d: Int, sparse: Bool, kind: Int, record: @escaping @Sendable () -> Void = {}
) -> [TimeDependentOperator] {
    let (csr, dense) = explicitSparseFixture(d)
    let adjoint = dense.conjugateTranspose
    let a = sparse ? ConstantOperator(csr) : ConstantOperator(dense)
    let b = sparse ? ConstantOperator(CSRMatrix(from: adjoint)) : ConstantOperator(adjoint)
    if kind == 0 { return [.constant(a), .constant(b)] }
    let coefficients: [ComplexTimeFunction]
    if kind == 1 {
        coefficients = [.constant(Complex(0.7, -0.3)), .constant(Complex(-0.2, 0.4))]
    } else {
        coefficients = [
            .generated { t in
                record()
                // Cross zero and revisit it after a rejected/repeated stage.
                return Complex(t - 0.2, 0.3 * (t - 0.2))
            },
            .generated { t in
                record()
                return Complex(-0.2 + t, 0.4 - 0.1 * t)
            },
        ]
    }
    let first = TimeDependentOperator.linearCombination(.init(
        coefficients: coefficients,
        operators: [a, kind == 3 ? ConstantOperator(adjoint) : b]))
    if kind == 4 {
        return [first, .generatedDense(.init { t, out in
            record()
            // Replacing storage must not leave stale pointers in latent views.
            out = .init(copying: dense)
            out.multiply(by: Complex(0.3 + t, -0.1))
        })]
    }
    return [first, .constant(b)]
}

@Suite("HEOM sparse bath propagation")
struct HEOMSparseBathTests {
    @Test("Latent bath actions keep CSR storage and match dense propagation", arguments: heomShifts, 0..<5)
    func bathActions(shift: HEOM.ShiftType, kind: Int) throws {
        for d in [2, 7] {
            let calls = Mutex(0)
            let sparseSources = heomBathFixture(d, sparse: true, kind: kind) { calls.withLock { $0 += 1 } }
            let denseSources = heomBathFixture(d, sparse: false, kind: kind)
            let model = hopsFixtureModel()
            func configuration(_ sources: [TimeDependentOperator], workers: Int) -> HEOM.Configuration {
                .init(hierarchy: .init(environment: .init(couplingOperators: sources, bath: model),
                                       truncation: .maximumTier(2)),
                      shiftType: shift, parallelism: .maximumWorkers(workers))
            }
            let referenceConfig = configuration(denseSources, workers: 1)
            let problem = DensityMatrixProblem(sparseFixtureProblem(d, sparseH: true, sparseC: true))
            let shiftCount = shift == .none ? 0 : model.poleCount
            let failure = HEOM.CPUEngine.Failure()
            var reference = try HEOM.CPUEngine.RightHandSide(
                problem: problem, configuration: referenceConfig, failure: failure, copies: 3)
            var state = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: referenceConfig.hierarchy.count, shiftCount: shiftCount, copies: 3)
            var expected = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: referenceConfig.hierarchy.count, shiftCount: shiftCount, copies: 3)
            var actual = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: referenceConfig.hierarchy.count, shiftCount: shiftCount, copies: 3)
            let count = state.ados.rows * d * d
            for i in 0..<count {
                state.ados.elements[i] = Complex(0.02 * Double.sin(Double(i + 1)), 0.01 * Double.cos(Double(i)))
            }
            for i in 0..<d { state.ados.elements[i * d + i] += Complex(1 / Double(d)) }
            for p in 0..<shiftCount { state.shifts[p] = Complex(0.03 * Double(p + 1), -0.02) }
            for workers in [1, 3] {
                calls.withLock { $0 = 0 }
                var rhs = try HEOM.CPUEngine.RightHandSide(
                    problem: problem, configuration: configuration(sparseSources, workers: workers),
                    failure: failure, copies: 3)
                #expect(calls.withLock { $0 } == 0)
                // Assert representation as well as numerical agreement: a
                // regression that silently densifies CSR must fail this test.
                for p in 0..<model.poleCount {
                    #expect({ rhs.bathViews[p].sparseLambda.count > 0 }())
                    #expect({ rhs.lambda[p].dense.rows == (kind >= 3 ? d : 0) }())
                    #expect({ rhs.memory[p].dense.rows == (kind >= 3 ? d : 0) }())
                }
                let firstView = rhs.bathViews[0].sparseLambda.terms[0].matrix
                let times = [0.0, 0.2, 0.1, 0.2, 0.4]
                for t in times {
                    rhs.evaluate(t: t, y: state, dy: &actual)
                    reference.evaluate(t: t, y: state, dy: &expected)
                    try failure.check()
                    expectHOPSClose((0..<count).map { actual.ados.elements[$0] },
                                    (0..<count).map { expected.ados.elements[$0] }, tolerance: 2e-12)
                    expectHOPSClose((0..<shiftCount).map { actual.shifts[$0] },
                                    (0..<shiftCount).map { expected.shifts[$0] }, tolerance: 2e-12)
                    let currentView = rhs.bathViews[0].sparseLambda.terms[0].matrix
                    #expect(currentView.values == firstView.values)
                    #expect(currentView.rows == firstView.rows)
                    #expect(currentView.columns == firstView.columns)
                }
                let callsPerStage = kind < 2 ? 0 : (kind == 4 ? 3 : 2)
                #expect(calls.withLock { $0 } == callsPerStage * times.count)
            }
        }
    }

    @Test("Empty CSR, zero latent weights and zero poles have no bath action", arguments: heomShifts)
    func emptyBathActions(shift: HEOM.ShiftType) throws {
        let d = 5
        let empty = CSRMatrix<Complex<Double>>(
            rows: d, columns: d, values: [], rowIndices: Array(repeating: 0, count: d + 1), columnIndices: [])
        let zero = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
        let problem = DensityMatrixProblem(initialState: heomIdentity(d), system: .init(zero))
        for model in [heomSingleBath(), heomSingleBath(strength: 0), CorrelatedBathModel.zero(channelCount: 1)] {
            let config = HEOM.Configuration(
                hierarchy: .init(environment: .init(couplingOperator: .constant(empty), bath: model),
                                 truncation: .maximumTier(2)),
                shiftType: shift, parallelism: .maximumWorkers(3))
            let shifts = shift == .none ? 0 : model.poleCount
            let failure = HEOM.CPUEngine.Failure()
            var rhs = try HEOM.CPUEngine.RightHandSide(problem: problem, configuration: config, failure: failure)
            let state = HEOM.CPUEngine.State(dimension: d, hierarchyCount: config.hierarchy.count, shiftCount: shifts)
            var derivative = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: config.hierarchy.count, shiftCount: shifts)
            state.ados.elements[0] = .one
            for i in 0..<derivative.ados.rows * d * d { derivative.ados.elements[i] = Complex(42) }
            rhs.evaluate(t: 0.1, y: state, dy: &derivative)
            try failure.check()
            #expect((0..<derivative.ados.rows * d * d).allSatisfy { derivative.ados.elements[$0] == .zero })
            #expect((0..<shifts).allSatisfy { derivative.shifts[$0] == .zero })
        }
    }

    @Test("All HEOM entry points preserve sparse bath actions", arguments: heomShifts, 0..<4)
    func entryPoints(shift: HEOM.ShiftType, api: Int) throws {
        let d = 5
        let problem = DensityMatrixProblem(sparseFixtureProblem(d, sparseH: true, sparseC: true))
        let propagation = hopsPropagation(end: 0.12, maximumStep: 0.01, output: .times([0.04, 0.08, 0.12]))
        let insertion = sparseFixtureOperator(d, sparse: true)
        func run(sparse: Bool, workers: Int) throws -> [Complex<Double>] {
            let config = HEOM.Configuration(
                hierarchy: .init(environment: .init(
                    couplingOperators: heomBathFixture(d, sparse: sparse, kind: 2), bath: hopsFixtureModel()),
                    truncation: .maximumTier(2)),
                shiftType: shift, parallelism: .maximumWorkers(workers))
            var values: [Complex<Double>] = []
            switch api {
            case 0:
                try HEOM.solve(problem: problem, configuration: config, propagation: propagation) { _, rho in
                    for i in 0..<d * d { values.append(rho.elements[i]) }
                    return .proceed
                }
            case 1:
                try HEOM.solveWithHierarchy(problem: problem, configuration: config, propagation: propagation) { _, view in
                    for ado in 0..<view.count {
                        view.withState(at: ado) { rho in
                            for i in 0..<d { for j in 0..<d { values.append(rho[i, j]) } }
                        }
                    }
                    return .proceed
                }
            case 2:
                try HEOM.solveTwoTimeCorrelation(problem: problem, configuration: config,
                    request: .init(insertionTime: 0.04, insertion: .right(insertion), observable: insertion),
                    propagation: propagation) { _, value in
                        values.append(value)
                        return .proceed
                    }
            default:
                try HEOM.solveMultiTimeOrderedCorrelation(problem: problem, configuration: config,
                    request: .init(insertions: [
                        .init(time: 0.04, insertion: .right(insertion)),
                        .init(time: 0.08, insertion: .left(insertion)),
                    ], observable: insertion), propagation: propagation) { _, value in
                        values.append(value)
                        return .proceed
                    }
            }
            #expect(!values.isEmpty)
            return values
        }
        let expected = try run(sparse: false, workers: 1)
        for workers in [1, 3] {
            expectHOPSClose(try run(sparse: true, workers: workers), expected, tolerance: 2e-10)
        }
    }
}
