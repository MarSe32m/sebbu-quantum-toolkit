// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("Explicit CSR solver equivalence")
struct SparseSolverTests {
    @Test(arguments: [2, 7], [false, true])
    func gkslAndMCWF(d: Int, expansion: Bool) {
        let dense = sparseFixtureProblem(d, sparseH: false, sparseC: false, expansion: expansion)
        for (h, c) in [(true, false), (false, true), (true, true)] {
            let sparse = sparseFixtureProblem(d, sparseH: h, sparseC: c, expansion: expansion)
            var gd = GKSL.CPUEngine.GKSLRightHandSide(DensityMatrixProblem(dense))
            var gs = GKSL.CPUEngine.GKSLRightHandSide(DensityMatrixProblem(sparse))
            let rho = Matrix(
                elements: (0..<d * d).map { Complex(0.01 * Double($0 + 1), -0.02 * Double($0)) }, rows: d,
                columns: d)
            let y = GKSL.CPUEngine.DensityMatrixState(rho)
            var a = GKSL.CPUEngine.DensityMatrixState(dimension: d)
            var b = GKSL.CPUEngine.DensityMatrixState(dimension: d)
            gd.evaluate(t: 0.13, y: y, dy: &a)
            gs.evaluate(t: 0.13, y: y, dy: &b)
            expectHOPSClose(
                (0..<d * d).map { a.densityMatrix.elements[$0] },
                (0..<d * d).map { b.densityMatrix.elements[$0] })
            var md = MCWF.CPUEngine.RightHandSide(
                hamiltonian: dense.system.hamiltonian, channels: dense.markovianChannels, dimension: d)
            var ms = MCWF.CPUEngine.RightHandSide(
                hamiltonian: sparse.system.hamiltonian, channels: sparse.markovianChannels, dimension: d)
            let psi = MCWF.CPUEngine.TrajectoryState(sparseFixtureState(d))
            var ma = MCWF.CPUEngine.TrajectoryState(dimension: d)
            var mb = MCWF.CPUEngine.TrajectoryState(dimension: d)
            md.evaluate(t: 0.13, y: psi, dy: &ma)
            ms.evaluate(t: 0.13, y: psi, dy: &mb)
            expectHOPSClose((0..<d).map { ma.wavefunction[$0] }, (0..<d).map { mb.wavefunction[$0] })
            #expect(abs(ma.cumulativeHazard - mb.cumulativeHazard) < 1e-12)
        }
    }

    @Test(arguments: [UInt64(1), 17, 99], [false, true])
    func jumps(seed: UInt64, expansion: Bool) throws {
        let d = 7
        let (_, matrix) = explicitSparseFixture(d)
        let second = MarkovianChannel(
            rate: .constant(0.17), collapseOperator: .constant(matrix.conjugateTranspose))
        let dense = sparseFixtureProblem(d, sparseH: false, sparseC: false, expansion: expansion)
        let sparse = sparseFixtureProblem(d, sparseH: true, sparseC: true, expansion: expansion)
        var a = MCWF.CPUEngine.JumpWorkspace(channels: dense.markovianChannels + [second], dimension: d)
        var b = MCWF.CPUEngine.JumpWorkspace(channels: sparse.markovianChannels + [second], dimension: d)
        var x = MCWF.CPUEngine.TrajectoryState(sparseFixtureState(d))
        var y = MCWF.CPUEngine.TrajectoryState(sparseFixtureState(d))
        var rngA = TrajectoryRandomNumberGenerator(seed: seed)
        var rngB = TrajectoryRandomNumberGenerator(seed: seed)
        try a.applyJump(at: 0.13, state: &x, using: &rngA)
        try b.applyJump(at: 0.13, state: &y, using: &rngB)
        for i in a.weights.indices { #expect(abs(a.weights[i] - b.weights[i]) < 1e-12) }
        expectHOPSClose((0..<d).map { x.wavefunction[$0] }, (0..<d).map { y.wavefunction[$0] })

        var ca = MCWF.CPUEngine.CorrelationJumpWorkspace(
            channels: dense.markovianChannels + [second], dimension: d)
        var cb = MCWF.CPUEngine.CorrelationJumpWorkspace(
            channels: sparse.markovianChannels + [second], dimension: d)
        var cx = MCWF.CPUEngine.CorrelationState(guideAndDyad: sparseFixtureState(d))
        var cy = MCWF.CPUEngine.CorrelationState(guideAndDyad: sparseFixtureState(d))
        for i in 0..<d {
            let ket = Complex(0.02 * Double(i + 1), -0.13)
            let bra = Complex(-0.03, 0.07 * Double(d - i))
            cx.ket[i] = ket; cy.ket[i] = ket
            cx.bra[i] = bra; cy.bra[i] = bra
        }
        try ca.applyJump(at: 0.13, state: &cx, using: &rngA)
        try cb.applyJump(at: 0.13, state: &cy, using: &rngB)
        for i in ca.weights.indices { #expect(abs(ca.weights[i] - cb.weights[i]) < 1e-12) }
        expectHOPSClose((0..<d).map { cx.guide[$0] }, (0..<d).map { cy.guide[$0] })
        expectHOPSClose((0..<d).map { cx.ket[$0] }, (0..<d).map { cy.ket[$0] })
        expectHOPSClose((0..<d).map { cx.bra[$0] }, (0..<d).map { cy.bra[$0] })
    }

    @Test(arguments: [QSD.EquationType.linear, .nonLinear, .nonLinearNormalized], [false, true])
    func qsd(equation: QSD.EquationType, expansion: Bool) {
        let d = 7
        let dense = sparseFixtureProblem(d, sparseH: false, sparseC: false, expansion: expansion)
        let sparse = sparseFixtureProblem(d, sparseH: true, sparseC: true, expansion: expansion)
        var a = QSD.CPUEngine.QSDRightHandSide(dense, equationType: equation, seed: 17, trajectoryID: 3)
        var b = QSD.CPUEngine.QSDRightHandSide(sparse, equationType: equation, seed: 17, trajectoryID: 3)
        let y = QSD.CPUEngine.StateVector(state: sparseFixtureState(d))
        var da = QSD.CPUEngine.StateVector(dimension: d)
        var db = QSD.CPUEngine.StateVector(dimension: d)
        a.drift(t: 0.13, y: y, into: &da)
        b.drift(t: 0.13, y: y, into: &db)
        expectHOPSClose((0..<d).map { da.state[$0] }, (0..<d).map { db.state[$0] })
        a.diffusion(t: 0.13, y: y, channel: 0, into: &da)
        b.diffusion(t: 0.13, y: y, channel: 0, into: &db)
        expectHOPSClose((0..<d).map { da.state[$0] }, (0..<d).map { db.state[$0] })
    }

    @Test(arguments: 0..<6, [false, true])
    func hops(variant: Int, expansion: Bool) throws {
        for d in [2, 7] {
            let dense = sparseFixtureProblem(d, sparseH: false, sparseC: false, expansion: expansion)
            let sparse = sparseFixtureProblem(d, sparseH: true, sparseC: true, expansion: expansion)
            func config(_ csr: Bool) -> HOPS.Configuration {
                .init(
                    hierarchy: .init(
                        environment: .init(
                            couplingOperator: sparseFixtureOperator(d, sparse: csr, expansion: expansion),
                            bath: heomSingleBath()),
                        truncation: .maximumTier(3)), equationType: hopsEquationTypes[variant / 2],
                    shiftType: variant % 2 == 0 ? .none : .meanField, noiseStepSize: 0.01)
            }
            let cd = config(false)
            let cs = config(true)
            let pd = try HOPS.CPUEngine.Preparation(
                problem: dense, configuration: cd, propagation: hopsPropagation())
            let ps = try HOPS.CPUEngine.Preparation(
                problem: sparse, configuration: cs, propagation: hopsPropagation())
            if d > 2 && !expansion { #expect({ ps.bathChannels[0].op.isSparse }()) }
            let n = cd.hierarchy.count * 3
            var y = HOPS.CPUEngine.State(dimension: d, hierarchyCount: n, shiftCount: pd.shiftCount)
            var a = HOPS.CPUEngine.State(dimension: d, hierarchyCount: n, shiftCount: pd.shiftCount)
            var b = HOPS.CPUEngine.State(dimension: d, hierarchyCount: n, shiftCount: pd.shiftCount)
            for i in 0..<n * d { y.amplitudes.elements[i] = Complex(0.1 + 0.002 * Double(i), -0.03) }
            for i in 0..<y.shifts.count { y.shifts[i] = Complex(0.04, -0.02) }
            var rd = HOPS.CPUEngine.RightHandSide(
                hamiltonian: dense.system.hamiltonian, preparation: pd, seed: 17, trajectoryID: 3)
            var rs = HOPS.CPUEngine.RightHandSide(
                hamiltonian: sparse.system.hamiltonian, preparation: ps, seed: 17, trajectoryID: 3)
            rd.physicalNoise[0] = Complex(0.12, -0.07)
            rs.physicalNoise[0] = Complex(0.12, -0.07)
            rd.evaluateWithCurrentNoise(t: 0.13, y: y, into: &a)
            rs.evaluateWithCurrentNoise(t: 0.13, y: y, into: &b)
            expectHOPSClose(
                (0..<n * d).map { a.amplitudes.elements[$0] }, (0..<n * d).map { b.amplitudes.elements[$0] })
            expectHOPSClose(
                (0..<a.shifts.count).map { a.shifts[$0] }, (0..<b.shifts.count).map { b.shifts[$0] })
            rd.diffusion(t: 0.13, y: y, channel: 0, into: &a)
            rs.diffusion(t: 0.13, y: y, channel: 0, into: &b)
            expectHOPSClose(
                (0..<n * d).map { a.amplitudes.elements[$0] }, (0..<n * d).map { b.amplitudes.elements[$0] })
        }
    }

    @Test(arguments: heomShifts, [false, true])
    func heom(shift: HEOM.ShiftType, expansion: Bool) throws {
        let d = 5
        let dense = DensityMatrixProblem(
            sparseFixtureProblem(d, sparseH: false, sparseC: false, expansion: expansion))
        let sparse = DensityMatrixProblem(
            sparseFixtureProblem(d, sparseH: true, sparseC: true, expansion: expansion))
        for workers in [1, 3] {
            func config(_ csr: Bool) -> HEOM.Configuration {
                var c = HEOM.Configuration(
                    hierarchy: .init(
                        environment: .init(
                            couplingOperator: sparseFixtureOperator(d, sparse: csr, expansion: expansion),
                            bath: heomSingleBath()),
                        truncation: .maximumTier(3)), shiftType: shift)
                c.parallelism = .maximumWorkers(workers)
                return c
            }
            let cd = config(false)
            let cs = config(true)
            let fa = HEOM.CPUEngine.Failure()
            let fb = HEOM.CPUEngine.Failure()
            var rd = try HEOM.CPUEngine.RightHandSide(
                problem: dense, configuration: cd, failure: fa, copies: 3)
            var rs = try HEOM.CPUEngine.RightHandSide(
                problem: sparse, configuration: cs, failure: fb, copies: 3)
            let shifts = shift == .none ? 0 : 1
            var y = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: cd.hierarchy.count, shiftCount: shifts, copies: 3)
            var a = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: cd.hierarchy.count, shiftCount: shifts, copies: 3)
            var b = HEOM.CPUEngine.State(
                dimension: d, hierarchyCount: cd.hierarchy.count, shiftCount: shifts, copies: 3)
            let count = y.ados.rows * y.ados.columns
            for i in 0..<count { y.ados.elements[i] = Complex(0.1 + 0.002 * Double(i), -0.03) }
            for i in 0..<shifts { y.shifts[i] = Complex(0.04, -0.02) }
            rd.evaluate(t: 0.13, y: y, dy: &a)
            rs.evaluate(t: 0.13, y: y, dy: &b)
            try fa.check()
            try fb.check()
            expectHOPSClose((0..<count).map { a.ados.elements[$0] }, (0..<count).map { b.ados.elements[$0] })
            expectHOPSClose((0..<shifts).map { a.shifts[$0] }, (0..<shifts).map { b.shifts[$0] })
        }
    }

    @Test func coefficientsAreStagedBeforeHierarchyActions() throws {
        let d = 5
        let calls = Mutex(0)
        let (csr, _) = explicitSparseFixture(d)
        let op = TimeDependentOperator.linearCombination(
            .init(
                coefficients: [
                    .generated { t in
                        calls.withLock { $0 += 1 }
                        return Complex(1 + t, -t)
                    }
                ], operators: [.init(csr)]))
        let problem = PureStateProblem(
            initialState: sparseFixtureState(d), system: .init(dimension: d, hamiltonian: op))
        let env = BathEnvironment(couplingOperator: .constant(csr), bath: heomSingleBath())
        let hc = HOPS.Configuration(
            hierarchy: .init(environment: env, truncation: .maximumTier(12)), equationType: .linear)
        let prep = try HOPS.CPUEngine.Preparation(
            problem: problem, configuration: hc, propagation: hopsPropagation())
        var rhs = HOPS.CPUEngine.RightHandSide(hamiltonian: op, preparation: prep, seed: 1, trajectoryID: 0)
        var y = HOPS.CPUEngine.State(
            dimension: d, hierarchyCount: 3 * hc.hierarchy.count, shiftCount: prep.shiftCount)
        var dy = HOPS.CPUEngine.State(
            dimension: d, hierarchyCount: 3 * hc.hierarchy.count, shiftCount: prep.shiftCount)
        y.amplitudes.elements[0] = .one
        calls.withLock { $0 = 0 }
        for _ in 0..<2 { rhs.evaluateWithCurrentNoise(t: 0.13, y: y, into: &dy) }
        #expect(calls.withLock { $0 } == 2)
        var ec = HEOM.Configuration(hierarchy: .init(environment: env, truncation: .maximumTier(12)))
        ec.parallelism = .maximumWorkers(3)
        let failure = HEOM.CPUEngine.Failure()
        var erhs = try HEOM.CPUEngine.RightHandSide(
            problem: DensityMatrixProblem(problem), configuration: ec, failure: failure, copies: 3)
        var ey = HEOM.CPUEngine.State(
            dimension: d, hierarchyCount: ec.hierarchy.count, shiftCount: 0, copies: 3)
        var edy = HEOM.CPUEngine.State(
            dimension: d, hierarchyCount: ec.hierarchy.count, shiftCount: 0, copies: 3)
        ey.ados.elements[0] = .one
        calls.withLock { $0 = 0 }
        for _ in 0..<2 { erhs.evaluate(t: 0.13, y: ey, dy: &edy) }
        try failure.check()
        #expect(calls.withLock { $0 } == 2)
    }
}
