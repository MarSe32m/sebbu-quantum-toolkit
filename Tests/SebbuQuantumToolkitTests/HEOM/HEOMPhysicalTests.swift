// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HEOM Physical tests")
struct HEOMPhysicalTests {
    @Test("Both HEOM variants recover analytic independent-boson coherence", arguments: heomShifts)
    func heomIndependentBoson(shift: HEOM.ShiftType) throws {
        let w = Complex<Double>(0.6, 1.2)
        let strength = 0.16
        let energy = 0.7
        let rho = hopsMatrix([Complex(0.5), Complex(0.5), Complex(0.5), Complex(0.5)])
        let problem = DensityMatrixProblem(
            initialState: rho, system: QuantumSystem(hopsMatrix([0, 0, 0, Complex(energy)])))
        let times = [0.0, 0.07, 0.25, 0.7, 1.4, 2.0]
        try HEOM.solve(
            problem: problem, configuration: heomConfiguration(shift: shift, depth: 12),
            propagation: hopsPropagation(end: 2, tolerance: 1e-11, output: .times(times))
        ) { time, result in
            let f = strength * (time / w - (1 - Complex<Double>.exp(-w * time)) / (w * w))
            let expected = 0.5 * Complex<Double>.exp(Complex(0, -energy * time) - f)
            #expect((result[1, 0] - expected).length < 2e-9)
            #expect((result[0, 1] - expected.conjugate).length < 2e-9)
            #expect((result[0, 0] - 0.5).length < 1e-12)
            #expect((result[1, 1] - 0.5).length < 1e-12)
            return .proceed
        }
    }
    
    @Test(
        "HEOM matches an explicit damped oscillator for Hermitian and non-Hermitian coupling",
        arguments: heomShifts, [false, true])
    func heomPseudomodeDynamics(shift: HEOM.ShiftType, nonHermitian: Bool) throws {
        let h = hopsMatrix([0, Complex(0.35), Complex(0.35), Complex(0.2)])
        let l = nonHermitian ? hopsMatrix([0, 1, 0, 0]) : hopsMatrix([0, 0, 0, 1])
        let rho = hopsMatrix([Complex(0.4), Complex(0, 0.2), Complex(0, -0.2), Complex(0.6)])
        let markov = MarkovianChannel(
            rate: .constant(0.13), collapseOperator: .constant(hopsMatrix([0, 1, 0, 0])))
        let problem = DensityMatrixProblem(
            initialState: rho, system: QuantumSystem(h), markovianChannels: [markov])
        let times = [0.0, 0.13, 0.5, 1.1]
        let propagation = hopsPropagation(end: 1.1, tolerance: 1e-10, output: .times(times))
        var reference: [[Complex<Double>]] = []
        let n = 7
        try GKSL.solve(
            problem: heomPseudomodeProblem(h: h, coupling: l, rho: rho, oscillatorDimension: n, rate: 0.13),
            propagation: propagation
        ) { _, joint in
            var reduced = Array(repeating: Complex<Double>.zero, count: 4)
            for i in 0..<2 {
                for j in 0..<2 { for k in 0..<n { reduced[i * 2 + j] += joint[i * n + k, j * n + k] } }
            }
            reference.append(reduced)
            return .proceed
        }
        var index = 0
        try HEOM.solveWithHierarchy(
            problem: problem, configuration: heomConfiguration(shift: shift, depth: 10, operators: [l]),
            propagation: propagation
        ) { _, hierarchy in
            hierarchy.withPhysicalState { root in
                expectHOPSClose(
                    [root[0, 0], root[0, 1], root[1, 0], root[1, 1]], reference[index], tolerance: 3e-8)
                #expect((root[0, 0] + root[1, 1] - 1).length < 1e-11)
                #expect((root[0, 1] - root[1, 0].conjugate).length < 1e-11)
                #expect(root[0, 0].real >= 0 && root[1, 1].real >= 0)
            }
            index += 1
            return .proceed
        }
        #expect(index == times.count)
    }
    
    @Test("Centering annihilates auxiliary occupation for a coupling eigenstate")
    func heomCenteredEigenstate() throws {
        let problem = DensityMatrixProblem(
            initialState: hopsMatrix([0, 0, 0, 1]), system: QuantumSystem(hopsMatrix([0, 0, 0, 0])))
        var maximumAuxiliary = 0.0
        try HEOM.solveWithHierarchy(
            problem: problem, configuration: heomConfiguration(shift: .meanField, depth: 3),
            propagation: hopsPropagation(end: 2)
        ) { _, hierarchy in
            for i in 1..<hierarchy.count {
                hierarchy.withState(at: i) { ado in
                    for row in 0..<2 {
                        for col in 0..<2 { maximumAuxiliary = max(maximumAuxiliary, ado[row, col].length) }
                    }
                }
            }
            return .proceed
        }
        #expect(maximumAuxiliary < 1e-13)
        var ordinaryAuxiliary = 0.0
        try HEOM.solveWithHierarchy(
            problem: problem, configuration: heomConfiguration(depth: 3),
            propagation: hopsPropagation(end: 2)
        ) { _, hierarchy in
            hierarchy.withState(at: 1) { ordinaryAuxiliary = $0[1, 1].length }
            return .proceed
        }
        #expect(ordinaryAuxiliary > 0.01)
    }
    
    @Test("HEOM converges with hierarchy depth and preserves ket-bra adjoints", arguments: heomShifts)
    func heomConvergenceAndAdjoints(shift: HEOM.ShiftType) throws {
        let problem = DensityMatrixProblem(hopsProblem())
        let high = try heomFinal(problem: problem, configuration: heomConfiguration(shift: shift, depth: 10))
        let low = try heomFinal(problem: problem, configuration: heomConfiguration(shift: shift, depth: 1))
        let mid = try heomFinal(problem: problem, configuration: heomConfiguration(shift: shift, depth: 5))
        let errorLow = zip(low, high).map { ($0 - $1).length }.max()!
        let errorMid = zip(mid, high).map { ($0 - $1).length }.max()!
        #expect(errorLow > 1e-6)
        #expect(errorMid < errorLow * 0.01)
        let config = heomConfiguration(shift: shift, depth: 4)
        let indices = heomIndices(config.hierarchy)
        let ids = Dictionary(uniqueKeysWithValues: indices.enumerated().map { ($0.element, $0.offset) })
        try HEOM.solveWithHierarchy(problem: problem, configuration: config, propagation: hopsPropagation(end: 1))
        { _, h in
            for i in indices.indices {
                let reversed = [indices[i][1], indices[i][0]]
                h.withState(at: i) { a in
                    h.withState(at: ids[reversed]!) { b in
                        for row in 0..<2 {
                            for col in 0..<2 { #expect((a[row, col] - b[col, row].conjugate).length < 1e-11) }
                        }
                    }
                }
            }
            return .proceed
        }
    }
    
    @Test("Correlated multi-pole HEOM agrees with ordinary and centered coordinates")
    func heomCorrelatedCentering() throws {
        let problem = DensityMatrixProblem(hopsProblem())
        let model = hopsFixtureModel()
        let ordinary = try heomFinal(
            problem: problem,
            configuration: heomConfiguration(depth: 6, model: model, operators: hopsFixtureOperators),
            propagation: hopsPropagation(end: 0.3, tolerance: 1e-10))
        let centered = try heomFinal(
            problem: problem,
            configuration: heomConfiguration(
                shift: .meanField, depth: 6, model: model, operators: hopsFixtureOperators),
            propagation: hopsPropagation(end: 0.3, tolerance: 1e-10))
        expectHOPSClose(ordinary, centered, tolerance: 2e-8)
    }
    
    @Test(
        "Centered expectations divide by the guide trace without renormalizing the state", arguments: heomShifts)
    func heomTraceScale(shift: HEOM.ShiftType) throws {
        let problem = DensityMatrixProblem(hopsProblem())
        var rho = problem.initialState
        for i in rho.elements.indices { rho.elements[i] *= 2.5 }
        let scaled = DensityMatrixProblem(initialState: rho, system: problem.system)
        let config = heomConfiguration(shift: shift)
        let first = try heomFinal(problem: problem, configuration: config)
        let second = try heomFinal(problem: scaled, configuration: config)
        expectHOPSClose(first.map { 2.5 * $0 }, second, tolerance: 2e-8)
    }
}
