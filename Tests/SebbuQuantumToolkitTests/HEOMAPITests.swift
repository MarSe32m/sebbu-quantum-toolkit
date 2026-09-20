// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Test("Zero-bath HEOM reduces to GKSL with dynamic Hamiltonian and Lindblad channels", arguments: heomShifts)
func heomZeroBathGKSL(shift: HEOM.ShiftType) throws {
    let h = ClosureHamiltonian { t, matrix in
        matrix[0, 0] = 0
        matrix[1, 1] = Complex(0.1 * t)
        matrix[0, 1] = Complex(0.3 * Double.cos(t))
        matrix[1, 0] = matrix[0, 1]
    }
    let collapse = TimeDependentOperator.generatedDense(
        .init { t, output in
            output.zeroElements()
            output[0, 1] = Complex(1 + 0.2 * t)
        })
    let problem = PureStateProblem(
        initialState: Vector<Complex<Double>>([0, 1]),
        system: QuantumSystem(dimension: 2, hamiltonian: h),
        markovianChannels: [.init(rate: .generated { t in 0.2 + t * t }, collapseOperator: collapse)])
    let configuration = heomConfiguration(shift: shift, model: .zero(channelCount: 1))
    let propagation = hopsPropagation(
        end: 1.1, start: 0.3, tolerance: 1e-10, output: .times([0.3, 0.41, 0.8, 1.1]))
    var expected: [[Complex<Double>]] = []
    try GKSL.solve(problem: problem, propagation: propagation) { _, rho in
        expected.append((0..<4).map { rho.elements[$0] })
        return .proceed
    }
    var index = 0
    try HEOM.CPUEngine().solve(problem: problem, configuration: configuration, propagation: propagation) {
        _, rho in
        expectHOPSClose((0..<4).map { rho.elements[$0] }, expected[index], tolerance: 2e-8)
        index += 1
        return .proceed
    }
    #expect(index == expected.count)
}

@Test("Dynamic bath operators are materialized at every integration stage", arguments: heomShifts)
func heomDynamicBath(shift: HEOM.ShiftType) throws {
    let base = hopsMatrix([0, 0, 0, 1])
    let dynamic = TimeDependentOperator.generatedDense(
        .init { t, out in
            out.zeroElements()
            out[1, 1] = Complex(1 + t)
        })
    let model = heomSingleBath()
    let config = HEOM.Configuration(
        hierarchy: .init(
            environment: .init(couplingOperator: dynamic, bath: model), truncation: .maximumTier(10)),
        shiftType: shift)
    let problem = DensityMatrixProblem(hopsProblem())
    let value = try heomFinal(
        problem: problem, configuration: config, propagation: hopsPropagation(end: 0.7, tolerance: 1e-10))
    // An independently constructed, time-dependent explicit oscillator checks
    // both Lambda(t) and M(t), including the centered mean at each RK stage.
    let n = 6
    let original = heomPseudomodeProblem(
        h: hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)]), coupling: base,
        rho: problem.initialState, oscillatorDimension: n)
    let free = heomPseudomodeProblem(
        h: hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)]), coupling: hopsMatrix([0, 0, 0, 0]),
        rho: problem.initialState, oscillatorDimension: n)
    var h0 = UniqueMatrix<Complex<Double>>.zeros(rows: 2 * n, columns: 2 * n)
    var hf = UniqueMatrix<Complex<Double>>.zeros(rows: 2 * n, columns: 2 * n)
    original.system.hamiltonian.hamiltonian(t: 0, into: &h0)
    free.system.hamiltonian.hamiltonian(t: 0, into: &hf)
    let full = Matrix(copying: h0)
    let uncoupled = Matrix(copying: hf)
    let h = ClosureHamiltonian { t, out in
        for i in 0..<(4 * n * n) {
            out.elements[i] = full.elements[i] + t * (full.elements[i] - uncoupled.elements[i])
        }
    }
    let joint = DensityMatrixProblem(
        initialState: original.initialState, system: QuantumSystem(dimension: 2 * n, hamiltonian: h),
        markovianChannels: original.markovianChannels)
    var expected = Array(repeating: Complex<Double>.zero, count: 4)
    try GKSL.solve(problem: joint, propagation: hopsPropagation(end: 0.7, tolerance: 1e-10)) { _, rho in
        for i in 0..<2 {
            for j in 0..<2 { for k in 0..<n { expected[i * 2 + j] += rho[i * n + k, j * n + k] } }
        }
        return .proceed
    }
    expectHOPSClose(value, expected, tolerance: 3e-8)
}

@Test("HEOM schedules and observer termination are consistent across all four entry points")
func heomSchedulesAndStops() throws {
    let config = heomConfiguration(model: .zero(channelCount: 1))
    let problem = hopsProblem(hopsMatrix([0, 0, 0, 0]))
    let identity = TimeDependentOperator.constant(heomIdentity(2))
    for api in 0..<4 {
        var times: [Double] = []
        let propagation = hopsPropagation(end: 1, maximumStep: 1, output: .times([0, 0.25, 1]))
        let result: PropagationRunSummary
        switch api {
        case 0:
            result = try HEOM.solve(problem: problem, configuration: config, propagation: propagation) {
                t, _ in
                times.append(t)
                return t >= 0.25 ? .stop : .proceed
            }
        case 1:
            result = try HEOM.CPUEngine().solveWithHierarchy(
                problem: problem, configuration: config, propagation: propagation
            ) { t, _ in
                times.append(t)
                return t >= 0.25 ? .stop : .proceed
            }
        case 2:
            result = try HEOM.solveTwoTimeCorrelation(
                problem: problem, configuration: config,
                request: .init(insertionTime: 0, insertion: .left(identity), observable: identity),
                propagation: propagation
            ) { t, _ in
                times.append(t)
                return t >= 0.25 ? .stop : .proceed
            }
        default:
            result = try HEOM.CPUEngine().solveMultiTimeOrderedCorrelation(
                problem: problem, configuration: config,
                request: .init(
                    insertions: [.init(time: 0, insertion: .left(identity))], observable: identity),
                propagation: propagation
            ) { t, _ in
                times.append(t)
                return t >= 0.25 ? .stop : .proceed
            }
        }
        #expect(times == [0, 0.25])
        #expect(result.finalTime == 0.25 && result.endReason == .stoppedByObserver)
    }
    for schedule in [OutputSchedule.uniform(step: 0.2), .everyAcceptedStep, .final, .times([])] {
        var times: [Double] = []
        try HEOM.solve(
            problem: problem, configuration: config,
            propagation: hopsPropagation(end: 1, maximumStep: 0.2, output: schedule)
        ) { t, _ in
            times.append(t)
            return .proceed
        }
        switch schedule {
        case .uniform: #expect(times.count == 6)
        case .everyAcceptedStep:
            #expect(times.first! > 0 && times.last! == 1 && times.count == Set(times).count)
        case .final: #expect(times == [1])
        case .times: #expect(times.isEmpty)
        }
    }
}

@Test("Equal-time, endpoint and zero-duration correlations are emitted after insertion")
func heomEndpointCorrelations() throws {
    let problem = PureStateProblem(
        initialState: Vector<Complex<Double>>([0, 1]), system: QuantumSystem(hopsMatrix([0, 0, 0, 0])))
    let config = heomConfiguration(shift: .meanField)
    let request = TwoTimeCorrelationRequest(
        insertionTime: 0.5, insertion: .left(.constant(hopsMatrix([0, 1, 0, 0]))),
        observable: .constant(hopsMatrix([0, 0, 1, 0])))
    for start in [0.0, 0.5] {
        for schedule in [OutputSchedule.final, .times([0.5]), .everyAcceptedStep] {
            var values: [Complex<Double>] = []
            let result = try HEOM.solveTwoTimeCorrelation(
                problem: problem, configuration: config, request: request,
                propagation: hopsPropagation(end: 0.5, start: start, output: schedule)
            ) { t, value in
                #expect(t == 0.5)
                values.append(value)
                return .proceed
            }
            #expect(result.finalTime == 0.5)
            if case .everyAcceptedStep = schedule {
                #expect(values.isEmpty)
            } else {
                expectHOPSClose(values, [1], tolerance: 1e-12)
            }
        }
    }
    var callbacks = 0
    let result = try HEOM.solveWithHierarchy(
        problem: problem, configuration: config,
        propagation: hopsPropagation(end: 0.5, start: 0.5, output: .final)
    ) { _, h in
        #expect(h.count == config.hierarchy.count)
        for i in 1..<h.count { h.withState(at: i) { #expect($0[1, 1] == .zero) } }
        callbacks += 1
        return .stop
    }
    #expect(callbacks == 1 && result.endReason == .stoppedByObserver)
}

@Test("HEOM progress includes preparation and finishes cleanly on observer stop")
func heomProgress() throws {
    let capture = ProgressCapture()
    var propagation = hopsPropagation(end: 1, maximumStep: 0.02, output: .times([0.7, 1]))
    propagation.progress = capture.reporting()
    let identity = TimeDependentOperator.constant(heomIdentity(2))
    let result = try HEOM.solveMultiTimeOrderedCorrelation(
        problem: hopsProblem(), configuration: heomConfiguration(shift: .meanField),
        request: .init(
            insertions: [
                .init(time: 0.3, insertion: .left(identity)), .init(time: 0.6, insertion: .right(identity)),
            ], observable: identity),
        propagation: propagation
    ) { _, _ in .stop }
    #expect(result.finalTime == 0.7)
    #expect(capture.percentages.first == 0 && capture.percentages.last == 70)
    #expect(capture.percentages.count > 10)
    #expect(capture.newlineCount == 1)
}
