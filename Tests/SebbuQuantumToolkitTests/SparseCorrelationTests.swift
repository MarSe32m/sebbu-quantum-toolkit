// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("Sparse public propagation and correlation paths", .serialized)
struct SparseCorrelationTests {
    @Test(arguments: 0..<5, [false, true])
    func correlations(method: Int, multi: Bool) throws {
        for variant in 0..<((method == 2 || method == 3) ? 3 : 1) {
            func run(_ sparse: Bool) throws -> [Complex<Double>] {
                let d = 3
                let problem = sparseFixtureProblem(d, sparseH: sparse, sparseC: sparse, expansion: true)
                let op = sparseFixtureOperator(d, sparse: sparse, expansion: true)
                let request = TwoTimeCorrelationRequest(
                    insertionTime: 0.04, insertion: .right(op), observable: op)
                let multiple = MultiTimeOrderedCorrelationRequest(
                    insertions: [
                        .init(time: 0.04, insertion: .right(op)), .init(time: 0.08, insertion: .left(op)),
                    ], observable: op)
                let propagation = hopsPropagation(
                    end: 0.12, maximumStep: 0.005, output: .times([0.04, 0.08, 0.12]))
                let execution = TrajectoryExecution(trajectories: 2, seed: 713, parallelism: .serial)
                let env = BathEnvironment(couplingOperator: op, bath: heomSingleBath(strength: 0.01))
                let hc = HOPS.Configuration(
                    hierarchy: .init(environment: env, truncation: .maximumTier(2)),
                    equationType: hopsEquationTypes[variant], shiftType: .meanField, noiseStepSize: 0.005)
                let ec = HEOM.Configuration(
                    hierarchy: .init(environment: env, truncation: .maximumTier(2)), shiftType: .meanField)
                let qc = QSD.Configuration(
                    equationType: [QSD.EquationType.linear, .nonLinear, .nonLinearNormalized][variant])
                var values: [Complex<Double>] = []
                switch (method, multi) {
                case (0, false):
                    try GKSL.solveTwoTimeCorrelation(
                        problem: problem, request: request, propagation: propagation
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (0, true):
                    try GKSL.solveMultiTimeOrderedCorrelation(
                        problem: problem, request: multiple, propagation: propagation
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (1, false):
                    try MCWF.solveTwoTimeCorrelation(
                        problem: problem, request: request, propagation: propagation, execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (1, true):
                    try MCWF.solveMultiTimeOrderedCorrelation(
                        problem: problem, request: multiple, propagation: propagation, execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (2, false):
                    try QSD.solveTwoTimeCorrelation(
                        problem: problem, configuration: qc, request: request, propagation: propagation,
                        execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (2, true):
                    try QSD.solveMultiTimeOrderedCorrelation(
                        problem: problem, configuration: qc, request: multiple, propagation: propagation,
                        execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (3, false):
                    try HOPS.solveTwoTimeCorrelation(
                        problem: problem, configuration: hc, request: request, propagation: propagation,
                        execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (3, true):
                    try HOPS.solveMultiTimeOrderedCorrelation(
                        problem: problem, configuration: hc, request: multiple, propagation: propagation,
                        execution: execution
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                case (4, false):
                    try HEOM.solveTwoTimeCorrelation(
                        problem: problem, configuration: ec, request: request, propagation: propagation
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                default:
                    try HEOM.solveMultiTimeOrderedCorrelation(
                        problem: problem, configuration: ec, request: multiple, propagation: propagation
                    ) { _, v in
                        values.append(v)
                        return .proceed
                    }
                }
                #expect(!values.isEmpty)
                return values
            }
            expectHOPSClose(try run(false), try run(true), tolerance: 2e-10)
        }
    }

    @Test(arguments: 0..<7)
    func seededTrajectories(method: Int) throws {
        func run(_ sparse: Bool) throws -> [Complex<Double>] {
            let d = 5
            let problem = sparseFixtureProblem(d, sparseH: sparse, sparseC: sparse)
            let propagation = hopsPropagation(end: 0.1, maximumStep: 0.005, output: .times([0, 0.05, 0.1]))
            var values: [Complex<Double>] = []
            if method == 0 {
                try MCWF.solveTrajectory(
                    problem: problem, propagation: propagation, seed: 153, trajectoryID: 2
                ) { _, state in
                    for i in 0..<state.count { values.append(state[i]) }
                    return .proceed
                }
            } else if method < 4 {
                try QSD.solveTrajectory(
                    problem: problem,
                    configuration: .init(
                        equationType: [QSD.EquationType.linear, .nonLinear, .nonLinearNormalized][method - 1]),
                    propagation: propagation, seed: 153, trajectoryID: 2
                ) { _, state in
                    for i in 0..<state.count { values.append(state[i]) }
                    return .proceed
                }
            } else {
                let env = BathEnvironment(
                    couplingOperator: sparseFixtureOperator(d, sparse: sparse),
                    bath: heomSingleBath(strength: 0.01))
                let config = HOPS.Configuration(
                    hierarchy: .init(environment: env, truncation: .maximumTier(2)),
                    equationType: hopsEquationTypes[method - 4], shiftType: .meanField, noiseStepSize: 0.005)
                try HOPS.solveTrajectory(
                    problem: problem, configuration: config, propagation: propagation, seed: 153,
                    trajectoryID: 2
                ) { _, state in
                    for i in 0..<state.count { values.append(state[i]) }
                    return .proceed
                }
            }
            #expect(values.count == 3 * d)
            return values
        }
        expectHOPSClose(try run(false), try run(true), tolerance: 2e-10)
    }
}
