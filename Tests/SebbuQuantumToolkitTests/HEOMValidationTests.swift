// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Test("HEOM validates two-time requests with the existing typed errors")
func heomTwoTimeValidation() throws {
    let problem = hopsProblem()
    let config = heomConfiguration()
    let op = TimeDependentOperator.constant(heomIdentity(2))
    let wrong = TimeDependentOperator.constant(heomIdentity(3))
    let propagation = hopsPropagation(end: 1)
    func run(_ request: TwoTimeCorrelationRequest) throws {
        try HEOM.solveTwoTimeCorrelation(
            problem: problem, configuration: config, request: request, propagation: propagation
        ) { _, _ in
            Issue.record("Invalid request reached observer")
            return .proceed
        }
    }
    #expect(throws: TwoTimeCorrelationError.nonFiniteInsertionTime) {
        try run(.init(insertionTime: .nan, insertion: .left(op), observable: op))
    }
    #expect(throws: TwoTimeCorrelationError.insertionTimeOutsideTimeSpan(insertionTime: 2, start: 0, end: 1))
    {
        try run(.init(insertionTime: 2, insertion: .left(op), observable: op))
    }
    #expect(
        throws: TwoTimeCorrelationError.insertionOperatorDimensionMismatch(expected: 2, rows: 3, columns: 3)
    ) {
        try run(.init(insertionTime: 0, insertion: .left(wrong), observable: op))
    }
    #expect(throws: TwoTimeCorrelationError.observableDimensionMismatch(expected: 2, rows: 3, columns: 3)) {
        try run(.init(insertionTime: 0, insertion: .left(op), observable: wrong))
    }
    let generated = TimeDependentOperator.generatedDense(
        .init { _, out in out = .zeros(rows: 3, columns: 3) })
    #expect(
        throws: TwoTimeCorrelationError.insertionOperatorDimensionMismatch(expected: 2, rows: 3, columns: 3)
    ) {
        try run(.init(insertionTime: 0.2, insertion: .right(generated), observable: op))
    }
    #expect(throws: TwoTimeCorrelationError.observableDimensionMismatch(expected: 2, rows: 3, columns: 3)) {
        try run(.init(insertionTime: 0, insertion: .left(op), observable: generated))
    }
}

@Test("HEOM validates ordering and indexed multi-time errors")
func heomMultiTimeValidation() throws {
    let op = TimeDependentOperator.constant(heomIdentity(2))
    func run(_ events: [TimedCorrelationInsertion]) throws {
        try HEOM.solveMultiTimeOrderedCorrelation(
            problem: hopsProblem(), configuration: heomConfiguration(),
            request: .init(insertions: events, observable: op), propagation: hopsPropagation(end: 1)
        ) { _, _ in
            Issue.record("Invalid events reached observer")
            return .proceed
        }
    }
    #expect(throws: MultiTimeOrderedCorrelationError.noInsertions) { try run([]) }
    #expect(throws: MultiTimeOrderedCorrelationError.nonFiniteInsertionTime(index: 0)) {
        try run([.init(time: .infinity, insertion: .left(op))])
    }
    #expect(
        throws: MultiTimeOrderedCorrelationError.insertionTimeOutsideTimeSpan(
            index: 0, insertionTime: -1, start: 0, end: 1)
    ) {
        try run([.init(time: -1, insertion: .left(op))])
    }
    #expect(
        throws: MultiTimeOrderedCorrelationError.insertionTimesNotStrictlyIncreasing(
            previousIndex: 0, previousTime: 0.5, index: 1, time: 0.5)
    ) {
        try run([.init(time: 0.5, insertion: .left(op)), .init(time: 0.5, insertion: .right(op))])
    }
    let wrong = TimeDependentOperator.generatedDense(.init { _, out in out = .zeros(rows: 3, columns: 3) })
    #expect(
        throws: MultiTimeOrderedCorrelationError.insertionOperatorDimensionMismatch(
            index: 1, expected: 2, rows: 3, columns: 3)
    ) {
        try run([.init(time: 0, insertion: .left(op)), .init(time: 0.1, insertion: .left(wrong))])
    }
}

@Test("HEOM rejects malformed bath and collapse operators without unsafe indexing")
func heomOperatorValidation() throws {
    let wrong = TimeDependentOperator.constant(heomIdentity(3))
    let generated = TimeDependentOperator.generatedDense(
        .init { _, out in out = .zeros(rows: 3, columns: 3) })
    for op in [wrong, generated] {
        let config = HEOM.Configuration(
            hierarchy: .init(
                environment: .init(couplingOperator: op, bath: heomSingleBath()), truncation: .maximumTier(2))
        )
        #expect(throws: HEOM.CPUEngine.SolverError.operatorDimensionMismatch) {
            try HEOM.solve(problem: hopsProblem(), configuration: config, propagation: hopsPropagation()) {
                _, _ in .proceed
            }
        }
        #expect(throws: HEOM.CPUEngine.SolverError.operatorDimensionMismatch) {
            try HEOM.solve(
                problem: hopsProblem(markovian: [.init(rate: .constant(0.2), collapseOperator: op)]),
                configuration: heomConfiguration(),
                propagation: hopsPropagation()
            ) { _, _ in .proceed }
        }
    }
}

@Test("HEOM reports invalid rates and centered traces and closes progress on failure")
func heomRuntimeValidation() throws {
    let capture = ProgressCapture()
    var propagation = hopsPropagation()
    propagation.progress = capture.reporting()
    let badRate = hopsProblem(markovian: [
        .init(rate: .generated { _ in -1 }, collapseOperator: .constant(hopsMatrix([0, 1, 0, 0])))
    ])
    #expect(throws: HEOM.CPUEngine.SolverError.invalidMarkovianRate(time: 0)) {
        try HEOM.solve(problem: badRate, configuration: heomConfiguration(), propagation: propagation) {
            _, _ in .proceed
        }
    }
    #expect(capture.newlineCount == 1)
    #expect(capture.percentages.last != 100)
    let empty = DensityMatrixProblem(
        initialState: hopsMatrix([0, 0, 0, 0]), system: QuantumSystem(hopsMatrix([0, 0, 0, 0])))
    #expect(throws: HEOM.CPUEngine.SolverError.invalidGuideTrace(time: 0)) {
        try HEOM.solve(
            problem: empty, configuration: heomConfiguration(shift: .meanField),
            propagation: hopsPropagation()
        ) { _, _ in .proceed }
    }
    let nonfinite = DensityMatrixProblem(
        initialState: hopsMatrix([Complex(.nan, 0), 0, 0, 1]),
        system: QuantumSystem(hopsMatrix([0, 0, 0, 0])))
    #expect(throws: HEOM.CPUEngine.SolverError.nonFiniteState(time: 0)) {
        try HEOM.solve(problem: nonfinite, configuration: heomConfiguration(), propagation: hopsPropagation())
        { _, _ in .proceed }
    }
}
