// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS correlation validation", .serialized)
struct HOPSCorrelationValidationTests {
	@Test("Two-time errors retain the public two-time error type")
	func twoTimeErrors() throws {
		let bad = TimeDependentOperator.constant(
			Matrix<Complex<Double>>.zeros(rows: 3, columns: 3))
		let generated = TimeDependentOperator.generatedDense(
			.init { _, output in output = .zeros(rows: 3, columns: 3) })
		let b = CorrelationInsertion.left(.constant(hcX))
		let a = TimeDependentOperator.constant(hcZ)
		let cases: [(TwoTimeCorrelationRequest, TwoTimeCorrelationError)] = [
			(
				.init(insertionTime: .nan, insertion: b, observable: a),
				.nonFiniteInsertionTime
			),
			(
				.init(insertionTime: .infinity, insertion: b, observable: a),
				.nonFiniteInsertionTime
			),
			(
				.init(insertionTime: -0.1, insertion: b, observable: a),
				.insertionTimeOutsideTimeSpan(
					insertionTime: -0.1, start: 0, end: 0.5)
			),
			(
				.init(insertionTime: 0.6, insertion: b, observable: a),
				.insertionTimeOutsideTimeSpan(
					insertionTime: 0.6, start: 0, end: 0.5)
			),
			(
				.init(insertionTime: 0.1, insertion: .left(bad), observable: a),
				.insertionOperatorDimensionMismatch(
					expected: 2, rows: 3, columns: 3)
			),
			(
				.init(
					insertionTime: 0.1, insertion: .right(generated),
					observable: a),
				.insertionOperatorDimensionMismatch(
					expected: 2, rows: 3, columns: 3)
			),
			(
				.init(insertionTime: 0.1, insertion: b, observable: bad),
				.observableDimensionMismatch(expected: 2, rows: 3, columns: 3)
			),
			(
				.init(insertionTime: 0.1, insertion: b, observable: generated),
				.observableDimensionMismatch(expected: 2, rows: 3, columns: 3)
			),
		]
		for (request, error) in cases {
			#expect(throws: error) {
				try HOPS.solveTwoTimeCorrelation(
					problem: hopsProblem(), configuration: hcConfiguration(5),
					request: request,
					propagation: hopsPropagation(end: 0.5),
					execution: .init(
						trajectories: 1, seed: 1, parallelism: .serial)
				) { _, _ in .proceed }
			}
		}
	}

	@Test("Multi-time requests validate event order and dynamic dimensions")
	func multiTimeErrors() throws {
		let insertion = CorrelationInsertion.left(.constant(hcX))
		let event = TimedCorrelationInsertion(time: 0.1, insertion: insertion)
		let bad = TimeDependentOperator.constant(
			Matrix<Complex<Double>>.zeros(rows: 2, columns: 3))
		let generated = TimeDependentOperator.generatedDense(
			.init { _, output in output = .zeros(rows: 3, columns: 2) })
		let cases: [([TimedCorrelationInsertion], MultiTimeOrderedCorrelationError)] = [
			([], .noInsertions),
			(
				[event, .init(time: .infinity, insertion: insertion)],
				.nonFiniteInsertionTime(index: 1)
			),
			(
				[event, .init(time: 0.6, insertion: insertion)],
				.insertionTimeOutsideTimeSpan(
					index: 1, insertionTime: 0.6, start: 0, end: 0.5)
			),
			(
				[event, event],
				.insertionTimesNotStrictlyIncreasing(
					previousIndex: 0, previousTime: 0.1, index: 1, time: 0.1)
			),
			(
				[event, .init(time: 0.05, insertion: insertion)],
				.insertionTimesNotStrictlyIncreasing(
					previousIndex: 0, previousTime: 0.1, index: 1, time: 0.05)
			),
			(
				[event, .init(time: 0.2, insertion: .right(bad))],
				.insertionOperatorDimensionMismatch(
					index: 1, expected: 2, rows: 2, columns: 3)
			),
			(
				[event, .init(time: 0.2, insertion: .left(generated))],
				.insertionOperatorDimensionMismatch(
					index: 1, expected: 2, rows: 3, columns: 2)
			),
		]
		for (events, error) in cases {
			#expect(throws: error) {
				try hcValues(
					problem: hopsProblem(), configuration: hcConfiguration(5),
					request: .init(
						insertions: events, observable: .constant(hcZ)),
					propagation: hopsPropagation(end: 0.5))
			}
		}
		#expect(
			throws: MultiTimeOrderedCorrelationError.observableDimensionMismatch(
				expected: 2, rows: 3, columns: 2)
		) {
			try hcValues(
				problem: hopsProblem(), configuration: hcConfiguration(5),
				request: .init(insertions: [event], observable: generated),
				propagation: hopsPropagation(end: 0.5))
		}
	}

	@Test("Propagation validation and nonfinite insertion failures reach the caller")
	func propagationErrors() throws {
		var config = hcConfiguration(5, colored: true)
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [.init(time: 0.1, insertion: .left(.constant(hcX)))],
			observable: .constant(hcZ))
		#expect(throws: TrajectoryEnsembleError.everyAcceptedStepOutputIsNotSupported) {
			try hcValues(
				problem: hopsProblem(), configuration: config, request: request,
				propagation: hopsPropagation(output: .everyAcceptedStep))
		}
		config.unravelling = .jump
		#expect(throws: HCT.SolverError.unsupportedUnravelling) {
			try hcValues(
				problem: hopsProblem(), configuration: config, request: request,
				propagation: hopsPropagation())
		}
		config.unravelling = .diffusive
		let nonfinite = TimeDependentOperator.generatedDense(
			.init { _, output in
				output.zeroElements()
				output[0, 0] = Complex(.nan)
			})
		#expect(throws: HCT.SolverError.nonFiniteState(time: 0.1)) {
			try hcValues(
				problem: hopsProblem(), configuration: config,
				request: .init(
					insertions: [.init(time: 0.1, insertion: .left(nonfinite))],
					observable: .constant(hcZ)),
				propagation: hopsPropagation())
		}
	}
}
