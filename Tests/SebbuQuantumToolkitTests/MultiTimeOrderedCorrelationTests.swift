// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuQuantumToolkit
import SebbuScience
import Testing

@Suite("Multi-time ordered correlations")
struct MultiTimeOrderedCorrelationTests {
	@Test("GKSL, MCWF, and QSD apply left and right insertions in order")
	func insertionOrdering() throws {
		let problem = multiTimeProblem(channels: [])
		let propagation = multiTimePropagation(
			end: 0.4,
			output: .times([0, 0.1, 0.2, 0.4])
		)
		let execution = TrajectoryExecution(
			trajectories: 1,
			seed: 0xA11C_E,
			parallelism: .serial
		)
		let requests = [
			MultiTimeOrderedCorrelationRequest(
				insertions: [
					.init(time: 0.1, insertion: .left(multiTimeLowering)),
					.init(time: 0.2, insertion: .left(multiTimeRaising)),
				],
				observable: multiTimeExcitedProjector
			),
			MultiTimeOrderedCorrelationRequest(
				insertions: [
					.init(time: 0.1, insertion: .left(multiTimeLowering)),
					.init(time: 0.2, insertion: .right(multiTimeRaising)),
				],
				observable: multiTimeGroundProjector
			),
		]

		for request in requests {
			var deterministic: [(Double, Complex<Double>)] = []
			try GKSL.solveMultiTimeOrderedCorrelation(
				problem: problem,
				request: request,
				propagation: propagation
			) { time, value in
				deterministic.append((time, value))
				return .proceed
			}
			try expectUnitResults(deterministic)

			for algorithm in [
				MCWF.JumpAlgorithm.waitingTime(
					eventTolerance: 1e-10,
					maximumEventIterations: 64
				),
				.discreteTime,
			] {
				var values: [(Double, Complex<Double>)] = []
				try MCWF.solveMultiTimeOrderedCorrelation(
					problem: problem,
					configuration: .init(jumpAlgorithm: algorithm),
					request: request,
					propagation: propagation,
					execution: execution
				) { time, value in
					values.append((time, value))
					return .proceed
				}
				try expectUnitResults(values)
			}

			for equationType in [
				QSD.EquationType.linear,
				.nonLinear,
				.nonLinearNormalized,
			] {
				var values: [(Double, Complex<Double>)] = []
				try QSD.solveMultiTimeOrderedCorrelation(
					problem: problem,
					configuration: .init(equationType: equationType),
					request: request,
					propagation: propagation,
					execution: execution
				) { time, value in
					values.append((time, value))
					return .proceed
				}
				try expectUnitResults(values, tolerance: 2e-10)
			}
		}
	}

	@Test("MCWF and QSD multi-time ensembles agree with GKSL")
	func dissipativeAgreement() throws {
		let system = QuantumSystem(
			Matrix<Complex<Double>>(
				elements: [.zero, Complex(0.25), Complex(0.25), Complex(0.15)],
				rows: 2,
				columns: 2
			)
		)
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>([.zero, .one]),
			system: system,
			markovianChannels: [
				MarkovianChannel(
					rate: .constant(0.6),
					collapseOperator: multiTimeLowering
				)
			]
		)
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.08, insertion: .left(multiTimeLowering)),
				.init(time: 0.2, insertion: .left(multiTimeRaising)),
			],
			observable: multiTimeExcitedProjector
		)
		let propagation = multiTimePropagation(
			end: 0.5,
			output: .times([0.2, 0.35, 0.5]),
			maximumStep: 0.002
		)
		var reference: [Complex<Double>] = []
		try GKSL.solveMultiTimeOrderedCorrelation(
			problem: problem,
			request: request,
			propagation: propagation
		) { _, value in
			reference.append(value)
			return .proceed
		}
		let execution = TrajectoryExecution(
			trajectories: 8192,
			seed: 0x4D55_71AE,
			parallelism: .maximumConcurrentTasks(4),
			batchSize: 16
		)

		var mcwf: [Complex<Double>] = []
		try MCWF.solveMultiTimeOrderedCorrelation(
			problem: problem,
			request: request,
			propagation: propagation,
			execution: execution
		) { _, value in
			mcwf.append(value)
			return .proceed
		}
		try expectClose(mcwf, reference)

		var qsd: [Complex<Double>] = []
		try QSD.solveMultiTimeOrderedCorrelation(
			problem: problem,
			request: request,
			propagation: propagation,
			execution: execution
		) { _, value in
			qsd.append(value)
			return .proceed
		}
		try expectClose(qsd, reference)
	}

	@Test("Requests validate the ordered insertion sequence")
	func validation() throws {
		let problem = multiTimeProblem(channels: [])
		let propagation = multiTimePropagation(end: 0.5, output: .final)
		#expect(throws: MultiTimeOrderedCorrelationError.noInsertions) {
			try GKSL.solveMultiTimeOrderedCorrelation(
				problem: problem,
				request: .init(insertions: [], observable: multiTimeRaising),
				propagation: propagation
			) { _, _ in .proceed }
		}
		#expect(
			throws: MultiTimeOrderedCorrelationError
				.insertionTimesNotStrictlyIncreasing(
					previousIndex: 0,
					previousTime: 0.3,
					index: 1,
					time: 0.2
				)
		) {
			try GKSL.solveMultiTimeOrderedCorrelation(
				problem: problem,
				request: .init(
					insertions: [
						.init(time: 0.3, insertion: .left(multiTimeLowering)),
						.init(time: 0.2, insertion: .left(multiTimeRaising)),
					],
					observable: multiTimeExcitedProjector
				),
				propagation: propagation
			) { _, _ in .proceed }
		}
	}
}

private var multiTimeLowering: TimeDependentOperator {
	.constant(.init(Matrix<Complex<Double>>(
		elements: [.zero, .one, .zero, .zero],
		rows: 2,
		columns: 2
	)))
}

private var multiTimeRaising: TimeDependentOperator {
	.constant(.init(Matrix<Complex<Double>>(
		elements: [.zero, .zero, .one, .zero],
		rows: 2,
		columns: 2
	)))
}

private var multiTimeGroundProjector: TimeDependentOperator {
	.constant(.init(Matrix<Complex<Double>>(
		elements: [.one, .zero, .zero, .zero],
		rows: 2,
		columns: 2
	)))
}

private var multiTimeExcitedProjector: TimeDependentOperator {
	.constant(.init(Matrix<Complex<Double>>(
		elements: [.zero, .zero, .zero, .one],
		rows: 2,
		columns: 2
	)))
}

private func multiTimeProblem(
	channels: [MarkovianChannel]
) -> PureStateProblem<ConstantHamiltonian> {
	PureStateProblem(
		initialState: Vector<Complex<Double>>([.zero, .one]),
		system: QuantumSystem(Matrix<Complex<Double>>.zeros(rows: 2, columns: 2)),
		markovianChannels: channels
	)
}

private func multiTimePropagation(
	end: Double,
	output: OutputSchedule,
	maximumStep: Double = 0.005
) -> PropagationOptions<IntegrationOptions> {
	.init(
		timeSpan: .init(start: 0, end: end),
		output: output,
		integration: .init(
			minimumStepSize: 0,
			maximumStepSize: maximumStep,
			absoluteTolerance: 1e-10,
			relativeTolerance: 1e-10
		)
	)
}

private func expectUnitResults(
	_ results: [(Double, Complex<Double>)],
	tolerance: Double = 1e-12
) throws {
	#expect(results.map(\.0) == [0.2, 0.4])
	for (_, value) in results { #expect((value - .one).length < tolerance) }
}

private func expectClose(
	_ values: [Complex<Double>],
	_ expected: [Complex<Double>]
) throws {
	#expect(values.count == expected.count)
	for (value, reference) in zip(values, expected) {
		#expect((value - reference).length < 0.07)
	}
}
