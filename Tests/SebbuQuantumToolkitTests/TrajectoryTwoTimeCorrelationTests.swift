// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuQuantumToolkit
import SebbuScience
import Testing

@Suite("Trajectory two-time correlations")
struct TrajectoryTwoTimeCorrelationTests {
	@Test("MCWF and every QSD formulation preserve unitary phase")
	func unitaryPhase() throws {
		let frequency = 1.7
		let insertionTime = 0.3
		let outputTimes = [0.0, insertionTime, 0.55, 1.0]
		let expectedOutputTimes = Array(outputTimes.dropFirst())
		let system = QuantumSystem(
			Matrix<Complex<Double>>(
				elements: [.zero, .zero, .zero, Complex(frequency)],
				rows: 2,
				columns: 2
			)
		)
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>([.zero, .one]),
			system: system
		)
		let request = TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(trajectoryCorrelationLoweringOperator),
			observable: trajectoryCorrelationRaisingOperator
		)
		let propagation = trajectoryCorrelationPropagation(
			end: outputTimes.last!,
			output: .times(outputTimes),
			maximumStep: 0.001
		)
		let execution = TrajectoryExecution(
			trajectories: 1,
			seed: 0x51A7_E,
			parallelism: .serial
		)

		var mcwfResults: [(Double, Complex<Double>)] = []
		try MCWF.solveTwoTimeCorrelation(
			problem: problem,
			request: request,
			propagation: propagation,
			execution: execution
		) { time, value in
			mcwfResults.append((time, value))
			return .proceed
		}
		try expectUnitaryCorrelation(
			mcwfResults,
			times: expectedOutputTimes,
			insertionTime: insertionTime,
			frequency: frequency,
			tolerance: 2e-8
		)

		for equationType in trajectoryCorrelationQSDEquationTypes {
			var qsdResults: [(Double, Complex<Double>)] = []
			try QSD.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(equationType: equationType),
				request: request,
				propagation: propagation,
				execution: execution
			) { time, value in
				qsdResults.append((time, value))
				return .proceed
			}
			try expectUnitaryCorrelation(
				qsdResults,
				times: expectedOutputTimes,
				insertionTime: insertionTime,
				frequency: frequency,
				tolerance: 3e-6
			)
		}
	}

	@Test("Left and right stochastic insertions preserve operator ordering")
	func insertionOrdering() throws {
		let problem = trajectoryCorrelationProblem(markovianChannels: [])
		let propagation = trajectoryCorrelationPropagation(
			end: 0.5,
			output: .times([0.25, 0.5]),
			maximumStep: 0.01
		)
		let execution = TrajectoryExecution(
			trajectories: 2,
			seed: 123,
			parallelism: .maximumConcurrentTasks(2)
		)

		for insertion in [
			CorrelationInsertion.left(trajectoryCorrelationLoweringOperator),
			CorrelationInsertion.right(trajectoryCorrelationLoweringOperator),
		] {
			let expected =
				if case .left = insertion {
					Complex<Double>.one
				} else {
					Complex<Double>.zero
				}
			let request = TwoTimeCorrelationRequest(
				insertionTime: 0.25,
				insertion: insertion,
				observable: trajectoryCorrelationRaisingOperator
			)

			var mcwfValues: [Complex<Double>] = []
			try MCWF.solveTwoTimeCorrelation(
				problem: problem,
				request: request,
				propagation: propagation,
				execution: execution
			) { _, value in
				mcwfValues.append(value)
				return .proceed
			}
			#expect(mcwfValues.count == 2)
			for value in mcwfValues {
				#expect((value - expected).length < 1e-12)
			}

			var qsdValues: [Complex<Double>] = []
			try QSD.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(equationType: .nonLinearNormalized),
				request: request,
				propagation: propagation,
				execution: execution
			) { _, value in
				qsdValues.append(value)
				return .proceed
			}
			#expect(qsdValues.count == 2)
			for value in qsdValues {
				#expect((value - expected).length < 1e-12)
			}
		}
	}

	@Test("Shared identity channels leave inserted dyads invariant")
	func identityChannelGaugeCancellation() throws {
		let identity = TimeDependentOperator.generatedDense(
			DynamicDenseOperator { _, output in
				output[0, 0] = .one
				output[0, 1] = .zero
				output[1, 0] = .zero
				output[1, 1] = .one
			}
		)
		let problem = trajectoryCorrelationProblem(
			markovianChannels: [
				MarkovianChannel(
					rate: .constant(12),
					collapseOperator: identity
				)
			]
		)
		let request = TwoTimeCorrelationRequest(
			insertionTime: 0,
			insertion: .left(trajectoryCorrelationLoweringOperator),
			observable: trajectoryCorrelationRaisingOperator
		)
		let propagation = trajectoryCorrelationPropagation(
			end: 0.4,
			output: .final,
			maximumStep: 0.002
		)
		let execution = TrajectoryExecution(
			trajectories: 4,
			seed: 0x1D_E17,
			parallelism: .maximumConcurrentTasks(2)
		)

		for algorithm in trajectoryCorrelationMCWFAlgorithms {
			var value = Complex<Double>.zero
			try MCWF.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(jumpAlgorithm: algorithm),
				request: request,
				propagation: propagation,
				execution: execution
			) { _, sample in
				value = sample
				return .proceed
			}
			#expect((value - .one).length < 2e-10)
		}

		for equationType in [
			QSD.EquationType.nonLinear,
			.nonLinearNormalized,
		] {
			var value = Complex<Double>.zero
			try QSD.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(equationType: equationType),
				request: request,
				propagation: propagation,
				execution: execution
			) { _, sample in
				value = sample
				return .proceed
			}
			#expect((value - .one).length < 2e-8)
		}
	}

	@Test("MCWF and QSD agree with GKSL for a driven dissipative qubit")
	func drivenDissipativeQubit() throws {
		let system = QuantumSystem(
			Matrix<Complex<Double>>(
				elements: [
					.zero, Complex(0.35), Complex(0.35), Complex(0.2),
				],
				rows: 2,
				columns: 2
			)
		)
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>([
				Complex(0.8), Complex(0, 0.6),
			]),
			system: system,
			markovianChannels: [trajectoryCorrelationAmplitudeDamping(rate: 0.6)]
		)
		let insertionTime = 0.12
		let outputTimes = [insertionTime, 0.28, 0.5]
		let request = TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(trajectoryCorrelationLoweringOperator),
			observable: trajectoryCorrelationRaisingOperator
		)
		let propagation = trajectoryCorrelationPropagation(
			end: outputTimes.last!,
			output: .times(outputTimes),
			maximumStep: 0.002
		)

		var reference: [Complex<Double>] = []
		try GKSL.solveTwoTimeCorrelation(
			problem: problem,
			request: request,
			propagation: propagation
		) { _, value in
			reference.append(value)
			return .proceed
		}
		#expect(reference.count == outputTimes.count)

		let execution = TrajectoryExecution(
			trajectories: 1024,
			seed: 0xD1_551_7A7E,
			parallelism: .maximumConcurrentTasks(4),
			batchSize: 16
		)
		for solve in [
			{ (observer: @escaping (Double, Complex<Double>) -> PropagationControl) throws in
				try MCWF.solveTwoTimeCorrelation(
					problem: problem,
					request: request,
					propagation: propagation,
					execution: execution,
					observing: observer
				)
			},
			{ (observer: @escaping (Double, Complex<Double>) -> PropagationControl) throws in
				try QSD.solveTwoTimeCorrelation(
					problem: problem,
					configuration: .init(equationType: .nonLinearNormalized),
					request: request,
					propagation: propagation,
					execution: execution,
					observing: observer
				)
			},
		] {
			var values: [Complex<Double>] = []
			_ = try solve { _, value in
				values.append(value)
				return .proceed
			}
			#expect(values.count == reference.count)
			for (value, expected) in zip(values, reference) {
				#expect((value - expected).length < 0.07)
			}
		}
	}

	@Test("MCWF and QSD agree with analytic amplitude-damping correlations")
	func amplitudeDamping() throws {
		let rate = 0.8
		let insertionTime = 0.1
		let outputTimes = [insertionTime, 0.4]
		let problem = trajectoryCorrelationProblem(
			markovianChannels: [trajectoryCorrelationAmplitudeDamping(rate: rate)]
		)
		let request = TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(trajectoryCorrelationLoweringOperator),
			observable: trajectoryCorrelationRaisingOperator
		)
		let propagation = trajectoryCorrelationPropagation(
			end: outputTimes.last!,
			output: .times(outputTimes),
			maximumStep: 0.002
		)
		let execution = TrajectoryExecution(
			trajectories: 1024,
			seed: 0xC0FF_EE,
			parallelism: .maximumConcurrentTasks(4),
			batchSize: 16
		)

		for algorithm in trajectoryCorrelationMCWFAlgorithms {
			var values: [Complex<Double>] = []
			try MCWF.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(jumpAlgorithm: algorithm),
				request: request,
				propagation: propagation,
				execution: execution
			) { _, value in
				values.append(value)
				return .proceed
			}
			try expectAmplitudeDampingCorrelation(
				values,
				times: outputTimes,
				insertionTime: insertionTime,
				rate: rate,
				tolerance: 0.07
			)
		}

		for equationType in trajectoryCorrelationQSDEquationTypes {
			var values: [Complex<Double>] = []
			try QSD.solveTwoTimeCorrelation(
				problem: problem,
				configuration: .init(equationType: equationType),
				request: request,
				propagation: propagation,
				execution: execution
			) { _, value in
				values.append(value)
				return .proceed
			}
			try expectAmplitudeDampingCorrelation(
				values,
				times: outputTimes,
				insertionTime: insertionTime,
				rate: rate,
				tolerance: 0.07
			)
		}
	}

	@Test("Trajectory correlations validate requests and support observer stop")
	func validationAndTermination() throws {
		let problem = trajectoryCorrelationProblem(markovianChannels: [])
		let propagation = trajectoryCorrelationPropagation(
			end: 0.5,
			output: .times([0.25, 0.5]),
			maximumStep: 0.01
		)
		let execution = TrajectoryExecution(
			trajectories: 1,
			seed: 1,
			parallelism: .serial
		)

		#expect(throws: TwoTimeCorrelationError.nonFiniteInsertionTime) {
			try MCWF.solveTwoTimeCorrelation(
				problem: problem,
				request: TwoTimeCorrelationRequest(
					insertionTime: .nan,
					insertion: .left(trajectoryCorrelationLoweringOperator),
					observable: trajectoryCorrelationRaisingOperator
				),
				propagation: propagation,
				execution: execution
			) { _, _ in .proceed }
		}

		let threeByThree = TimeDependentOperator.constant(
			ConstantOperator(
				Matrix<Complex<Double>>.zeros(rows: 3, columns: 3)
			)
		)
		#expect(
			throws: TwoTimeCorrelationError.observableDimensionMismatch(
				expected: 2,
				rows: 3,
				columns: 3
			)
		) {
			try QSD.solveTwoTimeCorrelation(
				problem: problem,
				request: TwoTimeCorrelationRequest(
					insertionTime: 0.25,
					insertion: .left(trajectoryCorrelationLoweringOperator),
					observable: threeByThree
				),
				propagation: propagation,
				execution: execution
			) { _, _ in .proceed }
		}

		#expect(
			throws: TrajectoryEnsembleError.everyAcceptedStepOutputIsNotSupported
		) {
			try MCWF.solveTwoTimeCorrelation(
				problem: problem,
				request: TwoTimeCorrelationRequest(
					insertionTime: 0.25,
					insertion: .left(trajectoryCorrelationLoweringOperator),
					observable: trajectoryCorrelationRaisingOperator
				),
				propagation: trajectoryCorrelationPropagation(
					end: 0.5,
					output: .everyAcceptedStep,
					maximumStep: 0.01
				),
				execution: execution
			) { _, _ in .proceed }
		}

		var observedTimes: [Double] = []
		let summary = try QSD.solveTwoTimeCorrelation(
			problem: problem,
			request: TwoTimeCorrelationRequest(
				insertionTime: 0.25,
				insertion: .left(trajectoryCorrelationLoweringOperator),
				observable: trajectoryCorrelationRaisingOperator
			),
			propagation: propagation,
			execution: execution
		) { time, _ in
			observedTimes.append(time)
			return .stop
		}
		#expect(observedTimes == [0.25])
		#expect(summary.propagation.finalTime == 0.25)
		#expect(summary.propagation.endReason == .stoppedByObserver)
	}
}

private let trajectoryCorrelationQSDEquationTypes: [QSD.EquationType] = [
	.linear,
	.nonLinear,
	.nonLinearNormalized,
]

private let trajectoryCorrelationMCWFAlgorithms: [MCWF.JumpAlgorithm] = [
	.waitingTime(eventTolerance: 1e-10, maximumEventIterations: 64),
	.discreteTime,
]

private var trajectoryCorrelationZeroSystem: QuantumSystem<ConstantHamiltonian> {
	QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private var trajectoryCorrelationLoweringOperator: TimeDependentOperator {
	.constant(
		ConstantOperator(
			Matrix<Complex<Double>>(
				elements: [.zero, .one, .zero, .zero],
				rows: 2,
				columns: 2
			)
		)
	)
}

private var trajectoryCorrelationRaisingOperator: TimeDependentOperator {
	.constant(
		ConstantOperator(
			Matrix<Complex<Double>>(
				elements: [.zero, .zero, .one, .zero],
				rows: 2,
				columns: 2
			)
		)
	)
}

private func trajectoryCorrelationProblem(
	markovianChannels: [MarkovianChannel]
) -> PureStateProblem<ConstantHamiltonian> {
	PureStateProblem(
		initialState: Vector<Complex<Double>>([.zero, .one]),
		system: trajectoryCorrelationZeroSystem,
		markovianChannels: markovianChannels
	)
}

private func trajectoryCorrelationAmplitudeDamping(
	rate: Double
) -> MarkovianChannel {
	MarkovianChannel(
		rate: .constant(rate),
		collapseOperator: trajectoryCorrelationLoweringOperator
	)
}

private func trajectoryCorrelationPropagation(
	end: Double,
	output: OutputSchedule,
	maximumStep: Double
) -> PropagationOptions<IntegrationOptions> {
	PropagationOptions(
		timeSpan: SimulationTimeSpan(start: 0, end: end),
		output: output,
		integration: IntegrationOptions(
			minimumStepSize: 0,
			maximumStepSize: maximumStep,
			absoluteTolerance: 1e-10,
			relativeTolerance: 1e-10
		)
	)
}

private func expectUnitaryCorrelation(
	_ results: [(Double, Complex<Double>)],
	times: [Double],
	insertionTime: Double,
	frequency: Double,
	tolerance: Double
) throws {
	#expect(results.map(\.0) == times)
	for (time, value) in results {
		let phase = frequency * (time - insertionTime)
		let expected = Complex<Double>(Double.cos(phase), Double.sin(phase))
		#expect((value - expected).length < tolerance)
	}
}

private func expectAmplitudeDampingCorrelation(
	_ values: [Complex<Double>],
	times: [Double],
	insertionTime: Double,
	rate: Double,
	tolerance: Double
) throws {
	#expect(values.count == times.count)
	for (time, value) in zip(times, values) {
		let expected = Double.exp(-rate * (time + insertionTime) / 2)
		#expect(abs(value.real - expected) < tolerance)
		#expect(abs(value.imaginary) < tolerance)
	}
}
