// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("CPU QSD engine")
struct CPUQSDEngineTests {
	@Test("Fixed stochastic output times are reached without interpolation")
	func fixedOutputSchedule() throws {
		let outputTimes = [0.0, 0.03, 0.07, 0.1]
		let problem = qsdProblem(markovianChannels: [])
		var observedTimes: [Double] = []
		var observedStates: [[Complex<Double>]] = []

		let summary = try QSD.solve(
			problem: problem,
			configuration: .init(equationType: .linear),
			propagation: qsdPropagationOptions(
				end: outputTimes.last!,
				output: .times(outputTimes),
				maximumStep: 0.04
			),
			seed: 123,
			trajectoryID: 7
		) { time, state in
			observedTimes.append(time)
			observedStates.append([state[0], state[1]])
			return .proceed
		}

		#expect(observedTimes == outputTimes)
		#expect(observedStates.count == outputTimes.count)
		for state in observedStates {
			#expect(state == [.one, .zero])
		}
		#expect(summary.trajectoryIDs == 7..<8)
		#expect(summary.masterSeed == 123)
		#expect(summary.propagation.finalTime == 0.1)
		#expect(summary.propagation.endReason == .reachedEndTime)
	}

	@Test("A QSD observer can terminate propagation")
	func observerTermination() throws {
		let problem = qsdProblem(markovianChannels: [])
		var observedTimes: [Double] = []

		let summary = try QSD.solve(
			problem: problem,
			configuration: .init(equationType: .linear),
			propagation: qsdPropagationOptions(
				end: 0.1,
				output: .everyAcceptedStep,
				maximumStep: 0.02
			),
			seed: 1,
			trajectoryID: 0
		) { time, _ in
			observedTimes.append(time)
			return time >= 0.04 ? .stop : .proceed
		}

		#expect(observedTimes == [0.02, 0.04])
		#expect(summary.propagation.finalTime == 0.04)
		#expect(summary.propagation.endReason == .stoppedByObserver)
	}

	@Test("All QSD formulations are replayable by seed and trajectory ID")
	func deterministicReplay() throws {
		let dephasingOperator = TimeDependentOperator.constant(
			ConstantOperator(
				Matrix<Complex<Double>>(
					elements: [.one, .zero, .zero, -.one],
					rows: 2,
					columns: 2
				)
			)
		)
		let component = Complex<Double>(1 / 2.0.squareRoot())
		let problem = qsdProblem(
			initialState: [component, component],
			markovianChannels: [
				MarkovianChannel(
					rate: .constant(0.7),
					collapseOperator: dephasingOperator
				)
			]
		)

		for equationType in qsdEquationTypes {
			let first = try finalQSDState(
				problem: problem,
				equationType: equationType,
				seed: 0x1234_5678_9ABC_DEF0,
				trajectoryID: 41
			)
			let replay = try finalQSDState(
				problem: problem,
				equationType: equationType,
				seed: 0x1234_5678_9ABC_DEF0,
				trajectoryID: 41
			)
			let otherTrajectory = try finalQSDState(
				problem: problem,
				equationType: equationType,
				seed: 0x1234_5678_9ABC_DEF0,
				trajectoryID: 42
			)

			#expect(first == replay)
			let trajectoryDifference = zip(first, otherTrajectory).reduce(0.0) {
				$0 + ($1.0 - $1.1).lengthSquared
			}
			#expect(trajectoryDifference > 1e-16)
		}
	}

	@Test("solveTrajectories runs the requested IDs on a shared output schedule")
	func solveTrajectories() throws {
		let outputTimes = [0.0, 0.03, 0.07]
		let problem = qsdProblem(markovianChannels: [])
		let observations = Mutex<[UInt64: [Double]]>([:])
		let execution = TrajectoryExecution(
			trajectoryIDs: UInt64(5)..<9,
			randomness: .seeded(0xCAFE_BABE),
			parallelism: .maximumConcurrentTasks(2),
			batchSize: 2
		)

		let summary = try QSD.solveTrajectories(
			problem: problem,
			configuration: .init(equationType: .linear),
			propagation: qsdPropagationOptions(
				end: 0.1,
				output: .times(outputTimes),
				maximumStep: 0.04
			),
			execution: execution
		) { trajectoryID, time, _ in
			observations.withLock { values in
				values[trajectoryID, default: []].append(time)
			}
		}

		let recorded = observations.withLock { $0 }
		#expect(recorded.count == execution.trajectoryIDs.count)
		for trajectoryID in execution.trajectoryIDs {
			#expect(recorded[trajectoryID] == outputTimes)
		}
		#expect(summary.trajectoryIDs == execution.trajectoryIDs)
		#expect(summary.masterSeed == 0xCAFE_BABE)
		#expect(summary.propagation.finalTime == 0.1)
		#expect(summary.propagation.endReason == .reachedEndTime)
	}

	@Test("Ensembles reject trajectory-dependent accepted-step output")
	func ensembleRejectsAcceptedStepOutput() {
		let problem = qsdProblem(markovianChannels: [])
		let execution = TrajectoryExecution(
			trajectories: 2,
			seed: 1,
			parallelism: .serial
		)

		#expect(
			throws: TrajectoryEnsembleError.everyAcceptedStepOutputIsNotSupported
		) {
			try QSD.solveEnsemble(
				problem: problem,
				configuration: .init(equationType: .linear),
				propagation: qsdPropagationOptions(
					end: 0.1,
					output: .everyAcceptedStep,
					maximumStep: 0.02
				),
				execution: execution
			) { _, _ in }
		}
	}

	@Test("solveEnsemble reports averaged density matrices on the fixed schedule")
	func fixedScheduleEnsemble() throws {
		let outputTimes = [0.0, 0.03, 0.07]
		let problem = qsdProblem(
			initialState: [.one, .one],
			markovianChannels: []
		)
		var observedTimes: [Double] = []
		var observedMatrices: [Matrix<Complex<Double>>] = []

		let summary = try QSD.solveEnsemble(
			problem: problem,
			configuration: .init(equationType: .nonLinearNormalized),
			propagation: qsdPropagationOptions(
				end: 0.1,
				output: .times(outputTimes),
				maximumStep: 0.04
			),
			execution: TrajectoryExecution(
				trajectories: 4,
				seed: 123,
				parallelism: .maximumConcurrentTasks(2),
				batchSize: 2
			)
		) { time, densityMatrix in
			observedTimes.append(time)
			observedMatrices.append(Matrix(copying: densityMatrix))
		}

		#expect(observedTimes == outputTimes)
		#expect(observedMatrices.count == outputTimes.count)
		for densityMatrix in observedMatrices {
			for element in densityMatrix.elements {
				#expect((element - Complex(0.5)).length < 1e-14)
			}
		}
		#expect(summary.propagation.finalTime == 0.1)
	}

	@Test("Ensemble reduction is reproducible across parallelism choices")
	func ensembleParallelismReplay() throws {
		let component = Complex<Double>(1 / 2.0.squareRoot())
		let problem = qsdProblem(
			initialState: [component, component],
			markovianChannels: [amplitudeDampingChannel(rate: 0.8)]
		)
		let propagation = qsdPropagationOptions(
			end: 0.1,
			output: .final,
			maximumStep: 0.002
		)

		func solve(parallelism: TrajectoryParallelism) throws
			-> Matrix<Complex<Double>>
		{
			var result = Matrix<Complex<Double>>.zeros(rows: 2, columns: 2)
			try QSD.solveEnsemble(
				problem: problem,
				configuration: .init(equationType: .nonLinearNormalized),
				propagation: propagation,
				execution: TrajectoryExecution(
					trajectories: 64,
					seed: 0xA11C_E,
					parallelism: parallelism,
					batchSize: 4
				)
			) { _, densityMatrix in
				result = Matrix(copying: densityMatrix)
			}
			return result
		}

		let serial = try solve(parallelism: .serial)
		let parallel = try solve(parallelism: .maximumConcurrentTasks(4))
        #expect(serial.isApproximatelyEqual(to: parallel, absoluteTolerance: 1e-14))
	}

	@Test("Adjacent trajectory IDs do not use shifted copies of one stream")
	func trajectoryStreamsAreSeparated() {
		var first = TrajectoryRandomNumberGenerator(
			seed: 0x1234_5678_9ABC_DEF0,
			trajectoryID: 0
		)
		var second = TrajectoryRandomNumberGenerator(
			seed: 0x1234_5678_9ABC_DEF0,
			trajectoryID: 1
		)
		let firstSamples = [first.next(), first.next(), first.next()]
		let secondSamples = [second.next(), second.next()]

		#expect(Array(firstSamples.dropFirst()) != secondSamples)
	}

	@Test("Normalized QSD normalizes its initial state and every accepted step")
	func normalizedStateNorm() throws {
		let problem = qsdProblem(
			initialState: [.one, .one],
			markovianChannels: [amplitudeDampingChannel(rate: 0.8)]
		)
		var maximumNormError = 0.0

		try QSD.solve(
			problem: problem,
			configuration: .init(equationType: .nonLinearNormalized),
			propagation: qsdPropagationOptions(
				end: 0.2,
				output: .everyAcceptedStep,
				maximumStep: 0.002
			),
			seed: 99,
			trajectoryID: 3
		) { _, state in
			maximumNormError = Swift.max(maximumNormError, abs(state.normSquared - 1))
			return .proceed
		}

		#expect(maximumNormError < 5e-14)
	}

	@Test("Each diffusion callback applies only its requested channel")
	func diffusionChannelIsolation() {
		let dynamicExcitedProjector = TimeDependentOperator.generatedDense(
			DynamicDenseOperator { _, output in
				output[0, 0] = .zero
				output[0, 1] = .zero
				output[1, 0] = .zero
				output[1, 1] = .one
			}
		)
		let constantGroundProjector = TimeDependentOperator.constant(
			ConstantOperator(
				Matrix<Complex<Double>>(
					elements: [.one, .zero, .zero, .zero],
					rows: 2,
					columns: 2
				)
			)
		)
		let problem = qsdProblem(
			initialState: [.one, .one],
			markovianChannels: [
				MarkovianChannel(
					rate: .constant(8),
					collapseOperator: dynamicExcitedProjector
				),
				MarkovianChannel(
					rate: .constant(2),
					collapseOperator: constantGroundProjector
				),
			]
		)
		var rhs = CPUQSDEngine.QSDRightHandSide(
			problem,
			equationType: .linear,
			seed: 0,
			trajectoryID: 0
		)
		let state = CPUQSDEngine.StateVector(state: problem.initialState)
		var diffusion = CPUQSDEngine.StateVector(dimension: 2)

		rhs.diffusion(t: 0, y: state, channel: 0, into: &diffusion)
		#expect(diffusion.state[0] == .zero)
		#expect(diffusion.state[1] == Complex(2))

		rhs.diffusion(t: 0, y: state, channel: 1, into: &diffusion)
		#expect(diffusion.state[0] == .one)
		#expect(diffusion.state[1] == .zero)
	}

	@Test("The normalized Stratonovich RHS is tangent to the unit sphere")
	func normalizedRightHandSideTangency() {
		let problem = qsdProblem(
			initialState: [Complex(0.3.squareRoot()), Complex(0.7.squareRoot())],
			markovianChannels: [amplitudeDampingChannel(rate: 0.9)]
		)
		var rhs = CPUQSDEngine.QSDRightHandSide(
			problem,
			equationType: .nonLinearNormalized,
			seed: 0,
			trajectoryID: 0
		)
		let state = CPUQSDEngine.StateVector(state: problem.initialState)
		var drift = CPUQSDEngine.StateVector(dimension: 2)
		var diffusion = CPUQSDEngine.StateVector(dimension: 2)

		rhs.drift(t: 0, y: state, into: &drift)
		rhs.diffusion(t: 0, y: state, channel: 0, into: &diffusion)

		#expect(abs(state.state.inner(drift.state).real) < 2e-15)
		#expect(state.state.inner(diffusion.state).length < 2e-15)
	}

	@Test("All QSD formulations reproduce amplitude damping in ensemble")
	func amplitudeDampingEnsemble() throws {
		let rate = 0.7
		let end = 0.2
		let trajectoryCount = 2048
		let expectedExcitedPopulation = Double.exp(-rate * end)
		let problem = qsdProblem(
			initialState: [.zero, .one],
			markovianChannels: [amplitudeDampingChannel(rate: rate)]
		)
		let propagation = qsdPropagationOptions(
			end: end,
			output: .final,
			maximumStep: 0.002
		)

		for equationType in qsdEquationTypes {
			var excitedPopulation = 0.0
			let summary = try QSD.solveEnsemble(
				problem: problem,
				configuration: .init(equationType: equationType),
				propagation: propagation,
				execution: TrajectoryExecution(
					trajectories: trajectoryCount,
					seed: 0xC0FF_EE,
					parallelism: .maximumConcurrentTasks(4),
					batchSize: 16
				)
			) { _, densityMatrix in
				excitedPopulation = densityMatrix[1, 1].real
			}

			#expect(abs(excitedPopulation - expectedExcitedPopulation) < 0.06)
			#expect(summary.propagation.finalTime == end)
			#expect(summary.propagation.endReason == .reachedEndTime)
		}
	}
}

private let qsdEquationTypes: [QSD.EquationType] = [
	.linear,
	.nonLinear,
	.nonLinearNormalized,
]

private var qsdZeroTwoLevelSystem: QuantumSystem<ConstantHamiltonian> {
	QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private func qsdProblem(
	initialState: [Complex<Double>] = [.one, .zero],
	markovianChannels: [MarkovianChannel]
) -> PureStateProblem<ConstantHamiltonian> {
	PureStateProblem(
		initialState: Vector(initialState),
		system: qsdZeroTwoLevelSystem,
		markovianChannels: markovianChannels
	)
}

private func amplitudeDampingChannel(rate: Double) -> MarkovianChannel {
	MarkovianChannel(
		rate: .constant(rate),
		collapseOperator: .constant(
			ConstantOperator(
				Matrix<Complex<Double>>(
					elements: [.zero, .one, .zero, .zero],
					rows: 2,
					columns: 2
				)
			)
		)
	)
}

private func qsdPropagationOptions(
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

private func finalQSDState(
	problem: PureStateProblem<ConstantHamiltonian>,
	equationType: QSD.EquationType,
	seed: UInt64,
	trajectoryID: UInt64
) throws -> [Complex<Double>] {
	var result: [Complex<Double>] = []
	try QSD.solve(
		problem: problem,
		configuration: .init(equationType: equationType),
		propagation: qsdPropagationOptions(
			end: 0.05,
			output: .final,
			maximumStep: 0.002
		),
		seed: seed,
		trajectoryID: trajectoryID
	) { _, state in
		result = [state[0], state[1]]
		return .proceed
	}
	return result
}
