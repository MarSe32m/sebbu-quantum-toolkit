// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("CPU MCWF engine")
struct CPUMCWFEngineTests {
	@Test("Fixed output schedules are dense and normalized")
	func fixedOutputSchedule() throws {
		let outputTimes = [0.0, 0.03, 0.07, 0.1]
		let problem = mcwfProblem(
			initialState: [.zero, Complex(2)],
			markovianChannels: []
		)

		for algorithm in mcwfAlgorithms {
			var observedTimes: [Double] = []
			var observedStates: [[Complex<Double>]] = []
			let summary = try MCWF.solveTrajectory(
				problem: problem,
				configuration: .init(jumpAlgorithm: algorithm),
				propagation: mcwfPropagationOptions(
					end: outputTimes.last!,
					output: .times(outputTimes),
					maximumStep: 0.04
				),
				seed: 123,
				trajectoryID: 7,
				observing: { time, state in
					observedTimes.append(time)
					observedStates.append([state[0], state[1]])
					return .proceed
				}
			)

			#expect(observedTimes == outputTimes)
			#expect(observedStates.count == outputTimes.count)
			for state in observedStates {
				#expect(state == [.zero, .one])
			}
			#expect(summary.trajectoryIDs == 7..<8)
			#expect(summary.masterSeed == 123)
			#expect(summary.propagation.finalTime == 0.1)
			#expect(summary.propagation.endReason == .reachedEndTime)
		}
	}

	@Test("An MCWF observer can terminate propagation")
	func observerTermination() throws {
		let problem = mcwfProblem(markovianChannels: [])
		var observedTimes: [Double] = []

		let summary = try MCWF.solveTrajectory(
			problem: problem,
			configuration: .init(),
			propagation: mcwfPropagationOptions(
				end: 0.1,
				output: .everyAcceptedStep,
				maximumStep: 0.02
			),
			seed: 1,
			trajectoryID: 0,
			observing: { time, _ in
				observedTimes.append(time)
				return time >= 0.04 ? .stop : .proceed
			}
		)

		#expect(observedTimes == [0.02, 0.04])
		#expect(summary.propagation.finalTime == 0.04)
		#expect(summary.propagation.endReason == .stoppedByObserver)
	}

	@Test("Waiting-time MCWF locates and reports a post-jump state")
	func waitingTimeJumpLocation() throws {
		let problem = mcwfProblem(
			initialState: [.zero, .one],
			markovianChannels: [mcwfAmplitudeDampingChannel(rate: 1)]
		)
		let firstUniformBits = UInt64(1) << 63
		var randomNumberGenerator = ScriptedRandomNumberGenerator(
			[firstUniformBits, 0, 0]
		)
		var observations: [(time: Double, ground: Double, excited: Double)] = []

		try MCWF.solveTrajectory(
			problem: problem,
			configuration: .init(
				jumpAlgorithm: .waitingTime(
					eventTolerance: 1e-12,
					maximumEventIterations: 64
				)
			),
			propagation: mcwfPropagationOptions(
				end: 1,
				output: .everyAcceptedStep,
				maximumStep: 1
			),
			rng: &randomNumberGenerator
		) { time, state in
			observations.append(
				(time, state[0].lengthSquared, state[1].lengthSquared)
			)
			return .proceed
		}

		let sampledUniform = (Double(UInt64(1) << 52) + 0.5) * 0x1.0p-53
		let expectedJumpTime = -Double.log(sampledUniform)
		#expect(observations.count == 2)
		#expect(abs(observations[0].time - expectedJumpTime) < 1e-11)
		#expect(abs(observations[0].ground - 1) < 1e-14)
		#expect(observations[0].excited < 1e-28)
		#expect(observations[1].time == 1)
		#expect(abs(observations[1].ground - 1) < 1e-14)
	}

	@Test("Discrete-time MCWF emits pre-jump interiors and a post-jump endpoint")
	func discreteTimeJumpAndOutputOrdering() throws {
		let problem = mcwfProblem(
			initialState: [.zero, .one],
			markovianChannels: [mcwfAmplitudeDampingChannel(rate: 1)]
		)
		var randomNumberGenerator = ScriptedRandomNumberGenerator([0, 0])
		var observations: [(time: Double, ground: Double, excited: Double)] = []

		try MCWF.solveTrajectory(
			problem: problem,
			configuration: .init(jumpAlgorithm: .discreteTime),
			propagation: mcwfPropagationOptions(
				end: 0.25,
				output: .times([0, 0.125, 0.25]),
				maximumStep: 0.25
			),
			rng: &randomNumberGenerator
		) { time, state in
			observations.append(
				(time, state[0].lengthSquared, state[1].lengthSquared)
			)
			return .proceed
		}

		#expect(observations.count == 3)
		#expect(observations[0].time == 0)
		#expect(observations[0].excited == 1)
		#expect(observations[1].time == 0.125)
		#expect(abs(observations[1].excited - 1) < 1e-14)
		#expect(observations[2].time == 0.25)
		#expect(abs(observations[2].ground - 1) < 1e-14)
		#expect(observations[2].excited < 1e-28)
	}

	@Test("Generated collapse operators and weighted channel selection are supported")
	func generatedOperatorAndChannelSelection() throws {
		let toGround = TimeDependentOperator.constant(
			ConstantOperator(
				Matrix<Complex<Double>>(
					elements: [
						.zero, .zero, .one,
						.zero, .zero, .zero,
						.zero, .zero, .zero,
					],
					rows: 3,
					columns: 3
				)
			)
		)
		let toMiddle = TimeDependentOperator.generatedDense(
			DynamicDenseOperator { _, output in
				output.zeroElements()
				output[1, 2] = .one
			}
		)
		let system = QuantumSystem(
			Matrix<Complex<Double>>(
				elements: [Complex<Double>](repeating: .zero, count: 9),
				rows: 3,
				columns: 3
			)
		)
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>([.zero, .zero, .one]),
			system: system,
			markovianChannels: [
				MarkovianChannel(
					rate: .constant(1),
					collapseOperator: toGround
				),
				MarkovianChannel(
					rate: .generated { _ in 1 },
					collapseOperator: toMiddle
				),
			]
		)
		var randomNumberGenerator = ScriptedRandomNumberGenerator(
			[UInt64(1) << 63, .max, 0]
		)
		var finalState: [Complex<Double>] = []

		try MCWF.solveTrajectory(
			problem: problem,
			configuration: .init(),
			propagation: mcwfPropagationOptions(
				end: 0.5,
				output: .final,
				maximumStep: 0.5
			),
			rng: &randomNumberGenerator
		) { _, state in
			finalState = [state[0], state[1], state[2]]
			return .proceed
		}

		#expect(finalState.count == 3)
		#expect(finalState[0].lengthSquared < 1e-28)
		#expect(abs(finalState[1].lengthSquared - 1) < 1e-14)
		#expect(finalState[2].lengthSquared < 1e-28)
	}

	@Test("Seed and trajectory ID replay both jump algorithms exactly")
	func deterministicReplay() throws {
		let problem = mcwfProblem(
			initialState: [.zero, .one],
			markovianChannels: [mcwfAmplitudeDampingChannel(rate: 0.8)]
		)

		for algorithm in mcwfAlgorithms {
			let first = try finalMCWFState(
				problem: problem,
				algorithm: algorithm,
				seed: 0x1234_5678_9ABC_DEF0,
				trajectoryID: 41
			)
			let replay = try finalMCWFState(
				problem: problem,
				algorithm: algorithm,
				seed: 0x1234_5678_9ABC_DEF0,
				trajectoryID: 41
			)

			#expect(first == replay)
		}
	}

	@Test("solveTrajectories runs the requested IDs on a shared output schedule")
	func solveTrajectories() throws {
		let outputTimes = [0.0, 0.03, 0.07]
		let problem = mcwfProblem(markovianChannels: [])
		let observations = Mutex<[UInt64: [Double]]>([:])
		let execution = TrajectoryExecution(
			trajectoryIDs: UInt64(11)..<15,
			randomness: .seeded(0xDEAD_BEEF),
			parallelism: .maximumConcurrentTasks(2),
			batchSize: 2
		)

		let summary = try MCWF.solveTrajectories(
			problem: problem,
			configuration: .init(),
			propagation: mcwfPropagationOptions(
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
		#expect(summary.masterSeed == 0xDEAD_BEEF)
		#expect(summary.propagation.finalTime == 0.1)
		#expect(summary.propagation.endReason == .reachedEndTime)
	}

	@Test("Ensemble reduction is reproducible across parallelism choices")
	func ensembleParallelismReplay() throws {
		let problem = mcwfProblem(
			initialState: [.zero, .one],
			markovianChannels: [mcwfAmplitudeDampingChannel(rate: 0.8)]
		)
		let propagation = mcwfPropagationOptions(
			end: 0.5,
			output: .final,
			maximumStep: 0.02
		)

		func solve(parallelism: TrajectoryParallelism) throws
			-> Matrix<Complex<Double>>
		{
			var result = Matrix<Complex<Double>>.zeros(rows: 2, columns: 2)
			try MCWF.solveEnsemble(
				problem: problem,
				configuration: .init(),
				propagation: propagation,
				execution: TrajectoryExecution(
					trajectories: 64,
					seed: 0xBADC_0DE,
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
        #expect(serial.isApproximatelyEqual(to: parallel, absoluteTolerance: 1e-15))
	}

	@Test("solveEnsemble reports averaged density matrices on the fixed schedule")
	func fixedScheduleEnsemble() throws {
		let outputTimes = [0.0, 0.03, 0.07]
		let problem = mcwfProblem(
			initialState: [.zero, Complex(2)],
			markovianChannels: []
		)
		var observedTimes: [Double] = []
		var observedMatrices: [Matrix<Complex<Double>>] = []

		let summary = try MCWF.solveEnsemble(
			problem: problem,
			configuration: .init(),
			propagation: mcwfPropagationOptions(
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
			#expect(densityMatrix[0, 0].length < 1e-14)
			#expect(densityMatrix[0, 1].length < 1e-14)
			#expect(densityMatrix[1, 0].length < 1e-14)
			#expect((densityMatrix[1, 1] - .one).length < 1e-14)
		}
		#expect(summary.propagation.finalTime == 0.1)
	}

	@Test("Both jump algorithms reproduce amplitude damping in ensemble")
	func amplitudeDampingEnsemble() throws {
		let rate = 0.7
		let end = 0.8
		let trajectoryCount = 2048
		let expectedExcitedPopulation = Double.exp(-rate * end)
		let problem = mcwfProblem(
			initialState: [.zero, .one],
			markovianChannels: [mcwfAmplitudeDampingChannel(rate: rate)]
		)
		let propagation = mcwfPropagationOptions(
			end: end,
			output: .final,
			maximumStep: 0.02
		)

		for algorithm in mcwfAlgorithms {
			var excitedPopulation = 0.0
			let summary = try MCWF.solveEnsemble(
				problem: problem,
				configuration: .init(jumpAlgorithm: algorithm),
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

	@Test("A zero initial state reports a typed solver error")
	func zeroInitialState() {
		let problem = mcwfProblem(
			initialState: [.zero, .zero],
			markovianChannels: []
		)

		#expect(throws: CPUMCWFEngine.SolverError.invalidStateNorm(time: 0)) {
			try MCWF.solveTrajectory(
				problem: problem,
				configuration: .init(),
				propagation: mcwfPropagationOptions(
					end: 0,
					output: .final,
					maximumStep: 0.1
				),
				seed: 0,
				trajectoryID: 0,
				observing: { _, _ in .proceed }
			)
		}
	}
}

private let mcwfAlgorithms: [MCWF.JumpAlgorithm] = [
	.waitingTime(eventTolerance: 1e-10, maximumEventIterations: 64),
	.discreteTime,
]

private var mcwfZeroTwoLevelSystem: QuantumSystem<ConstantHamiltonian> {
	QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private func mcwfProblem(
	initialState: [Complex<Double>] = [.one, .zero],
	markovianChannels: [MarkovianChannel]
) -> PureStateProblem<ConstantHamiltonian> {
	PureStateProblem(
		initialState: Vector(initialState),
		system: mcwfZeroTwoLevelSystem,
		markovianChannels: markovianChannels
	)
}

private func mcwfAmplitudeDampingChannel(rate: Double) -> MarkovianChannel {
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

private func mcwfPropagationOptions(
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
			absoluteTolerance: 1e-11,
			relativeTolerance: 1e-11
		)
	)
}

private func finalMCWFState(
	problem: PureStateProblem<ConstantHamiltonian>,
	algorithm: MCWF.JumpAlgorithm,
	seed: UInt64,
	trajectoryID: UInt64
) throws -> [Complex<Double>] {
	var result: [Complex<Double>] = []
	try MCWF.solveTrajectory(
		problem: problem,
		configuration: .init(jumpAlgorithm: algorithm),
		propagation: mcwfPropagationOptions(
			end: 0.5,
			output: .final,
			maximumStep: 0.02
		),
		seed: seed,
		trajectoryID: trajectoryID,
		observing: { _, state in
			result = [state[0], state[1]]
			return .proceed
		}
	)
	return result
}

private struct ScriptedRandomNumberGenerator: RandomNumberGenerator {
	private var values: [UInt64]
	private var index = 0
	private let fallback: UInt64

	init(_ values: [UInt64], fallback: UInt64 = 0) {
		self.values = values
		self.fallback = fallback
	}

	mutating func next() -> UInt64 {
		guard index < values.count else { return fallback }
		defer { index += 1 }
		return values[index]
	}
}
