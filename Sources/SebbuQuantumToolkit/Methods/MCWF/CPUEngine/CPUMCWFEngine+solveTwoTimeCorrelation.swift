// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience
import Synchronization

extension CPUMCWFEngine {
	/// Computes an ensemble two-time correlation with a shared-record
	/// guide--companion construction.
	///
	/// The guide determines every waiting time and jump channel. The companion
	/// is transformed by the same conditional propagator and jump, preserving
	/// the inserted dyad rather than turning it into an independent trajectory.
	/// Scheduled output must be fixed across trajectories.
    @inlinable
	@discardableResult
	public func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		Self.validate(configuration: configuration)
		try _validateTwoTimeCorrelationRequest(
			request,
			timeSpan: propagation.timeSpan,
			dimension: problem.system.dimension
		)

		let outputTimes = try _fixedEnsembleOutputTimes(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		).filter { $0 >= request.insertionTime }
		let masterSeed = execution.resolvedMasterSeed()
		let trajectoryCount = execution.trajectoryIDs.count
		let maximumConcurrentTasks = Swift.min(
			execution.resolvedMaximumConcurrentTasks,
			trajectoryCount
		)

		let currentThreadCount = BLAS.getNumThreads()
		BLAS.setNumThreads(1)
		defer { BLAS.setNumThreads(currentThreadCount) }

		let ensembleSums = Mutex(
			[Complex<Double>](repeating: .zero, count: outputTimes.count)
		)
		let currentTrajectoryID = Atomic(execution.trajectoryIDs.lowerBound)
		let results: [_TrajectoryEnsembleBatchResult] = FixedWorkerPool.with(
			workers: maximumConcurrentTasks
		) { _ in
			var localSums = [Complex<Double>](
				repeating: .zero,
				count: outputTimes.count
			)
			var trajectoryID = currentTrajectoryID.wrappingAdd(
				1,
				ordering: .relaxed
			).oldValue
			var completedTrajectoryCount = 0

			while trajectoryID < execution.trajectoryIDs.upperBound {
				var randomNumberGenerator = TrajectoryRandomNumberGenerator(
					seed: masterSeed,
					trajectoryID: trajectoryID,
					purpose: .mcwfJumps
				)
				var sampleIndex = 0
				do {
					let summary = try solveCorrelationTrajectory(
						problem: problem,
						configuration: configuration,
						request: request,
						propagation: propagation,
						rng: &randomNumberGenerator
					) { time, value in
						precondition(
							sampleIndex < outputTimes.count
								&& time == outputTimes[sampleIndex],
							"An MCWF correlation trajectory produced an unexpected output time"
						)
						localSums[sampleIndex] += value
						sampleIndex += 1
					}
					precondition(
						sampleIndex == outputTimes.count,
						"An MCWF correlation trajectory omitted an output sample"
					)
					precondition(
						summary.finalTime == propagation.timeSpan.end
							&& summary.endReason == .reachedEndTime,
						"An MCWF correlation trajectory must reach the end time"
					)
				} catch {
					return _TrajectoryEnsembleBatchResult(
						trajectoryCount: completedTrajectoryCount,
						failure: _TrajectoryFailure(
							trajectoryID: trajectoryID,
							error: error
						)
					)
				}

				completedTrajectoryCount += 1
				trajectoryID =
					currentTrajectoryID.add(
						1,
						ordering: .relaxed
					).oldValue
			}

			ensembleSums.withLock { sums in
				for index in sums.indices {
					sums[index] += localSums[index]
				}
			}
			return _TrajectoryEnsembleBatchResult(
				trajectoryCount: completedTrajectoryCount,
				failure: nil
			)
		}

		if let failure = results.compactMap(\.failure).first {
			throw failure.error
		}
		let completedTrajectories = results.reduce(into: 0) {
			$0 += $1.trajectoryCount
		}
		precondition(
			completedTrajectories == trajectoryCount,
			"The MCWF correlation reduction omitted one or more trajectories"
		)

		let inverseTrajectoryCount = 1 / Double(trajectoryCount)
		var propagationSummary = PropagationRunSummary(
			finalTime: propagation.timeSpan.end,
			endReason: .reachedEndTime
		)
		ensembleSums.withLock { sums in
			for index in outputTimes.indices {
				guard propagationSummary.endReason == .reachedEndTime else {
					break
				}
				let value = inverseTrajectoryCount * sums[index]
				if observer(outputTimes[index], value) == .stop {
					propagationSummary = PropagationRunSummary(
						finalTime: outputTimes[index],
						endReason: .stoppedByObserver
					)
				}
			}
		}

		return TrajectoryRunSummary(
			trajectoryIDs: execution.trajectoryIDs,
			masterSeed: masterSeed,
			propagation: propagationSummary
		)
	}
}

extension CPUMCWFEngine {
    @inlinable
	internal func solveCorrelationTrajectory<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		observing observer: (Double, Complex<Double>) -> Void
	) throws -> PropagationRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let insertionTime = request.insertionTime
		let dimension = problem.system.dimension

		var guide: Vector<Complex<Double>>
		if start < insertionTime {
			var propagatedGuide: Vector<Complex<Double>>?
			let guideSummary = try solveTrajectory(
				problem: problem,
				configuration: configuration,
				propagation: PropagationOptions(
					timeSpan: SimulationTimeSpan(
						start: start,
						end: insertionTime
					),
					output: .final,
					integration: propagation.integration
				),
				rng: &rng
			) { _, state in
				propagatedGuide = Vector(copying: state)
				return .proceed
			}
			precondition(
				guideSummary.finalTime == insertionTime
					&& guideSummary.endReason == .reachedEndTime,
				"The MCWF guide must reach the insertion time"
			)
			guard let propagatedGuide else {
				preconditionFailure("The MCWF guide did not emit its final state")
			}
			guide = propagatedGuide
		} else {
			var initialState = TrajectoryState(problem.initialState)
			try Self.normalize(&initialState, at: start)
			guide = Vector(copying: initialState.wavefunction)
		}

		var state = CorrelationState(guide: guide)
		var operatorStorage = UniqueMatrix<Complex<Double>>.zeros(
			rows: dimension,
			columns: dimension
		)
		var actionStorage = UniqueVector<Complex<Double>>.zero(dimension)
		try _applyCorrelationInsertion(
			request.insertion,
			at: insertionTime,
			dimension: dimension,
			guide: state.guide,
			companion: &state.companion,
			operatorStorage: &operatorStorage
		)

		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		outputCursor.discardTimes(before: insertionTime)
		while let outputTime = outputCursor.nextTime(through: insertionTime) {
			observer(
				outputTime,
				try _correlationSample(
					request: request,
					at: outputTime,
					dimension: dimension,
					guide: state.guide,
					companion: state.companion,
					normalization: state.guide.normSquared,
					operatorStorage: &operatorStorage,
					actionStorage: &actionStorage
				)
			)
		}

		guard insertionTime < end else {
			return PropagationRunSummary(
				finalTime: end,
				endReason: .reachedEndTime
			)
		}

		let channels = Self.prepareChannels(
			problem.markovianChannels,
			dimension: dimension
		)
		let rhs = CorrelationRightHandSide(
			hamiltonian: problem.system.hamiltonian,
			channels: channels,
			dimension: dimension
		)
		var solver = UniqueDOPRISolver(
			t: insertionTime,
			dt: Swift.min(
				propagation.integration.maximumStepSize,
				end - insertionTime
			),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y4: CorrelationState(dimension: dimension),
			k1: CorrelationState(dimension: dimension),
			k2: CorrelationState(dimension: dimension),
			k3: CorrelationState(dimension: dimension),
			k4: CorrelationState(dimension: dimension),
			k5: CorrelationState(dimension: dimension),
			k6: CorrelationState(dimension: dimension),
			k7: CorrelationState(dimension: dimension),
			temporary: CorrelationState(dimension: dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)
		var outputState = CorrelationState(dimension: dimension)
		var jumpWorkspace = CorrelationJumpWorkspace(
			channels: channels,
			dimension: dimension
		)

		var waitingTimeTarget = Double.infinity
		if case .waitingTime = configuration.jumpAlgorithm, !channels.isEmpty {
			waitingTimeTarget = Self.nextWaitingTime(using: &rng)
		}

		func emit(
			_ time: Double,
			_ sampleState: borrowing CorrelationState
		) throws {
			observer(
				time,
				try _correlationSample(
					request: request,
					at: time,
					dimension: dimension,
					guide: sampleState.guide,
					companion: sampleState.companion,
					normalization: sampleState.guide.normSquared,
					operatorStorage: &operatorStorage,
					actionStorage: &actionStorage
				)
			)
		}

		while solver.t < end {
			let step = try solver.step(y: &state, upTo: end)

			switch configuration.jumpAlgorithm {
			case .waitingTime(let eventTolerance, let maximumEventIterations):
				if state.hazard >= waitingTimeTarget {
					let hazardFunctional = CorrelationHazardFunctional()
					let timeScale = Swift.max(
						1,
						Swift.max(abs(step.startTime), abs(step.endTime))
					)
					let effectiveTolerance = Swift.max(
						eventTolerance,
						64 * Double.ulpOfOne * timeScale
					)
					var lowerTime = step.startTime
					var upperTime = step.endTime
					let lowerValue =
						solver.interpolateLastStep(
							at: lowerTime,
							linearFunctional: hazardFunctional
						) - waitingTimeTarget
					let upperValue =
						solver.interpolateLastStep(
							at: upperTime,
							linearFunctional: hazardFunctional
						) - waitingTimeTarget

					guard lowerValue.isFinite else {
						throw SolverError.invalidHazard(time: lowerTime)
					}
					guard upperValue.isFinite else {
						throw SolverError.invalidHazard(time: upperTime)
					}

					let jumpTime: Double
					if lowerValue == .zero {
						jumpTime = lowerTime
					} else if upperValue == .zero {
						jumpTime = upperTime
					} else {
						guard lowerValue < .zero && upperValue > .zero
						else {
							throw SolverError.eventNotBracketed(
								stepStart: step.startTime,
								stepEnd: step.endTime
							)
						}

						var iterations = 0
						while upperTime - lowerTime > effectiveTolerance,
							iterations < maximumEventIterations
						{
							let middleTime =
								0.5 * (lowerTime + upperTime)
							if middleTime == lowerTime
								|| middleTime == upperTime
							{
								break
							}
							let middleValue =
								solver.interpolateLastStep(
									at: middleTime,
									linearFunctional:
										hazardFunctional
								) - waitingTimeTarget
							guard middleValue.isFinite else {
								throw SolverError.invalidHazard(
									time: middleTime)
							}
							if middleValue < .zero {
								lowerTime = middleTime
							} else {
								upperTime = middleTime
							}
							iterations += 1
						}

						guard
							upperTime - lowerTime <= effectiveTolerance
								|| lowerTime.nextUp >= upperTime
						else {
							throw
								SolverError
								.eventLocationDidNotConverge(
									stepStart: step.startTime,
									stepEnd: step.endTime,
									iterations: iterations
								)
						}
						jumpTime = 0.5 * (lowerTime + upperTime)
					}

					while let outputTime = outputCursor.nextTime(
						before: jumpTime)
					{
						solver.interpolateLastStep(
							at: outputTime,
							into: &outputState
						)
						try emit(outputTime, outputState)
					}

					solver.truncateLastStep(at: jumpTime, restoring: &state)
					try jumpWorkspace.applyJump(
						at: jumpTime,
						state: &state,
						using: &rng
					)
					state.hazard = .zero
					solver.stateDidChange()
					waitingTimeTarget = Self.nextWaitingTime(using: &rng)

					while let outputTime = outputCursor.nextTime(
						through: jumpTime)
					{
						precondition(
							outputTime == jumpTime,
							"A pre-jump MCWF correlation sample was not emitted before the event"
						)
						try emit(outputTime, state)
					}
				} else {
					while let outputTime = outputCursor.nextTime(
						through: step.endTime)
					{
						if outputTime == step.endTime {
							try emit(outputTime, state)
						} else {
							solver.interpolateLastStep(
								at: outputTime,
								into: &outputState
							)
							try emit(outputTime, outputState)
						}
					}
				}

			case .discreteTime:
				while let outputTime = outputCursor.nextTime(before: step.endTime) {
					solver.interpolateLastStep(
						at: outputTime,
						into: &outputState
					)
					try emit(outputTime, outputState)
				}

				let hazardTolerance =
					128 * Double.ulpOfOne * Swift.max(1, abs(state.hazard))
				guard
					state.hazard.isFinite,
					state.hazard >= -hazardTolerance
				else {
					throw SolverError.invalidHazard(time: step.endTime)
				}
				let integratedHazard = Swift.max(.zero, state.hazard)
				let jumpProbability = 1 - Double.exp(-integratedHazard)
				if jumpProbability > .zero,
					rng.nextUnitDouble() < jumpProbability
				{
					try jumpWorkspace.applyJump(
						at: step.endTime,
						state: &state,
						using: &rng
					)
				}
				state.hazard = .zero
				solver.stateDidChange()

				while let outputTime = outputCursor.nextTime(through: step.endTime)
				{
					precondition(
						outputTime == step.endTime,
						"An interior MCWF correlation sample was not emitted before the step endpoint"
					)
					try emit(outputTime, state)
				}
			}
		}

		return PropagationRunSummary(
			finalTime: end,
			endReason: .reachedEndTime
		)
	}
}

extension MCWF {
	@inlinable
	@inline(__always)
	@discardableResult
	public static func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let engine = CPUMCWFEngine()
		return try engine.solveTwoTimeCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			execution: execution,
			observing: observer
		)
	}
}
