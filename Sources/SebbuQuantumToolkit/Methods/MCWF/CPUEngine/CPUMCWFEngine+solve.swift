// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	@discardableResult
	public func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		precondition(
			trajectoryID < UInt64.max,
			"UInt64.max cannot be represented in the half-open trajectory-ID range"
		)

		var randomNumberGenerator = TrajectoryRandomNumberGenerator(
			seed: seed,
			trajectoryID: trajectoryID,
			purpose: .mcwfJumps
		)
		let propagationSummary = try solveTrajectory(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			rng: &randomNumberGenerator,
			observing: observer
		)
		return TrajectoryRunSummary(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1),
			masterSeed: seed,
			propagation: propagationSummary
		)
	}

	/// Propagates one trajectory with a caller-owned random-number generator.
	///
	/// This overload is useful for deterministic unit tests and for callers
	/// which manage random streams outside ``TrajectoryExecution``. Because an
	/// arbitrary generator has no master-seed or trajectory-ID metadata, it
	/// returns a ``PropagationRunSummary`` rather than a
	/// ``TrajectoryRunSummary``.
	@discardableResult
	public func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> PropagationRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		try solveTrajectory(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			rng: &rng,
			observing: observer
		)
	}
}

extension CPUMCWFEngine {
	internal func solveTrajectory<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> PropagationRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		Self.validate(configuration: configuration)

		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let dimension = problem.system.dimension
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		var state = TrajectoryState(problem.initialState)
		try Self.normalize(&state, at: start)
		var outputState = TrajectoryState(dimension: dimension)

		if let initialTime = outputCursor.takeInitialTime() {
			outputState.assign(state)
			if try Self.observeNormalized(
				time: initialTime,
				state: &outputState,
				observer: observer
			) == .stop {
				return PropagationRunSummary(
					finalTime: initialTime,
					endReason: .stoppedByObserver
				)
			}
		}

		guard start < end else {
			return PropagationRunSummary(
				finalTime: start,
				endReason: .reachedEndTime
			)
		}

		let channels = Self.prepareChannels(
			problem.markovianChannels,
			dimension: dimension
		)
		let rhs = RightHandSide(
			hamiltonian: problem.system.hamiltonian,
			channels: channels,
			dimension: dimension
		)
		var solver = UniqueDOPRISolver(
			t: start,
			dt: Swift.min(propagation.integration.maximumStepSize, end - start),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y4: TrajectoryState(dimension: dimension),
			k1: TrajectoryState(dimension: dimension),
			k2: TrajectoryState(dimension: dimension),
			k3: TrajectoryState(dimension: dimension),
			k4: TrajectoryState(dimension: dimension),
			k5: TrajectoryState(dimension: dimension),
			k6: TrajectoryState(dimension: dimension),
			k7: TrajectoryState(dimension: dimension),
			temporary: TrajectoryState(dimension: dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)
		var jumpWorkspace = JumpWorkspace(
			channels: channels,
			dimension: dimension
		)

		var waitingTimeTarget = Double.infinity
		if case .waitingTime = configuration.jumpAlgorithm, !channels.isEmpty {
			waitingTimeTarget = Self.nextWaitingTime(using: &rng)
		}

		while solver.t < end {
			let step = try solver.step(y: &state, upTo: end)

			switch configuration.jumpAlgorithm {
			case .waitingTime(let eventTolerance, let maximumEventIterations):
				if state.hazard >= waitingTimeTarget {
					let hazardFunctional = HazardFunctional()
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
						if try Self.observeNormalized(
							time: outputTime,
							state: &outputState,
							observer: observer
						) == .stop {
							return PropagationRunSummary(
								finalTime: outputTime,
								endReason: .stoppedByObserver
							)
						}
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
							"A pre-jump MCWF output time was not emitted before the event"
						)
						outputState.assign(state)
						if try Self.observeNormalized(
							time: outputTime,
							state: &outputState,
							observer: observer
						) == .stop {
							return PropagationRunSummary(
								finalTime: outputTime,
								endReason: .stoppedByObserver
							)
						}
					}
				} else {
					while let outputTime = outputCursor.nextTime(
						through: step.endTime)
					{
						if outputTime == step.endTime {
							outputState.assign(state)
						} else {
							solver.interpolateLastStep(
								at: outputTime,
								into: &outputState
							)
						}
						if try Self.observeNormalized(
							time: outputTime,
							state: &outputState,
							observer: observer
						) == .stop {
							return PropagationRunSummary(
								finalTime: outputTime,
								endReason: .stoppedByObserver
							)
						}
					}
				}

			case .discreteTime:
				while let outputTime = outputCursor.nextTime(before: step.endTime) {
					solver.interpolateLastStep(
						at: outputTime,
						into: &outputState
					)
					if try Self.observeNormalized(
						time: outputTime,
						state: &outputState,
						observer: observer
					) == .stop {
						return PropagationRunSummary(
							finalTime: outputTime,
							endReason: .stoppedByObserver
						)
					}
				}

				let hazardTolerance =
					128 * Double.ulpOfOne
					* Swift.max(1, abs(state.hazard))
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
						"A discrete MCWF output time was not emitted before the step endpoint"
					)
					outputState.assign(state)
					if try Self.observeNormalized(
						time: outputTime,
						state: &outputState,
						observer: observer
					) == .stop {
						return PropagationRunSummary(
							finalTime: outputTime,
							endReason: .stoppedByObserver
						)
					}
				}
			}
		}

		return PropagationRunSummary(
			finalTime: end,
			endReason: .reachedEndTime
		)
	}

	@inline(__always)
	internal static func validate(configuration: MCWF.Configuration) {
		if case .waitingTime(let eventTolerance, let maximumEventIterations) =
			configuration.jumpAlgorithm
		{
			precondition(
				eventTolerance.isFinite && eventTolerance > .zero,
				"The MCWF event tolerance must be positive and finite"
			)
			precondition(
				maximumEventIterations > 0,
				"The maximum MCWF event-iteration count must be positive"
			)
		}
	}

	@inline(__always)
	internal static func normalize(
		_ state: inout TrajectoryState,
		at time: Double
	) throws {
		let normSquared = state.wavefunction.normSquared
		guard normSquared.isFinite && normSquared > .zero else {
			throw SolverError.invalidStateNorm(time: time)
		}
		state.wavefunction.divideBLAS(by: normSquared.squareRoot())
	}

	@inline(__always)
	internal static func observeNormalized(
		time: Double,
		state: inout TrajectoryState,
		observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> PropagationControl {
		try normalize(&state, at: time)
		return observer(time, state.wavefunction)
	}

	@inline(__always)
	internal static func nextWaitingTime<RNG: RandomNumberGenerator>(
		using randomNumberGenerator: inout RNG
	) -> Double {
		-Double.log(randomNumberGenerator.nextUnitDoubleOpen())
	}
}

extension MCWF {
	@inlinable
	@inline(always)
	@discardableResult
	public static func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration = .init(),
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUMCWFEngine()
		return try implementation.solve(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID,
			observing: observer
		)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration = .init(),
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		rng: inout RNG,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> PropagationRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let implementation = CPUMCWFEngine()
		return try implementation.solve(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			rng: &rng,
			observing: observer
		)
	}
    
    @inlinable
    @inline(always)
    @discardableResult
    public static func solve<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: Configuration = .init(),
        propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
        observing observer: (
            Double,
            borrowing UniqueVector<Complex<Double>>
        ) -> PropagationControl
    ) throws -> PropagationRunSummary
    where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        var rng = SystemRandomNumberGenerator()
        return try implementation.solve(
            problem: problem,
            configuration: configuration,
            propagation: propagation,
            rng: &rng,
            observing: observer
        )
    }
}
