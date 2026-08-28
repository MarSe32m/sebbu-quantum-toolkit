// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUQSDEngine: Sendable {
	@inlinable
	public init() {}

	public enum SolverError: Error, Equatable {
		case invalidStateNorm(time: Double)
	}
}

extension CPUQSDEngine: QSD.Implementation {
	@inlinable
	@discardableResult
	public func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
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

		let propagationSummary = try solveTrajectory(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID,
			observing: observer
		)
		return TrajectoryRunSummary(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1),
			masterSeed: seed,
			propagation: propagationSummary
		)
	}

	@inlinable
	@discardableResult
	public func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		throw ImplementationError.notImplemented
	}

	@inlinable
	public func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		throw ImplementationError.notImplemented
	}

	@inlinable
	internal func solveTrajectory<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		var state = StateVector(state: problem.initialState)

		switch configuration.equationType {
		case .linear:
			break
		case .nonLinear:
			try Self.validateStateNorm(state, at: start)
		case .nonLinearNormalized:
			try Self.normalize(&state, at: start)
		}

		if let initialTime = outputCursor.takeInitialTime() {
			if observer(initialTime, state.state) == .stop {
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

		let dimension = problem.system.dimension
		var noiseStorage = [Complex<Double>](
			repeating: .zero,
			count: problem.markovianChannels.count
		)
		let noises = noiseStorage.mutableSpan
		let rhs = QSDRightHandSide(
			problem,
			equationType: configuration.equationType,
			seed: seed,
			trajectoryID: trajectoryID
		)
		var solver = UniqueSRK2Solver(
			t: start,
			dt: Swift.min(propagation.integration.maximumStepSize, end - start),
			rhs: rhs,
			drift0: .init(dimension: dimension),
			drift1: .init(dimension: dimension),
			noise0: .init(dimension: dimension),
			noise1: .init(dimension: dimension),
			temporary: .init(dimension: dimension),
			noises: noises
		)

		while solver.t < end {
			let stepLimit = Swift.min(
				outputCursor.nextRequiredStepBoundary ?? end,
				end
			)
			precondition(
				stepLimit > solver.t,
				"The stochastic output boundary must advance integration time"
			)
			let step = solver.step(y: &state, upTo: stepLimit)

			switch configuration.equationType {
			case .linear:
				break
			case .nonLinear:
				try Self.validateStateNorm(state, at: step.endTime)
			case .nonLinearNormalized:
				try Self.normalize(&state, at: step.endTime)
				solver.stateDidChange()
			}

			while let outputTime = outputCursor.nextTime(through: step.endTime) {
				// Fixed stochastic schedules are hard step boundaries. Unlike an
				// ODE solve, no deterministic interpolation of the state is used.
				precondition(
					outputTime == step.endTime,
					"A stochastic output time must coincide with a step boundary"
				)
				if observer(outputTime, state.state) == .stop {
					return PropagationRunSummary(
						finalTime: outputTime,
						endReason: .stoppedByObserver
					)
				}
			}
		}

		return PropagationRunSummary(
			finalTime: end,
			endReason: .reachedEndTime
		)
	}

	@inlinable
	internal static func validateStateNorm(
		_ state: borrowing StateVector,
		at time: Double
	) throws {
		let normSquared = state.state.normSquared
		guard normSquared.isFinite && normSquared > .zero else {
			throw SolverError.invalidStateNorm(time: time)
		}
	}

	@inlinable
	internal static func normalize(
		_ state: inout StateVector,
		at time: Double
	) throws {
		let normSquared = state.state.normSquared
		guard normSquared.isFinite && normSquared > .zero else {
			throw SolverError.invalidStateNorm(time: time)
		}
		state.state.divideBLAS(by: normSquared.squareRoot())
	}
}

extension QSD {
	@inlinable
	@inline(always)
	@discardableResult
	public static func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (
			Double,
			borrowing UniqueVector<Complex<Double>>
		) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUQSDEngine()
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
	public static func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUQSDEngine()
		return try implementation.solveEnsemble(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			execution: execution,
			forEach
		)
	}

	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUQSDEngine()
		return try implementation.solveTrajectories(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			execution: execution,
			forEach
		)
	}
}
