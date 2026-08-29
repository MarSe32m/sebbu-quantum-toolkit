// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuCollections
import SebbuScience

extension CPUMCWFEngine {
	@discardableResult
	public func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let masterSeed = execution.resolvedMasterSeed()
		let parallelism = Swift.min(
			execution.resolvedMaximumConcurrentTasks,
			execution.trajectoryIDs.count
		)
		let results: [_TrajectorySolveResult] = execution.trajectoryIDs.parallelMap(
			parallelism: parallelism,
			blockSize: execution.resolvedBatchSize
		) { trajectoryID in
			var randomNumberGenerator = TrajectoryRandomNumberGenerator(
				seed: masterSeed,
				trajectoryID: trajectoryID,
				purpose: .mcwfJumps
			)
			do {
				let summary = try solveTrajectory(
					problem: problem,
					configuration: configuration,
					propagation: propagation,
					rng: &randomNumberGenerator
				) { time, state in
					forEach(trajectoryID, time, state)
					return .proceed
				}
				precondition(
					summary.finalTime == propagation.timeSpan.end
						&& summary.endReason == .reachedEndTime,
					"A trajectory observer without termination control must reach the end time"
				)
			} catch {
				return _TrajectorySolveResult(
					trajectoryID: trajectoryID,
					error: error
				)
			}
			return _TrajectorySolveResult(
				trajectoryID: trajectoryID,
				error: nil
			)
		}
		if let error = results.first(where: { $0.error != nil })?.error {
			throw error
		}
		return TrajectoryRunSummary(
			trajectoryIDs: execution.trajectoryIDs,
			masterSeed: masterSeed,
			propagation: .init(
				finalTime: propagation.timeSpan.end,
				endReason: .reachedEndTime
			)
		)
	}
}

extension MCWF {
	@inlinable
	@inline(always)
	@discardableResult
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUMCWFEngine()
		return try implementation.solveTrajectories(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			execution: execution,
			forEach
		)
	}
}
