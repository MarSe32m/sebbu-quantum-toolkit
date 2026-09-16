// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import BasicContainers
import Numerics
import SebbuBLAS
import SebbuScience
import Synchronization

extension HOPS.CPUEngine {
    @inlinable
	@discardableResult
	public func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> HOPS.EnsembleRunResult where Hamiltonian: HamiltonianFunction {
		let outputTimes = try _fixedEnsembleOutputTimes(
			timeSpan: propagation.timeSpan, schedule: propagation.output)
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let seed = execution.resolvedMasterSeed()
		let count = execution.trajectoryIDs.count
		let workers = min(execution.resolvedMaximumConcurrentTasks, count)
		let next = Atomic<Int>(0)
		let sums = Mutex(
			_emptyEnsembleSums(
				count: outputTimes.count, dimension: problem.system.dimension))
		let blasThreads = BLAS.getNumThreads()
		BLAS.setNumThreads(1)
		defer { BLAS.setNumThreads(blasThreads) }

		let results: [_TrajectoryEnsembleBatchResult] = FixedWorkerPool.with(
			workers: workers
		) { _ in
			var local = _emptyEnsembleSums(
				count: outputTimes.count, dimension: problem.system.dimension)
			var completed = 0
			while true {
				// Count offsets rather than IDs so an upper bound of UInt64.max
				// cannot wrap the work queue back to trajectory zero.
				let offset = next.wrappingAdd(1, ordering: .relaxed).oldValue
				if offset >= count { break }
				let id = execution.trajectoryIDs.lowerBound + UInt64(offset)
				var sample = 0
				do {
					_ = try _solveTrajectory(
						problem: problem, preparation: preparation,
						propagation: propagation,
						seed: seed, trajectoryID: id
					) { t, state in
						precondition(
							sample < outputTimes.count
								&& t == outputTimes[sample])
						let normalization =
							configuration.equationType == .linear
							? 1 : state.normSquared
						_accumulateStateProjector(
							state, normalization: normalization,
							into: &local[sample])
						sample += 1
						return .proceed
					}
					precondition(sample == outputTimes.count)
					completed += 1
				} catch {
					return .init(
						trajectoryCount: completed,
						failure: .init(trajectoryID: id, error: error))
				}
			}
			sums.withLock { _mergeEnsembleSums(local, into: &$0) }
			return .init(trajectoryCount: completed, failure: nil)
		}
		// Report solver failures before asserting successful completion. Do not
		// emit a partially averaged ensemble when any trajectory has failed.
		if let failure = results.compactMap(\.failure).min(by: {
			$0.trajectoryID < $1.trajectoryID
		}) {
			throw failure.error
		}
		precondition(results.reduce(0) { $0 + $1.trajectoryCount } == count)
		sums.withLock { values in
			for i in outputTimes.indices {
				values[i].multiply(by: 1 / Double(count))
				forEach(outputTimes[i], values[i])
			}
		}
		let summary = TrajectoryRunSummary(
			trajectoryIDs: execution.trajectoryIDs, masterSeed: seed,
			propagation: .init(
				finalTime: propagation.timeSpan.end, endReason: .reachedEndTime))
		let definition = HOPS.BathNoiseDefinition(
			model: configuration.hierarchy.environment.bath,
			timeSpan: propagation.timeSpan,
			stepSize: preparation.noise.step)
		return .init(
			summary: summary,
			bathNoise: .init(
				definition: definition, masterSeed: seed,
				trajectoryIDs: execution.trajectoryIDs))
	}

    @inlinable
	@discardableResult
	public func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> HOPS.EnsembleRunResult where Hamiltonian: HamiltonianFunction {
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let seed = execution.resolvedMasterSeed()
		let count = execution.trajectoryIDs.count
		let next = Atomic<Int>(0)
		let blasThreads = BLAS.getNumThreads()
		BLAS.setNumThreads(1)
		defer { BLAS.setNumThreads(blasThreads) }
		let failures: [_TrajectoryFailure?] = FixedWorkerPool.with(
			workers: min(count, execution.resolvedMaximumConcurrentTasks)
		) { _ in
			while true {
				let offset = next.wrappingAdd(1, ordering: .relaxed).oldValue
				if offset >= count { return nil }
				let id = execution.trajectoryIDs.lowerBound + UInt64(offset)
				do {
					_ = try _solveTrajectory(
						problem: problem, preparation: preparation,
						propagation: propagation,
						seed: seed, trajectoryID: id
					) { t, state in
						forEach(id, t, state)
						return .proceed
					}
				} catch { return .init(trajectoryID: id, error: error) }
			}
		}
		if let failure = failures.compactMap({ $0 }).min(by: {
			$0.trajectoryID < $1.trajectoryID
		}) {
			throw failure.error
		}
		let summary = TrajectoryRunSummary(
			trajectoryIDs: execution.trajectoryIDs, masterSeed: seed,
			propagation: .init(
				finalTime: propagation.timeSpan.end, endReason: .reachedEndTime))
		let definition = HOPS.BathNoiseDefinition(
			model: configuration.hierarchy.environment.bath,
			timeSpan: propagation.timeSpan,
			stepSize: preparation.noise.step)
		return .init(
			summary: summary,
			bathNoise: .init(
				definition: definition, masterSeed: seed,
				trajectoryIDs: execution.trajectoryIDs))
	}
}
