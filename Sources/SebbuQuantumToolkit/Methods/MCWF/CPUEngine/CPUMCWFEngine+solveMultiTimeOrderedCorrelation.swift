// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience
import Synchronization

extension CPUMCWFEngine: MCWF.MultiTimeOrderedCorrelationImplementation {
	@discardableResult
	public func solveMultiTimeOrderedCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		Self.validate(configuration: configuration)
		try _validateMultiTimeOrderedCorrelationRequest(
			request,
			timeSpan: propagation.timeSpan,
			dimension: problem.system.dimension
		)
		let lastInsertionTime = request.insertions.last!.time
		let outputTimes = try _fixedEnsembleOutputTimes(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		).filter { $0 >= lastInsertionTime }
		let masterSeed = execution.resolvedMasterSeed()
		let trajectoryCount = execution.trajectoryIDs.count
		let workerCount = Swift.min(
			execution.resolvedMaximumConcurrentTasks,
			trajectoryCount
		)

		let currentThreadCount = BLAS.getNumThreads()
		BLAS.setNumThreads(1)
		defer { BLAS.setNumThreads(currentThreadCount) }
		let sums = Mutex([Complex<Double>](repeating: .zero, count: outputTimes.count))
		let currentID = Atomic(execution.trajectoryIDs.lowerBound)
		let results: [_TrajectoryEnsembleBatchResult] = FixedWorkerPool.with(
			workers: workerCount
		) { _ in
			var local = [Complex<Double>](repeating: .zero, count: outputTimes.count)
			var trajectoryID = currentID.wrappingAdd(1, ordering: .relaxed).oldValue
			var completed = 0
			while trajectoryID < execution.trajectoryIDs.upperBound {
				var rng = TrajectoryRandomNumberGenerator(
					seed: masterSeed,
					trajectoryID: trajectoryID,
					purpose: .mcwfJumps
				)
				var sampleIndex = 0
				do {
					let summary = try solveMultiTimeCorrelationTrajectory(
						problem: problem,
						configuration: configuration,
						request: request,
						propagation: propagation,
						rng: &rng
					) { time, value in
						precondition(
							sampleIndex < outputTimes.count
								&& time == outputTimes[sampleIndex]
						)
						local[sampleIndex] += value
						sampleIndex += 1
					}
					precondition(sampleIndex == outputTimes.count)
					precondition(
						summary.finalTime == propagation.timeSpan.end
							&& summary.endReason == .reachedEndTime
					)
				} catch {
					return .init(
						trajectoryCount: completed,
						failure: .init(trajectoryID: trajectoryID, error: error)
					)
				}
				completed += 1
				trajectoryID = currentID.add(1, ordering: .relaxed).oldValue
			}
			sums.withLock { total in
				for index in total.indices { total[index] += local[index] }
			}
			return .init(trajectoryCount: completed, failure: nil)
		}

		if let failure = results.compactMap(\.failure).first {
			throw failure.error
		}
		precondition(results.reduce(0) { $0 + $1.trajectoryCount } == trajectoryCount)
		let inverseCount = 1 / Double(trajectoryCount)
		var summary = PropagationRunSummary(
			finalTime: propagation.timeSpan.end,
			endReason: .reachedEndTime
		)
		sums.withLock { values in
			for index in outputTimes.indices where summary.endReason == .reachedEndTime {
				if observer(outputTimes[index], inverseCount * values[index]) == .stop {
					summary = .init(
						finalTime: outputTimes[index],
						endReason: .stoppedByObserver
					)
				}
			}
		}
		return .init(
			trajectoryIDs: execution.trajectoryIDs,
			masterSeed: masterSeed,
			propagation: summary
		)
	}
}

extension MCWF {
	@discardableResult
	public static func solveMultiTimeOrderedCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration = .init(),
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		try CPUMCWFEngine().solveMultiTimeOrderedCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			execution: execution,
			observing: observer
		)
	}
}
