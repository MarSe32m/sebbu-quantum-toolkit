// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuCollections
import SebbuScience
import BasicContainers
import Synchronization

extension CPUMCWFEngine {
    @discardableResult
    public func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: MCWF.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        _ forEach: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> Void
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let outputTimes = try _fixedEnsembleOutputTimes(
            timeSpan: propagation.timeSpan,
            schedule: propagation.output
        )
        let masterSeed = execution.resolvedMasterSeed()
        let maximumConcurrentTasks = Swift.min(
            execution.resolvedMaximumConcurrentTasks,
            execution.trajectoryIDs.count
        )
        let trajectoryCount = execution.trajectoryIDs.count
        let dimension = problem.system.dimension
        let ensembleSums: Mutex<UniqueArray<UniqueMatrix<Complex<Double>>>> = .init(
            _emptyEnsembleSums(
                count: outputTimes.count,
                dimension: dimension
            )
        )
        
        let currentTrajectoryID: Atomic<UInt64> = Atomic(execution.trajectoryIDs.lowerBound)
        let results: [_TrajectoryEnsembleBatchResult] = FixedWorkerPool.with(workers: maximumConcurrentTasks) { workerID in
            var localSums = _emptyEnsembleSums(
                count: outputTimes.count,
                dimension: dimension
            )
            var trajectoryID: UInt64 = currentTrajectoryID.wrappingAdd(1, ordering: .relaxed).oldValue
            var trajectoryCount = 0
            while trajectoryID < execution.trajectoryIDs.upperBound {
                var sampleIndex = 0
                var randomNumberGenerator =
                TrajectoryRandomNumberGenerator(
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
                        precondition(
                            sampleIndex
                            < outputTimes.count
                            && time
                            == outputTimes[
                                sampleIndex
                            ],
                            "An ensemble trajectory produced an unexpected output time"
                        )
                        _accumulateStateProjector(
                            state,
                            normalization: state
                                .normSquared,
                            into: &localSums[
                                sampleIndex]
                        )
                        sampleIndex += 1
                        return .proceed
                    }
                    precondition(
                        sampleIndex == outputTimes.count,
                        "An ensemble trajectory did not produce every output sample"
                    )
                    precondition(
                        summary.finalTime
                        == propagation.timeSpan.end
                        && summary.endReason
                        == .reachedEndTime,
                        "An ensemble trajectory must reach the end time"
                    )
                } catch {
                    return _TrajectoryEnsembleBatchResult(
                        trajectoryCount: trajectoryCount,
                        failure: _TrajectoryFailure(
                            trajectoryID: trajectoryID,
                            error: error
                        )
                    )
                }
                trajectoryID = currentTrajectoryID.add(1, ordering: .relaxed).oldValue
                trajectoryCount += 1
            }
            ensembleSums.withLock { sums in _mergeEnsembleSums(localSums, into: &sums) }
            return _TrajectoryEnsembleBatchResult(
                trajectoryCount: trajectoryCount,
                failure: nil
            )
        }
        let completedTrajectories = results.reduce(into: 0) { $0 = $0 + $1.trajectoryCount }
        
        precondition(
            completedTrajectories == trajectoryCount,
            "The ensemble reduction omitted one or more trajectories"
        )
        let inverseTrajectoryCount = 1 / Double(trajectoryCount)
        ensembleSums.withLock { sums in
            for index in outputTimes.indices {
                sums[index].multiply(by: inverseTrajectoryCount)
                forEach(outputTimes[index], sums[index])
            }
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
    public static func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: Configuration,
        propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
        execution: TrajectoryExecution,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        return try implementation.solveEnsemble(
            problem: problem,
            configuration: configuration,
            propagation: propagation,
            execution: execution,
            forEach
        )
    }
}
