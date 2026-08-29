// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import SebbuCollections

extension CPUQSDEngine {
    @usableFromInline
    internal struct _SolveResult: Sendable {
        @usableFromInline
        let id: UInt64
        @usableFromInline
        let error: (any Error)?
        
        @inlinable
        init(id: UInt64, error: (any Error)?) {
            self.id = id
            self.error = error
        }
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
        let masterSeed: UInt64 = switch execution.randomness {
            case .nondeterministic: .random(in: .min ... .max)
            case .seeded(let seed): seed
        }
        let batchSize = execution.batchSize ?? 1
        let parallelism: Int? = switch execution.parallelism {
            case .serial: 1
            case .automatic: nil
            case .maximumConcurrentTasks(let tasks): tasks
        }
        let results: [_SolveResult] = execution.trajectoryIDs.parallelMap(parallelism: parallelism, blockSize: batchSize) { trajectoryID in
            do {
                try solve(problem: problem, configuration: configuration, propagation: propagation, seed: masterSeed, trajectoryID: trajectoryID) { time, state in
                    forEach(trajectoryID, time, state)
                    return .proceed
                }
            } catch {
                return _SolveResult(id: trajectoryID, error: error)
            }
            return _SolveResult(id: trajectoryID, error: nil)
        }
        if let error = results.first(where: { $0.error != nil })?.error {
            throw error
        }
        let endTime = switch propagation.output {
        case .everyAcceptedStep, .final, .uniform(step: _): propagation.timeSpan.end
        case .times(let times): times.last!
        }
        return TrajectoryRunSummary(trajectoryIDs: execution.trajectoryIDs, masterSeed: masterSeed, propagation: .init(finalTime: endTime, endReason: .reachedEndTime))
    }
}

extension QSD {
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
