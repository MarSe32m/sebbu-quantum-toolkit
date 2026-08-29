// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import BasicContainers

public enum TrajectoryRandomness: Sendable {
	/// Derive each trajectory stream from the master seed and trajectory ID.
	case seeded(UInt64)

	/// Generate and record a master seed when the run begins.
	case nondeterministic
}

public enum TrajectoryParallelism: Sendable {
	case serial
	case automatic
	case maximumConcurrentTasks(Int)
}

public enum TrajectoryEnsembleError: Error, Equatable, Sendable {
	/// Accepted-step times can differ between independently propagated trajectories.
	case everyAcceptedStepOutputIsNotSupported
}

public struct TrajectoryExecution: Sendable {
	/// Global trajectory identifiers.
	public var trajectoryIDs: Range<UInt64>

	public var randomness: TrajectoryRandomness
	public var parallelism: TrajectoryParallelism

	/// Optional CPU scheduling and reduction-batch hint, or GPU batch-size hint.
	/// `nil` lets the implementation choose. Changing it does not change any
	/// trajectory's random stream, although floating-point ensemble reductions
	/// may be grouped differently.
	public var batchSize: Int?

	public init(
		trajectoryIDs: Range<UInt64>,
		randomness: TrajectoryRandomness,
		parallelism: TrajectoryParallelism = .automatic,
		batchSize: Int? = nil
	) {
		precondition(!trajectoryIDs.isEmpty)
		precondition(batchSize == nil || batchSize! > 0)
		if case .maximumConcurrentTasks(let tasks) = parallelism {
			precondition(
				tasks > 0, "The maximum concurrent task count must be positive")
		}

		self.trajectoryIDs = trajectoryIDs
		self.randomness = randomness
		self.parallelism = parallelism
		self.batchSize = batchSize
	}

	public init(
		trajectories count: Int,
		startingAt firstTrajectoryID: UInt64 = 0,
		seed: UInt64 = .random(in: .min ... .max),
		parallelism: TrajectoryParallelism = .automatic,
		batchSize: Int? = nil
	) {
		precondition(count > 0)
		let unsignedCount = UInt64(count)
		precondition(
			unsignedCount <= UInt64.max - firstTrajectoryID,
			"The requested trajectory-ID range overflows UInt64"
		)

		self.init(
			trajectoryIDs:
				firstTrajectoryID..<(firstTrajectoryID + unsignedCount),
			randomness: .seeded(seed),
			parallelism: parallelism,
			batchSize: batchSize
		)
	}
}

extension TrajectoryExecution {
	internal func resolvedMasterSeed() -> UInt64 {
		switch randomness {
		case .seeded(let seed):
			return seed
		case .nondeterministic:
			return .random(in: .min ... .max)
		}
	}

	internal var resolvedMaximumConcurrentTasks: Int {
		precondition(!trajectoryIDs.isEmpty, "At least one trajectory is required")
		switch parallelism {
		case .serial:
			return 1
		case .automatic:
            return Swift.min(Platform.activeProcessorCount, trajectoryIDs.count)
		case .maximumConcurrentTasks(let tasks):
			precondition(
				tasks > 0, "The maximum concurrent task count must be positive")
			return tasks
		}
	}

	internal var resolvedBatchSize: Int {
		let result = batchSize ?? 1
		precondition(result > 0, "The trajectory batch size must be positive")
		return result
	}
}

internal struct _TrajectoryFailure: Sendable {
	internal let trajectoryID: UInt64
	internal let error: any Error
}

internal struct _TrajectorySolveResult: Sendable {
	internal let trajectoryID: UInt64
	internal let error: (any Error)?
}

internal struct _TrajectoryEnsembleBatchResult: Sendable {
    internal let trajectoryCount: Int
	internal let failure: _TrajectoryFailure?
}

internal func _fixedEnsembleOutputTimes(
	timeSpan: SimulationTimeSpan,
	schedule: OutputSchedule
) throws -> [Double] {
	if case .everyAcceptedStep = schedule {
		throw TrajectoryEnsembleError.everyAcceptedStepOutputIsNotSupported
	}

	var cursor = OutputCursor(timeSpan: timeSpan, schedule: schedule)
	var result: [Double] = []
	if let initialTime = cursor.takeInitialTime() {
		result.append(initialTime)
	}
	while let time = cursor.nextTime(through: timeSpan.end) {
		result.append(time)
	}
	return result
}

internal func _emptyEnsembleSums(
	count: Int,
	dimension: Int
) -> UniqueArray<UniqueMatrix<Complex<Double>>> {
    UniqueArray(capacity: count) { span in
        for _ in 0..<count {
            span.append(.zeros(rows: dimension, columns: dimension))
        }
    }
}

internal func _accumulateStateProjector(
	_ state: borrowing UniqueVector<Complex<Double>>,
	normalization: Double,
	into result: inout UniqueMatrix<Complex<Double>>
) {
	precondition(
		normalization.isFinite && normalization > .zero,
		"A trajectory projector requires a finite, positive normalization"
	)
	precondition(
		result.rows == state.count && result.columns == state.count,
		"The density-matrix accumulator dimension does not match the state"
	)

	let inverseNormalization = 1 / normalization
	for row in 0..<state.count {
		let scaledRow = inverseNormalization * state[row]
		for column in 0..<state.count {
			result[row, column] += scaledRow * state[column].conjugate
		}
	}
}

internal func _mergeEnsembleSums(
	_ contribution: borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
	into result: inout UniqueArray<UniqueMatrix<Complex<Double>>>
) {
	precondition(
		result.count == contribution.count,
		"Ensemble batches produced different output counts"
	)
	for index in result.indices {
		result[index].add(contribution[index])
	}
}

public struct TrajectoryRunSummary: Sendable {
	public let trajectoryIDs: Range<UInt64>

	/// The actual seed, including one generated for `.nondeterministic`.
	public let masterSeed: UInt64

	public let propagation: PropagationRunSummary

	@inlinable
	public init(
		trajectoryIDs: Range<UInt64>, masterSeed: UInt64, propagation: PropagationRunSummary
	) {
		self.trajectoryIDs = trajectoryIDs
		self.masterSeed = masterSeed
		self.propagation = propagation
	}
}
