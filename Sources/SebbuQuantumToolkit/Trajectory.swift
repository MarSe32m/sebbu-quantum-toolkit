// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuScience
import Numerics

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

public struct TrajectoryExecution: Sendable {
    /// Global trajectory identifiers.
    public var trajectoryIDs: Range<UInt64>

    public var randomness: TrajectoryRandomness
    public var parallelism: TrajectoryParallelism

    /// Optional CPU chunk or GPU batch-size hint.
    /// `nil` lets the implementation choose.
    public var batchSize: Int?

    public init(
        trajectoryIDs: Range<UInt64>,
        randomness: TrajectoryRandomness,
        parallelism: TrajectoryParallelism = .automatic,
        batchSize: Int? = nil
    ) {
        precondition(!trajectoryIDs.isEmpty)
        precondition(batchSize == nil || batchSize! > 0)

        self.trajectoryIDs = trajectoryIDs
        self.randomness = randomness
        self.parallelism = parallelism
        self.batchSize = batchSize
    }

    public init(
        trajectories count: Int,
        startingAt firstTrajectoryID: UInt64 = 0,
        seed: UInt64,
        parallelism: TrajectoryParallelism = .automatic,
        batchSize: Int? = nil
    ) {
        precondition(count > 0)

        self.init(
            trajectoryIDs:
                firstTrajectoryID..<(firstTrajectoryID + UInt64(count)),
            randomness: .seeded(seed),
            parallelism: parallelism,
            batchSize: batchSize
        )
    }
}

public struct TrajectoryRunSummary: Sendable {
    public let trajectoryIDs: Range<UInt64>

    /// The actual seed, including one generated for `.nondeterministic`.
    public let masterSeed: UInt64
}
