// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuScience

/// Philox4x64 stream used to derive a trajectory from a master seed, channel number,
/// trajectory identifier and a purpose.
@usableFromInline
internal struct TrajectoryRandomNumberGenerator: RandomNumberGenerator, Sendable {
    @usableFromInline
    internal enum Purpose: UInt64 {
        case unspecified = 0
        case gaussianWhiteNoise
        case coloredNoiseGeneration
        case mcwfJumps
    }
    
    @usableFromInline
    internal var generator: Philox4x64

    @inlinable
    internal init(seed: UInt64, channel: UInt64 = 0, trajectoryID: UInt64 = 0, purpose: Purpose = .unspecified) {
        var splitMix = SplitMix64(seed: seed)
        self.generator = Philox4x64(
            key: .init(splitMix.next(), splitMix.next()),
            counter: .init(
                0/* block index */,
                channel/* channel */,
                trajectoryID /* trajectory id */,
                purpose.rawValue/* purpose */
            )
        )
    }

    @inlinable
    internal mutating func next() -> UInt64 {
        generator.next()
    }
}
