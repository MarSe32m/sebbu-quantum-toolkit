// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct SimulationTimeSpan: Sendable {
    public var start: Double
    public var end: Double

    @inlinable
    public init(start: Double, end: Double) {
        precondition(end >= start)
        self.start = start
        self.end = end
    }
}
