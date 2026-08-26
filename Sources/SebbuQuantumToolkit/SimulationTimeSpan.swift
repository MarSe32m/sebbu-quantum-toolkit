// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct SimulationTimeSpan: Sendable {
	public var start: Double
	public var end: Double

	@inlinable
	public init(start: Double, end: Double) {
		precondition(
			start.isFinite && end.isFinite,
			"Simulation times must be finite"
		)
		precondition(end >= start, "Simulation end must not precede its start")
		self.start = start
		self.end = end
	}
}
