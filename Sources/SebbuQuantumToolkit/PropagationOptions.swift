// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct PropagationOptions<Integration: Sendable>: Sendable {
	public var timeSpan: SimulationTimeSpan
	public var output: OutputSchedule
	public var integration: Integration

	@inlinable
	public init(
		timeSpan: SimulationTimeSpan,
		output: OutputSchedule,
		integration: Integration
	) {
		self.timeSpan = timeSpan
		self.output = output
		self.integration = integration
	}
}
