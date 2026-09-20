// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct PropagationOptions<Integration: Sendable>: Sendable {
	public var timeSpan: SimulationTimeSpan
	public var output: OutputSchedule
	public var integration: Integration

	/// Optional progress for the whole invocation. Defaults to no output.
	/// Deterministic methods report time steps. Trajectory methods report completions.
	public var progress: ProgressReporting

	@inlinable
	public init(
		timeSpan: SimulationTimeSpan,
		output: OutputSchedule,
		integration: Integration,
		progress: ProgressReporting = .none
	) {
		self.timeSpan = timeSpan
		self.output = output
		self.integration = integration
		self.progress = progress
	}

	/// Worker trajectories and preparation segments must not create nested displays.
	@inlinable
	internal var withoutProgressReporting: Self {
		var copy = self
		copy.progress = .none
		return copy
	}
}

public enum PropagationControl: Sendable, Equatable {
    case proceed
    case stop
}

public enum PropagationEndReason: Sendable, Equatable {
    case reachedEndTime
    case stoppedByObserver
}

public struct PropagationRunSummary: Sendable {
    public let finalTime: Double
    public let endReason: PropagationEndReason
    
    @inlinable
    public init(finalTime: Double, endReason: PropagationEndReason) {
        self.finalTime = finalTime
        self.endReason = endReason
    }
}
