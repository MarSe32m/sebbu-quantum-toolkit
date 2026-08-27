// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

@usableFromInline
internal struct OutputCursor {
	@usableFromInline
	internal let timeSpan: SimulationTimeSpan
	@usableFromInline
	internal let schedule: OutputSchedule

	@usableFromInline
	internal var explicitTimeIndex = 0
	@usableFromInline
	internal var uniformTimeIndex = 0
	@usableFromInline
	internal var lastAcceptedStepTime: Double?
	@usableFromInline
	internal var emittedFinalState = false

	@inlinable
	internal init(timeSpan: SimulationTimeSpan, schedule: OutputSchedule) {
		switch schedule {
		case .times(let times):
			var previous: Double?
			for time in times {
				precondition(
					time.isFinite
						&& time >= timeSpan.start
						&& time <= timeSpan.end,
					"Explicit output times must be finite and lie inside the simulation time span"
				)
				if let previous {
					precondition(
						time > previous,
						"Explicit output times must be strictly increasing"
					)
				}
				previous = time
			}

		case .uniform(let step):
			precondition(
				step.isFinite && step > .zero,
				"Uniform output spacing must be positive and finite"
			)
			precondition(
				timeSpan.start == timeSpan.end
					|| timeSpan.start + step > timeSpan.start,
				"Uniform output spacing is too small to advance time"
			)

		case .everyAcceptedStep, .final:
			break
		}

		self.timeSpan = timeSpan
		self.schedule = schedule
	}

	/// Returns an output at the initial time when the schedule requests one.
	@inlinable
	internal mutating func takeInitialTime() -> Double? {
		switch schedule {
		case .everyAcceptedStep:
			return nil

		case .times(let times):
			guard
				explicitTimeIndex < times.count,
				times[explicitTimeIndex] == timeSpan.start
			else {
				return nil
			}
			explicitTimeIndex += 1
			return timeSpan.start

		case .uniform:
			guard uniformTimeIndex == 0 else { return nil }
			uniformTimeIndex = 1
			return timeSpan.start

		case .final:
			guard timeSpan.start == timeSpan.end, !emittedFinalState else {
				return nil
			}
			emittedFinalState = true
			return timeSpan.start
		}
	}

	/// Returns the next scheduled time no later than an accepted step end.
	@inlinable
	internal mutating func nextTime(through acceptedStepEnd: Double) -> Double? {
		switch schedule {
		case .everyAcceptedStep:
			guard lastAcceptedStepTime != acceptedStepEnd else { return nil }
			lastAcceptedStepTime = acceptedStepEnd
			return acceptedStepEnd

		case .times(let times):
			guard explicitTimeIndex < times.count else { return nil }
			let time = times[explicitTimeIndex]
			guard time <= acceptedStepEnd else { return nil }
			explicitTimeIndex += 1
			return time

		case .uniform(let step):
			let time = timeSpan.start + Double(uniformTimeIndex) * step
			guard time <= acceptedStepEnd && time <= timeSpan.end else {
				return nil
			}
			uniformTimeIndex += 1
			return time

		case .final:
			guard acceptedStepEnd >= timeSpan.end, !emittedFinalState else {
				return nil
			}
			emittedFinalState = true
			return timeSpan.end
		}
	}
}
