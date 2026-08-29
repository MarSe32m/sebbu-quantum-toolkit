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

	/// Advances fixed output schedules to their first time at or after
	/// `lowerBound` without emitting the skipped times.
	///
	/// This is used when an observable becomes defined only partway through a
	/// propagation, as with a two-time-correlation insertion. Accepted-step
	/// and final-only schedules do not contain fixed times to discard.
	@inlinable
	internal mutating func discardTimes(before lowerBound: Double) {
		precondition(
			lowerBound.isFinite
				&& lowerBound >= timeSpan.start
				&& lowerBound <= timeSpan.end,
			"The output lower bound must lie inside the simulation time span"
		)

		switch schedule {
		case .times(let times):
			// Find the first explicit time that is not less than lowerBound.
			var lower = explicitTimeIndex
			var upper = times.count
			while lower < upper {
				let middle = lower + (upper - lower) / 2
				if times[middle] < lowerBound {
					lower = middle + 1
				} else {
					upper = middle
				}
			}
			explicitTimeIndex = lower

		case .uniform(let step):
			guard lowerBound > timeSpan.start else { return }

			let quotient = (lowerBound - timeSpan.start) / step
			precondition(
				quotient.isFinite && quotient < Double(Int.max),
				"The uniform output schedule contains too many samples"
			)

			// Starting from floor(quotient), rather than ceil(quotient), keeps
			// an exactly aligned grid point when the division rounded slightly
			// upward. The loop corrects either rounding direction.
			var index = Swift.max(
				uniformTimeIndex,
				Int(quotient.rounded(.down))
			)
			while timeSpan.start + Double(index) * step < lowerBound {
				index += 1
			}
			uniformTimeIndex = index

		case .everyAcceptedStep, .final:
			break
		}
	}

	/// Returns the next fixed output time that an integrator must land on.
	///
	/// Deterministic integrators can interpolate inside an accepted step, but
	/// stochastic integrators generally cannot do so without retaining the
	/// driving Brownian path. Such integrators use this value as a hard step
	/// boundary and then call ``nextTime(through:)`` after accepting the step.
	@inlinable
	internal var nextRequiredStepBoundary: Double? {
		switch schedule {
		case .everyAcceptedStep:
			return nil

		case .times(let times):
			guard explicitTimeIndex < times.count else { return nil }
			return times[explicitTimeIndex]

		case .uniform(let step):
			let time = timeSpan.start + Double(uniformTimeIndex) * step
			return time <= timeSpan.end ? time : nil

		case .final:
			return emittedFinalState ? nil : timeSpan.end
		}
	}

	/// Returns the next fixed output time strictly before `upperBound`.
	///
	/// Event-driven propagators use this to emit dense-output samples which
	/// precede a discontinuity. The state at the discontinuity itself can then
	/// be emitted separately, after the jump has been applied. Accepted-step
	/// and final-only schedules have no pre-event sample to return.
	@inlinable
	internal mutating func nextTime(before upperBound: Double) -> Double? {
		switch schedule {
		case .everyAcceptedStep, .final:
			return nil

		case .times(let times):
			guard explicitTimeIndex < times.count else { return nil }
			let time = times[explicitTimeIndex]
			guard time < upperBound else { return nil }
			explicitTimeIndex += 1
			return time

		case .uniform(let step):
			let time = timeSpan.start + Double(uniformTimeIndex) * step
			guard time < upperBound && time <= timeSpan.end else {
				return nil
			}
			uniformTimeIndex += 1
			return time
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
