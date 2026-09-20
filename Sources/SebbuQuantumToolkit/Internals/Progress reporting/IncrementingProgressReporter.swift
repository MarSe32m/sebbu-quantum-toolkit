// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Synchronization

@usableFromInline
internal final class IncrementingProgressReporter: Sendable {
	@usableFromInline internal let minValue: Int
	@usableFromInline internal let maxValue: Int
	@usableFromInline internal let initialValue: Int
	@usableFromInline internal let state: State

	internal var currentValue: Int { state.currentValue.load(ordering: .relaxed) }

	@usableFromInline
	internal init(
		minValue: Int, maxValue: Int, currentValue: Int,
		display: ProgressReporting.Display
	) {
		precondition(maxValue > minValue)
		precondition(currentValue >= minValue && currentValue <= maxValue)
		self.minValue = minValue
		self.maxValue = maxValue
		self.initialValue = currentValue
		self.state = State(
			currentValue: currentValue,
			lastReportedPercentage: Self.percentage(
				for: currentValue, minValue: minValue, maxValue: maxValue),
			display: display)
	}

	@usableFromInline
	internal convenience init(
		minValue: Int, maxValue: Int, currentValue: Int, printTimeLeft: Bool = false
	) {
		self.init(
			minValue: minValue, maxValue: maxValue, currentValue: currentValue,
			display: .init(
				style: .percentage, label: "Progress", printTimeLeft: printTimeLeft,
				barWidth: 30, write: { ProgressLine.write($0, to: .standardOutput) }
			))
	}

	@usableFromInline
	internal convenience init(total: Int, printTimeLeft: Bool = false) {
		self.init(
			minValue: 0, maxValue: total, currentValue: 0, printTimeLeft: printTimeLeft)
	}

	@inlinable
	internal func increment(by amount: Int = 1) {
		precondition(amount > 0)
		let (previous, current) = state.currentValue.wrappingAdd(amount, ordering: .relaxed)
		precondition(current > previous, "Progress counter overflow")
		let previousPercentage = Self.percentage(
			for: previous, minValue: minValue, maxValue: maxValue)
		let percentage = Self.percentage(
			for: current, minValue: minValue, maxValue: maxValue)
		guard percentage > previousPercentage else { return }
		// At most one reporting attempt per percentage boundary, not per trajectory.
		state.reporting.withLock { reporting in
			guard
				percentage > reporting.lastReportedPercentage
					&& !reporting.progressLine.isFinished
			else { return }
			reporting.lastReportedPercentage = percentage
			reporting.progressLine.update(
				state.display.message(
					percentage: percentage,
					fraction: Self.fraction(
						for: current, minValue: initialValue,
						maxValue: maxValue),
					elapsed: state.startTime.duration(to: .now)))
		}
	}

	@usableFromInline
	internal func finish() {
		state.reporting.withLock { $0.progressLine.finish() }
	}

	@inlinable
	internal static func fraction(for value: Int, minValue: Int, maxValue: Int) -> Double {
		if value >= maxValue { return 1 }
		if value <= minValue { return 0 }
		// Subtract as integers before converting, including ranges near Int.max.
		let completed = UInt(bitPattern: value) &- UInt(bitPattern: minValue)
		let total = UInt(bitPattern: maxValue) &- UInt(bitPattern: minValue)
		return Double(completed) / Double(total)
	}

	@inlinable
	internal static func percentage(for value: Int, minValue: Int, maxValue: Int) -> Int {
		if value >= maxValue { return 100 }
		return min(
			99, Int(fraction(for: value, minValue: minValue, maxValue: maxValue) * 100))
	}
}

extension IncrementingProgressReporter {
	@usableFromInline
	internal struct ReportingState: Sendable {
		@usableFromInline internal var progressLine: ProgressLine
		@usableFromInline internal var lastReportedPercentage: Int
	}

	@usableFromInline
	internal struct State: ~Copyable, Sendable {
		@usableFromInline internal let currentValue: Atomic<Int>
		@usableFromInline internal let reporting: Mutex<ReportingState>
		@usableFromInline internal let startTime: ContinuousClock.Instant
		@usableFromInline internal let display: ProgressReporting.Display

		@usableFromInline
		internal init(
			currentValue: Int, lastReportedPercentage: Int,
			display: ProgressReporting.Display
		) {
			self.currentValue = .init(currentValue)
			self.startTime = .now
			self.display = display
			var line = ProgressLine(write: display.write)
			line.update(
				display.message(
					percentage: lastReportedPercentage, fraction: 0,
					elapsed: .zero))
			self.reporting = .init(
				ReportingState(
					progressLine: line,
					lastReportedPercentage: lastReportedPercentage))
		}
	}
}

@inlinable
internal func withIncrementingProgressReporter<R>(
	minValue: Int,
	maxValue: Int,
	currentValue: Int,
	printTimeLeft: Bool = false,
	_ body: (IncrementingProgressReporter) throws -> R
) rethrows -> R {
	let reporter = IncrementingProgressReporter(
		minValue: minValue,
		maxValue: maxValue,
		currentValue: currentValue,
		printTimeLeft: printTimeLeft
	)
	defer { reporter.finish() }
	return try body(reporter)
}

@inlinable
internal func withIncrementingProgressReporter<R>(
	total: Int,
	printTimeLeft: Bool = false,
	_ body: (IncrementingProgressReporter) throws -> R
) rethrows -> R {
	let reporter = IncrementingProgressReporter(
		total: total,
		printTimeLeft: printTimeLeft
	)
	defer { reporter.finish() }
	return try body(reporter)
}
