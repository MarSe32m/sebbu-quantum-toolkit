// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

@usableFromInline
internal final class ContinuousProgressReporter {
	@usableFromInline internal let minValue: Double
	@usableFromInline internal let maxValue: Double
	@usableFromInline internal let initialValue: Double
	@usableFromInline internal var currentValue: Double
	@usableFromInline internal let display: ProgressReporting.Display
	@usableFromInline internal var progressLine: ProgressLine
	@usableFromInline internal let startTime: ContinuousClock.Instant
	@usableFromInline internal var lastReportedPercentage: Int

	@usableFromInline
	internal init(
		minValue: Double, maxValue: Double, currentValue: Double,
		display: ProgressReporting.Display
	) {
		precondition(minValue.isFinite && maxValue.isFinite && maxValue >= minValue)
		precondition(
			currentValue.isFinite && currentValue >= minValue
				&& currentValue <= maxValue)
		self.minValue = minValue
		self.maxValue = maxValue
		self.initialValue = currentValue
		self.currentValue = currentValue
		self.display = display
		self.progressLine = ProgressLine(write: display.write)
		self.startTime = .now
		// A zero-duration solve reports completion only after it returns successfully.
		self.lastReportedPercentage =
			minValue == maxValue
			? 0
			: Self.percentage(
				for: currentValue, minValue: minValue, maxValue: maxValue)
		progressLine.update(
			display.message(
				percentage: lastReportedPercentage, fraction: 0, elapsed: .zero))
	}

	@usableFromInline
	internal convenience init(
		minValue: Double, maxValue: Double, currentValue: Double,
		printTimeLeft: Bool = false
	) {
		self.init(
			minValue: minValue, maxValue: maxValue, currentValue: currentValue,
			display: .init(
				style: .percentage, label: "Progress", printTimeLeft: printTimeLeft,
				barWidth: 30, write: { ProgressLine.write($0, to: .standardOutput) }
			))
	}

	@usableFromInline
	internal convenience init(range: ClosedRange<Double>, printTimeLeft: Bool = false) {
		self.init(
			minValue: range.lowerBound, maxValue: range.upperBound,
			currentValue: range.lowerBound, printTimeLeft: printTimeLeft)
	}

	@inlinable
	internal func setValue(to value: Double) {
		precondition(value.isFinite)
		currentValue = max(currentValue, min(max(value, minValue), maxValue))
		let percentage = Self.percentage(
			for: currentValue, minValue: minValue, maxValue: maxValue)
		guard percentage > lastReportedPercentage && !progressLine.isFinished else {
			return
		}
		lastReportedPercentage = percentage
		// Clock reads and formatting occur only after the percentage gate.
		progressLine.update(
			display.message(
				percentage: percentage,
				fraction: Self.fraction(
					for: currentValue, minValue: initialValue,
					maxValue: maxValue),
				elapsed: startTime.duration(to: .now)))
	}

	@usableFromInline
	internal func finish() { progressLine.finish() }

	@inlinable
	internal static func fraction(for value: Double, minValue: Double, maxValue: Double)
		-> Double
	{
		if value >= maxValue { return 1 }
		if value <= minValue { return 0 }
		let width = maxValue - minValue
		if width.isFinite { return (value - minValue) / width }
		return (value / 2 - minValue / 2) / (maxValue / 2 - minValue / 2)
	}

	@inlinable
	internal static func percentage(for value: Double, minValue: Double, maxValue: Double)
		-> Int
	{
		if value >= maxValue { return 100 }
		// Floating-point rounding must not report completion before the endpoint.
		return min(
			99, Int(fraction(for: value, minValue: minValue, maxValue: maxValue) * 100))
	}
}

@inlinable
internal func withContinuousProgressReporter<R>(
	minValue: Double,
	maxValue: Double,
	currentValue: Double,
	printTimeLeft: Bool = false,
	_ body: (ContinuousProgressReporter) throws -> R
) rethrows -> R {
	let reporter = ContinuousProgressReporter(
		minValue: minValue,
		maxValue: maxValue,
		currentValue: currentValue,
		printTimeLeft: printTimeLeft
	)
	defer { reporter.finish() }
	return try body(reporter)
}

@inlinable
internal func withContinuousProgressReporter<R>(
	range: ClosedRange<Double>,
	printTimeLeft: Bool = false,
	_ body: (ContinuousProgressReporter) throws -> R
) rethrows -> R {
	let reporter = ContinuousProgressReporter(
		range: range,
		printTimeLeft: printTimeLeft
	)
	defer { reporter.finish() }
	return try body(reporter)
}
