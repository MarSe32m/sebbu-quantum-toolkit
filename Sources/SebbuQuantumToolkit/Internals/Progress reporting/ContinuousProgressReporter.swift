// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

@usableFromInline
internal final class ContinuousProgressReporter {
    public let minValue: Double
    public let maxValue: Double
    @usableFromInline
    internal var currentValue: Double

    public let printTimeLeft: Bool

    @usableFromInline
    internal var progressLine: ProgressLine

    @usableFromInline
    internal let startTime: ContinuousClock.Instant

    @usableFromInline
    internal var lastReportedPercentage: Int

    @inlinable
    public init(
        minValue: Double,
        maxValue: Double,
        currentValue: Double,
        printTimeLeft: Bool = false
    ) {
        precondition(maxValue > minValue, "maxValue must be greater than minValue")
        precondition(
            currentValue >= minValue && currentValue <= maxValue,
            "currentValue must lie within minValue...maxValue"
        )

        self.minValue = minValue
        self.maxValue = maxValue
        self.currentValue = currentValue
        self.progressLine = ProgressLine()
        self.startTime = .now
        self.printTimeLeft = printTimeLeft
        self.lastReportedPercentage = Self.percentage(
            for: currentValue,
            minValue: minValue,
            maxValue: maxValue
        )
    }

    @inlinable
    public convenience init(
        range: ClosedRange<Double>,
        printTimeLeft: Bool = false
    ) {
        self.init(
            minValue: range.lowerBound,
            maxValue: range.upperBound,
            currentValue: range.lowerBound,
            printTimeLeft: printTimeLeft
        )
    }

    @inlinable
    public func setValue(to value: Double) {
        currentValue = value

        let currentPercentage = Self.percentage(
            for: value,
            minValue: minValue,
            maxValue: maxValue
        )
        guard currentPercentage > lastReportedPercentage else {
            return
        }
        lastReportedPercentage = currentPercentage

        guard printTimeLeft else {
            progressLine.update("Progress \(currentPercentage)%")
            return
        }

        let clampedValue = min(max(value, minValue), maxValue)
        guard clampedValue > minValue else {
            progressLine.update("Progress \(currentPercentage)%")
            return
        }

        let totalDuration = startTime.duration(to: .now)
        let completed = clampedValue - minValue
        let remaining = maxValue - clampedValue
        let remainingDuration = totalDuration * remaining / completed
        progressLine.update(
            "Progress \(currentPercentage)%, estimated time left: \(remainingDuration)"
        )
    }

    @inlinable
    public func finish() {
        progressLine.finish()
    }

    @inlinable
    @inline(always)
    internal static func percentage(
        for value: Double,
        minValue: Double,
        maxValue: Double
    ) -> Int {
        let fraction = min(max((value - minValue) / (maxValue - minValue), 0.0), 1.0)
        return Int(fraction * 100.0)
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
