// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Synchronization

@usableFromInline
internal final class IncrementingProgressReporter: Sendable {
    public let minValue: Int
    public let maxValue: Int
    public var currentValue: Int {
        state.currentValue.load(ordering: .relaxed)
    }

    public let printTimeLeft: Bool

    @usableFromInline
    internal let state: State

    @inlinable
    public init(
        minValue: Int,
        maxValue: Int,
        currentValue: Int,
        printTimeLeft: Bool = false
    ) {
        precondition(maxValue > minValue, "maxValue must be greater than minValue")
        precondition(
            currentValue >= minValue && currentValue <= maxValue,
            "currentValue must lie within minValue...maxValue"
        )

        self.minValue = minValue
        self.maxValue = maxValue
        self.state = State(
            currentValue: currentValue,
            lastReportedPercentage: Self.percentage(
                for: currentValue,
                minValue: minValue,
                maxValue: maxValue
            )
        )
        self.printTimeLeft = printTimeLeft
    }

    @inlinable
    public convenience init(
        total: Int,
        printTimeLeft: Bool = false
    ) {
        self.init(
            minValue: 0,
            maxValue: total,
            currentValue: 0,
            printTimeLeft: printTimeLeft
        )
    }

    @inlinable
    public func increment(by amount: Int = 1) {
        precondition(amount > 0, "Increment amount must be positive")

        let (previous, current) = state.currentValue.wrappingAdd(amount, ordering: .relaxed)
        let previousPercentage = Self.percentage(
            for: previous,
            minValue: minValue,
            maxValue: maxValue
        )
        let currentPercentage = Self.percentage(
            for: current,
            minValue: minValue,
            maxValue: maxValue
        )

        guard currentPercentage > previousPercentage else {
            return
        }

        state.reporting.withLock { reporting in
            guard currentPercentage > reporting.lastReportedPercentage else {
                return
            }
            reporting.lastReportedPercentage = currentPercentage

            guard printTimeLeft else {
                reporting.progressLine.update("Progress \(currentPercentage)%")
                return
            }

            let clampedCurrent = min(max(current, minValue), maxValue)
            guard clampedCurrent > minValue else {
                reporting.progressLine.update("Progress \(currentPercentage)%")
                return
            }

            let totalDuration = state.startTime.duration(to: .now)
            let completed = Double(clampedCurrent) - Double(minValue)
            let remaining = Double(maxValue) - Double(clampedCurrent)
            let remainingDuration = totalDuration * remaining / completed
            reporting.progressLine.update(
                "Progress \(currentPercentage)%, estimated time left: \(remainingDuration)"
            )
        }
    }

    @inlinable
    public func finish() {
        state.reporting.withLock {
            $0.progressLine.finish()
        }
    }

    @inlinable
    @inline(always)
    internal static func percentage(
        for value: Int,
        minValue: Int,
        maxValue: Int
    ) -> Int {
        let fraction = min(
            max(
                (Double(value) - Double(minValue)) /
                    (Double(maxValue) - Double(minValue)),
                0.0
            ),
            1.0
        )
        return Int(fraction * 100.0)
    }
}

internal extension IncrementingProgressReporter {
    @usableFromInline
    struct ReportingState: Sendable {
        @usableFromInline
        internal var progressLine: ProgressLine
        @usableFromInline
        internal var lastReportedPercentage: Int
        
        @inlinable
        internal init(progressLine: ProgressLine, lastReportedPercentage: Int) {
            self.progressLine = progressLine
            self.lastReportedPercentage = lastReportedPercentage
        }
    }

    @usableFromInline
    struct State: ~Copyable, Sendable {
        @usableFromInline
        internal let currentValue: Atomic<Int>
        @usableFromInline
        internal let reporting: Mutex<ReportingState>
        @usableFromInline
        internal let startTime: ContinuousClock.Instant

        @inlinable
        internal init(currentValue: Int, lastReportedPercentage: Int) {
            self.currentValue = .init(currentValue)
            self.reporting = .init(
                ReportingState(
                    progressLine: ProgressLine(),
                    lastReportedPercentage: lastReportedPercentage
                )
            )
            self.startTime = .now
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
