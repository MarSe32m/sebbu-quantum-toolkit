// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Dispatch
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

final class ProgressCapture: Sendable {
	private let storage = Mutex<[String]>([])

	var chunks: [String] { storage.withLock { $0 } }
	var percentages: [Int] {
		chunks.compactMap { text in
			guard let end = text.firstIndex(of: "%") else { return nil }
			let digits = text[..<end].reversed().prefix(while: { $0.isNumber })
				.reversed()
			return Int(String(digits))
		}
	}
	var newlineCount: Int { chunks.filter { $0 == "\n" }.count }

	func reporting(style: ProgressReporting.Style = .percentage, eta: Bool = false)
		-> ProgressReporting
	{
		.init(
			display: .init(
				style: style, label: "Test", printTimeLeft: eta, barWidth: 10,
				write: { [self] text in storage.withLock { $0.append(text) } }))
	}
}

@Suite("Progress reporters")
struct ProgressReporterTests {
	@Test("Default reporting has no reporters or output machinery")
	func disabled() {
		let propagation = PropagationOptions(
			timeSpan: .init(start: 0, end: 1), output: .final, integration: 0)
		#expect(propagation.progress.display == nil)
		#expect(propagation.progress.continuous(in: propagation.timeSpan) == nil)
		#expect(propagation.progress.incrementing(total: 10) == nil)
		var enabled = propagation
		enabled.progress = .console()
		#expect(enabled.withoutProgressReporting.progress.display == nil)
		#expect(enabled.progress.display != nil)
	}

	@Test("Bars, percentages and elapsed-work time estimates share one renderer")
	func formatting() {
		let capture = ProgressCapture()
		let display = capture.reporting(style: .bar, eta: true).display!
		#expect(
			display.message(percentage: 50, fraction: 0.5, elapsed: .seconds(61))
				== "Test [=====     ] 50%, ETA 1m 1s")
		#expect(
			display.message(percentage: 0, fraction: 0, elapsed: .zero)
				== "Test [          ] 0%, ETA --")
		#expect(
			display.message(percentage: 100, fraction: 1, elapsed: .seconds(3))
				== "Test [==========] 100%, ETA 0s")
		#expect(
			capture.reporting().display!.message(
				percentage: 25, fraction: 0.25, elapsed: .seconds(4)) == "Test 25%")
		#expect(ProgressReporting.Display.duration(0.01) == "1s")
		#expect(ProgressReporting.Display.duration(3601) == "1h 0m 1s")
		#expect(ProgressReporting.Display.duration(.infinity) == "--")
		#expect(ProgressReporting.Display.duration(.greatestFiniteMagnitude) == ">100y")
		let sanitized = ProgressReporting.Display(
			style: .percentage, label: "A\nB\rC\u{001B}", printTimeLeft: false,
			barWidth: 1, write: { _ in })
		#expect(
			sanitized.message(percentage: 5, fraction: 0.05, elapsed: .zero)
				== "A B C  5%")
	}

	@Test("Continuous updates are bounded by percentages, not time-step count")
	func continuous() {
		let capture = ProgressCapture()
		let reporter = capture.reporting().continuous(in: .init(start: -2, end: 2))!
		for i in 0...10_000 { reporter.setValue(to: -2 + 4 * Double(i) / 10_000) }
		reporter.setValue(to: 2)
		reporter.setValue(to: -1)
		reporter.finish()
		reporter.finish()
		reporter.setValue(to: 3)
		#expect(capture.percentages == Array(0...100))
		#expect(capture.newlineCount == 1)
		#expect(capture.chunks.last == "\n")
	}

	@Test("Partial and zero-duration runs finalize without fabricated completion")
	func partialAndZero() {
		let partial = ProgressCapture()
		let reporter = partial.reporting().continuous(in: .init(start: -1, end: 3))!
		reporter.setValue(to: 0)
		reporter.finish()
		#expect(partial.percentages == [0, 25])
		#expect(partial.newlineCount == 1)
		let zero = ProgressCapture()
		let empty = zero.reporting(eta: true).continuous(in: .init(start: 2, end: 2))!
		#expect(zero.percentages == [0])
		empty.setValue(to: 2)
		empty.finish()
		#expect(zero.percentages == [0, 100])
		#expect(!zero.chunks.joined().contains("nan"))
	}

	@Test("Concurrent completions produce a single monotonically advancing line")
	func concurrentCompletions() {
		let capture = ProgressCapture()
		let reporter = capture.reporting(style: .bar, eta: true).incrementing(
			total: 10_000)!
		DispatchQueue.concurrentPerform(iterations: 8) { _ in
			for _ in 0..<1250 { reporter.increment() }
		}
		reporter.finish()
		reporter.finish()
		#expect(reporter.currentValue == 10_000)
		let values = capture.percentages
		#expect(values.first == 0 && values.last == 100)
		#expect(values.count <= 101)
		#expect(zip(values, values.dropFirst()).allSatisfy { $0 < $1 })
		#expect(capture.newlineCount == 1 && capture.chunks.last == "\n")
	}

	@Test("Integer and floating-point endpoints never round up to premature completion")
	func endpoints() {
		typealias I = IncrementingProgressReporter
		typealias C = ContinuousProgressReporter
		#expect(I.percentage(for: Int.max - 1, minValue: 0, maxValue: Int.max) == 99)
		#expect(
			I.percentage(for: Int.max - 5, minValue: Int.max - 10, maxValue: Int.max)
				== 50)
		#expect(I.percentage(for: 0, minValue: Int.min, maxValue: Int.max) == 50)
		#expect(C.percentage(for: Double(1).nextDown, minValue: 0, maxValue: 1) < 100)
		#expect(
			C.percentage(
				for: 0, minValue: -.greatestFiniteMagnitude,
				maxValue: .greatestFiniteMagnitude) == 50)
		let capture = ProgressCapture()
		let reporter = capture.reporting().incrementing(total: 7)!
		reporter.increment(by: 2)
		reporter.increment(by: 3)
		reporter.finish()
		#expect(capture.percentages == [0, 28, 71])
	}

	@Test("A shorter redraw clears the previous suffix and finishing is idempotent")
	func lineCleanup() {
		let chunks = Mutex<[String]>([])
		var line = ProgressLine(write: { text in chunks.withLock { $0.append(text) } })
		line.update("long value")
		line.update("short")
		line.finish()
		line.finish()
		line.update("ignored")
		#expect(chunks.withLock { $0 } == ["\rlong value", "\rshort     ", "\n"])
	}
}
