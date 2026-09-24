// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Optional terminal progress for a solver invocation.
///
/// GKSL and HEOM advances after accepted integration steps. QSD, MCWF and HOPS advance
/// after whole trajectories complete, including correlation and hierarchy solves.
/// Each invocation owns its reporter, so these options can be reused safely.
public struct ProgressReporting: Sendable {
	public enum Style: Sendable {
		case percentage
		case bar
	}

	public enum Stream: Sendable {
		case standardOutput
		case standardError
	}

	@usableFromInline
	internal let display: Display?

	/// No reporting, reporter allocation, clock reads, locking or output.
	public static let none = ProgressReporting(display: nil)

	/// Prints one progress line, redrawn only when the integer percentage increases.
	/// Concurrent console reporters targeting the same stream are stacked automatically.
	///
	/// The time estimate uses elapsed wall time and completed work. It is an
	/// estimate: adaptive time steps and trajectories can have different costs.
	/// A run stopped early retains its actual progress rather than showing 100%.
	public static func console(
		style: Style = .bar,
		label: String = "Progress",
		printTimeLeft: Bool = true,
		barWidth: Int = 30,
		stream: Stream = .standardError
	) -> Self {
		Self(
			display: Display(
				style: style, label: label, printTimeLeft: printTimeLeft,
				barWidth: barWidth, stream: stream))
	}

	@usableFromInline
	internal init(display: Display?) {
		self.display = display
	}

	@inlinable
	internal func continuous(in span: SimulationTimeSpan) -> ContinuousProgressReporter? {
		guard let display else { return nil }
		return ContinuousProgressReporter(
			minValue: span.start, maxValue: span.end, currentValue: span.start,
			display: display)
	}

	@inlinable
	internal func incrementing(total: Int) -> IncrementingProgressReporter? {
		guard let display else { return nil }
		return IncrementingProgressReporter(
			minValue: 0, maxValue: total, currentValue: 0, display: display)
	}
}

extension ProgressReporting {
	@usableFromInline
	internal struct Display: Sendable {
		let style: Style
		let label: String
		let printTimeLeft: Bool
		let barWidth: Int
		@usableFromInline let write: @Sendable (String) -> Void
		@usableFromInline let stream: Stream?

		@usableFromInline
		init(
			style: Style, label: String, printTimeLeft: Bool, barWidth: Int,
			write: @escaping @Sendable (String) -> Void
		) {
			self.init(
				style: style, label: label, printTimeLeft: printTimeLeft,
				barWidth: barWidth, write: write, stream: nil)
		}

		@usableFromInline
		init(
			style: Style, label: String, printTimeLeft: Bool, barWidth: Int,
			stream: Stream
		) {
			self.init(
				style: style, label: label, printTimeLeft: printTimeLeft,
				barWidth: barWidth,
				write: { ProgressLine.write($0, to: stream) }, stream: stream)
		}

		@usableFromInline
		init(
			style: Style, label: String, printTimeLeft: Bool, barWidth: Int,
			write: @escaping @Sendable (String) -> Void, stream: Stream?
		) {
			precondition(
				barWidth > 0 && barWidth <= 200,
				"Progress bar width must be in 1...200")
			self.style = style
			// A progress display occupies one terminal row.
			self.label = label.replacingProgressControlCharacters()
			self.printTimeLeft = printTimeLeft
			self.barWidth = barWidth
			self.write = write
			self.stream = stream
		}

		@usableFromInline
		func message(percentage: Int, fraction: Double, elapsed: Duration) -> String {
			var result = label
			if style == .bar {
				let filled = Int(Double(barWidth) * Double(percentage) / 100)
				result +=
					" [" + String(repeating: "=", count: filled)
					+ String(repeating: " ", count: barWidth - filled) + "]"
			}
			result += " \(percentage)%"
			if printTimeLeft {
				let components = elapsed.components
				let seconds =
					Double(components.seconds) + Double(components.attoseconds)
					* 1e-18
				let remaining =
					fraction > 0
					? seconds * (1 - fraction) / fraction : .infinity
				result += ", ETA " + Self.duration(remaining)
			}
			return result
		}

		internal static func duration(_ seconds: Double) -> String {
			guard seconds.isFinite && seconds >= 0 else { return "--" }
			// Keep conversion to Int safe, including extreme time-span ratios.
			guard seconds < Double(Int.max / 2) else { return ">100y" }
			let total = Int(seconds.rounded(.up))
			if total < 60 { return "\(total)s" }
			if total < 3600 { return "\(total / 60)m \(total % 60)s" }
			return "\(total / 3600)h \((total / 60) % 60)m \(total % 60)s"
		}
	}
}

extension String {
	fileprivate func replacingProgressControlCharacters() -> String {
		String(
			unicodeScalars.map { scalar in
				scalar.value < 32 || (127...159).contains(scalar.value)
					? " " : Character(scalar)
			})
	}
}
