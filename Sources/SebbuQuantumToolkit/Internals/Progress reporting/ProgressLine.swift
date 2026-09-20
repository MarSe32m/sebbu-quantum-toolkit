// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#if os(Windows)
	import ucrt
#elseif canImport(Darwin)
	import Darwin
#elseif canImport(Glibc)
	@preconcurrency import Glibc
#elseif canImport(Musl)
	import Musl
#else
	#error("Unsupported platform")
#endif

@usableFromInline
internal struct ProgressLine: Sendable {
	@usableFromInline internal var previousLength = 0
	@usableFromInline internal var isFinished = false
	@usableFromInline internal let write: @Sendable (String) -> Void

	@usableFromInline
	init(write: @escaping @Sendable (String) -> Void = { Self.write($0, to: .standardOutput) })
	{
		self.write = write
	}

	@usableFromInline
	mutating func update(_ message: String) {
		guard !isFinished else { return }
		let padding = String(repeating: " ", count: max(0, previousLength - message.count))
		write("\r\(message)\(padding)")
		previousLength = message.count
	}

	/// Ends the line once, without inventing a successful completion.
	@usableFromInline
	mutating func finish() {
		guard !isFinished else { return }
		isFinished = true
		write("\n")
		previousLength = 0
	}

	@usableFromInline
	static func write(_ text: String, to stream: ProgressReporting.Stream) {
        text.withCString { pointer in
            switch stream {
            case .standardOutput:
                _ = fputs(pointer, stdout)
                fflush(stdout)
            case .standardError:
                _ = fputs(pointer, stderr)
                fflush(stderr)
            }
        }

	}
}
