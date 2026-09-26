// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Synchronization

#if os(Windows)
	import ucrt
	nonisolated(unsafe) let stdout = __acrt_iob_func(1)
	nonisolated(unsafe) let stderr = __acrt_iob_func(2)
#elseif canImport(Darwin)
	import Darwin
#elseif canImport(Glibc)
	@preconcurrency import Glibc
#elseif canImport(Musl)
	import Musl
#else
	#error("Unsupported platform")
#endif

/// Serializes and redraws all live progress lines targeting one terminal stream.
///
/// The cursor is kept at the end of the last progress row. Updating any row moves
/// back to the first row, redraws the complete stack and leaves the cursor at the
/// end of the last row again. Finished rows remain in the stack until the last
/// contemporaneous progress line finishes, which keeps row positions stable.
@usableFromInline
internal final class ProgressLineCoordinator: Sendable {
	@usableFromInline
	internal struct Entry: Sendable {
		@usableFromInline internal let id: UInt64
		@usableFromInline internal var message: String
		@usableFromInline internal var isFinished: Bool
	}

	@usableFromInline
	internal struct State: Sendable {
		@usableFromInline internal var nextID: UInt64 = 0
		@usableFromInline internal var entries: [Entry] = []
		@usableFromInline internal var renderedLengths: [Int] = []
	}

	@usableFromInline internal let state = Mutex(State())
	@usableFromInline internal let write: @Sendable (String) -> Void

	@usableFromInline
	internal init(write: @escaping @Sendable (String) -> Void) {
		self.write = write
	}

	@usableFromInline
	internal func register(_ message: String) -> UInt64 {
		state.withLock { state in
			let id = state.nextID
			state.nextID &+= 1
			state.entries.append(Entry(id: id, message: message, isFinished: false))
			write(Self.redraw(&state))
			return id
		}
	}

	@usableFromInline
	internal func update(_ id: UInt64, message: String) {
		state.withLock { state in
			guard
				let index = state.entries.firstIndex(where: { $0.id == id }),
				!state.entries[index].isFinished
			else { return }
			state.entries[index].message = message
			write(Self.redraw(&state))
		}
	}

	@usableFromInline
	internal func finish(_ id: UInt64) {
		state.withLock { state in
			guard
				let index = state.entries.firstIndex(where: { $0.id == id }),
				!state.entries[index].isFinished
			else { return }
			state.entries[index].isFinished = true

			// Keep completed rows fixed in place while sibling solves are still active.
			guard state.entries.allSatisfy(\.isFinished) else { return }

			// The cursor currently sits at the end of the last row. One newline commits
			// the complete stack, matching the old single-line finish behaviour.
			write("\n")
			state.entries.removeAll(keepingCapacity: true)
			state.renderedLengths.removeAll(keepingCapacity: true)
		}
	}

	@usableFromInline
	internal static func redraw(_ state: inout State) -> String {
		precondition(!state.entries.isEmpty)

		var output = ""
		let previousCount = state.renderedLengths.count

		// The cursor is at the end of the previous last row. Move to the first row.
		if previousCount > 1 {
			output += "\u{001B}[\(previousCount - 1)A"
		}
		output += "\r"

		var nextLengths: [Int] = []
		nextLengths.reserveCapacity(state.entries.count)
		for index in state.entries.indices {
			let message = state.entries[index].message
			let previousLength =
				index < previousCount ? state.renderedLengths[index] : 0
			output += message
			if previousLength > message.count {
				output += String(
					repeating: " ", count: previousLength - message.count)
			}
			nextLengths.append(message.count)
			if index != state.entries.index(before: state.entries.endIndex) {
				output += "\n"
			}
		}
		state.renderedLengths = nextLengths
		return output
	}
}

@usableFromInline
internal struct ProgressLine: Sendable {
	@usableFromInline internal var previousLength = 0
	@usableFromInline internal var isFinished = false
	@usableFromInline internal let write: (@Sendable (String) -> Void)?
	@usableFromInline internal let coordinator: ProgressLineCoordinator?
	@usableFromInline internal var registrationID: UInt64?

	@usableFromInline
	internal static let standardOutputCoordinator = ProgressLineCoordinator(
		write: { Self.write($0, to: .standardOutput) })
	@usableFromInline
	internal static let standardErrorCoordinator = ProgressLineCoordinator(
		write: { Self.write($0, to: .standardError) })

	/// Creates an independent line for injected writers, primarily tests and
	/// non-terminal internal sinks. These retain the previous single-line behaviour.
	@usableFromInline
	init(write: @escaping @Sendable (String) -> Void = { Self.write($0, to: .standardOutput) })
	{
		self.write = write
		self.coordinator = nil
		self.registrationID = nil
	}

	/// Creates a line participating in an explicitly supplied coordinator.
	/// This is internal so stacking can be tested without writing to a real terminal.
	@usableFromInline
	init(coordinator: ProgressLineCoordinator) {
		self.write = nil
		self.coordinator = coordinator
		self.registrationID = nil
	}

	@usableFromInline
	init(display: ProgressReporting.Display) {
		if let stream = display.stream {
			self.init(coordinator: Self.coordinator(for: stream))
		} else {
			self.init(write: display.write)
		}
	}

	@usableFromInline
	mutating func update(_ message: String) {
		guard !isFinished else { return }
		if let coordinator {
			if let registrationID {
				coordinator.update(registrationID, message: message)
			} else {
				registrationID = coordinator.register(message)
			}
			return
		}

		guard let write else { return }
		let padding = String(repeating: " ", count: max(0, previousLength - message.count))
		write("\r\(message)\(padding)")
		previousLength = message.count
	}

	/// Ends the line once, without inventing a successful completion.
	@usableFromInline
	mutating func finish() {
		guard !isFinished else { return }
		isFinished = true
		if let coordinator {
			if let registrationID { coordinator.finish(registrationID) }
		} else {
			write?("\n")
		}
		previousLength = 0
	}

	@usableFromInline
	internal static func coordinator(for stream: ProgressReporting.Stream)
		-> ProgressLineCoordinator
	{
		switch stream {
		case .standardOutput:
			return standardOutputCoordinator
		case .standardError:
			return standardErrorCoordinator
		}
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
