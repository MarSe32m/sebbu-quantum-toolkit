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
    @usableFromInline
    internal var previousLength = 0

    @inlinable
    init() { }
    
    @inlinable
    mutating func update(_ message: String) {
        let paddingCount = max(0, previousLength - message.count)
        let padding = String(repeating: " ", count: paddingCount)

        writeToStandardOutput("\r\(message)\(padding)")
        previousLength = message.count
    }

    @inlinable
    mutating func finish() {
        writeToStandardOutput("\n")
        previousLength = 0
    }

    @inlinable
    internal func writeToStandardOutput(_ text: String) {
        text.withCString { pointer in
            _ = fputs(pointer, stdout)
        }
        fflush(stdout)
    }
}
