// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#if canImport(WinSDK)
import WinSDK
#elseif canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

@usableFromInline
internal enum Platform: Sendable {
    @usableFromInline
    static let activeProcessorCount: Int = {
        #if canImport(WinSDK)
        let count = GetActiveProcessorCount(WORD(ALL_PROCESSOR_GROUPS))
        #elseif canImport(Darwin) || canImport(Glibc) || canImport(Musl)
        let count = sysconf(Int32(_SC_NPROCESSORS_ONLN))
        #else
        let count = 1
        #endif
        return Swift.max(1, Int(count))
    }()
}
