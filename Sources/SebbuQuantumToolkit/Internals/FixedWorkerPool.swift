// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Dispatch
import Synchronization

@usableFromInline
internal final class FixedWorkerPool: Sendable {
    @inlinable
    internal static func with<R>(workers: Int, _ body: @Sendable (Int) -> R) -> [R] where R: Sendable {
        if workers == 1 {
            return [body(0)]
        } else {
            //TODO: Maybe replace with a lock-free queue in the future
            let results: Mutex<[R]> = Mutex([])
            DispatchQueue.concurrentPerform(iterations: workers) { workerIndex in
                let result = body(workerIndex)
                results.withLock { $0.append(result) }
            }
            return results.withLock { $0 }
        }
    }
}
