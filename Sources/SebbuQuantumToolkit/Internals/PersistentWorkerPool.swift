// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Dispatch
import Synchronization

#if canImport(WinSDK)
    import WinSDK
#elseif canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

/// A synchronous, single-caller pool. Worker zero runs on the calling thread.
/// Native threads avoid occupying libdispatch's shared pool while waiting for
/// the next stage, including when several independent solves run concurrently.
/// Jobs and synchronization objects are reused; no tasks are created by run().
@usableFromInline
internal final class PersistentWorkerPool<Job: Sendable> {
    @usableFromInline
    internal final class Shared: Sendable {
        @usableFromInline
        let job = Mutex<Job?>(nil)
        @usableFromInline
        let completed = DispatchSemaphore(value: 0)
        @usableFromInline
        let execute: @Sendable (Job, Int) -> Void

        @inlinable
        init(execute: @escaping @Sendable (Job, Int) -> Void) {
            self.execute = execute
        }

        @inlinable
        func work(index: Int, ready: DispatchSemaphore) {
            while true {
                ready.wait()
                guard let current = job.withLock({ $0 }) else { return }
                execute(current, index)
                completed.signal()
            }
        }
    }

    @usableFromInline
    internal struct Worker {
        @usableFromInline
        let ready: DispatchSemaphore
        @usableFromInline
        let thread: NativeWorkerThread
        
        @inlinable
        init(ready: DispatchSemaphore, thread: NativeWorkerThread) {
            self.ready = ready
            self.thread = thread
        }
    }

    @usableFromInline
    internal let shared: Shared
    @usableFromInline
    internal let workers: [Worker]
    /// Includes the calling thread. Resource limits may reduce the request.
    @usableFromInline
    internal var workerCount: Int { workers.count + 1 }

    @inlinable
    internal init(workers requested: Int, execute: @escaping @Sendable (Job, Int) -> Void) {
        precondition(requested > 0)
        let shared = Shared(execute: execute)
        var workers: [Worker] = []
        for index in 1..<requested {
            let ready = DispatchSemaphore(value: 0)
            guard let thread = NativeWorkerThread({ shared.work(index: index, ready: ready) }) else {
                // Keep any threads already created and divide subsequent jobs
                // among the actual count. This also permits a serial fallback.
                break
            }
            workers.append(Worker(ready: ready, thread: thread))
        }
        self.shared = shared
        self.workers = workers
    }

    /// Returns only after every worker has stopped accessing the job's buffers.
    /// The caller must not invoke run concurrently on the same pool.
    @inlinable
    internal func run(_ job: Job) {
        if workers.isEmpty {
            shared.execute(job, 0)
            return
        }
        shared.job.withLock { $0 = job }
        for worker in workers { worker.ready.signal() }
        shared.execute(job, 0)
        for _ in workers { shared.completed.wait() }
        shared.job.withLock { $0 = nil }
    }

    @inlinable
    deinit {
        // No job is outstanding: run is synchronous. A nil job tells each
        // sleeper to exit. Join before releasing any worker-owned resources.
        for worker in workers { worker.ready.signal() }
        for worker in workers { worker.thread.join() }
    }
}

/// The owner joins each successfully created thread exactly once.
@usableFromInline
internal struct NativeWorkerThread {
    @usableFromInline
    internal final class Entry: Sendable {
        @usableFromInline let body: @Sendable () -> Void
        @inlinable
        init(_ body: @escaping @Sendable () -> Void) { self.body = body }
    }

    #if canImport(WinSDK)
        @usableFromInline let handle: HANDLE
    #elseif canImport(Darwin) || canImport(Glibc) || canImport(Musl)
        @usableFromInline let handle: pthread_t
    #endif

    @inlinable
    init?(_ body: @escaping @Sendable () -> Void) {
        let entry = Unmanaged.passRetained(Entry(body))
        #if canImport(WinSDK)
            guard
                let handle = CreateThread(
                    nil, 0,
                    { pointer in
                        let entry = Unmanaged<Entry>.fromOpaque(pointer!).takeRetainedValue()
                        entry.body()
                        return 0
                    }, entry.toOpaque(), 0, nil)
            else {
                entry.release()
                return nil
            }
            self.handle = handle
        #elseif canImport(Darwin) || canImport(Glibc) || canImport(Musl)
            #if canImport(Darwin) || canImport(Musl)
                var handle: pthread_t? = nil
            #elseif canImport(Glibc)
                var handle = pthread_t()
            #endif
            let result = pthread_create(
                &handle, nil,
                { pointer in
                    #if canImport(Darwin)
                    let entry = Unmanaged<Entry>.fromOpaque(pointer).takeRetainedValue()
                    #elseif canImport(Musl) || canImport(Glibc)
                    let entry = Unmanaged<Entry>.fromOpaque(pointer!).takeRetainedValue()
                    #endif
                    entry.body()
                    return nil
                }, entry.toOpaque())
            guard result == 0 else {
                entry.release()
                return nil
            }
            #if canImport(Darwin) || canImport(Musl)
                self.handle = handle!
            #else
                self.handle = handle
            #endif
        #else
            entry.release()
            return nil
        #endif
    }

    @inlinable
    func join() {
        #if canImport(WinSDK)
            let result = WaitForSingleObject(handle, INFINITE)
            precondition(result == WAIT_OBJECT_0)
            CloseHandle(handle)
        #elseif canImport(Darwin) || canImport(Glibc) || canImport(Musl)
            let result = pthread_join(handle, nil)
            precondition(result == 0)
        #endif
    }
}
