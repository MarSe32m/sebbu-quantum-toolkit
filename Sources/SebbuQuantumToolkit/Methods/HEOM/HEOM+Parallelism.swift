// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

extension HEOM {
    /// Controls ADO work within each Runge–Kutta stage. Workers are retained
    /// for one propagation, including correlation insertions and restarts.
    public enum Parallelism: Sendable, Equatable {
        /// Execute on the calling thread, without creating a worker pool.
        case serial
        /// Choose a conservative worker count from the dimension and ADO work.
        /// Small problems remain serial; the count never exceeds available cores.
        case automatic
        /// Use at most this many workers, including the calling thread.
        /// Must be positive. Also capped by the available cores and ADO count.
        case maximumWorkers(Int)

        @inlinable
        internal func workerCount(
            dimension: Int, adoCount: Int, poleCount: Int, collapseCount: Int,
            availableCores: Int = Platform.activeProcessorCount
        ) -> Int {
            let limit = max(1, min(availableCores, adoCount))
            switch self {
            case .serial:
                return 1
            case .maximumWorkers(let count):
                precondition(count > 0, "The HEOM worker count must be positive.")
                return min(count, limit)
            case .automatic:
                // Dense products cost O(d^3). Estimate two Hamiltonian products,
                // four per pole and two per collapse channel. Require enough
                // work per worker to amortize the stage synchronization.
                // Double arithmetic avoids overflow for very large hierarchies.
                let d = Double(dimension)
                let products = 2 + 4 * Double(poleCount) + 2 * Double(collapseCount)
                let work = Double(adoCount) * d * d * d * products
                // BLAS processes the same cubic work much faster than the tiny
                // scalar kernel, so require larger batches on its path.
                let minimumWork = dimension <= 4 ? 32_768.0 : 1_048_576.0
                let workers = min(Double(limit), max(1, work / minimumWork))
                return Int(workers)
            }
        }
    }
}
