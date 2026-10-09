// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

extension HOPS.CPUEngine {
    /// Independent cursors preserve the actual origins of noncommensurate grids.
    @usableFromInline
    internal struct NoiseGridCursor {
        @usableFromInline let grid: (origin: Double, step: Double)?
        @usableFromInline var index: Int

        @inlinable
        init(_ grid: (origin: Double, step: Double)?, start: Double) {
            self.grid = grid
            self.index = 1
            if let grid {
                precondition(grid.origin.isFinite && grid.origin <= start
                             && grid.step.isFinite && grid.step > 0
                             && (grid.origin + grid.step).isFinite
                             && grid.origin + grid.step > grid.origin,
                             "Invalid noise integration grid.")
                let coordinate = ((start - grid.origin) / grid.step).rounded(.down)
                precondition(coordinate.isFinite && coordinate >= 0 && coordinate < 0x1p52,
                             "The noise integration grid has too many nodes.")
                index = Int(coordinate)
                // Compare physical times to avoid skipping a boundary after a
                // rounded-up quotient. boundary(after:) advances beyond start.
                if index > 0 && grid.origin.addingProduct(Double(index), grid.step) > start {
                    index -= 1
                }
            }
        }

        @inlinable
        mutating func limit(_ limit: Double, after time: Double) -> Double {
            guard let grid else { return limit }
            var boundary = grid.origin.addingProduct(Double(index), grid.step)
            while boundary <= time {
                precondition(index < Int.max - 1, "The noise grid index overflowed.")
                index += 1
                let next = grid.origin.addingProduct(Double(index), grid.step)
                precondition(next.isFinite && next > boundary, "The noise grid is no longer resolvable.")
                boundary = next
            }
            return min(limit, boundary)
        }
    }
}
