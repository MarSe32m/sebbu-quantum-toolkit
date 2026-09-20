// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Common factorially scaled index tables for HOPS and doubled-index HEOM.
@usableFromInline
internal struct _BathHierarchyTables: ~Copyable {
    @usableFromInline let childIndices: UniqueArray<Int>
    @usableFromInline let parentIndices: UniqueArray<Int>
    @usableFromInline let kWArray: UniqueArray<Complex<Double>>
    @usableFromInline let multiIndices: UniqueArray<Int>
    @usableFromInline let tiers: UniqueArray<Int>
    @usableFromInline let parentWeights: UniqueArray<Double>
    @usableFromInline let childWeights: UniqueArray<Double>
    @usableFromInline let multiIndexCount: Int
    @usableFromInline let count: Int
    @usableFromInline let maximumTier: Int

    @inlinable
    init(poles: [Complex<Double>], truncation: BathHierarchyTruncation) {
        let directions = poles.count
        let root = [Int](repeating: 0, count: directions)
        let expectedCount: Int?
        switch truncation {
        case .maximumTier(let limit):
            precondition(limit >= 0, "The maximum hierarchy tier must be nonnegative.")
            expectedCount = _bathHierarchyCount(directions: directions, maximumTier: limit)
        case .custom(let accepts):
            precondition(accepts(root.span), "A custom hierarchy must retain the physical root.")
            expectedCount = nil
        }

        var nodes = [root]
        var nodeTiers = [0]
        // Rejected candidates also enter the dictionary, so each predicate
        // evaluation is performed only once and cannot depend on a path.
        var ids: [[Int]: Int] = [root: 0]
        if let expectedCount {
            _bathHierarchyCheckStorage(count: expectedCount, directions: directions)
            nodes.reserveCapacity(expectedCount)
            nodeTiers.reserveCapacity(expectedCount)
            ids.reserveCapacity(expectedCount)
        }
        var cursor = 0
        while cursor < nodes.count {
            let node = nodes[cursor]
            let tier = nodeTiers[cursor]
            if case .maximumTier(let limit) = truncation, tier == limit {
                cursor += 1
                continue
            }
            for p in (0..<directions).reversed() {
                precondition(tier < Int.max, "The hierarchy tier overflows Int.")
                var child = node
                child[p] += 1
                if ids[child] != nil { continue }
                if case .custom(let accepts) = truncation {
                    if !accepts(child.span) {
                        ids[child] = -1
                        continue
                    }
                    // BFS has already discovered the entire parent tier.
                    for q in 0..<directions where child[q] > 0 {
                        var parent = child
                        parent[q] -= 1
                        precondition(
                            (ids[parent] ?? -1) >= 0,
                            "A custom hierarchy must retain every parent of each retained state.")
                    }
                }
                _bathHierarchyCheckStorage(count: nodes.count + 1, directions: directions)
                ids[child] = nodes.count
                nodes.append(child)
                nodeTiers.append(tier + 1)
            }
            cursor += 1
        }

        let highestTier = nodeTiers.last!
        precondition(highestTier < Int.max, "Hierarchy weights overflow Int.")
        // Evaluate sqrt once per distinct occupation, only during setup.
        let squareRoots = (0...(highestTier + 1)).map { Double($0).squareRoot() }
        let entries = nodes.count * directions
        var parents = UniqueArray<Int>(repeating: -1, count: entries)
        var children = UniqueArray<Int>(repeating: -1, count: entries)
        var occupations = UniqueArray<Int>(repeating: 0, count: entries)
        var down = UniqueArray<Double>(repeating: 0, count: entries)
        var up = UniqueArray<Double>(repeating: 0, count: entries)
        var damping = UniqueArray<Complex<Double>>(repeating: .zero, count: nodes.count)
        for h in nodes.indices {
            let node = nodes[h]
            var value = Complex<Double>.zero
            for p in 0..<directions {
                let offset = h * directions + p
                let n = node[p]
                occupations[offset] = n
                down[offset] = squareRoots[n]
                up[offset] = squareRoots[n + 1]
                value -= Double(n) * poles[p]
                if n > 0 {
                    var parent = node
                    parent[p] -= 1
                    parents[offset] = ids[parent]!
                }
                var child = node
                child[p] += 1
                children[offset] = ids[child] ?? -1
            }
            precondition(
                value.real.isFinite && value.imaginary.isFinite,
                "The hierarchy damping coefficient is not representable.")
            damping[h] = value
        }
        self.multiIndexCount = directions
        self.count = nodes.count
        self.maximumTier = highestTier
        self.multiIndices = occupations
        self.parentIndices = parents
        self.childIndices = children
        self.parentWeights = down
        self.childWeights = up
        self.kWArray = damping
        self.tiers = UniqueArray(copying: nodeTiers)
    }
}

/// Binomial(P + D, D), cancelling before multiplication to avoid intermediate
/// overflow even when the final count is representable.
@usableFromInline
internal func _bathHierarchyCount(directions: Int, maximumTier: Int) -> Int {
    let small = min(directions, maximumTier)
    let large = max(directions, maximumTier)
    var count = 1
    if small == 0 { return count }
    for i in 1...small {
        let (numerator, sumOverflow) = large.addingReportingOverflow(i)
        precondition(!sumOverflow, "The hierarchy state count overflows Int.")
        var a = count
        var b = i
        while b != 0 { (a, b) = (b, a % b) }
        let (next, overflow) = (count / a).multipliedReportingOverflow(by: numerator / (i / a))
        precondition(!overflow, "The hierarchy state count overflows Int.")
        count = next
    }
    return count
}

@usableFromInline
internal func _bathHierarchyCheckStorage(count: Int, directions: Int) {
    let (entries, overflow) = count.multipliedReportingOverflow(by: directions)
    precondition(
        !overflow && entries <= Int.max / max(MemoryLayout<Int>.stride, MemoryLayout<Double>.stride)
            && count <= Int.max / MemoryLayout<Complex<Double>>.stride,
        "The hierarchy storage size overflows Int.")
}

// Keep the borrow of a noncopyable class field alive for the entire callback.
// Swift 6.3 cannot extend that borrow through a chained span extraction.
@inlinable
@inline(always)
internal func _bathHierarchyWithRow<Element>(
    _ storage: borrowing UniqueArray<Element>, range: Range<Int>,
    _ body: (borrowing Span<Element>) -> Void
) {
    let span = storage.span
    let row = span.extracting(range)
    body(row)
}
