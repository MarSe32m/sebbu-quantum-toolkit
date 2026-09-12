// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
#if swift(<6.5)
import BasicContainers
#endif

extension HOPS {
    /// Immutable latent-pole hierarchy, shared by all trajectories of a bath.
    ///
    /// Directions follow bath order, then pole order within each bath, exactly
    /// as in the correlated OU sampler. State zero is the physical root. IDs
    /// are assigned breadth-first, visiting directions from last to first for
    /// each state. No sorting or reduction of the supplied bath is performed.
    ///
    /// Each flat table uses `state * multiIndexCount + direction`. Missing
    /// neighbours have index `-1`. Weights are the nonnegative factorial-scaling
    /// factors, even for excluded children; the RHS must check the neighbour
    /// index and apply the upward minus sign separately.
    public final class Hierarchy: Sendable {
        public typealias Index = Int

        @usableFromInline package let childIndices: UniqueArray<Index>
        @usableFromInline package let parentIndices: UniqueArray<Index>
        @usableFromInline package let kWArray: UniqueArray<Complex<Double>>
        @usableFromInline package let multiIndices: UniqueArray<Int>
        @usableFromInline package let tiers: UniqueArray<Int>
        @usableFromInline package let parentWeights: UniqueArray<Double>
        @usableFromInline package let childWeights: UniqueArray<Double>

        /// Number of latent-pole directions in every multi-index.
        public let multiIndexCount: Int
        /// Number of retained states, including the physical root.
        public let count: Int
        /// Highest tier actually present, including for custom truncations.
        public let maximumTier: Int
        public let environment: Environment

        /// Constructs all indices, damping coefficients and square-root weights.
        /// Accessors subsequently borrow storage without allocating.
        @inlinable
        public init(environment: Environment, truncation: Truncation) {
            let poles = environment.bath.latentBaths.flatMap(\.poles)
            let directions = poles.count
            let root = [Int](repeating: 0, count: directions)
            let expectedCount: Int?
            switch truncation {
            case .maximumTier(let limit):
                precondition(limit >= 0, "The maximum hierarchy tier must be nonnegative.")
                expectedCount = _hopsHierarchyCount(directions: directions, maximumTier: limit)
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
                _hopsCheckStorage(count: expectedCount, directions: directions)
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
                            precondition((ids[parent] ?? -1) >= 0,
                                         "A custom hierarchy must retain every parent of each retained state.")
                        }
                    }
                    _hopsCheckStorage(count: nodes.count + 1, directions: directions)
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
                precondition(value.real.isFinite && value.imaginary.isFinite,
                             "The hierarchy damping coefficient is not representable.")
                damping[h] = value
            }
            self.environment = environment
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

        @inlinable
        @inline(always)
        public func tier(at index: Index) -> Int {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            return tiers[index]
        }

        /// The precomputed complex coefficient `-sum_p n_p W_p`.
        @inlinable
        @inline(always)
        public func damping(at index: Index) -> Complex<Double> {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            return kWArray[index]
        }

        @inlinable
        @inline(always)
        public func multiIndex(of index: Index, indices: (borrowing Span<Int>) -> Void) {
            _hopsWithRow(multiIndices, range: rowRange(at: index), indices)
        }

        @inlinable
        @inline(always)
        public func parentIndices(of index: Index, indices: (borrowing Span<Index>) -> Void) {
            _hopsWithRow(parentIndices, range: rowRange(at: index), indices)
        }

        @inlinable
        @inline(always)
        public func childIndices(of index: Index, indices: (borrowing Span<Index>) -> Void) {
            _hopsWithRow(childIndices, range: rowRange(at: index), indices)
        }

        /// `sqrt(n_p)` for factorially scaled auxiliaries.
        @inlinable
        @inline(always)
        public func parentWeights(of index: Index, weights: (borrowing Span<Double>) -> Void) {
            _hopsWithRow(parentWeights, range: rowRange(at: index), weights)
        }

        /// `sqrt(n_p + 1)`, including at the truncation boundary.
        /// Missing children must still be skipped using their `-1` index.
        @inlinable
        @inline(always)
        public func childWeights(of index: Index, weights: (borrowing Span<Double>) -> Void) {
            _hopsWithRow(childWeights, range: rowRange(at: index), weights)
        }

        @inlinable
        package func rowRange(at index: Index) -> Range<Int> {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            let start = index * multiIndexCount
            return start..<(start + multiIndexCount)
        }
    }
    
    public struct Environment: Sendable {
        public let couplingOperators: [TimeDependentOperator]
        public let bath: CorrelatedBathModel
        
        @inlinable
        public init(couplingOperators: [TimeDependentOperator], bath: CorrelatedBathModel) {
            precondition(couplingOperators.count == bath.channelCount, "There must be one coupling operator per physical bath channel.")
            self.couplingOperators = couplingOperators
            self.bath = bath
        }
        
        @inlinable
        public init(couplingOperator: TimeDependentOperator, bath: CorrelatedBathModel) {
            self.init(couplingOperators: [couplingOperator], bath: bath)
        }
    }
    
    public enum Truncation: Sendable {
        /// Retains every nonnegative multi-index whose total occupation is at
        /// most this nonnegative tier. Missing children use a hard cutoff.
        case maximumTier(Int)

        /// Returns true for retained multi-indices. The predicate must be pure,
        /// include the root, and describe a finite, downward-closed set: every
        /// parent of a retained state must also be retained. Rejected states
        /// prune their descendants. A predicate accepting infinitely many
        /// reachable states cannot finish construction.
        case custom(@Sendable (borrowing Span<Int>) -> Bool)
    }
}

extension HOPS {
    public struct HierarchyStateView: ~Copyable, ~Escapable {
        @usableFromInline
        package let systemDimension: Int
        // Total state
        @usableFromInline
        package let states: Span<Complex<Double>>

        @inlinable
        public var count: Int {
            states.count / systemDimension
        }

        @_lifetime(copy states)
        @inlinable
        package init(systemDimension: Int, states: Span<Complex<Double>>) {
            self.systemDimension = systemDimension
            self.states = states
        }
        
        @inlinable
        @inline(always)
        public func withPhysicalState<Result>(
            _ body: (
                borrowing StateVectorView
            ) -> Result
        ) -> Result {
            withState(at: 0, body)
        }
        
        @inlinable
        @inline(always)
        public func withState<Result>(
            at index: HOPS.Hierarchy.Index,
            _ body: (borrowing StateVectorView) -> Result
        ) -> Result {
            precondition(index >= 0 && index < count)
            let span = states.extracting(index &* systemDimension ..< (index &+ 1) &* systemDimension)
            let stateVectorView = StateVectorView(elements: span)
            return body(stateVectorView)
        }
    }
}

/// Binomial(P + D, D), cancelling before multiplication to avoid intermediate
/// overflow even when the final count is representable.
@usableFromInline
internal func _hopsHierarchyCount(directions: Int, maximumTier: Int) -> Int {
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
internal func _hopsCheckStorage(count: Int, directions: Int) {
    let (entries, overflow) = count.multipliedReportingOverflow(by: directions)
    precondition(!overflow && entries <= Int.max / max(MemoryLayout<Int>.stride, MemoryLayout<Double>.stride)
                 && count <= Int.max / MemoryLayout<Complex<Double>>.stride,
                 "The hierarchy storage size overflows Int.")
}

// Keep the borrow of a noncopyable class field alive for the entire callback.
// Swift 6.3 cannot extend that borrow through a chained span extraction.
@inlinable
@inline(always)
internal func _hopsWithRow<Element>(
    _ storage: borrowing UniqueArray<Element>, range: Range<Int>,
    _ body: (borrowing Span<Element>) -> Void
) {
    let span = storage.span
    let row = span.extracting(range)
    body(row)
}
