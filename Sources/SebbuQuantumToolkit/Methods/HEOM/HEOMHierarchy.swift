// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM {
    /// Factorially scaled ADOs indexed by `(m_0,...,m_(P-1), n_0,...,n_(P-1))`.
    /// The root has ID zero; missing neighbours have ID -1. Maximum tier
    /// bounds the combined ket and bra occupation, |m| + |n|.
    /// Custom predicates must be finite and downward closed. Choose a set
    /// symmetric under ket/bra exchange to preserve ADO adjoint symmetry.
    public final class Hierarchy: Sendable {
        public typealias Index = Int

        @usableFromInline package let childIndices: UniqueArray<Index>
        @usableFromInline package let parentIndices: UniqueArray<Index>
        @usableFromInline package let kWArray: UniqueArray<Complex<Double>>
        @usableFromInline package let multiIndices: UniqueArray<Int>
        @usableFromInline package let tiers: UniqueArray<Int>
        @usableFromInline package let parentWeights: UniqueArray<Double>
        @usableFromInline package let childWeights: UniqueArray<Double>

        /// Number of directions: twice the latent pole count (ket, then bra).
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
            let tables = _BathHierarchyTables(
                poles: poles + poles.map(\.conjugate), truncation: truncation)
            self.environment = environment
            self.childIndices = tables.childIndices
            self.parentIndices = tables.parentIndices
            self.kWArray = tables.kWArray
            self.multiIndices = tables.multiIndices
            self.tiers = tables.tiers
            self.parentWeights = tables.parentWeights
            self.childWeights = tables.childWeights
            self.multiIndexCount = tables.multiIndexCount
            self.count = tables.count
            self.maximumTier = tables.maximumTier
        }

        @inlinable
        @inline(always)
        public func tier(at index: Index) -> Int {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            return tiers[index]
        }

        /// The coefficient `-sum_p (m_p W_p + n_p W_p.conjugate)`.
        @inlinable
        @inline(always)
        public func damping(at index: Index) -> Complex<Double> {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            return kWArray[index]
        }

        @inlinable
        @inline(always)
        public func multiIndex(of index: Index, indices: (borrowing Span<Int>) -> Void) {
            _bathHierarchyWithRow(multiIndices, range: rowRange(at: index), indices)
        }

        @inlinable
        @inline(always)
        public func parentIndices(of index: Index, indices: (borrowing Span<Index>) -> Void) {
            _bathHierarchyWithRow(parentIndices, range: rowRange(at: index), indices)
        }

        @inlinable
        @inline(always)
        public func childIndices(of index: Index, indices: (borrowing Span<Index>) -> Void) {
            _bathHierarchyWithRow(childIndices, range: rowRange(at: index), indices)
        }

        /// `sqrt(n_p)` for factorially scaled auxiliaries.
        @inlinable
        @inline(always)
        public func parentWeights(of index: Index, weights: (borrowing Span<Double>) -> Void) {
            _bathHierarchyWithRow(parentWeights, range: rowRange(at: index), weights)
        }

        /// `sqrt(n_p + 1)`, including at the truncation boundary.
        /// Missing children must still be skipped using their `-1` index.
        @inlinable
        @inline(always)
        public func childWeights(of index: Index, weights: (borrowing Span<Double>) -> Void) {
            _bathHierarchyWithRow(childWeights, range: rowRange(at: index), weights)
        }

        @inlinable
        package func rowRange(at index: Index) -> Range<Int> {
            precondition(index >= 0 && index < count, "Invalid hierarchy state index.")
            let start = index * multiIndexCount
            return start..<(start + multiIndexCount)
        }
    }

}
