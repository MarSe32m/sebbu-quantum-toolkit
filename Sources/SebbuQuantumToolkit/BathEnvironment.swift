// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Physical coupling operators and their common latent bath model, usable by
/// both HOPS and HEOM without conversion or a second bath fit.
public struct BathEnvironment: Sendable {
    public let couplingOperators: [TimeDependentOperator]
    public let bath: CorrelatedBathModel

    @inlinable
    public init(couplingOperators: [TimeDependentOperator], bath: CorrelatedBathModel) {
        precondition(
            couplingOperators.count == bath.channelCount,
            "There must be one coupling operator per physical bath channel.")
        self.couplingOperators = couplingOperators
        self.bath = bath
    }

    @inlinable
    public init(couplingOperator: TimeDependentOperator, bath: CorrelatedBathModel) {
        self.init(couplingOperators: [couplingOperator], bath: bath)
    }
}

public enum BathHierarchyTruncation: Sendable {
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
