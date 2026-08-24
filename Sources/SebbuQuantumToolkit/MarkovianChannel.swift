// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct MarkovianChannel: Sendable {
    public let rate: ScalarTimeFunction
    public let collapseOperator: TimeDependentOperator
    
    @inlinable
    public init(rate: ScalarTimeFunction, collapseOperator: TimeDependentOperator) {
        self.rate = rate
        self.collapseOperator = collapseOperator
    }
    
    //TODO: Convenience methods
}
