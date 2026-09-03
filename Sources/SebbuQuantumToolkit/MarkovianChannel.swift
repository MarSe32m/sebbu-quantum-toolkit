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

    @inlinable
    public init(rate: Double, collapseOperator: Matrix<Complex<Double>>) {
        self.rate = .constant(rate)
        self.collapseOperator = .constant(collapseOperator)
    }
    
    @inlinable
    public init(rate: Double, collapseOperator: borrowing UniqueMatrix<Complex<Double>>) {
        self.rate = .constant(rate)
        self.collapseOperator = .constant(collapseOperator)
    }
    
    //TODO: More convenience methods
}

public enum MarkovianUnravelling: Sendable {
	case diffusive
	case jump
}
