// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum CorrelationInsertion: Sendable {
	/// Bρ(s), producing ⟨A(t)B(s)⟩.
	case left(TimeDependentOperator)

	/// ρ(s)B, producing ⟨B(s)A(t)⟩.
	case right(TimeDependentOperator)
}

public struct TwoTimeCorrelationRequest: Sendable {
	/// Earlier operator-insertion time s.
	public var insertionTime: Double

	/// Bρ(s) or ρ(s)B.
	public var insertion: CorrelationInsertion

	/// Later operator A, evaluated at each reported time t.
	public var observable: TimeDependentOperator

    @inlinable
	public init(
		insertionTime: Double,
		insertion: CorrelationInsertion,
		observable: TimeDependentOperator
	) {
		self.insertionTime = insertionTime
		self.insertion = insertion
		self.observable = observable
	}
}
