// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum CorrelationInsertion: Sendable {
	/// Applies `B rho(s)`, producing `<A(t) B(s)>` under the quantum
	/// regression theorem.
	case left(TimeDependentOperator)

	/// Applies `rho(s) B`, producing `<B(s) A(t)>` under the quantum
	/// regression theorem.
	case right(TimeDependentOperator)
}

/// A request for a two-time correlation with an operator insertion at the
/// earlier time `s` and observations at later absolute times `t`.
///
/// Output times are taken from the propagation's ``OutputSchedule``. Times
/// before ``insertionTime`` are skipped because the requested correlation is
/// only defined for `t >= s`.
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

/// Validation errors for a two-time-correlation request.
public enum TwoTimeCorrelationError: Error, Equatable, Sendable {
	/// The insertion time is NaN or infinite.
	case nonFiniteInsertionTime

	/// The insertion time does not lie in the propagation time span.
	case insertionTimeOutsideTimeSpan(
		insertionTime: Double,
		start: Double,
		end: Double
	)

	/// The insertion operator is not square with the system dimension.
	case insertionOperatorDimensionMismatch(
		expected: Int,
		rows: Int,
		columns: Int
	)

	/// The observable is not square with the system dimension.
	case observableDimensionMismatch(
		expected: Int,
		rows: Int,
		columns: Int
	)
}
