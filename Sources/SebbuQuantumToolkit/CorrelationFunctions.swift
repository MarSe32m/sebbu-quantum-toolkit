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

@inlinable
@inline(always)
internal func _correlationInsertionOperator(
	_ insertion: CorrelationInsertion
) -> TimeDependentOperator {
	switch insertion {
	case .left(let value), .right(let value):
		return value
	}
}

@inlinable
internal func _validateTwoTimeCorrelationRequest(
	_ request: TwoTimeCorrelationRequest,
	timeSpan: SimulationTimeSpan,
	dimension: Int
) throws {
	guard request.insertionTime.isFinite else {
		throw TwoTimeCorrelationError.nonFiniteInsertionTime
	}
	guard
		request.insertionTime >= timeSpan.start
			&& request.insertionTime <= timeSpan.end
	else {
		throw TwoTimeCorrelationError.insertionTimeOutsideTimeSpan(
			insertionTime: request.insertionTime,
			start: timeSpan.start,
			end: timeSpan.end
		)
	}

	let insertionOperator = _correlationInsertionOperator(request.insertion)
	if let mismatch = insertionOperator.firstDimensionMismatch(
		expected: dimension
	) {
		throw TwoTimeCorrelationError.insertionOperatorDimensionMismatch(
			expected: dimension,
			rows: mismatch.rows,
			columns: mismatch.columns
		)
	}
	if let mismatch = request.observable.firstDimensionMismatch(
		expected: dimension
	) {
		throw TwoTimeCorrelationError.observableDimensionMismatch(
			expected: dimension,
			rows: mismatch.rows,
			columns: mismatch.columns
		)
	}
}

/// Applies the earlier operator to a guide ket and materializes the second ket
/// needed to represent the inserted dyad.
///
/// A left insertion is represented as `|companion><guide|`, with
/// `|companion> = B|guide>`. A right insertion is represented as
/// `|guide><companion|`, with `|companion> = B^dagger|guide>`.
@inlinable
internal func _applyCorrelationInsertion(
	_ insertion: CorrelationInsertion,
	at time: Double,
	dimension: Int,
	guide: borrowing UniqueVector<Complex<Double>>,
	companion: inout UniqueVector<Complex<Double>>,
	operatorStorage: inout UniqueMatrix<Complex<Double>>
) throws {
	let insertionOperator = _correlationInsertionOperator(insertion)
	insertionOperator.insert(t: time, into: &operatorStorage)
	guard
		operatorStorage.rows == dimension
			&& operatorStorage.columns == dimension
	else {
		throw TwoTimeCorrelationError.insertionOperatorDimensionMismatch(
			expected: dimension,
			rows: operatorStorage.rows,
			columns: operatorStorage.columns
		)
	}

	switch insertion {
	case .left:
		operatorStorage.dotBLAS(guide, into: &companion)

	case .right:
		// Form B^dagger|guide> directly. This insertion occurs once per
		// trajectory, so a compact O(d^2) loop avoids another matrix buffer.
		for row in 0..<dimension {
			var value = Complex<Double>.zero
			for column in 0..<dimension {
				value +=
					operatorStorage[column, row].conjugate
					* guide[column]
			}
			companion[row] = value
		}
	}
}

/// Evaluates the trace of `observable` with the guide-companion dyad.
@inlinable
internal func _correlationSample(
	request: TwoTimeCorrelationRequest,
	at time: Double,
	dimension: Int,
	guide: borrowing UniqueVector<Complex<Double>>,
	companion: borrowing UniqueVector<Complex<Double>>,
	normalization: Double,
	operatorStorage: inout UniqueMatrix<Complex<Double>>,
	actionStorage: inout UniqueVector<Complex<Double>>
) throws -> Complex<Double> {
	precondition(
		normalization.isFinite && normalization > .zero,
		"A correlation trajectory requires a positive finite normalization"
	)

	request.observable.insert(t: time, into: &operatorStorage)
	guard
		operatorStorage.rows == dimension
			&& operatorStorage.columns == dimension
	else {
		throw TwoTimeCorrelationError.observableDimensionMismatch(
			expected: dimension,
			rows: operatorStorage.rows,
			columns: operatorStorage.columns
		)
	}

	let value: Complex<Double>
	switch request.insertion {
	case .left:
		operatorStorage.dotBLAS(companion, into: &actionStorage)
		value = guide.inner(actionStorage)

	case .right:
		operatorStorage.dotBLAS(guide, into: &actionStorage)
		value = companion.inner(actionStorage)
	}
	return value / normalization
}

extension TimeDependentOperator {
    @inlinable
	internal func firstDimensionMismatch(
		expected dimension: Int
	) -> (rows: Int, columns: Int)? {
		switch self {
		case .constant(let constantOperator):
			let matrix = constantOperator.matrix
			guard matrix.rows != dimension || matrix.columns != dimension else {
				return nil
			}
			return (matrix.rows, matrix.columns)

		case .linearCombination(let expansion):
			for component in expansion.operators {
				let matrix = component.matrix
				if matrix.rows != dimension || matrix.columns != dimension {
					return (matrix.rows, matrix.columns)
				}
			}
			return nil

		case .generatedDense:
			// Generated operators materialize into dimensioned caller-owned
			// storage. Validate that storage after invoking the generator.
			return nil
		}
	}
}
