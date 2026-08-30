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

/// An operator insertion at a fixed absolute simulation time.
public struct TimedCorrelationInsertion: Sendable {
	/// The absolute time at which the insertion is applied.
	public var time: Double

	/// The operator and density-operator side on which it is applied.
	public var insertion: CorrelationInsertion

	@inlinable
	public init(time: Double, insertion: CorrelationInsertion) {
		self.time = time
		self.insertion = insertion
	}
}

/// A request for a chronologically ordered multi-time correlation.
///
/// The insertion events represent the fixed-time operators
/// `A_1(t_1), ..., A_(n-1)(t_(n-1))` and must be supplied in strictly
/// increasing time order. The final operator `A_n(t)` is ``observable`` and
/// is evaluated at each scheduled output time at or after the last insertion.
/// For the usual time-ordered correlation, use a left insertion for every
/// event. Right insertions are also supported.
public struct MultiTimeOrderedCorrelationRequest: Sendable {
	public var insertions: [TimedCorrelationInsertion]
	public var observable: TimeDependentOperator

	@inlinable
	public init(
		insertions: [TimedCorrelationInsertion],
		observable: TimeDependentOperator
	) {
		self.insertions = insertions
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

/// Validation errors for a multi-time ordered correlation request.
public enum MultiTimeOrderedCorrelationError: Error, Equatable, Sendable {
	case noInsertions
	case nonFiniteInsertionTime(index: Int)
	case insertionTimeOutsideTimeSpan(
		index: Int,
		insertionTime: Double,
		start: Double,
		end: Double
	)
	case insertionTimesNotStrictlyIncreasing(
		previousIndex: Int,
		previousTime: Double,
		index: Int,
		time: Double
	)
	case insertionOperatorDimensionMismatch(
		index: Int,
		expected: Int,
		rows: Int,
		columns: Int
	)
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

internal func _validateMultiTimeOrderedCorrelationRequest(
	_ request: MultiTimeOrderedCorrelationRequest,
	timeSpan: SimulationTimeSpan,
	dimension: Int
) throws {
	guard !request.insertions.isEmpty else {
		throw MultiTimeOrderedCorrelationError.noInsertions
	}
	for index in request.insertions.indices {
		let event = request.insertions[index]
		guard event.time.isFinite else {
			throw MultiTimeOrderedCorrelationError.nonFiniteInsertionTime(index: index)
		}
		guard event.time >= timeSpan.start && event.time <= timeSpan.end else {
			throw MultiTimeOrderedCorrelationError.insertionTimeOutsideTimeSpan(
				index: index,
				insertionTime: event.time,
				start: timeSpan.start,
				end: timeSpan.end
			)
		}
		if index > request.insertions.startIndex {
			let previousIndex = index - 1
			let previousTime = request.insertions[previousIndex].time
			guard previousTime < event.time else {
				throw MultiTimeOrderedCorrelationError
					.insertionTimesNotStrictlyIncreasing(
						previousIndex: previousIndex,
						previousTime: previousTime,
						index: index,
						time: event.time
					)
			}
		}
		let insertionOperator = _correlationInsertionOperator(event.insertion)
		if let mismatch = insertionOperator.firstDimensionMismatch(expected: dimension) {
			throw MultiTimeOrderedCorrelationError
				.insertionOperatorDimensionMismatch(
					index: index,
					expected: dimension,
					rows: mismatch.rows,
					columns: mismatch.columns
				)
		}
	}
	if let mismatch = request.observable.firstDimensionMismatch(expected: dimension) {
		throw MultiTimeOrderedCorrelationError.observableDimensionMismatch(
			expected: dimension,
			rows: mismatch.rows,
			columns: mismatch.columns
		)
	}
}

internal func _multiTimeOrderedRequest(
	_ request: TwoTimeCorrelationRequest
) -> MultiTimeOrderedCorrelationRequest {
	MultiTimeOrderedCorrelationRequest(
		insertions: [
			TimedCorrelationInsertion(
				time: request.insertionTime,
				insertion: request.insertion
			)
		],
		observable: request.observable
	)
}

internal func _mapMultiTimeOrderedErrorToTwoTime(
	_ error: MultiTimeOrderedCorrelationError
) -> TwoTimeCorrelationError {
	switch error {
	case .noInsertions, .insertionTimesNotStrictlyIncreasing:
		preconditionFailure(
			"A converted two-time request always contains exactly one insertion"
		)
	case .nonFiniteInsertionTime:
		return .nonFiniteInsertionTime
	case .insertionTimeOutsideTimeSpan(_, let time, let start, let end):
		return .insertionTimeOutsideTimeSpan(
			insertionTime: time,
			start: start,
			end: end
		)
	case .insertionOperatorDimensionMismatch(
		_, let expected, let rows, let columns
	):
		return .insertionOperatorDimensionMismatch(
			expected: expected,
			rows: rows,
			columns: columns
		)
	case .observableDimensionMismatch(let expected, let rows, let columns):
		return .observableDimensionMismatch(
			expected: expected,
			rows: rows,
			columns: columns
		)
	}
}

/// Applies one insertion to the ket or bra side of `|ket><bra|`.
internal func _applyMultiTimeCorrelationInsertion(
	_ insertion: CorrelationInsertion,
	index: Int,
	at time: Double,
	dimension: Int,
	ket: inout UniqueVector<Complex<Double>>,
	bra: inout UniqueVector<Complex<Double>>,
	operatorStorage: inout UniqueMatrix<Complex<Double>>,
	actionStorage: inout UniqueVector<Complex<Double>>
) throws {
	let insertionOperator = _correlationInsertionOperator(insertion)
	insertionOperator.insert(t: time, into: &operatorStorage)
	guard operatorStorage.rows == dimension && operatorStorage.columns == dimension else {
		throw MultiTimeOrderedCorrelationError.insertionOperatorDimensionMismatch(
			index: index,
			expected: dimension,
			rows: operatorStorage.rows,
			columns: operatorStorage.columns
		)
	}
	switch insertion {
	case .left:
		operatorStorage.dotBLAS(ket, into: &actionStorage)
		ket.copyComponents(from: actionStorage)
	case .right:
		for row in 0..<dimension {
			var value = Complex<Double>.zero
			for column in 0..<dimension {
				value += operatorStorage[column, row].conjugate * bra[column]
			}
			actionStorage[row] = value
		}
		bra.copyComponents(from: actionStorage)
	}
}

internal func _multiTimeCorrelationSample(
	observable: TimeDependentOperator,
	at time: Double,
	dimension: Int,
	ket: borrowing UniqueVector<Complex<Double>>,
	bra: borrowing UniqueVector<Complex<Double>>,
	normalization: Double,
	operatorStorage: inout UniqueMatrix<Complex<Double>>,
	actionStorage: inout UniqueVector<Complex<Double>>
) throws -> Complex<Double> {
	precondition(normalization.isFinite && normalization > .zero)
	observable.insert(t: time, into: &operatorStorage)
	guard operatorStorage.rows == dimension && operatorStorage.columns == dimension else {
		throw MultiTimeOrderedCorrelationError.observableDimensionMismatch(
			expected: dimension,
			rows: operatorStorage.rows,
			columns: operatorStorage.columns
		)
	}
	operatorStorage.dotBLAS(ket, into: &actionStorage)
	return bra.inner(actionStorage) / normalization
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
