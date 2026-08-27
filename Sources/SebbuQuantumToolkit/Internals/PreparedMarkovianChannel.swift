// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Precomputed matrices for a time-independent collapse operator.
///
/// A reference type is intentional here. It lets the containing RHS use an
/// ordinary `Array` while each channel still uniquely owns its matrix
/// allocations. This avoids exercising arrays of noncopyable elements in the
/// Swift 6.3 compiler and costs only one allocation per channel during setup.
@usableFromInline
internal final class _PreparedConstantMarkovianChannel {
	@usableFromInline
	internal let rate: ScalarTimeFunction
	@usableFromInline
	internal let collapseOperator: UniqueMatrix<Complex<Double>>
	@usableFromInline
	internal let collapseOperatorAdjoint: UniqueMatrix<Complex<Double>>
	@usableFromInline
	internal let lossOperator: UniqueMatrix<Complex<Double>>

	@inlinable
	internal init(
		collapseOperator: consuming UniqueMatrix<Complex<Double>>,
		rate: ScalarTimeFunction
	) {
		let adjoint = collapseOperator.conjugateTranspose
		let lossOperator = adjoint.dot(collapseOperator)

		self.rate = rate
		self.collapseOperator = collapseOperator
		self.collapseOperatorAdjoint = adjoint
		self.lossOperator = lossOperator
	}

	@inlinable
	internal convenience init?(
		_ channel: MarkovianChannel,
		dimension: Int
	) {
		switch channel.collapseOperator {
		case .constant(let constantOperator):
			Self.validate(
				constantOperator.matrix,
				dimension: dimension
			)
			self.init(
				collapseOperator: UniqueMatrix(copying: constantOperator.matrix),
				rate: channel.rate
			)

		case .linearCombination(let expansion):
			guard expansion.isConstant else { return nil }

			var collapseOperator = UniqueMatrix<Complex<Double>>.zeros(
				rows: dimension,
				columns: dimension
			)
			for index in expansion.operators.indices {
				let matrix = expansion.operators[index].matrix
				Self.validate(matrix, dimension: dimension)
				guard
					case .constant(let coefficient) = expansion.coefficients[
						index]
				else {
					preconditionFailure(
						"A constant operator expansion contains a generated coefficient"
					)
				}
				collapseOperator.add(matrix, multiplied: coefficient)
			}

			self.init(
				collapseOperator: collapseOperator,
				rate: channel.rate
			)

		case .generatedDense:
			return nil
		}
	}

	@inlinable
	internal static func validate(
		_ matrix: Matrix<Complex<Double>>,
		dimension: Int
	) {
		precondition(
			matrix.rows == dimension && matrix.columns == dimension,
			"Collapse-operator dimensions do not match the quantum system"
		)
	}
}

/// A channel whose collapse operator must be materialized at every RHS time.
@usableFromInline
internal struct _PreparedDynamicMarkovianChannel {
	@usableFromInline
	internal let rate: ScalarTimeFunction
	@usableFromInline
	internal let collapseOperator: TimeDependentOperator

	@inlinable
	internal init(_ channel: MarkovianChannel, dimension: Int) {
		switch channel.collapseOperator {
		case .constant(let constantOperator):
			Self.validate(constantOperator.matrix, dimension: dimension)
		case .linearCombination(let expansion):
			for operatorComponent in expansion.operators {
				Self.validate(operatorComponent.matrix, dimension: dimension)
			}
		case .generatedDense:
			break
		}

		self.rate = channel.rate
		self.collapseOperator = channel.collapseOperator
	}

	@inlinable
	@inline(always)
	internal func insert(
		t: Double,
		into output: inout UniqueMatrix<Complex<Double>>
	) {
		collapseOperator.insert(t: t, into: &output)
	}

	@inlinable
	@inline(always)
	internal static func validate(
		_ matrix: Matrix<Complex<Double>>,
		dimension: Int
	) {
		precondition(
			matrix.rows == dimension && matrix.columns == dimension,
			"Collapse-operator dimensions do not match the quantum system"
		)
	}
}
