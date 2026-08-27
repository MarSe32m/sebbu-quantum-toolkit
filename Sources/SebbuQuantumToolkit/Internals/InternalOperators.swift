// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension TimeDependentOperator {
	/// Materializes this operator in caller-owned storage.
	///
	/// Keeping the public, copyable operator representation here avoids a
	/// second hierarchy of noncopyable operator enums in the numerical
	/// implementations. The latter substantially complicates ownership
	/// lowering without saving any per-step allocations.
	@inlinable
	internal func insert(
		t: Double,
		into output: inout UniqueMatrix<Complex<Double>>
	) {
		switch self {
		case .constant(let constantOperator):
			precondition(
				output.rows == constantOperator.matrix.rows
					&& output.columns == constantOperator.matrix.columns,
				"Operator dimensions do not match the output buffer"
			)
			output.copyElements(from: constantOperator.matrix)

		case .linearCombination(let expansion):
			precondition(
				expansion.coefficients.count == expansion.operators.count
					&& !expansion.operators.isEmpty,
				"An operator expansion must contain matching, nonempty coefficient and operator arrays"
			)

			for index in expansion.operators.indices {
				let matrix = expansion.operators[index].matrix
				precondition(
					output.rows == matrix.rows
						&& output.columns == matrix.columns,
					"Operator dimensions do not match the output buffer"
				)
				let coefficient = expansion.coefficients[index](t)
				if index == expansion.operators.startIndex {
					output.copyElements(from: matrix, multiplied: coefficient)
				} else {
					output.add(matrix, multiplied: coefficient)
				}
			}

		case .generatedDense(let dynamicOperator):
			dynamicOperator.generator(t, &output)
		}
	}
}
