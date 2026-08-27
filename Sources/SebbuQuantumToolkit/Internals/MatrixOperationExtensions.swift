// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension MatrixOperations {
	/// Computes `Tr(A B)` directly in O(n^2) operations without allocating
	/// space for the matrix product.
	@inlinable
	static func traceOfProduct<T: AlgebraicField>(
		_ lhs: borrowing UniqueMatrix<T>,
		_ rhs: borrowing UniqueMatrix<T>
	) -> T {
		precondition(
			lhs.columns == rhs.rows,
			"The matrices must be compatible for multiplication"
		)
		precondition(
			lhs.rows == rhs.columns,
			"The resulting matrix must be square"
		)

		var result: T = .zero
		for row in 0..<lhs.rows {
			for column in 0..<lhs.columns {
				result += lhs[row, column] * rhs[column, row]
			}
		}
		return result
	}
}
