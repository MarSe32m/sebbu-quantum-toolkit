// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

internal extension MatrixOperations {
    @inlinable
    static func trace<T: AlgebraicField>(_ A: Matrix<T>) -> T {
        return A.trace
    }
    
    @inlinable
    static func trace<T: AlgebraicField>(_ A: borrowing UniqueMatrix<T>) -> T {
        return A.trace
    }
    
    @inlinable
    static func trace<T: AlgebraicField>(_ A: Matrix<T>, _ B: Matrix<T>) -> T {
        precondition(A.columns == B.rows, "The matrices must be compatible for multiplication.")
        precondition(A.rows == B.columns, "The resulting matrix must be square.")
        var result: T = .zero
        for i in 0..<A.rows {
            for j in 0..<A.columns {
                result += A[i, j] * B[j, i]
            }
        }
        return result
    }
    
    @inlinable
    static func trace<T: AlgebraicField>(
        _ A: borrowing UniqueMatrix<T>,
        _ B: borrowing UniqueMatrix<T>
    ) -> T {
        precondition(A.columns == B.rows, "The matrices must be compatible for multiplication.")
        precondition(A.rows == B.columns, "The resulting matrix must be square.")
        var result: T = .zero
        for i in 0..<A.rows {
            for j in 0..<A.columns {
                result += A[i, j] * B[j, i]
            }
        }
        return result
    }
}
