// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension ConstantOperator {
    /// Scatter only stored CSR entries; prepared hot paths retain CSR storage.
    @inlinable
    internal func add(into output: inout UniqueMatrix<Complex<Double>>, scale: Complex<Double>) {
        switch storage {
        case .dense(let matrix):
            precondition(
                output.rows == matrix.rows && output.columns == matrix.columns,
                "Operator dimensions do not match the output buffer")
            output.add(matrix, multiplied: scale)
        case .sparse(let matrix):
            precondition(
                output.rows == matrix.rows && output.columns == matrix.columns,
                "Operator dimensions do not match the output buffer")
            matrix.withRowIndices { rows in
                matrix.withColumnIndices { columns in
                    matrix.withValues { values in
                        for row in 0..<matrix.rows {
                            for k in rows[row]..<rows[row + 1] {
                                output[unchecked: row, unchecked: columns[k]] += scale * values[k]
                            }
                        }
                    }
                }
            }
        }
    }
}

extension TimeDependentOperator {
    /// Materialization fallback, correct for mixed dense/CSR expansions too.
    @inlinable
    internal func insert(t: Double, into output: inout UniqueMatrix<Complex<Double>>) {
        switch self {
        case .constant(let op):
            if case .dense(let matrix) = op.storage {
                precondition(
                    output.rows == matrix.rows && output.columns == matrix.columns,
                    "Operator dimensions do not match the output buffer")
                output.copyElements(from: matrix)
            } else {
                output.zeroElements()
                op.add(into: &output, scale: .one)
            }
        case .linearCombination(let expansion):
            output.zeroElements()
            for i in expansion.operators.indices {
                expansion.operators[i].add(into: &output, scale: expansion.coefficients[i](t))
            }
        case .generatedDense(let op): op.generator(t, &output)
        }
    }
}
