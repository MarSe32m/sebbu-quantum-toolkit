// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

extension HOPS.CPUEngine {
	/// Ordinary row-major operators. Each contiguous hierarchy row is a ket.
	@usableFromInline
	internal enum OperatorApplication {
		@inlinable
		@inline(always)
		static func vector(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			adjoint: Bool = false, x: UnsafePointer<Complex<Double>>,
			y: UnsafeMutablePointer<Complex<Double>>,
			coefficient: Complex<Double> = .one,
			adding: Bool
		) {
			let d = matrix.rows
			if d <= 4 {
				for i in 0..<d {
					var value = Complex<Double>.zero
					for j in 0..<d {
						let a =
							adjoint
							? matrix[unchecked: j, unchecked: i]
								.conjugate
							: matrix[unchecked: i, unchecked: j]
						value += a * x[j]
					}
					if adding {
						y[i] += coefficient * value
					} else {
						y[i] = coefficient * value
					}
				}
			} else {
                
				BLAS.zgemv(
					layout: .rowMajor,
					transpose: adjoint ? .conjugateTranspose : .noTranspose,
					m: d, n: d, alpha: coefficient, a: matrix.elements, lda: d,
					x: x, incX: 1, beta: adding ? .one : .zero, y: y, incY: 1)
			}
		}
        
		@inlinable
        @inline(always)
		static func apply(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			to states: borrowing UniqueMatrix<Complex<Double>>,
			multiplied coefficient: Complex<Double> = .one,
			adding: Bool, into output: inout UniqueMatrix<Complex<Double>>
		) {
			precondition(
				matrix.rows == matrix.columns && states.columns == matrix.rows
					&& output.rows == states.rows
					&& output.columns == states.columns)
            if adding {
                for h in 0..<states.rows {
                    matrix.unsafeDot(states.elements + h &* states.columns, multiplied: coefficient, addingInto: output.elements + h &* states.columns)
                }
            } else {
                for h in 0..<states.rows {
                    matrix.unsafeDot(states.elements + h &* states.columns, multiplied: coefficient, into: output.elements + h &* states.columns)
                }
            }
		}

        // Forms L^dagger L into output
		@inlinable
		static func loss(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			let d = matrix.rows
			precondition(matrix.columns == d && output.rows == d && output.columns == d)
            //TODO: Implement UniqueMatrix.adjointDot(UniqueMatrix) in sebbu-science
            for i in 0..<d {
                for j in 0..<d {
                    var value = Complex<Double>.zero
                    for k in 0..<d {
                        value +=
                            matrix[unchecked: k, unchecked: i]
                            .conjugate
                            * matrix[unchecked: k, unchecked: j]
                    }
                    output[unchecked: i, unchecked: j] = value
                }
            }
		}
	}
}
