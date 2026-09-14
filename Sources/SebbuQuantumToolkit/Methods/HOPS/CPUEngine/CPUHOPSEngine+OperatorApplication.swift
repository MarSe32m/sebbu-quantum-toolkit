// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

extension HOPS.CPUEngine {

	/// Ordinary row-major operators; each contiguous hierarchy row is a ket.
	/// No GEMM packing buffers or operator transposes are needed during propagation.
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
			if matrix.rows == 2 {
				// Hoist these loads out of the row loop. This is the fluorescence
				// case, where even a GEMV call per two-component ket is expensive.
				let a = coefficient * matrix[unchecked: 0, unchecked: 0]
				let b = coefficient * matrix[unchecked: 0, unchecked: 1]
				let c = coefficient * matrix[unchecked: 1, unchecked: 0]
				let d = coefficient * matrix[unchecked: 1, unchecked: 1]
				for h in 0..<states.rows {
					let x = states.elements[2 * h]
					let z = states.elements[2 * h + 1]
					if adding {
						output.elements[2 * h] += a * x + b * z
						output.elements[2 * h + 1] += c * x + d * z
					} else {
						output.elements[2 * h] = a * x + b * z
						output.elements[2 * h + 1] = c * x + d * z
					}
				}
			} else {
				for h in 0..<states.rows {
					vector(
						matrix, x: states.elements + h * states.columns,
						y: output.elements + h * states.columns,
						coefficient: coefficient, adding: adding)
				}
			}
		}

		@inlinable
		static func loss(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			let d = matrix.rows
			precondition(matrix.columns == d && output.rows == d && output.columns == d)
			if d <= 4 {
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
			} else {
				// Each output column is L dagger times a column of L. In
				// particular, dynamic collapse operators never invoke GEMM.
				for j in 0..<d {
					BLAS.zgemv(
						layout: .rowMajor, transpose: .conjugateTranspose,
						m: d, n: d, alpha: .one, a: matrix.elements, lda: d,
						x: matrix.elements + j, incX: d, beta: .zero,
						y: output.elements + j, incY: d)
				}
			}
		}
	}
}
