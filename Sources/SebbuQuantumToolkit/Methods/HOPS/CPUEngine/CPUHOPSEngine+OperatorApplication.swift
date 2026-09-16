// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// Ordinary row-major operators. Each contiguous hierarchy row is a ket.
	///
	/// HOPS deliberately uses the scalar matrix-vector kernels from
	/// `sebbu-science` here. The hierarchy calls these kernels very frequently
	/// on relatively small vectors and dispatching each action through BLAS
	/// performs poorly when many trajectories execute concurrently.
    //TODO: Can we build OpenBLAS with some settings where it doesn't perform this poorly??
	@usableFromInline
	internal enum OperatorApplication {
		@inlinable
		@inline(always)
		static func vector(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			adjoint: Bool = false,
			x: UnsafePointer<Complex<Double>>,
			y: UnsafeMutablePointer<Complex<Double>>,
			coefficient: Complex<Double> = .one,
			adding: Bool
		) {
			if adjoint {
				if adding {
					matrix.unsafeAdjointDot(
						x, multiplied: coefficient, addingInto: y)
				} else {
					matrix.unsafeAdjointDot(
						x, multiplied: coefficient, into: y)
				}
			} else {
				if adding {
					matrix.unsafeDot(
						x, multiplied: coefficient, addingInto: y)
				} else {
					matrix.unsafeDot(
						x, multiplied: coefficient, into: y)
				}
			}
		}

		@inlinable
		@inline(always)
		static func apply(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			to states: borrowing UniqueMatrix<Complex<Double>>,
			multiplied coefficient: Complex<Double> = .one,
			adding: Bool,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			precondition(
				matrix.rows == matrix.columns && states.columns == matrix.rows
					&& output.rows == states.rows
					&& output.columns == states.columns)

			if adding {
				for h in 0..<states.rows {
					matrix.unsafeDot(
						states.elements + h &* states.columns,
						multiplied: coefficient,
						addingInto: output.elements + h &* states.columns)
				}
			} else {
				for h in 0..<states.rows {
					matrix.unsafeDot(
						states.elements + h &* states.columns,
						multiplied: coefficient,
						into: output.elements + h &* states.columns)
				}
			}
		}

		// Forms L^dagger L into output.
		@inlinable
		static func loss(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			let d = matrix.rows
			precondition(
				matrix.columns == d && output.rows == d && output.columns == d)
			for i in 0..<d {
				for j in 0..<d {
					var value = Complex<Double>.zero
					for k in 0..<d {
						value +=
							matrix[unchecked: k, unchecked: i].conjugate
							* matrix[unchecked: k, unchecked: j]
					}
					output[unchecked: i, unchecked: j] = value
				}
			}
		}
	}
}
