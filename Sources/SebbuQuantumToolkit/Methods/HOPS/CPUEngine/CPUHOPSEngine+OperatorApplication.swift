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

		/// CSR SpMV used by sparse physical HOPS bath operators. Adjoint
		/// application deliberately receives the separately prepared CSR
		/// conjugate transpose rather than scattering through the forward CSR.
		@inlinable
		@inline(always)
		static func vector(
			_ matrix: borrowing UniqueCSRMatrix<Complex<Double>>,
			x: UnsafePointer<Complex<Double>>,
			y: UnsafeMutablePointer<Complex<Double>>,
			coefficient: Complex<Double> = .one,
			adding: Bool
		) {
			if adding {
				matrix.dot(
					x, multiplied: coefficient, addingInto: y)
			} else {
				matrix.dot(
					x, multiplied: coefficient, into: y)
			}
		}

		/// <psi|L|psi> directly from CSR storage.
		@inlinable
		@inline(always)
		static func expectation(
			_ matrix: borrowing UniqueCSRMatrix<Complex<Double>>,
			state: UnsafePointer<Complex<Double>>
		) -> Complex<Double> {
			var result = Complex<Double>.zero
			matrix.withRowIndices { rowIndices in
				matrix.withColumnIndices { columnIndices in
					matrix.withValues { values in
						for row in 0..<matrix.rows {
							var value = Complex<Double>.zero
							for index in rowIndices[row]..<rowIndices[row + 1] {
								value +=
									values[index]
									* state[columnIndices[index]]
							}
							result += state[row].conjugate * value
						}
					}
				}
			}
			return result
		}

		/// Add a*L + b*L^dagger into a dense common generator in one traversal
		/// of the forward CSR representation.
		@inlinable
		@inline(always)
		static func addSparseBathContributions(
			_ matrix: borrowing UniqueCSRMatrix<Complex<Double>>,
			forwardCoefficient: Complex<Double>,
			adjointCoefficient: Complex<Double>,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			matrix.withRowIndices { rowIndices in
				matrix.withColumnIndices { columnIndices in
					matrix.withValues { values in
						for row in 0..<matrix.rows {
							for index in rowIndices[row]..<rowIndices[row + 1] {
								let column = columnIndices[index]
								let value = values[index]
								if forwardCoefficient != .zero {
									output[unchecked: row, unchecked: column] +=
										forwardCoefficient * value
								}
								if adjointCoefficient != .zero {
									output[unchecked: column, unchecked: row] +=
										adjointCoefficient * value.conjugate
								}
							}
						}
					}
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
