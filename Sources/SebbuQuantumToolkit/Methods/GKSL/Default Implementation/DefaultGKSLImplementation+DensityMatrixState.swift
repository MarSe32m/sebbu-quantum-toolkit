// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension DefaultGKSLImplementation {
	@usableFromInline
	internal struct DensityMatrixState: ~Copyable, AdaptiveStepODESolverState {
		@usableFromInline
		internal var densityMatrix: UniqueMatrix<Complex<Double>>

		@usableFromInline
		internal var norm: Double { densityMatrix.frobeniusNorm }

		@inlinable
		internal init(_ densityMatrix: borrowing UniqueMatrix<Complex<Double>>) {
			self.densityMatrix = .init(copying: densityMatrix)
		}

		@inlinable
		internal init(_ densityMatrix: Matrix<Complex<Double>>) {
			self.densityMatrix = .init(copying: densityMatrix)
		}

		@inlinable
		internal init(dimension: Int) {
			self.densityMatrix = .zeros(rows: dimension, columns: dimension)
		}

		@inlinable
		internal func errorNorm(
			to other: borrowing DensityMatrixState
		) -> Double {
			densityMatrix.frobeniusDistance(to: other.densityMatrix)
		}

		/// Component-wise weighted RMS error for the complex density-matrix
		/// entries. This avoids making tolerance semantics depend on Hilbert-
		/// space dimension or on a single global Frobenius scale.
		@inlinable
		internal func normalizedError(
			comparedTo lowerOrderEstimate: borrowing DensityMatrixState,
			relativeTo stepStart: borrowing DensityMatrixState,
			absoluteTolerance: Double,
			relativeTolerance: Double
		) -> Double {
			let componentCount = densityMatrix.rows * densityMatrix.columns
			var sumOfSquares = 0.0

			for row in 0..<densityMatrix.rows {
				for column in 0..<densityMatrix.columns {
					let higherOrder = densityMatrix[row, column]
					let lowerOrder = lowerOrderEstimate.densityMatrix[
						row, column]
					let initial = stepStart.densityMatrix[row, column]
					let difference = (higherOrder - lowerOrder).length
					let scale =
						absoluteTolerance
						+ relativeTolerance
						* Swift.max(higherOrder.length, initial.length)

					guard difference.isFinite, scale.isFinite, scale >= .zero
					else {
						return .infinity
					}
					if scale == .zero {
						if difference != .zero { return .infinity }
						continue
					}

					let ratio = difference / scale
					sumOfSquares += ratio * ratio
					if !sumOfSquares.isFinite { return .infinity }
				}
			}

			return (sumOfSquares / Double(componentCount)).squareRoot()
		}

		@inlinable
		internal mutating func assign(
			_ other: borrowing DensityMatrixState
		) {
			densityMatrix.copyElements(from: other.densityMatrix)
		}

		@inlinable
		internal mutating func add(
			_ other: borrowing DensityMatrixState,
			multiplied coefficient: Double
		) {
			densityMatrix.add(other.densityMatrix, multiplied: coefficient)
		}

		@inlinable
		internal mutating func assign(
			_ base: borrowing DensityMatrixState,
			adding direction: borrowing DensityMatrixState,
			multipliedBy coefficient: Double
		) {
			densityMatrix.copyElements(
				from: base.densityMatrix,
				adding: direction.densityMatrix,
				multiplied: Complex(coefficient)
			)
		}
	}
}
