// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	@usableFromInline
	internal struct TrajectoryState: ~Copyable, AdaptiveStepODESolverState,
		PDPState
	{
		@usableFromInline
		internal var wavefunction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var cumulativeHazard: Double

		@inlinable
		internal init(_ state: Vector<Complex<Double>>) {
			self.wavefunction = .init(copying: state)
			self.cumulativeHazard = .zero
		}

		@inlinable
		internal init(dimension: Int) {
			self.wavefunction = .zero(dimension)
			self.cumulativeHazard = .zero
		}

		@inlinable
		internal var norm: Double {
			Swift.max(wavefunction.norm, abs(cumulativeHazard))
		}

		@inlinable
		internal func errorNorm(
			to other: borrowing TrajectoryState
		) -> Double {
			.hypot(
				wavefunction.euclideanDistance(to: other.wavefunction),
				cumulativeHazard - other.cumulativeHazard
			)
		}

		/// Component-wise weighted RMS error for the wavefunction and a
		/// separately scaled error for the accumulated hazard.
		@inlinable
		internal func normalizedError(
			comparedTo lowerOrderEstimate: borrowing TrajectoryState,
			relativeTo stepStart: borrowing TrajectoryState,
			absoluteTolerance: Double,
			relativeTolerance: Double
		) -> Double {
			var sumOfSquares = 0.0
			for index in 0..<wavefunction.count {
				let higherOrder = wavefunction[index]
				let lowerOrder = lowerOrderEstimate.wavefunction[index]
				let initial = stepStart.wavefunction[index]
				let difference = (higherOrder - lowerOrder).length
				let scale =
					absoluteTolerance
					+ relativeTolerance
					* Swift.max(higherOrder.length, initial.length)

				guard difference.isFinite, scale.isFinite, scale >= .zero else {
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

			let hazardDifference = abs(cumulativeHazard - lowerOrderEstimate.cumulativeHazard)
			let hazardScale =
				absoluteTolerance
				+ relativeTolerance * Swift.max(abs(cumulativeHazard), abs(stepStart.cumulativeHazard))
			guard
				hazardDifference.isFinite,
				hazardScale.isFinite,
				hazardScale >= .zero
			else {
				return .infinity
			}
			if hazardScale == .zero {
				if hazardDifference != .zero { return .infinity }
			} else {
				let ratio = hazardDifference / hazardScale
				sumOfSquares += ratio * ratio
				if !sumOfSquares.isFinite { return .infinity }
			}

			return (sumOfSquares / Double(wavefunction.count + 1)).squareRoot()
		}

		@inlinable
		@inline(__always)
		internal mutating func assign(_ other: borrowing TrajectoryState) {
			wavefunction.copyComponents(from: other.wavefunction)
			cumulativeHazard = other.cumulativeHazard
		}

		@inlinable
		@inline(__always)
		internal mutating func add(
			_ other: borrowing TrajectoryState,
			multiplied coefficient: Double
		) {
			wavefunction.add(other.wavefunction, multiplied: coefficient)
			cumulativeHazard += coefficient * other.cumulativeHazard
		}

		@inlinable
		@inline(__always)
		internal mutating func assign(
			_ base: borrowing TrajectoryState,
			adding direction: borrowing TrajectoryState,
			multipliedBy coefficient: Double
		) {
			wavefunction.copyComponents(
				from: base.wavefunction,
				adding: direction.wavefunction,
				multiplied: coefficient
			)
			cumulativeHazard = base.cumulativeHazard + coefficient * direction.cumulativeHazard
		}
	}
}
