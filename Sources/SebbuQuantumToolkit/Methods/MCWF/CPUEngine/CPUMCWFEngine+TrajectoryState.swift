// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	@usableFromInline
	internal struct TrajectoryState: ~Copyable, AdaptiveStepODESolverState {
		@usableFromInline
		internal var wavefunction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var hazard: Double

		@inlinable
		internal init(_ state: Vector<Complex<Double>>) {
			self.wavefunction = .init(copying: state)
			self.hazard = .zero
		}

		@inlinable
		internal init(dimension: Int) {
			self.wavefunction = .zero(dimension)
			self.hazard = .zero
		}

		@inlinable
		internal var norm: Double {
			Swift.max(wavefunction.norm, abs(hazard))
		}

		@inlinable
		internal func errorNorm(
			to other: borrowing TrajectoryState
		) -> Double {
			.hypot(
				wavefunction.euclideanDistance(to: other.wavefunction),
				hazard - other.hazard
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

			let hazardDifference = abs(hazard - lowerOrderEstimate.hazard)
			let hazardScale =
				absoluteTolerance
				+ relativeTolerance * Swift.max(abs(hazard), abs(stepStart.hazard))
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
			hazard = other.hazard
		}

		@inlinable
		@inline(__always)
		internal mutating func add(
			_ other: borrowing TrajectoryState,
			multiplied coefficient: Double
		) {
			wavefunction.add(other.wavefunction, multiplied: coefficient)
			hazard += coefficient * other.hazard
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
			hazard = base.hazard + coefficient * direction.hazard
		}
	}

	@usableFromInline
	internal struct HazardFunctional: ODEStateLinearFunctional {
		@inlinable
		@inline(__always)
		internal func evaluate(_ state: borrowing TrajectoryState) -> Double {
			state.hazard
		}
	}
}
