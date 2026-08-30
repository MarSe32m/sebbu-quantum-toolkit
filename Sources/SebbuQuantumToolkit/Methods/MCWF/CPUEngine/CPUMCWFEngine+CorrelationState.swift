// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
    @usableFromInline
	internal struct CorrelationState: ~Copyable, AdaptiveStepODESolverState {
		@usableFromInline
        internal var guide: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var companion: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var hazard: Double

        @inlinable
		internal init(guide: Vector<Complex<Double>>) {
			self.guide = .init(copying: guide)
			self.companion = .zero(guide.count)
			self.hazard = .zero
		}

        @inlinable
		internal init(dimension: Int) {
			self.guide = .zero(dimension)
			self.companion = .zero(dimension)
			self.hazard = .zero
		}

		@inlinable
        internal var norm: Double {
			Swift.max(
				Swift.max(guide.norm, companion.norm),
				abs(hazard)
			)
		}

		@inlinable
        internal func errorNorm(
			to other: borrowing CorrelationState
		) -> Double {
			.hypot(
				guide.euclideanDistance(to: other.guide),
				.hypot(
					companion.euclideanDistance(to: other.companion),
					hazard - other.hazard
				)
			)
		}

        @inlinable
		internal func normalizedError(
			comparedTo lowerOrderEstimate: borrowing CorrelationState,
			relativeTo stepStart: borrowing CorrelationState,
			absoluteTolerance: Double,
			relativeTolerance: Double
		) -> Double {
			var sumOfSquares = 0.0
			for index in 0..<guide.count {
				let guideDifference =
					(guide[index] - lowerOrderEstimate.guide[index]).length
				let guideScale =
					absoluteTolerance
					+ relativeTolerance
					* Swift.max(
						guide[index].length,
						stepStart.guide[index].length
					)
				guard
					guideDifference.isFinite,
					guideScale.isFinite,
					guideScale >= .zero
				else {
					return .infinity
				}
				if guideScale == .zero {
					if guideDifference != .zero { return .infinity }
				} else {
					let ratio = guideDifference / guideScale
					sumOfSquares += ratio * ratio
					if !sumOfSquares.isFinite { return .infinity }
				}

				let companionDifference =
					(companion[index] - lowerOrderEstimate.companion[index])
					.length
				let companionScale =
					absoluteTolerance
					+ relativeTolerance
					* Swift.max(
						companion[index].length,
						stepStart.companion[index].length
					)
				guard
					companionDifference.isFinite,
					companionScale.isFinite,
					companionScale >= .zero
				else {
					return .infinity
				}
				if companionScale == .zero {
					if companionDifference != .zero { return .infinity }
				} else {
					let ratio = companionDifference / companionScale
					sumOfSquares += ratio * ratio
					if !sumOfSquares.isFinite { return .infinity }
				}
			}

			let hazardDifference = abs(hazard - lowerOrderEstimate.hazard)
			let hazardScale =
				absoluteTolerance
				+ relativeTolerance
				* Swift.max(abs(hazard), abs(stepStart.hazard))
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

			return (sumOfSquares / Double(2 * guide.count + 1)).squareRoot()
		}

        @inlinable
        @inline(always)
		internal mutating func assign(_ other: borrowing CorrelationState) {
			guide.copyComponents(from: other.guide)
			companion.copyComponents(from: other.companion)
			hazard = other.hazard
		}

        @inlinable
        @inline(always)
		internal mutating func add(
			_ other: borrowing CorrelationState,
			multiplied coefficient: Double
		) {
			guide.add(other.guide, multiplied: coefficient)
			companion.add(other.companion, multiplied: coefficient)
			hazard += coefficient * other.hazard
		}

        @inlinable
        @inline(always)
		internal mutating func assign(
			_ base: borrowing CorrelationState,
			adding direction: borrowing CorrelationState,
			multipliedBy coefficient: Double
		) {
			guide.copyComponents(
				from: base.guide,
				adding: direction.guide,
				multiplied: coefficient
			)
			companion.copyComponents(
				from: base.companion,
				adding: direction.companion,
				multiplied: coefficient
			)
			hazard = base.hazard + coefficient * direction.hazard
		}
	}
    
    @usableFromInline
    internal struct CorrelationHazardFunctional: ODEStateLinearFunctional {
        @inlinable
        internal init() {}
        
        @inlinable
        @inline(always)
		internal func evaluate(
			_ state: borrowing CorrelationState
		) -> Double {
			state.hazard
		}
	}
}
