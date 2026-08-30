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
		internal var ket: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var bra: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var hazard: Double

        @inlinable
		internal init(guide: Vector<Complex<Double>>) {
			self.guide = .init(copying: guide)
			self.ket = .zero(guide.count)
			self.bra = .zero(guide.count)
			self.hazard = .zero
		}

		@inlinable
		internal init(guideAndDyad guide: Vector<Complex<Double>>) {
			self.guide = .init(copying: guide)
			self.ket = .init(copying: guide)
			self.bra = .init(copying: guide)
			self.hazard = .zero
		}

        @inlinable
		internal init(dimension: Int) {
			self.guide = .zero(dimension)
			self.ket = .zero(dimension)
			self.bra = .zero(dimension)
			self.hazard = .zero
		}

		@inlinable
        internal var norm: Double {
			Swift.max(
				Swift.max(Swift.max(guide.norm, ket.norm), bra.norm),
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
					ket.euclideanDistance(to: other.ket),
					.hypot(
						bra.euclideanDistance(to: other.bra),
						hazard - other.hazard
					)
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

				let ketDifference =
					(ket[index] - lowerOrderEstimate.ket[index])
					.length
				let ketScale =
					absoluteTolerance
					+ relativeTolerance
					* Swift.max(
						ket[index].length,
						stepStart.ket[index].length
					)
				guard
					ketDifference.isFinite,
					ketScale.isFinite,
					ketScale >= .zero
				else {
					return .infinity
				}
				if ketScale == .zero {
					if ketDifference != .zero { return .infinity }
				} else {
					let ratio = ketDifference / ketScale
					sumOfSquares += ratio * ratio
					if !sumOfSquares.isFinite { return .infinity }
				}

				let braDifference =
					(bra[index] - lowerOrderEstimate.bra[index]).length
				let braScale =
					absoluteTolerance
					+ relativeTolerance
					* Swift.max(
						bra[index].length,
						stepStart.bra[index].length
					)
				guard braDifference.isFinite, braScale.isFinite, braScale >= .zero else {
					return .infinity
				}
				if braScale == .zero {
					if braDifference != .zero { return .infinity }
				} else {
					let ratio = braDifference / braScale
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

			return (sumOfSquares / Double(3 * guide.count + 1)).squareRoot()
		}

        @inlinable
        @inline(always)
		internal mutating func assign(_ other: borrowing CorrelationState) {
			guide.copyComponents(from: other.guide)
			ket.copyComponents(from: other.ket)
			bra.copyComponents(from: other.bra)
			hazard = other.hazard
		}

        @inlinable
        @inline(always)
		internal mutating func add(
			_ other: borrowing CorrelationState,
			multiplied coefficient: Double
		) {
			guide.add(other.guide, multiplied: coefficient)
			ket.add(other.ket, multiplied: coefficient)
			bra.add(other.bra, multiplied: coefficient)
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
			ket.copyComponents(
				from: base.ket,
				adding: direction.ket,
				multiplied: coefficient
			)
			bra.copyComponents(
				from: base.bra,
				adding: direction.bra,
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
