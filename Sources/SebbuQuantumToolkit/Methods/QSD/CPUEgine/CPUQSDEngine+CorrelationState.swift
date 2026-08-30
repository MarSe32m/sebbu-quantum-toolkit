// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
    @usableFromInline
	internal struct CorrelationState: ~Copyable, FixedStepSDESolverState {
        @usableFromInline
		internal var guide: UniqueVector<Complex<Double>>
        @usableFromInline
		internal var companion: UniqueVector<Complex<Double>>

        @inlinable
		internal init(guide: Vector<Complex<Double>>) {
			self.guide = .init(copying: guide)
			self.companion = .zero(guide.count)
		}

        @inlinable
		internal init(dimension: Int) {
			self.guide = .zero(dimension)
			self.companion = .zero(dimension)
		}

        @inlinable
		@inline(always)
		internal mutating func zero() {
			guide.zeroComponents()
			companion.zeroComponents()
		}

        @inlinable
        @inline(always)
		internal mutating func assign(_ other: borrowing CorrelationState) {
			guide.copyComponents(from: other.guide)
			companion.copyComponents(from: other.companion)
		}

        @inlinable
        @inline(always)
		internal mutating func add(
			_ other: borrowing CorrelationState,
			multiplied coefficient: Double
		) {
			guide.add(other.guide, multiplied: coefficient)
			companion.add(other.companion, multiplied: coefficient)
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
		}

        @inlinable
        @inline(always)
		internal mutating func add(
			_ other: borrowing CorrelationState,
			scaledBy noise: borrowing Complex<Double>
		) {
			guide.add(other.guide, multiplied: noise)
			companion.add(other.companion, multiplied: noise)
		}
	}
}
