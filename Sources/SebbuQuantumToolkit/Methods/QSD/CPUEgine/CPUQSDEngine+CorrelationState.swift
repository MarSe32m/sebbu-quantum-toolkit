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
		internal var ket: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var bra: UniqueVector<Complex<Double>>

        @inlinable
		internal init(guide: Vector<Complex<Double>>) {
			self.guide = .init(copying: guide)
			self.ket = .zero(guide.count)
			self.bra = .zero(guide.count)
		}

        @inlinable
		internal init(dimension: Int) {
			self.guide = .zero(dimension)
			self.ket = .zero(dimension)
			self.bra = .zero(dimension)
		}

		internal mutating func initializeDyadFromGuide() {
			let guideCopy = UniqueVector(copying: guide)
			ket.copyComponents(from: guideCopy)
			bra.copyComponents(from: guideCopy)
		}

        @inlinable
		@inline(always)
		internal mutating func zero() {
			guide.zeroComponents()
			ket.zeroComponents()
			bra.zeroComponents()
		}

        @inlinable
        @inline(always)
		internal mutating func assign(_ other: borrowing CorrelationState) {
			guide.copyComponents(from: other.guide)
			ket.copyComponents(from: other.ket)
			bra.copyComponents(from: other.bra)
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
		}

        @inlinable
        @inline(always)
		internal mutating func add(
			_ other: borrowing CorrelationState,
			scaledBy noise: borrowing Complex<Double>
		) {
			guide.add(other.guide, multiplied: noise)
			ket.add(other.ket, multiplied: noise)
			bra.add(other.bra, multiplied: noise)
		}
	}
}
