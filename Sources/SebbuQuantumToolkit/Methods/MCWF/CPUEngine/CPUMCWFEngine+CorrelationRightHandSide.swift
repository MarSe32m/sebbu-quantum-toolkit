// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension MCWF.CPUEngine {
    @usableFromInline
	internal struct CorrelationRightHandSide: ~Copyable, ODERHSFunction {
        @usableFromInline
		internal var hamiltonian: _PreparedOperator
        @usableFromInline
		internal var channels: UniqueArray<_PreparedMarkovianChannel>
		@usableFromInline
        internal var scratch: UniqueVector<Complex<Double>>
        @usableFromInline
        internal var guideAction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var ketAction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var braAction: UniqueVector<Complex<Double>>

        @inlinable
		internal init(
			hamiltonian: TimeDependentOperator,
			channels: [MarkovianChannel],
			dimension: Int
		) {
			self.hamiltonian = .init(hamiltonian, dimension: dimension)
			self.channels = _prepareMarkovianChannels(channels, dimension: dimension)
			self.scratch = .zero(dimension)
            self.guideAction = .zero(dimension)
			self.ketAction = .zero(dimension)
			self.braAction = .zero(dimension)
		}

		@inlinable
        internal mutating func evaluate(
			t: Double,
			y: borrowing CorrelationState,
			dy: inout CorrelationState
		) {
			let normSquared = y.guide.normSquared
			precondition(
				normSquared.isFinite && normSquared > .zero,
				"The MCWF guide state must have a positive finite norm"
			)

            hamiltonian.updateChecked(at: t)
            hamiltonian.action.vector(y.guide.components, into: dy.guide.components, scale: -.i, blas: true)
            hamiltonian.action.vector(y.ket.components, into: dy.ket.components, scale: -.i, blas: true)
            hamiltonian.action.vector(y.bra.components, into: dy.bra.components, scale: -.i, blas: true)

			var totalHazardRate = 0.0
			for i in 0..<channels.count {
				let rate = Self.checkedRate(channels[i].rate(t), at: t)
				guard rate > .zero else { continue }

                channels[i].updateChecked(at: t)
                Self.accumulate(rate: rate, lossOperator: channels[i].action,
                    y: y, normSquared: normSquared, guideAction: &guideAction,
                    ketAction: &ketAction, braAction: &braAction, scratch: &scratch,
                    dy: &dy, totalHazardRate: &totalHazardRate)

			}

			let normalizationDrift = Complex(0.5 * totalHazardRate)
			dy.guide.add(y.guide, multiplied: normalizationDrift)
			dy.ket.add(y.ket, multiplied: normalizationDrift)
			dy.bra.add(y.bra, multiplied: normalizationDrift)
			dy.cumulativeHazard = totalHazardRate
		}

        @inlinable
        @inline(always)
		internal static func accumulate(
			rate: Double,
			lossOperator: _CollapseAction,
			y: borrowing CorrelationState,
			normSquared: Double,
			guideAction: inout UniqueVector<Complex<Double>>,
			ketAction: inout UniqueVector<Complex<Double>>,
			braAction: inout UniqueVector<Complex<Double>>,
			scratch: inout UniqueVector<Complex<Double>>,
            dy: inout CorrelationState,
			totalHazardRate: inout Double
		) {
			lossOperator.lossVector(y.guide.components, into: guideAction.components, scratch: scratch.components, blas: true)
			lossOperator.lossVector(y.ket.components, into: ketAction.components, scratch: scratch.components, blas: true)
			lossOperator.lossVector(y.bra.components, into: braAction.components, scratch: scratch.components, blas: true)

			let rawExpectation = y.guide.inner(guideAction).real / normSquared
			precondition(
				rawExpectation.isFinite,
				"An MCWF jump-channel expectation is non-finite"
			)
			let positivityTolerance =
				128 * Double.ulpOfOne * Swift.max(1, abs(rawExpectation))
			precondition(
				rawExpectation >= -positivityTolerance,
				"C^dagger C must be positive semidefinite"
			)
			let expectation = Swift.max(.zero, rawExpectation)
			totalHazardRate += rate * expectation
			precondition(
				totalHazardRate.isFinite,
				"The total MCWF hazard rate is non-finite"
			)

			let coefficient = Complex(-0.5 * rate)
			dy.guide.add(guideAction, multiplied: coefficient)
			dy.ket.add(ketAction, multiplied: coefficient)
			dy.bra.add(braAction, multiplied: coefficient)
		}

        @inlinable
        @inline(always)
		internal static func checkedRate(_ rate: Double, at time: Double) -> Double {
			precondition(
				rate.isFinite && rate >= .zero,
				"An MCWF channel rate must be finite and nonnegative at t = \(time)"
			)
			return rate
		}
	}
}
