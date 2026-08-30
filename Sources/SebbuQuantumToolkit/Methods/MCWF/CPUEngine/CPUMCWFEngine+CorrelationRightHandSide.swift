// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
    @usableFromInline
	internal struct CorrelationRightHandSide<
		Hamiltonian: HamiltonianFunction
	>: ~Copyable, ODERHSFunction {
        @usableFromInline
		internal let hamiltonian: Hamiltonian
        @usableFromInline
		internal let channels: [PreparedChannel]

		@usableFromInline
        internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var adjointBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var lossBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var guideAction: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var companionAction: UniqueVector<Complex<Double>>

        @inlinable
		internal init(
			hamiltonian: Hamiltonian,
			channels: [PreparedChannel],
			dimension: Int
		) {
			self.hamiltonian = hamiltonian
			self.channels = channels
			self.hamiltonianBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
			self.collapseBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
			self.adjointBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
			self.lossBuffer = .zeros(rows: dimension, columns: dimension)
			self.guideAction = .zero(dimension)
			self.companionAction = .zero(dimension)
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

			hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
			hamiltonianBuffer.dotBLAS(
				y.guide,
				multiplied: -.i,
				into: &dy.guide
			)
			hamiltonianBuffer.dotBLAS(
				y.companion,
				multiplied: -.i,
				into: &dy.companion
			)

			var totalHazardRate = 0.0
			for channel in channels {
				let rate = Self.checkedRate(channel.rate(t), at: t)
				guard rate > .zero else { continue }

				switch channel {
				case .constant(let channel):
					Self.accumulate(
						rate: rate,
						lossOperator: channel.lossOperator,
						y: y,
						normSquared: normSquared,
						guideAction: &guideAction,
						companionAction: &companionAction,
						dy: &dy,
						totalHazardRate: &totalHazardRate
					)

				case .dynamic(let channel):
					channel.insert(t: t, into: &collapseBuffer)
					for row in 0..<collapseBuffer.rows {
						for column in 0..<collapseBuffer.columns {
							adjointBuffer[column, row] =
								collapseBuffer[row, column]
								.conjugate
						}
					}
					adjointBuffer.dotBLAS(
						collapseBuffer,
						into: &lossBuffer
					)
					Self.accumulate(
						rate: rate,
						lossOperator: lossBuffer,
						y: y,
						normSquared: normSquared,
						guideAction: &guideAction,
						companionAction: &companionAction,
						dy: &dy,
						totalHazardRate: &totalHazardRate
					)
				}
			}

			let normalizationDrift = Complex(0.5 * totalHazardRate)
			dy.guide.add(y.guide, multiplied: normalizationDrift)
			dy.companion.add(y.companion, multiplied: normalizationDrift)
			dy.hazard = totalHazardRate
		}

        @inlinable
        @inline(always)
		internal static func accumulate(
			rate: Double,
			lossOperator: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing CorrelationState,
			normSquared: Double,
			guideAction: inout UniqueVector<Complex<Double>>,
			companionAction: inout UniqueVector<Complex<Double>>,
			dy: inout CorrelationState,
			totalHazardRate: inout Double
		) {
			lossOperator.dotBLAS(y.guide, into: &guideAction)
			lossOperator.dotBLAS(y.companion, into: &companionAction)

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
			dy.companion.add(companionAction, multiplied: coefficient)
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
