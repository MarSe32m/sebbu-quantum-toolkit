// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension MCWF.CPUEngine {
    @usableFromInline
	internal struct RightHandSide: ~Copyable, ODERHSFunction {
        @usableFromInline
		internal var hamiltonian: _PreparedOperator
        @usableFromInline
		internal var channels: UniqueArray<_PreparedMarkovianChannel>
		@usableFromInline
        internal var scratch: UniqueVector<Complex<Double>>
        @usableFromInline
        internal var operatorAction: UniqueVector<Complex<Double>>

        @inlinable
		internal init(
			hamiltonian: TimeDependentOperator,
			channels: [MarkovianChannel],
			dimension: Int
		) {
			self.hamiltonian = .init(hamiltonian, dimension: dimension)
			self.channels = _prepareMarkovianChannels(channels, dimension: dimension)
			self.scratch = .zero(dimension)
            self.operatorAction = .zero(dimension)
		}

        @inlinable
		internal mutating func evaluate(
			t: Double,
			y: borrowing TrajectoryState,
			dy: inout TrajectoryState
		) {
			let normSquared = y.wavefunction.normSquared
			precondition(
				normSquared.isFinite && normSquared > .zero,
				"The conditional MCWF state must have a positive finite norm"
			)

            hamiltonian.updateChecked(at: t)
            hamiltonian.action.vector(y.wavefunction.components, into: dy.wavefunction.components, scale: -.i, blas: true)

			var totalHazardRate = 0.0
			for i in 0..<channels.count {
				let rate = Self.checkedRate(channels[i].rate(t), at: t)
				guard rate > .zero else { continue }

                channels[i].updateChecked(at: t)
                channels[i].action.lossVector(y.wavefunction.components,
                    into: operatorAction.components, scratch: scratch.components, blas: true)

				let rawExpectation =
					y.wavefunction.inner(operatorAction).real / normSquared
				precondition(
					rawExpectation.isFinite,
					"An MCWF jump-channel expectation is non-finite at t = \(t)"
				)
				let positivityTolerance =
					128 * Double.ulpOfOne
					* Swift.max(1, abs(rawExpectation))
				precondition(
					rawExpectation >= -positivityTolerance,
					"C^dagger C must be positive semidefinite at t = \(t)"
				)
				let expectation = Swift.max(.zero, rawExpectation)
				totalHazardRate += rate * expectation
				precondition(
					totalHazardRate.isFinite,
					"The total MCWF hazard rate is non-finite at t = \(t)"
				)

				dy.wavefunction.add(
					operatorAction,
					multiplied: Complex(-0.5 * rate)
				)
			}

			dy.wavefunction.add(
				y.wavefunction,
				multiplied: Complex(0.5 * totalHazardRate)
			)
			dy.cumulativeHazard = totalHazardRate
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
