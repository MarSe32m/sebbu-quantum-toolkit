// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

extension MCWF.CPUEngine {
    @usableFromInline
	internal struct CorrelationJumpWorkspace: ~Copyable, PDPJumpKernel {
		@usableFromInline
		internal typealias State = CorrelationState
		@usableFromInline
		internal typealias Failure = SolverError

		@usableFromInline
        internal var channels: UniqueArray<_PreparedMarkovianChannel>
		@usableFromInline
		internal var actionBuffer: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var weights: [Double]

        @inlinable
		internal init(channels: [MarkovianChannel], dimension: Int) {
			self.channels = _prepareMarkovianChannels(channels, dimension: dimension)
			self.actionBuffer = .zero(dimension)
			self.weights = [Double](repeating: .zero, count: channels.count)
		}

        @inlinable
		internal mutating func applyJump<RNG: RandomNumberGenerator>(
			at time: Double,
			state: inout CorrelationState,
			using randomNumberGenerator: inout RNG
		) throws(SolverError) {
			var totalWeight = 0.0
			for index in 0..<channels.count {

				let rate = Self.checkedRate(channels[index].rate(time), at: time)
				guard rate > .zero else {
					weights[index] = .zero
					continue
				}

                channels[index].updateChecked(at: time)
                channels[index].action.op.vector(state.guide.components, into: actionBuffer.components, blas: true)

				let weight = rate * actionBuffer.normSquared
				guard weight.isFinite && weight >= .zero else {
					throw SolverError.invalidJumpWeight(
						time: time,
						channel: index
					)
				}
				weights[index] = weight
				totalWeight += weight
			}

			guard totalWeight.isFinite && totalWeight > .zero else {
				throw SolverError.noAvailableJump(time: time)
			}

			let selection = totalWeight * randomNumberGenerator.nextUnitDouble()
			var cumulativeWeight = 0.0
			var selectedIndex: Int?
			for index in 0..<channels.count where weights[index] > .zero {
				cumulativeWeight += weights[index]
				selectedIndex = index
				if selection < cumulativeWeight { break }
			}
			guard let selectedIndex else {
				throw SolverError.noAvailableJump(time: time)
			}

            // Reuse the stage used for weights, including all companions.
            let op = channels[selectedIndex].action.op
            op.vector(state.guide.components, into: actionBuffer.components, blas: true)
            let normSquared = actionBuffer.normSquared
            guard normSquared.isFinite && normSquared > 0 else {
                throw SolverError.noAvailableJump(time: time)
            }
            let inverseNorm = 1 / normSquared.squareRoot()
            state.guide.copyComponents(from: actionBuffer, multiplied: inverseNorm)
            op.vector(state.ket.components, into: actionBuffer.components, blas: true)
            state.ket.copyComponents(from: actionBuffer, multiplied: inverseNorm)
            op.vector(state.bra.components, into: actionBuffer.components, blas: true)
            state.bra.copyComponents(from: actionBuffer, multiplied: inverseNorm)

		}

		@inlinable
		internal mutating func sampleAndApplyJump<RNG: RandomNumberGenerator>(
            at time: Double,
			to state: inout CorrelationState,
			using rng: inout RNG
		) throws(SolverError) {
			try applyJump(at: time, state: &state, using: &rng)
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
