// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

extension CPUMCWFEngine {
    @usableFromInline
	internal struct CorrelationJumpWorkspace: ~Copyable, PDPJumpKernel {
		@usableFromInline
		internal typealias State = CorrelationState
		@usableFromInline
		internal typealias Failure = SolverError

		@usableFromInline
        internal let channels: [PreparedChannel]
		@usableFromInline
        internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var actionBuffer: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var weights: [Double]

        @inlinable
		internal init(channels: [PreparedChannel], dimension: Int) {
			self.channels = channels
			self.collapseBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
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
			for index in channels.indices {
				let channel = channels[index]
				let rate = Self.checkedRate(channel.rate(time), at: time)
				guard rate > .zero else {
					weights[index] = .zero
					continue
				}
                    
                switch channel {
                case .constant(let channel):
                    channel.collapseOperator.dotBLAS(state.guide, into: &actionBuffer)

                case .dynamic(let channel):
                    channel.insert(t: time, into: &collapseBuffer)
                    collapseBuffer.dotBLAS(state.guide, into: &actionBuffer)
                }
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
			for index in channels.indices where weights[index] > .zero {
				cumulativeWeight += weights[index]
				selectedIndex = index
				if selection < cumulativeWeight { break }
			}
			guard let selectedIndex else {
				throw SolverError.noAvailableJump(time: time)
			}

            // Perform jump
			let selectedChannel = channels[selectedIndex]
            switch selectedChannel {
            case .constant(let channel):
                // Perform jump on the guide and record its norm
                channel.collapseOperator.dotBLAS(state.guide, into: &actionBuffer)
                let normSquared = actionBuffer.normSquared
                guard normSquared.isFinite && normSquared > .zero else {
                    throw SolverError.noAvailableJump(time: time)
                }
                let inverseNorm = 1 / normSquared.squareRoot()
                state.guide.copyComponents(from: actionBuffer, multiplied: inverseNorm)
                // Apply jump to ket and normalize according to guide
                channel.collapseOperator.dotBLAS(state.ket, into: &actionBuffer)
                state.ket.copyComponents(from: actionBuffer, multiplied: inverseNorm)
                channel.collapseOperator.dotBLAS(state.bra, into: &actionBuffer)
                state.bra.copyComponents(from: actionBuffer, multiplied: inverseNorm)

            case .dynamic(let channel):
                // Perform jump on the guide and record its norm
                channel.insert(t: time, into: &collapseBuffer)
                collapseBuffer.dotBLAS(state.guide, into: &actionBuffer)
                let normSquared = actionBuffer.normSquared
                guard normSquared.isFinite && normSquared > .zero else {
                    throw SolverError.noAvailableJump(time: time)
                }
                let inverseNorm = 1 / normSquared.squareRoot()
                state.guide.copyComponents(from: actionBuffer, multiplied: inverseNorm)
                // Apply jump to ket and normalize according to guide
                collapseBuffer.dotBLAS(state.ket, into: &actionBuffer)
                state.ket.copyComponents(from: actionBuffer, multiplied: inverseNorm)
                // Apply jump to bra and normalize according to guide
                collapseBuffer.dotBLAS(state.bra, into: &actionBuffer)
                state.bra.copyComponents(from: actionBuffer, multiplied: inverseNorm)
            }
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
