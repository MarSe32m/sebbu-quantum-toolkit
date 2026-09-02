// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
    @usableFromInline
	internal struct JumpWorkspace: ~Copyable, PDPJumpKernel {
		@usableFromInline
		internal typealias State = TrajectoryState
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
			self.collapseBuffer = .zeros(rows: dimension, columns: dimension)
			self.actionBuffer = .zero(dimension)
			self.weights = [Double](repeating: .zero, count: channels.count)
		}

        @inlinable
		internal mutating func applyJump<RNG: RandomNumberGenerator>(
			at time: Double,
			state: inout TrajectoryState,
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

				apply(
					channel,
					at: time,
					to: state.wavefunction
				)
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

			apply(
				channels[selectedIndex],
				at: time,
				to: state.wavefunction
			)
			state.wavefunction.copyComponents(from: actionBuffer)
			try CPUMCWFEngine.normalize(&state, at: time)
		}

		@inlinable
		internal mutating func sampleAndApplyJump<
			RNG: RandomNumberGenerator
		>(
			at time: Double,
			to state: inout TrajectoryState,
			using rng: inout RNG
		) throws(SolverError) {
			try applyJump(at: time, state: &state, using: &rng)
		}

        @inlinable
        @inline(__always)
        //@inline(always) TODO: Crashed. Report the bug to swiftlang when you have a reproducer...
		internal mutating func apply(
			_ channel: PreparedChannel,
			at time: Double,
			to state: borrowing UniqueVector<Complex<Double>>
		) {
			switch channel {
			case .constant(let channel):
				channel.collapseOperator.dotBLAS(state, into: &actionBuffer)

			case .dynamic(let channel):
				channel.insert(t: time, into: &collapseBuffer)
				collapseBuffer.dotBLAS(state, into: &actionBuffer)
			}
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
