// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

extension CPUMCWFEngine {
    @usableFromInline
	internal struct CorrelationJumpWorkspace: ~Copyable {
		@usableFromInline
        internal let channels: [PreparedChannel]
		@usableFromInline
        internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var guideAction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var ketAction: UniqueVector<Complex<Double>>
		@usableFromInline
		internal var braAction: UniqueVector<Complex<Double>>
		@usableFromInline
        internal var weights: [Double]

        @inlinable
		internal init(channels: [PreparedChannel], dimension: Int) {
			self.channels = channels
			self.collapseBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
			self.guideAction = .zero(dimension)
			self.ketAction = .zero(dimension)
			self.braAction = .zero(dimension)
			self.weights = [Double](repeating: .zero, count: channels.count)
		}

        @inlinable
		internal mutating func applyJump<RNG: RandomNumberGenerator>(
			at time: Double,
			state: inout CorrelationState,
			using randomNumberGenerator: inout RNG
		) throws {
			var totalWeight = 0.0
			for index in channels.indices {
				let channel = channels[index]
				let rate = Self.checkedRate(channel.rate(time), at: time)
				guard rate > .zero else {
					weights[index] = .zero
					continue
				}

				applyToGuide(channel, at: time, state: state.guide)
				let weight = rate * guideAction.normSquared
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

			let selectedChannel = channels[selectedIndex]
			applyToGuide(
				selectedChannel,
				at: time,
				state: state.guide
			)
			applyToKet(
				selectedChannel,
				at: time,
				state: state.ket
			)
			applyToBra(
				selectedChannel,
				at: time,
				state: state.bra
			)
			let normSquared = guideAction.normSquared
			guard normSquared.isFinite && normSquared > .zero else {
				throw SolverError.noAvailableJump(time: time)
			}
			let inverseNorm = 1 / normSquared.squareRoot()
			state.guide.copyComponents(
				from: guideAction,
				multiplied: inverseNorm
			)
			state.ket.copyComponents(
				from: ketAction,
				multiplied: inverseNorm
			)
			state.bra.copyComponents(
				from: braAction,
				multiplied: inverseNorm
			)
		}

        @inlinable
        @inline(always)
		internal mutating func applyToGuide(
			_ channel: PreparedChannel,
			at time: Double,
			state: borrowing UniqueVector<Complex<Double>>
		) {
			switch channel {
			case .constant(let channel):
				channel.collapseOperator.dotBLAS(state, into: &guideAction)

			case .dynamic(let channel):
				channel.insert(t: time, into: &collapseBuffer)
				collapseBuffer.dotBLAS(state, into: &guideAction)
			}
		}

        @inlinable
        @inline(always)
		internal mutating func applyToKet(
			_ channel: PreparedChannel,
			at time: Double,
			state: borrowing UniqueVector<Complex<Double>>
		) {
			switch channel {
			case .constant(let channel):
				channel.collapseOperator.dotBLAS(
					state,
					into: &ketAction
				)

			case .dynamic(let channel):
				channel.insert(t: time, into: &collapseBuffer)
				collapseBuffer.dotBLAS(state, into: &ketAction)
			}
		}

		@inlinable
		@inline(always)
		internal mutating func applyToBra(
			_ channel: PreparedChannel,
			at time: Double,
			state: borrowing UniqueVector<Complex<Double>>
		) {
			switch channel {
			case .constant(let channel):
				channel.collapseOperator.dotBLAS(state, into: &braAction)
			case .dynamic(let channel):
				channel.insert(t: time, into: &collapseBuffer)
				collapseBuffer.dotBLAS(state, into: &braAction)
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
