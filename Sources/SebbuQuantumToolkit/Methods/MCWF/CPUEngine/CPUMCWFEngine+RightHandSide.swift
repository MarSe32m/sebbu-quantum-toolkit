// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	internal struct RightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable,
		ODERHSFunction
	{
		internal let hamiltonian: Hamiltonian
		internal let channels: [PreparedChannel]

		internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
		internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		internal var adjointBuffer: UniqueMatrix<Complex<Double>>
		internal var lossBuffer: UniqueMatrix<Complex<Double>>
		internal var operatorAction: UniqueVector<Complex<Double>>

		internal init(
			hamiltonian: Hamiltonian,
			channels: [PreparedChannel],
			dimension: Int
		) {
			self.hamiltonian = hamiltonian
			self.channels = channels
			self.hamiltonianBuffer = .zeros(rows: dimension, columns: dimension)
			self.collapseBuffer = .zeros(rows: dimension, columns: dimension)
			self.adjointBuffer = .zeros(rows: dimension, columns: dimension)
			self.lossBuffer = .zeros(rows: dimension, columns: dimension)
			self.operatorAction = .zero(dimension)
		}

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

			hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
			hamiltonianBuffer.dotBLAS(
				y.wavefunction,
				multiplied: -.i,
				into: &dy.wavefunction
			)

			var totalHazardRate = 0.0
			for channel in channels {
				let rate = Self.checkedRate(channel.rate(t), at: t)
				guard rate > .zero else { continue }

				switch channel {
				case .constant(let channel):
					channel.lossOperator.dotBLAS(
						y.wavefunction,
						into: &operatorAction
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
					adjointBuffer.dotBLAS(collapseBuffer, into: &lossBuffer)
					lossBuffer.dotBLAS(
						y.wavefunction,
						into: &operatorAction
					)
				}

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
			dy.hazard = totalHazardRate
		}

		@inline(__always)
		internal static func checkedRate(_ rate: Double, at time: Double) -> Double {
			precondition(
				rate.isFinite && rate >= .zero,
				"An MCWF channel rate must be finite and nonnegative at t = \(time)"
			)
			return rate
		}
	}

	internal static func prepareChannels(
		_ markovianChannels: [MarkovianChannel],
		dimension: Int
	) -> [PreparedChannel] {
		var channels: [PreparedChannel] = []
		channels.reserveCapacity(markovianChannels.count)
		for channel in markovianChannels {
			if let prepared = _PreparedConstantMarkovianChannel(
				channel,
				dimension: dimension
			) {
				channels.append(.constant(prepared))
			} else {
				channels.append(
					.dynamic(
						_PreparedDynamicMarkovianChannel(
							channel,
							dimension: dimension
						)
					)
				)
			}
		}
		return channels
	}
}
