// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
	@usableFromInline
	internal struct QSDRightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable,
		SDERHSFunction
	{
		@usableFromInline
		internal let equationType: QSD.EquationType
		@usableFromInline
		internal let hamiltonian: Hamiltonian
		@usableFromInline
		internal let channels: [PreparedChannel]
		@usableFromInline
		internal var randomNumberGenerator: TrajectoryRandomNumberGenerator

		@usableFromInline
		internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var adjointBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var productBuffer: UniqueMatrix<Complex<Double>>

		@inlinable
		internal init(
			_ problem: borrowing PureStateProblem<Hamiltonian>,
			equationType: QSD.EquationType,
			seed: UInt64,
			trajectoryID: UInt64
		) {
			let dimension = problem.system.dimension
			precondition(dimension > 0, "The quantum-system dimension must be positive")

			var channels: [PreparedChannel] = []
			channels.reserveCapacity(problem.markovianChannels.count)
			for channel in problem.markovianChannels {
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

			self.equationType = equationType
			self.hamiltonian = problem.system.hamiltonian
			self.channels = channels
			self.randomNumberGenerator = TrajectoryRandomNumberGenerator(
				seed: seed,
				trajectoryID: trajectoryID,
                purpose: .gaussianWhiteNoise
			)
			self.hamiltonianBuffer = .zeros(rows: dimension, columns: dimension)
			self.collapseBuffer = .zeros(rows: dimension, columns: dimension)
			self.adjointBuffer = .zeros(rows: dimension, columns: dimension)
			self.productBuffer = .zeros(rows: dimension, columns: dimension)
		}

		@inlinable
		internal mutating func drift(
			t: Double,
			y: borrowing StateVector,
			into dy: inout StateVector
		) {
			hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
			hamiltonianBuffer.dotBLAS(y.state, multiplied: -.i, into: &dy.state)

			let normSquared: Double
			switch equationType {
			case .linear:
				normSquared = 1
			case .nonLinear, .nonLinearNormalized:
				normSquared = y.state.normSquared
				precondition(
					normSquared.isFinite && normSquared > .zero,
					"Non-linear QSD requires a finite, nonzero state norm"
				)
			}

			for channel in channels {
				switch channel {
				case .constant(let channel):
					Self.accumulateDrift(
						equationType: equationType,
						rate: Self.checkedRate(channel.rate(t)),
						collapseOperator: channel.collapseOperator,
						collapseOperatorAdjoint: channel
							.collapseOperatorAdjoint,
						lossOperator: channel.lossOperator,
						y: y,
						normSquared: normSquared,
						into: &dy
					)

				case .dynamic(let channel):
					channel.insert(t: t, into: &collapseBuffer)
					for row in 0..<collapseBuffer.rows {
						for column in 0..<collapseBuffer.columns {
							adjointBuffer[
								unchecked: column, unchecked: row] =
								collapseBuffer[
									unchecked: row,
									unchecked: column
								].conjugate
						}
					}
					adjointBuffer.dotBLAS(collapseBuffer, into: &productBuffer)
					Self.accumulateDrift(
						equationType: equationType,
						rate: Self.checkedRate(channel.rate(t)),
						collapseOperator: collapseBuffer,
						collapseOperatorAdjoint: adjointBuffer,
						lossOperator: productBuffer,
						y: y,
						normSquared: normSquared,
						into: &dy
					)
				}
			}
		}

		@inlinable
		internal mutating func diffusion(
			t: Double,
			y: borrowing StateVector,
			channel channelIndex: Int,
			into dy: inout StateVector
		) {
			precondition(
				channels.indices.contains(channelIndex), "Invalid QSD noise channel"
			)
			dy.state.zeroComponents()

			let normSquared: Double
			switch equationType {
			case .linear, .nonLinear:
				normSquared = 1
			case .nonLinearNormalized:
				normSquared = y.state.normSquared
				precondition(
					normSquared.isFinite && normSquared > .zero,
					"Normalized QSD requires a finite, nonzero state norm"
				)
			}

			switch channels[channelIndex] {
			case .constant(let channel):
				Self.assignDiffusion(
					equationType: equationType,
					rate: Self.checkedRate(channel.rate(t)),
					collapseOperator: channel.collapseOperator,
					y: y,
					normSquared: normSquared,
					into: &dy
				)

			case .dynamic(let channel):
				channel.insert(t: t, into: &collapseBuffer)
				Self.assignDiffusion(
					equationType: equationType,
					rate: Self.checkedRate(channel.rate(t)),
					collapseOperator: collapseBuffer,
					y: y,
					normSquared: normSquared,
					into: &dy
				)
			}
		}

		@inlinable
		internal mutating func sampleNormalizedNoises(
			t: Double,
			stepSize: Double,
			into noises: inout MutableSpan<Complex<Double>>
		) {
			precondition(
				noises.count == channels.count,
				"There must be one noise value for each Markovian channel"
			)
			for index in noises.indices {
				let sample: Complex<Double> = randomNumberGenerator.nextNormal()
				noises[unchecked: index] = sample
			}
		}

		@inlinable
		@inline(always)
		internal static func checkedRate(_ rate: Double) -> Double {
			precondition(
				rate.isFinite && rate >= .zero,
				"A Markovian-channel rate must be finite and nonnegative"
			)
			return rate
		}

		@inlinable
		internal static func accumulateDrift(
			equationType: QSD.EquationType,
			rate: Double,
			collapseOperator: borrowing UniqueMatrix<Complex<Double>>,
			collapseOperatorAdjoint: borrowing UniqueMatrix<Complex<Double>>,
			lossOperator: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing StateVector,
			normSquared: Double,
			into dy: inout StateVector
		) {
			guard rate != .zero else { return }

			// The numerical generator samples the real and imaginary parts of
			// each complex Wiener value with unit variance, hence sqrt(rate / 2).
			lossOperator.dotBLAS(
				y.state,
				multiplied: Complex(-0.5 * rate),
				addingInto: &dy.state
			)

			switch equationType {
			case .linear:
				return

			case .nonLinear, .nonLinearNormalized:
				let expectationAdjoint =
					y.state.inner(metric: collapseOperatorAdjoint, y.state)
					/ normSquared
				collapseOperator.dotBLAS(
					y.state,
					multiplied: rate * expectationAdjoint,
					addingInto: &dy.state
				)

				guard case .nonLinearNormalized = equationType else { return }

				let expectation = expectationAdjoint.conjugate
				let lossExpectation =
					y.state.inner(metric: lossOperator, y.state).real
					/ normSquared

				let scalar =
					rate * (0.5 * lossExpectation - expectation.lengthSquared)
				dy.state.add(y.state, multiplied: scalar)
			}
		}

		@inlinable
		internal static func assignDiffusion(
			equationType: QSD.EquationType,
			rate: Double,
			collapseOperator: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing StateVector,
			normSquared: Double,
			into dy: inout StateVector
		) {
			guard rate != .zero else { return }
			let coefficient = (0.5 * rate).squareRoot()
			collapseOperator.dotBLAS(
				y.state,
				multiplied: Complex(coefficient),
				addingInto: &dy.state
			)

			guard equationType == .nonLinearNormalized else { return }
			let expectation =
				y.state.inner(metric: collapseOperator, y.state) / normSquared
			dy.state.add(y.state, multiplied: -coefficient * expectation)
		}
	}
}
