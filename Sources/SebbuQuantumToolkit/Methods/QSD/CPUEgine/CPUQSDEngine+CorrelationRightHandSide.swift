// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
    @usableFromInline
	internal struct CorrelationRightHandSide<
		Hamiltonian: HamiltonianFunction
	>: ~Copyable, SDERHSFunction {
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
			precondition(
				dimension > 0,
				"The quantum-system dimension must be positive"
			)

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
			self.productBuffer = .zeros(
				rows: dimension,
				columns: dimension
			)
		}

        @inlinable
		internal mutating func drift(
			t: Double,
			y: borrowing CorrelationState,
			into dy: inout CorrelationState
		) {
			hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
			hamiltonianBuffer.dotBLAS(
				y.guide,
				multiplied: -.i,
				into: &dy.guide
			)
			hamiltonianBuffer.dotBLAS(
				y.ket,
				multiplied: -.i,
				into: &dy.ket
			)
			hamiltonianBuffer.dotBLAS(
				y.bra,
				multiplied: -.i,
				into: &dy.bra
			)

			let normSquared: Double
			switch equationType {
			case .linear:
				normSquared = 1
			case .nonLinear, .nonLinearNormalized:
				normSquared = y.guide.normSquared
				precondition(
					normSquared.isFinite && normSquared > .zero,
					"Non-linear QSD requires a finite, nonzero guide norm"
				)
			}

			for channel in channels {
				switch channel {
				case .constant(let channel):
					Self.accumulateDrift(
						equationType: equationType,
						rate: Self.checkedRate(channel.rate(t)),
						collapseOperator: channel.collapseOperator,
						collapseOperatorAdjoint:
							channel.collapseOperatorAdjoint,
						lossOperator: channel.lossOperator,
						y: y,
						normSquared: normSquared,
						into: &dy
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
						into: &productBuffer
					)
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
			y: borrowing CorrelationState,
			channel channelIndex: Int,
			into dy: inout CorrelationState
		) {
			precondition(
				channels.indices.contains(channelIndex),
				"Invalid QSD noise channel"
			)
			dy.zero()

			let normSquared: Double
			switch equationType {
			case .linear, .nonLinear:
				normSquared = 1
			case .nonLinearNormalized:
				normSquared = y.guide.normSquared
				precondition(
					normSquared.isFinite && normSquared > .zero,
					"Normalized QSD requires a finite, nonzero guide norm"
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
			y: borrowing CorrelationState,
			normSquared: Double,
			into dy: inout CorrelationState
		) {
			guard rate > .zero else { return }

			let lossCoefficient = Complex(-0.5 * rate)
			lossOperator.dotBLAS(
				y.guide,
				multiplied: lossCoefficient,
				addingInto: &dy.guide
			)
			lossOperator.dotBLAS(
				y.ket,
				multiplied: lossCoefficient,
				addingInto: &dy.ket
			)
			lossOperator.dotBLAS(
				y.bra,
				multiplied: lossCoefficient,
				addingInto: &dy.bra
			)

			switch equationType {
			case .linear:
				return

			case .nonLinear, .nonLinearNormalized:
				let expectationAdjoint =
					y.guide.inner(metric: collapseOperatorAdjoint, y.guide)
					/ normSquared
				let shiftCoefficient = rate * expectationAdjoint
				collapseOperator.dotBLAS(
					y.guide,
					multiplied: shiftCoefficient,
					addingInto: &dy.guide
				)
				collapseOperator.dotBLAS(
					y.ket,
					multiplied: shiftCoefficient,
					addingInto: &dy.ket
				)
				collapseOperator.dotBLAS(
					y.bra,
					multiplied: shiftCoefficient,
					addingInto: &dy.bra
				)

				guard case .nonLinearNormalized = equationType else {
					return
				}
				let expectation = expectationAdjoint.conjugate
				let lossExpectation =
					y.guide.inner(metric: lossOperator, y.guide).real
					/ normSquared
				let scalar =
					rate
					* (0.5 * lossExpectation - expectation.lengthSquared)
				dy.guide.add(y.guide, multiplied: scalar)
				dy.ket.add(y.ket, multiplied: scalar)
				dy.bra.add(y.bra, multiplied: scalar)
			}
		}

        @inlinable
		internal static func assignDiffusion(
			equationType: QSD.EquationType,
			rate: Double,
			collapseOperator: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing CorrelationState,
			normSquared: Double,
			into dy: inout CorrelationState
		) {
			guard rate > .zero else { return }
			let coefficient = (0.5 * rate).squareRoot()
			collapseOperator.dotBLAS(
				y.guide,
				multiplied: Complex(coefficient),
				addingInto: &dy.guide
			)
			collapseOperator.dotBLAS(
				y.ket,
				multiplied: Complex(coefficient),
				addingInto: &dy.ket
			)
			collapseOperator.dotBLAS(
				y.bra,
				multiplied: Complex(coefficient),
				addingInto: &dy.bra
			)

			guard equationType == .nonLinearNormalized else { return }
			let expectation =
				y.guide.inner(metric: collapseOperator, y.guide)
				/ normSquared
			let shift = -coefficient * expectation
			dy.guide.add(y.guide, multiplied: shift)
			dy.ket.add(y.ket, multiplied: shift)
			dy.bra.add(y.bra, multiplied: shift)
		}
	}
}
