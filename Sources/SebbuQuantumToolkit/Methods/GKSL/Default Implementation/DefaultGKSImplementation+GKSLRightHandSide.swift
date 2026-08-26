// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension DefaultGKSLImplementation {
	@usableFromInline
	internal struct GKSLRightHandSide<Hamiltonian: HamiltonianFunction>:
		~Copyable, ODERHSFunction
	{
		@usableFromInline
		internal let hamiltonian: Hamiltonian
		@usableFromInline
		internal let constantChannels: [_PreparedConstantMarkovianChannel]
		@usableFromInline
		internal let dynamicChannels: [_PreparedDynamicMarkovianChannel]

		@usableFromInline
		internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var collapseBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var adjointBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var productBuffer: UniqueMatrix<Complex<Double>>
		@usableFromInline
		internal var temporaryBuffer: UniqueMatrix<Complex<Double>>

		@inlinable
		internal init(_ problem: borrowing DensityMatrixProblem<Hamiltonian>) {
			let dimension = problem.system.dimension
			precondition(dimension > 0, "The quantum-system dimension must be positive")

			var constantChannels: [_PreparedConstantMarkovianChannel] = []
			var dynamicChannels: [_PreparedDynamicMarkovianChannel] = []
			constantChannels.reserveCapacity(problem.markovianChannels.count)
			dynamicChannels.reserveCapacity(problem.markovianChannels.count)

			for channel in problem.markovianChannels {
				if let prepared = _PreparedConstantMarkovianChannel(
					channel,
					dimension: dimension
				) {
					constantChannels.append(prepared)
				} else {
					dynamicChannels.append(
						_PreparedDynamicMarkovianChannel(
							channel,
							dimension: dimension
						)
					)
				}
			}

			self.hamiltonian = problem.system.hamiltonian
			self.constantChannels = constantChannels
			self.dynamicChannels = dynamicChannels
			self.hamiltonianBuffer = .zeros(rows: dimension, columns: dimension)
			self.collapseBuffer = .zeros(rows: dimension, columns: dimension)
			self.adjointBuffer = .zeros(rows: dimension, columns: dimension)
			self.productBuffer = .zeros(rows: dimension, columns: dimension)
			self.temporaryBuffer = .zeros(rows: dimension, columns: dimension)
		}

		@inlinable
		internal mutating func evaluate(
			t: Double,
			y: borrowing DensityMatrixState,
			dy: inout DensityMatrixState
		) {
			// Copying these copyable descriptors locally prevents accesses to
			// channel storage from overlapping the inout accesses to scratch
			// matrices in `self`. Their arrays and matrices retain COW storage.
			let hamiltonian = self.hamiltonian
			let constantChannels = self.constantChannels
			let dynamicChannels = self.dynamicChannels

			// -i[H(t), rho]
			hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
			hamiltonianBuffer.dotBLAS(
				y.densityMatrix,
				multiplied: -.i,
				into: &dy.densityMatrix
			)
			y.densityMatrix.dotBLAS(
				hamiltonianBuffer,
				multiplied: .i,
				addingInto: &dy.densityMatrix
			)

			// gamma * (C rho C^dagger - 1/2 {C^dagger C, rho})
			for channel in constantChannels {
				let rate = channel.rate(t)
				Self.validate(rate: rate)
				if rate == .zero { continue }

				channel.collapseOperator.dotBLAS(
					y.densityMatrix,
					into: &productBuffer
				)
				productBuffer.dotBLAS(
					channel.collapseOperatorAdjoint,
					multiplied: Complex(rate),
					addingInto: &dy.densityMatrix
				)

				let lossScale = Complex(-0.5 * rate)
				channel.lossOperator.dotBLAS(
					y.densityMatrix,
					multiplied: lossScale,
					addingInto: &dy.densityMatrix
				)
				y.densityMatrix.dotBLAS(
					channel.lossOperator,
					multiplied: lossScale,
					addingInto: &dy.densityMatrix
				)
			}

			for channel in dynamicChannels {
				let rate = channel.rate(t)
				Self.validate(rate: rate)
				if rate == .zero { continue }

				channel.insert(t: t, into: &collapseBuffer)
				for row in 0..<collapseBuffer.rows {
					for column in 0..<collapseBuffer.columns {
						adjointBuffer[column, row] =
							collapseBuffer[row, column].conjugate
					}
				}

				adjointBuffer.dotBLAS(
					collapseBuffer,
					into: &productBuffer
				)
				collapseBuffer.dotBLAS(
					y.densityMatrix,
					into: &temporaryBuffer
				)
				temporaryBuffer.dotBLAS(
					adjointBuffer,
					multiplied: Complex(rate),
					addingInto: &dy.densityMatrix
				)

				let lossScale = Complex(-0.5 * rate)
				productBuffer.dotBLAS(
					y.densityMatrix,
					multiplied: lossScale,
					addingInto: &dy.densityMatrix
				)
				y.densityMatrix.dotBLAS(
					productBuffer,
					multiplied: lossScale,
					addingInto: &dy.densityMatrix
				)
			}
		}

		@inlinable
		@inline(always)
		internal static func validate(rate: Double) {
			precondition(
				rate.isFinite && rate >= .zero,
				"A GKSL channel rate must be finite and nonnegative"
			)
		}
	}
}
