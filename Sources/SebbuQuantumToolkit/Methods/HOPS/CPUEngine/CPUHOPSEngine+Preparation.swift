// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// All expensive, trajectory-independent preparation is shared by an ensemble.
	@usableFromInline
	internal final class Preparation: Sendable {
		@usableFromInline
		struct Direction: Sendable {
			@usableFromInline
			let index: Int
			@usableFromInline
			let upward: Complex<Double>  // R_ip
			@usableFromInline
			let downward: Complex<Double>  // sum_q K_pq conj(R_iq)

			@inlinable
			init(index: Int, upward: Complex<Double>, downward: Complex<Double>) {
				self.index = index
				self.upward = upward
				self.downward = downward
			}
		}

		@usableFromInline
		struct BathChannel: ~Copyable, Sendable {
			@usableFromInline
			let physicalIndex: Int
			/// Index into trajectory-local storage for a time-dependent operator.
			/// Constant operators use -1 and remain owned by Preparation.
			@usableFromInline
			let dynamicMatrixIndex: Int
			@usableFromInline
			let op: PreparedOperator
			@usableFromInline
			let directions: UniqueArray<Direction>

			@inlinable
			init(
				physicalIndex: Int,
				dynamicMatrixIndex: Int,
				op: consuming PreparedOperator,
				directions: consuming UniqueArray<Direction>
			) {
				precondition(dynamicMatrixIndex >= -1)
				self.physicalIndex = physicalIndex
				self.dynamicMatrixIndex = dynamicMatrixIndex
				self.op = op
				self.directions = directions
			}
		}

		@usableFromInline
		let configuration: HOPS.Configuration
		@usableFromInline
		let dimension: Int
		@usableFromInline
		let poles: UniqueVector<Complex<Double>>
		@usableFromInline
		let shiftCount: Int
		@usableFromInline
		let bathChannels: UniqueArray<BathChannel>
		/// Number of physical-bath matrices that each trajectory must own.
		/// Constant coupling operators remain shared by Preparation.
		@usableFromInline
		let dynamicBathMatrixCount: Int
		@usableFromInline
		let connections: BathConnections
		@usableFromInline
		let markovianOperators: UniqueArray<PreparedOperator>
		@usableFromInline
		let rates: UniqueArray<PreparedTimeFunction<Double>>
		@usableFromInline
		let noise: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator

		@inlinable
		init<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegrationOptions>
		) throws {
			guard configuration.unravelling == .diffusive else {
				throw SolverError.unsupportedUnravelling
			}
			let dimension = problem.system.dimension
			let model = configuration.hierarchy.environment.bath
			let step =
				configuration.noiseStepSize
				?? propagation.integration.maximumStepSize
			precondition(
				step.isFinite && step > 0,
				"The OU mesh step must be finite and positive.")
			self.configuration = configuration
			self.dimension = dimension
			self.poles = .init(model.latentBaths.flatMap(\.poles))
			self.shiftCount =
				configuration.equationType != .linear
					|| configuration.shiftType == .meanField
				? model.poleCount : 0
			self.noise = .init(
				model: model,
				windowDuration: propagation.integration.maximumStepSize,
				start: propagation.timeSpan.start, step: step)

			// Correlations are contracted once, without factoring one-sided
			// exponential residues or changing the sampler's latent basis.
			let coefficients = _LatentBathCoefficients(model)
			var bathChannels = UniqueArray<BathChannel>(
				minimumCapacity: model.channelCount)
			var dynamicBathMatrixCount = 0
			for i in 0..<model.channelCount {
				let source =
					configuration.hierarchy.environment.couplingOperators[i]
				let op = try PreparedOperator(
					source, dimension: dimension, needsLoss: false)
				var directions = UniqueArray<Direction>()
				for p in coefficients.poles.indices {
					let up = coefficients.upward[i, p]
					let down = coefficients.downward[i, p]
					if up != .zero || down != .zero {
						directions.append(.init(index: p, upward: up, downward: down))
					}
				}
				if !directions.isEmpty {
					let dynamicMatrixIndex: Int
					if source.isConstant {
						dynamicMatrixIndex = -1
					} else {
						dynamicMatrixIndex = dynamicBathMatrixCount
						dynamicBathMatrixCount += 1
					}
					bathChannels.append(
						.init(
							physicalIndex: i,
							dynamicMatrixIndex: dynamicMatrixIndex,
							op: op,
							directions: directions))
				}
			}

			self.dynamicBathMatrixCount = dynamicBathMatrixCount
			self.connections = BathConnections(
				hierarchy: configuration.hierarchy,
				dimension: dimension,
				channels: bathChannels)
			self.bathChannels = bathChannels

			var markovianOperators = UniqueArray<PreparedOperator>(
				minimumCapacity: problem.markovianChannels.count)
			for channel in problem.markovianChannels {
				markovianOperators.append(
					try PreparedOperator(
						channel.collapseOperator, dimension: dimension,
						needsLoss: true))
			}
			self.markovianOperators = markovianOperators
			var rates = UniqueArray<PreparedTimeFunction<Double>>(
				minimumCapacity: problem.markovianChannels.count)
			for channel in problem.markovianChannels {
				rates.append(PreparedTimeFunction(channel.rate))
			}
			self.rates = rates
		}
	}

	@usableFromInline
	internal struct PreparedOperator: ~Copyable, Sendable {
		@usableFromInline
		let source: PreparedSource
		@usableFromInline
		let constant: OperatorMatrices?

		@inlinable
		init(
			_ source: TimeDependentOperator,
			dimension: Int,
			needsLoss: Bool
		) throws {
			switch source {
			case .constant(let op):
				guard op.matrix.rows == dimension && op.matrix.columns == dimension
				else {
					throw SolverError.operatorDimensionMismatch
				}
			case .linearCombination(let expansion):
				for op in expansion.operators {
					guard
						op.matrix.rows == dimension
							&& op.matrix.columns == dimension
					else {
						throw SolverError.operatorDimensionMismatch
					}
				}
			case .generatedDense:
				break
			}
			self.source = PreparedSource(source)
			if source.isConstant {
				var original = UniqueMatrix<Complex<Double>>.zeros(
					rows: dimension, columns: dimension)
				source.insert(t: 0, into: &original)
				self.constant = OperatorMatrices(
					original, needsLoss: needsLoss)
			} else {
				self.constant = nil
			}
		}
	}

	/// Immutable operators in their original orientation. The optional loss
	/// matrix is prepared once for constant Markovian channels.
	@usableFromInline
	internal struct OperatorMatrices: ~Copyable, Sendable {
		@usableFromInline
		let matrix: UniqueMatrix<Complex<Double>>
		@usableFromInline
		let loss: UniqueMatrix<Complex<Double>>

		@inlinable
		init(
			_ original: borrowing UniqueMatrix<Complex<Double>>,
			needsLoss: Bool
		) {
			let n = original.rows
			var loss = UniqueMatrix<Complex<Double>>.zeros(
				rows: needsLoss ? n : 1,
				columns: needsLoss ? n : 1)
			if needsLoss {
				OperatorApplication.loss(original, into: &loss)
			}
			self.matrix = .init(copying: original)
			self.loss = loss
		}
	}
}
