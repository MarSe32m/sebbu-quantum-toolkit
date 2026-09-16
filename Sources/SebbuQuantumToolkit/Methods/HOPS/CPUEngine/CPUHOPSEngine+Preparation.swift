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
			@usableFromInline
			let op: PreparedOperator
			@usableFromInline
			let directions: UniqueArray<Direction>
			@usableFromInline
			let connections: BathConnections

			@inlinable
			init(
				physicalIndex: Int, op: consuming PreparedOperator,
				directions: consuming UniqueArray<Direction>,
				connections: consuming BathConnections
			) {
				self.physicalIndex = physicalIndex
				self.op = op
				self.directions = directions
				self.connections = connections
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
			let covariances = model.latentBaths.map(\.stationaryCovariance)
			var bathChannels = UniqueArray<BathChannel>(
				minimumCapacity: model.channelCount)
			for i in 0..<model.channelCount {
				let source = configuration.hierarchy.environment.couplingOperators[
					i]
				let op = try PreparedOperator(
					source, dimension: dimension, needsLoss: false)
				var directions = UniqueArray<Direction>()
				var offset = 0
				for a in model.latentBaths.indices {
					let bath = model.latentBaths[a]
					for p in 0..<bath.poleCount {
						var down = Complex<Double>.zero
						for q in 0..<bath.poleCount {
							down +=
								covariances[a][p, q]
								* bath.residues[i, q].conjugate
						}
						let up = bath.residues[i, p]
						if up != .zero || down != .zero {
							directions.append(
								.init(
									index: offset + p,
									upward: up, downward: down))
						}
					}
					offset += bath.poleCount
				}
				if !directions.isEmpty {
					let connections = BathConnections(
						hierarchy: configuration.hierarchy,
						dimension: dimension, directions: directions)
					bathChannels.append(
						.init(
							physicalIndex: i, op: op,
							directions: directions,
							connections: connections))
				}
			}
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
		init(_ source: TimeDependentOperator, dimension: Int, needsLoss: Bool) throws {
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
			case .generatedDense: break
			}
			self.source = PreparedSource(source)
			if source.isConstant {
				var original = UniqueMatrix<Complex<Double>>.zeros(
					rows: dimension, columns: dimension)
				source.insert(t: 0, into: &original)
				self.constant = OperatorMatrices(original, needsLoss: needsLoss)
			} else {
				self.constant = nil
			}
		}
	}

	/// Immutable operators in their original orientation. The optional loss
	/// matrix is prepared once for constant Markovian channels.
	@usableFromInline
	internal struct OperatorMatrices: ~Copyable, Sendable {
		@usableFromInline let matrix: UniqueMatrix<Complex<Double>>
		@usableFromInline let loss: UniqueMatrix<Complex<Double>>

		@inlinable
		init(_ original: borrowing UniqueMatrix<Complex<Double>>, needsLoss: Bool) {
			let n = original.rows
			var loss = UniqueMatrix<Complex<Double>>.zeros(
				rows: needsLoss ? n : 1, columns: needsLoss ? n : 1)
			if needsLoss { OperatorApplication.loss(original, into: &loss) }
			self.matrix = .init(copying: original)
			self.loss = loss
		}
	}
}
