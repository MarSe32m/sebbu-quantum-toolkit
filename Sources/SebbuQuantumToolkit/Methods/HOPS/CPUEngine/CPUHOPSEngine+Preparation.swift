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
		struct BathChannel: Sendable {
			@usableFromInline
            let physicalIndex: Int
			@usableFromInline
            let op: PreparedOperator
			@usableFromInline
            let directions: [Direction]
            
            @inlinable
            init(physicalIndex: Int, op: PreparedOperator, directions: [Direction]) {
                self.physicalIndex = physicalIndex
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
        let bathChannels: [BathChannel]
		@usableFromInline
        let markovianOperators: [PreparedOperator]
		@usableFromInline
        let rates: [ScalarTimeFunction]
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
			var bathChannels: [BathChannel] = []
			for i in 0..<model.channelCount {
				let source = configuration.hierarchy.environment.couplingOperators[
					i]
				let op = try PreparedOperator(
					source, dimension: dimension, needsLoss: false)
				var directions: [Direction] = []
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
					bathChannels.append(
						.init(
							physicalIndex: i, op: op,
							directions: directions))
				}
			}
			self.bathChannels = bathChannels
			self.markovianOperators = try problem.markovianChannels.map {
				try PreparedOperator(
					$0.collapseOperator, dimension: dimension, needsLoss: true)
			}
			self.rates = problem.markovianChannels.map(\.rate)
		}
	}

    @usableFromInline
	internal struct PreparedOperator: Sendable {
		@usableFromInline
        let source: TimeDependentOperator
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
			self.source = source
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

	/// States occupy rows, so applying O to every ket is the GEMM Y O^T.
	@usableFromInline
    internal final class OperatorMatrices: Sendable {
		@usableFromInline
        let transpose: UniqueMatrix<Complex<Double>>
		@usableFromInline
        let adjointTranspose: UniqueMatrix<Complex<Double>>
		@usableFromInline
        let lossTranspose: UniqueMatrix<Complex<Double>>

        @inlinable
		init(_ original: borrowing UniqueMatrix<Complex<Double>>, needsLoss: Bool) {
			let n = original.rows
			var transpose = UniqueMatrix<Complex<Double>>.zeros(rows: n, columns: n)
			var adjointTranspose = UniqueMatrix<Complex<Double>>.zeros(
				rows: n, columns: n)
			Self.transpose(original, into: &transpose, adjointInto: &adjointTranspose)
			var loss = UniqueMatrix<Complex<Double>>.zeros(
				rows: needsLoss ? n : 1, columns: needsLoss ? n : 1)
			if needsLoss { transpose.dotBLAS(adjointTranspose, into: &loss) }
			self.transpose = transpose
			self.adjointTranspose = adjointTranspose
			self.lossTranspose = loss
		}

        @inlinable
		static func transpose(
			_ original: borrowing UniqueMatrix<Complex<Double>>,
			into transpose: inout UniqueMatrix<Complex<Double>>,
			adjointInto adjoint: inout UniqueMatrix<Complex<Double>>
		) {
			precondition(
				original.rows == transpose.rows
					&& original.columns == transpose.columns,
				"Generated operator dimensions do not match the system.")
			for i in 0..<original.rows {
				for j in 0..<original.columns {
					transpose[unchecked: j, unchecked: i] =
						original[unchecked: i, unchecked: j]
					adjoint[unchecked: i, unchecked: j] =
						original[unchecked: i, unchecked: j].conjugate
				}
			}
		}
	}
}
