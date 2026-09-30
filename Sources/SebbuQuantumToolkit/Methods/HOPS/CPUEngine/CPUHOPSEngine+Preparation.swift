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
			let op: PreparedBathOperator
			@usableFromInline
			let directions: UniqueArray<Direction>

			@inlinable
			init(
				physicalIndex: Int,
				op: consuming PreparedBathOperator,
				directions: consuming UniqueArray<Direction>
			) {
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
		let bathChannels: UniqueArray<BathChannel>
		/// Number of physical-bath matrices that each trajectory must own.
		/// Constant coupling operators remain shared by Preparation.
		@usableFromInline
		let dynamicBathMatrixCount: Int
		@usableFromInline
		let connections: BathConnections
        @usableFromInline
        let markovianChannels: [MarkovianChannel]
		@usableFromInline
        let noiseGenerator: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator?
        @usableFromInline
        let noiseDefinition: HOPS.BathNoiseDefinition

        @inlinable
        var noise: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator {
            // Only seeded entry points prepare the standard OU factory.
            noiseGenerator!
        }

        @inlinable
        func makeBathNoise(
            seed: UInt64, trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent
        ) -> HOPS.BathNoiseSampler {
            .init(path: .init(definition: noiseDefinition, masterSeed: seed,
                             trajectoryID: trajectoryID, ensembleSampling: ensembleSampling),
                  preparedGenerator: noise)
        }

		@inlinable
		init(
			problem: borrowing PureStateProblem,
			configuration: HOPS.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            preparingStandardNoise: Bool = true
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
            self.noiseDefinition = .init(model: model, timeSpan: propagation.timeSpan, stepSize: step)
            self.noiseGenerator = preparingStandardNoise ? .init(
                model: model,
                windowDuration: propagation.integration.maximumStepSize,
                start: propagation.timeSpan.start, step: step) : nil

			// Correlations are contracted once, without factoring one-sided
			// exponential residues or changing the sampler's latent basis.
			let coefficients = _LatentBathCoefficients(model)
			var bathChannels = UniqueArray<BathChannel>(
				minimumCapacity: model.channelCount)
			var dynamicBathMatrixCount = 0
			for i in 0..<model.channelCount {
				let source =
					configuration.hierarchy.environment.couplingOperators[i]
				var directions = UniqueArray<Direction>()
				for p in coefficients.poles.indices {
					let up = coefficients.upward[i, p]
					let down = coefficients.downward[i, p]
					if up != .zero || down != .zero {
						directions.append(.init(index: p, upward: up, downward: down))
					}
				}
				if !directions.isEmpty || !preparingStandardNoise {
					let dynamicMatrixIndex =
						source.isConstant ? -1 : dynamicBathMatrixCount
					let op = try PreparedBathOperator(
						source,
						dimension: dimension,
						policyCode: configuration._bathOperatorStoragePolicyCode,
						dynamicMatrixIndex: dynamicMatrixIndex)
					if !source.isConstant {
						dynamicBathMatrixCount += 1
					}
					bathChannels.append(
						.init(
							physicalIndex: i,
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

            for channel in problem.markovianChannels {
                guard channel.collapseOperator.firstDimensionMismatch(expected: dimension) == nil else {
                    throw SolverError.operatorDimensionMismatch
                }
            }
            self.markovianChannels = problem.markovianChannels
		}
	}

	/// Prepared representation used only for physical HOPS bath operators.
	///
	/// Constant sparse operators keep both CSR orientations because HOPS uses
	/// both L*x and L^dagger*x at every hierarchy action. Dynamic operators stay
	/// dense and materialize into trajectory-local buffers exactly as before.
	@usableFromInline
	internal enum PreparedBathOperator: ~Copyable, Sendable {
		case dense(UniqueMatrix<Complex<Double>>)
		case sparse(SparseBathOperator)
		case dynamic(PreparedSource, matrixIndex: Int)

		@inlinable
		init(
			_ source: TimeDependentOperator,
			dimension: Int,
			policyCode: UInt8,
			dynamicMatrixIndex: Int
		) throws {
			switch source {
			case .constant(let op):
                switch op.storage {
                    case .dense(let matrix):
                        guard matrix.rows == dimension && matrix.columns == dimension else {
                            throw SolverError.operatorDimensionMismatch
                        }
                    case .sparse(let matrix):
                        guard matrix.rows == dimension && matrix.columns == dimension else {
                            throw SolverError.operatorDimensionMismatch
                        }
                }
			case .linearCombination(let expansion):
				for op in expansion.operators {
                    switch op.storage {
                        case .dense(let matrix):
                            guard matrix.rows == dimension && matrix.columns == dimension else {
                                throw SolverError.operatorDimensionMismatch
                            }
                        case .sparse(let matrix):
                            guard matrix.rows == dimension && matrix.columns == dimension else {
                                throw SolverError.operatorDimensionMismatch
                            }
                    }
				}
			case .generatedDense:
				break
			}

            if case .constant(let op) = source, case .sparse(let csr) = op.storage,
               policyCode != 1 && dimension != 2 {
                let matrix = UniqueCSRMatrix(copying: csr)
                let adjoint = matrix.conjugateTranspose
                self = .sparse(.init(matrix: matrix, adjoint: adjoint))
                return
            }
			if source.isConstant {
				var original = UniqueMatrix<Complex<Double>>.zeros(
					rows: dimension, columns: dimension)
				source.insert(t: 0, into: &original)
				if BathOperatorStorage.shouldUseSparse(
					original, policyCode: policyCode)
				{
					let matrix = UniqueCSRMatrix<Complex<Double>>(from: original)
					let adjoint = matrix.conjugateTranspose
					self = .sparse(
						.init(matrix: matrix, adjoint: adjoint))
				} else {
					self = .dense(original)
				}
				return
			}

			precondition(dynamicMatrixIndex >= 0)
			self = .dynamic(
				PreparedSource(source),
				matrixIndex: dynamicMatrixIndex)
		}

		@inlinable
		var isSparse: Bool {
			switch self {
			case .sparse: true
            case .dense: false
            case .dynamic: false
			}
		}

		@inlinable
		var isDynamic: Bool {
			switch self {
			case .dynamic: true
            case .dense: false
            case .sparse: false
			}
		}
	}

	@usableFromInline
	internal struct SparseBathOperator: ~Copyable, Sendable {
		@usableFromInline
		let matrix: UniqueCSRMatrix<Complex<Double>>
		@usableFromInline
		let adjoint: UniqueCSRMatrix<Complex<Double>>

		@inlinable
		init(
			matrix: consuming UniqueCSRMatrix<Complex<Double>>,
			adjoint: consuming UniqueCSRMatrix<Complex<Double>>
		) {
			self.matrix = matrix
			self.adjoint = adjoint
		}
	}

	/// Centralized crossover policy. The cutoff keeps tiny matrices on the
	/// existing scalar dense kernels; exact structural zeros determine density.
	@usableFromInline
	internal enum BathOperatorStorage {
		@usableFromInline static let automaticMinimumDimension = 8
		@usableFromInline static let automaticMaximumDensity = 0.25

		@inlinable
		static func shouldUseSparse(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			policyCode: UInt8
		) -> Bool {
			// Preserve the hand-specialized TLS path unconditionally.
			if matrix.rows == 2 { return false }
			if policyCode == 1 { return false }
			if policyCode == 2 { return true }
			precondition(policyCode == 0)
			if matrix.rows < automaticMinimumDimension { return false }
			var nnz = 0
			for i in 0..<matrix.rows {
				for j in 0..<matrix.columns {
					if matrix[unchecked: i, unchecked: j] != .zero {
						nnz += 1
					}
				}
			}
			return Double(nnz)
				<= automaticMaximumDensity
					* Double(matrix.rows * matrix.columns)
		}
	}

}


extension HOPS.CPUEngine {
    @_lifetime(borrow preparation)
    @inlinable
    internal static func makeSeededRightHandSide(
        hamiltonian: TimeDependentOperator, preparation: borrowing Preparation,
        seed: UInt64, trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent
    ) -> RightHandSide<HOPS.BathNoiseSampler, GaussianDiffusiveNoise> {
        RightHandSide(
            hamiltonian: hamiltonian, preparation: preparation,
            bathNoise: preparation.makeBathNoise(
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling),
            markovianNoise: GaussianDiffusiveNoise(
                channelCount: preparation.markovianChannels.count, seed: seed,
                trajectoryID: trajectoryID, ensembleSampling: ensembleSampling))
    }
}
