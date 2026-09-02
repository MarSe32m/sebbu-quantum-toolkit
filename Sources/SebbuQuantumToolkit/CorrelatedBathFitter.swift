// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Numerics
import SebbuScience

/// Fits a physical latent rational factor to a matrix-valued spectral density
/// or bath correlation function.
public enum CorrelatedBathFitter {
	public enum Domain: String, Sendable {
		case bathCorrelation
		case spectralDensity
	}

	public enum FittingError: Error, Sendable, CustomStringConvertible {
		case invalidOptions(String)
		case invalidGrid(String)
		case invalidWeights
		case invalidChannelScales
		case inconsistentMatrixDimensions(index: Int)
		case nonFiniteSample(index: Int)
		case nonHermitianSample(index: Int, relativeError: Double)
		case nonPositiveSemidefiniteSample(
			index: Int,
			minimumEigenvalue: Double,
			tolerance: Double
		)
		case insufficientResiduals(parameters: Int, residuals: Int)
		case linearAlgebraFailure(String)
		case optimizationFailure(String)

		public var description: String {
			switch self {
			case .invalidOptions(let message), .invalidGrid(let message):
				message
			case .invalidWeights:
				"The sample weights must be finite, nonnegative, and contain a positive value."
			case .invalidChannelScales:
				"Channel scales must contain one finite positive value per channel."
			case .inconsistentMatrixDimensions(let index):
				"Sample \(index) has dimensions inconsistent with the other samples."
			case .nonFiniteSample(let index):
				"Sample \(index) contains a non-finite value."
			case .nonHermitianSample(let index, let relativeError):
				"Sample \(index) is not Hermitian (relative defect \(relativeError))."
			case .nonPositiveSemidefiniteSample(
				let index,
				let minimumEigenvalue,
				let tolerance
			):
				"Sample \(index) is not positive semidefinite: minimum eigenvalue "
					+ "\(minimumEigenvalue), tolerance \(tolerance)."
			case .insufficientResiduals(let parameters, let residuals):
				"Levenberg-Marquardt needs at least as many real residuals as parameters "
					+ "(parameters: \(parameters), residuals: \(residuals))."
			case .linearAlgebraFailure(let message):
				"A linear-algebra operation failed: \(message)"
			case .optimizationFailure(let message):
				"Levenberg-Marquardt failed: \(message)"
			}
		}
	}

	public struct Options: Sendable {
		/// Maximum number of distinct pole locations retained from the matrix
		/// pencil initializer. A pole may initially be present in more than
		/// one latent bath when its matrix residue has rank greater than one.
		public var maximumPencilPoleCount: Int

		/// Minimum total number of latent pole states retained by backward
		/// elimination for a nonzero target.
		public var minimumPoleCount: Int

		/// Number of independent latent inputs. When omitted, the fitter uses
		/// the largest sampled spectral rank. For a time-domain target, the
		/// rank is estimated from the spectrum reconstructed by matrix pencil.
		public var latentBathCount: Int?

		/// A pruning candidate is accepted while its weighted relative RMS
		/// error is below this target. Set to `nil` to use only the permitted
		/// increase from the full initial fit.
		public var targetRelativeRMSError: Double?

		/// When the target error is not attainable, pruning may increase the
		/// full-model relative RMS error by at most this fraction.
		public var maximumRelativeRMSEIncrease: Double

		/// Strict lower bound used in
		/// `Re(W) = minimumDamping + softplus(u)`.
		public var minimumDamping: Double

		/// Relative singular-value threshold used to estimate how many latent
		/// copies of each matrix-pencil pole are initially required.
		public var matrixPencilResidueRankTolerance: Double

		/// Relative singular-value threshold used to estimate the numerical
		/// rank of the block Hankel matrix before invoking matrix pencil.
		public var matrixPencilHankelRankTolerance: Double

		/// Relative distance below which initial matrix-pencil poles are merged.
		public var duplicatePoleTolerance: Double

		/// Relative tolerance for Hermiticity and small negative eigenvalues.
		/// Only violations within this tolerance are projected away.
		public var physicalityTolerance: Double

		/// Optional diagonal channel scales `S_i`. Residual element `(i,j)`
		/// is divided by `sqrt(S_i S_j)`. When omitted, scales are inferred.
		public var channelScales: [Double]?

		public var pencilParameter: Int?
		public var functionTolerance: Double
		public var stepTolerance: Double
		public var gradientTolerance: Double
		public var maximumFunctionEvaluations: Int?

		/// Limits expensive nonlinear removal trials at each model order. The
		/// candidates are ranked first using their unrefined errors.
		public var maximumPruningCandidatesPerStep: Int

		public init(
			maximumPencilPoleCount: Int = 8,
			minimumPoleCount: Int = 1,
			latentBathCount: Int? = nil,
			targetRelativeRMSError: Double? = 1e-3,
			maximumRelativeRMSEIncrease: Double = 0.05,
			minimumDamping: Double = 1e-10,
			matrixPencilResidueRankTolerance: Double = 1e-7,
			matrixPencilHankelRankTolerance: Double = 1e-8,
			duplicatePoleTolerance: Double = 1e-5,
			physicalityTolerance: Double = 1e-10,
			channelScales: [Double]? = nil,
			pencilParameter: Int? = nil,
			functionTolerance: Double = 1e-9,
			stepTolerance: Double = 1e-9,
			gradientTolerance: Double = 1e-9,
			maximumFunctionEvaluations: Int? = nil,
			maximumPruningCandidatesPerStep: Int = 4
		) {
			self.maximumPencilPoleCount = maximumPencilPoleCount
			self.minimumPoleCount = minimumPoleCount
			self.latentBathCount = latentBathCount
			self.targetRelativeRMSError = targetRelativeRMSError
			self.maximumRelativeRMSEIncrease = maximumRelativeRMSEIncrease
			self.minimumDamping = minimumDamping
			self.matrixPencilResidueRankTolerance = matrixPencilResidueRankTolerance
			self.matrixPencilHankelRankTolerance = matrixPencilHankelRankTolerance
			self.duplicatePoleTolerance = duplicatePoleTolerance
			self.physicalityTolerance = physicalityTolerance
			self.channelScales = channelScales
			self.pencilParameter = pencilParameter
			self.functionTolerance = functionTolerance
			self.stepTolerance = stepTolerance
			self.gradientTolerance = gradientTolerance
			self.maximumFunctionEvaluations = maximumFunctionEvaluations
			self.maximumPruningCandidatesPerStep = maximumPruningCandidatesPerStep
		}
	}

	public struct Diagnostics: Sendable {
		public let domain: Domain
		public let matrixPencilPoleCount: Int
		public let initialPoleCount: Int
		public let finalPoleCount: Int
		public let initialRelativeRMSError: Double
		public let relativeRMSError: Double
		public let maximumRelativeError: Double
		public let acceptedPoleRemovals: Int
		public let pruningTrials: Int
		public let functionEvaluations: Int
		public let converged: Bool
		public let optimizerInfo: Int32

		public init(
			domain: Domain,
			matrixPencilPoleCount: Int,
			initialPoleCount: Int,
			finalPoleCount: Int,
			initialRelativeRMSError: Double,
			relativeRMSError: Double,
			maximumRelativeError: Double,
			acceptedPoleRemovals: Int,
			pruningTrials: Int,
			functionEvaluations: Int,
			converged: Bool,
			optimizerInfo: Int32
		) {
			self.domain = domain
			self.matrixPencilPoleCount = matrixPencilPoleCount
			self.initialPoleCount = initialPoleCount
			self.finalPoleCount = finalPoleCount
			self.initialRelativeRMSError = initialRelativeRMSError
			self.relativeRMSError = relativeRMSError
			self.maximumRelativeError = maximumRelativeError
			self.acceptedPoleRemovals = acceptedPoleRemovals
			self.pruningTrials = pruningTrials
			self.functionEvaluations = functionEvaluations
			self.converged = converged
			self.optimizerInfo = optimizerInfo
		}
	}

	public struct Result: Sendable {
		public let model: CorrelatedBathModel
		public let diagnostics: Diagnostics

		public init(model: CorrelatedBathModel, diagnostics: Diagnostics) {
			self.model = model
			self.diagnostics = diagnostics
		}
	}

    /// Fits positive-lag BCF samples directly in the time domain.
    public static func fitBathCorrelation(
        times: [Double],
        values: [Complex<Double>],
        weights: [Double]? = nil,
        options: Options = Options()
    ) throws -> Result {
        try fit(
            domain: .bathCorrelation,
            axis: times,
            values: values.map { Matrix(elements: [$0], rows: 1, columns: 1) },
            weights: weights,
            options: options
        )
    }
    
	/// Fits positive-lag BCF samples directly in the time domain.
	public static func fitBathCorrelation(
		times: [Double],
		values: [Matrix<Complex<Double>>],
		weights: [Double]? = nil,
		options: Options = Options()
	) throws -> Result {
		try fit(
			domain: .bathCorrelation,
			axis: times,
			values: values,
			weights: weights,
			options: options
		)
	}

	/// Samples and fits a matrix-valued BCF closure.
	public static func fitBathCorrelation(
		times: [Double],
		weights: [Double]? = nil,
		options: Options = Options(),
		evaluating bathCorrelation: (Double) -> Matrix<Complex<Double>>
	) throws -> Result {
		try fitBathCorrelation(
			times: times,
			values: times.map(bathCorrelation),
			weights: weights,
			options: options
		)
	}

	/// Fits Hermitian positive-semidefinite spectral-density samples.
	/// The nonlinear objective is evaluated directly as `J = H H^dagger`.
	public static func fitSpectralDensity(
		frequencies: [Double],
		values: [Matrix<Complex<Double>>],
		weights: [Double]? = nil,
		options: Options = Options()
	) throws -> Result {
		try fit(
			domain: .spectralDensity,
			axis: frequencies,
			values: values,
			weights: weights,
			options: options
		)
	}

	/// Samples and fits a matrix-valued spectral-density closure.
	public static func fitSpectralDensity(
		frequencies: [Double],
		weights: [Double]? = nil,
		options: Options = Options(),
		evaluating spectralDensity: (Double) -> Matrix<Complex<Double>>
	) throws -> Result {
		try fitSpectralDensity(
			frequencies: frequencies,
			values: frequencies.map(spectralDensity),
			weights: weights,
			options: options
		)
	}
}

extension CorrelatedBathFitter {
	fileprivate struct Target {
		let domain: Domain
		let axis: [Double]
		let values: [Matrix<Complex<Double>>]
		let weights: [Double]
		let channelScales: [Double]
		let channelCount: Int
		let targetNormSquared: Double
		let residualNormalization: Double

		var residualCount: Int {
			2 * axis.count * channelCount * channelCount
		}

		func prediction(
			from model: CorrelatedBathModel,
			at index: Int
		) -> Matrix<Complex<Double>> {
			switch domain {
			case .bathCorrelation:
				model.bathCorrelation(at: axis[index])
			case .spectralDensity:
				model.spectralDensity(at: axis[index])
			}
		}

		func residuals(for model: CorrelatedBathModel) -> Vector<Double> {
			var residuals: [Double] = []
			residuals.reserveCapacity(residualCount)
			for sample in axis.indices {
				let fitted = prediction(from: model, at: sample)
				let sampleFactor =
					weights[sample].squareRoot() * residualNormalization
				for i in 0..<channelCount {
					for j in 0..<channelCount {
						let factor =
							sampleFactor
							/ (channelScales[i] * channelScales[j])
							.squareRoot()
						let difference = fitted[i, j] - values[sample][i, j]
						residuals.append(factor * difference.real)
						residuals.append(factor * difference.imaginary)
					}
				}
			}
			return Vector(residuals)
		}

		func errors(for model: CorrelatedBathModel) -> (
			relativeRMS: Double,
			maximumRelative: Double
		) {
			var errorSquared = 0.0
			var maximumRelative = 0.0
			for sample in axis.indices {
				let fitted = prediction(from: model, at: sample)
				var sampleError = 0.0
				var sampleTarget = 0.0
				for i in 0..<channelCount {
					for j in 0..<channelCount {
						let scale = channelScales[i] * channelScales[j]
						let difference = fitted[i, j] - values[sample][i, j]
						sampleError += squaredMagnitude(difference) / scale
						sampleTarget +=
							squaredMagnitude(values[sample][i, j])
							/ scale
					}
				}
				errorSquared += weights[sample] * sampleError
				let denominator = max(
					sampleTarget,
					targetNormSquared / Double(max(axis.count, 1)) * 1e-24
				)
				maximumRelative = max(
					maximumRelative,
					(sampleError / denominator).squareRoot()
				)
			}
			return (
				(errorSquared / targetNormSquared).squareRoot(),
				maximumRelative
			)
		}
	}

	fileprivate struct PencilCandidate {
		var pole: Complex<Double>
		var residue: Matrix<Complex<Double>>
		var score: Double
	}

	fileprivate struct PoleState {
		var pole: Complex<Double>
		var residue: [Complex<Double>]
	}

	fileprivate struct BathState {
		var poles: [PoleState]
	}

	fileprivate struct ModelState {
		let channelCount: Int
		var baths: [BathState]

		var poleCount: Int {
			baths.reduce(0) { $0 + $1.poles.count }
		}

		var parameterCount: Int {
			2 * poleCount * (channelCount + 1)
		}

		var model: CorrelatedBathModel {
			let latentBaths = baths.compactMap {
				bath -> CorrelatedBathModel.LatentBath? in
				guard !bath.poles.isEmpty else { return nil }
				let residues = Matrix<Complex<Double>>(
					rows: channelCount,
					columns: bath.poles.count
				) { elements in
					for pole in bath.poles.indices {
						for channel in 0..<channelCount {
							elements[
								channel * bath.poles.count + pole] =
								bath.poles[pole].residue[channel]
						}
					}
				}
				return CorrelatedBathModel.LatentBath(
					poles: bath.poles.map(\.pole),
					residues: residues
				)
			}
			return CorrelatedBathModel(
				channelCount: channelCount, latentBaths: latentBaths)
		}

		func encoded(minimumDamping: Double) -> Vector<Double> {
			var parameters: [Double] = []
			parameters.reserveCapacity(parameterCount)
			for bath in baths {
				for state in bath.poles {
					parameters.append(
						inverseSoftplus(
							max(
								state.pole.real - minimumDamping,
								minimumDamping))
					)
					parameters.append(state.pole.imaginary)
					for residue in state.residue {
						parameters.append(residue.real)
						parameters.append(residue.imaginary)
					}
				}
			}
			return Vector(parameters)
		}

		func decoded(
			from parameters: Vector<Double>,
			minimumDamping: Double
		) -> ModelState {
			precondition(parameters.count == parameterCount)
			var result = self
			var index = 0
			for bath in result.baths.indices {
				for pole in result.baths[bath].poles.indices {
					let damping = minimumDamping + softplus(parameters[index])
					let frequency = parameters[index + 1]
					index += 2
					result.baths[bath].poles[pole].pole = Complex(
						damping, frequency)
					for channel in 0..<channelCount {
						result.baths[bath].poles[pole].residue[channel] =
							Complex(
								parameters[index],
								parameters[index + 1]
							)
						index += 2
					}
				}
			}
			return result
		}

		func removing(bath: Int, pole: Int) -> ModelState {
			var result = self
			result.baths[bath].poles.remove(at: pole)
			result.baths.removeAll { $0.poles.isEmpty }
			return result
		}
	}

	fileprivate struct OptimizedState {
		let state: ModelState
		let relativeRMS: Double
		let maximumRelative: Double
		let functionEvaluations: Int
		let converged: Bool
		let info: Int32
	}

	fileprivate struct RemovalCandidate {
		let bath: Int
		let pole: Int
		let estimatedError: Double
	}

	fileprivate static func fit(
		domain: Domain,
		axis: [Double],
		values: [Matrix<Complex<Double>>],
		weights suppliedWeights: [Double]?,
		options: Options
	) throws -> Result {
		try validate(options)
		let target = try prepareSamples(
			domain: domain,
			axis: axis,
			values: values,
			weights: suppliedWeights,
			options: options
		)

		if target.targetNormSquared <= Double.leastNormalMagnitude {
			let model = CorrelatedBathModel.zero(channelCount: target.channelCount)
			return Result(
				model: model,
				diagnostics: Diagnostics(
					domain: domain,
					matrixPencilPoleCount: 0,
					initialPoleCount: 0,
					finalPoleCount: 0,
					initialRelativeRMSError: 0,
					relativeRMSError: 0,
					maximumRelativeError: 0,
					acceptedPoleRemovals: 0,
					pruningTrials: 0,
					functionEvaluations: 0,
					converged: true,
					optimizerInfo: 1
				)
			)
		}

		let pencilSamples = try makePencilSamples(target: target, options: options)
		let candidates = try matrixPencilCandidates(
			samples: pencilSamples.values,
			step: pencilSamples.step,
			options: options
		)
		let latentBathCount = try inferLatentBathCount(
			target: target,
			requested: options.latentBathCount,
			candidates: candidates,
			tolerance: options.matrixPencilResidueRankTolerance
		)
		let parametersPerPole = 2 * (target.channelCount + 1)
		guard latentBathCount <= target.residualCount / parametersPerPole else {
			throw FittingError.insufficientResiduals(
				parameters: latentBathCount * parametersPerPole,
				residuals: target.residualCount
			)
		}
		let initialState = try makeInitialState(
			candidates: candidates,
			alphaZero: pencilSamples.values[0],
			channelCount: target.channelCount,
			latentBathCount: latentBathCount,
			options: options
		)
		guard options.minimumPoleCount <= initialState.poleCount else {
			throw FittingError.invalidOptions(
				"The minimum pole count exceeds the matrix-pencil initialized model order."
			)
		}
		guard initialState.parameterCount <= target.residualCount else {
			throw FittingError.insufficientResiduals(
				parameters: initialState.parameterCount,
				residuals: target.residualCount
			)
		}

		var optimized = try optimize(initialState, target: target, options: options)
		let initialPoleCount = optimized.state.poleCount
		let initialError = optimized.relativeRMS
		let acceptableError = max(
			options.targetRelativeRMSError ?? 0,
			max(initialError * (1 + options.maximumRelativeRMSEIncrease), 1e-14)
		)

		var acceptedRemovals = 0
		var pruningTrials = 0
		var totalFunctionEvaluations = optimized.functionEvaluations

		while optimized.state.poleCount > options.minimumPoleCount {
			var removals: [RemovalCandidate] = []
			for bath in optimized.state.baths.indices {
				for pole in optimized.state.baths[bath].poles.indices {
					let candidate = optimized.state.removing(
						bath: bath, pole: pole)
					guard candidate.poleCount >= options.minimumPoleCount else {
						continue
					}
					removals.append(
						RemovalCandidate(
							bath: bath,
							pole: pole,
							estimatedError: target.errors(
								for: candidate.model
							).relativeRMS
						)
					)
				}
			}
			removals.sort { $0.estimatedError < $1.estimatedError }

			var accepted: OptimizedState?
			for removal in removals.prefix(options.maximumPruningCandidatesPerStep) {
				let candidate = optimized.state.removing(
					bath: removal.bath,
					pole: removal.pole
				)
				pruningTrials += 1
				let trial: OptimizedState
				do {
					trial = try optimize(
						candidate, target: target, options: options)
				} catch {
					continue
				}
				totalFunctionEvaluations += trial.functionEvaluations
				if trial.relativeRMS <= acceptableError,
					accepted == nil || trial.relativeRMS < accepted!.relativeRMS
				{
					accepted = trial
				}
			}

			guard let accepted else { break }
			optimized = accepted
			acceptedRemovals += 1
		}

		return Result(
			model: optimized.state.model,
			diagnostics: Diagnostics(
				domain: domain,
				matrixPencilPoleCount: candidates.count,
				initialPoleCount: initialPoleCount,
				finalPoleCount: optimized.state.poleCount,
				initialRelativeRMSError: initialError,
				relativeRMSError: optimized.relativeRMS,
				maximumRelativeError: optimized.maximumRelative,
				acceptedPoleRemovals: acceptedRemovals,
				pruningTrials: pruningTrials,
				functionEvaluations: totalFunctionEvaluations,
				converged: optimized.converged,
				optimizerInfo: optimized.info
			)
		)
	}

	fileprivate static func validate(_ options: Options) throws {
		guard options.maximumPencilPoleCount > 0,
			options.minimumPoleCount > 0,
			options.latentBathCount.map({ $0 > 0 }) ?? true,
			options.targetRelativeRMSError.map({ $0.isFinite && $0 >= 0 }) ?? true,
			options.maximumRelativeRMSEIncrease.isFinite,
			options.maximumRelativeRMSEIncrease >= 0,
			options.minimumDamping.isFinite,
			options.minimumDamping > 0,
			options.matrixPencilResidueRankTolerance.isFinite,
			options.matrixPencilResidueRankTolerance > 0,
			options.matrixPencilHankelRankTolerance.isFinite,
			options.matrixPencilHankelRankTolerance > 0,
			options.duplicatePoleTolerance.isFinite,
			options.duplicatePoleTolerance >= 0,
			options.physicalityTolerance.isFinite,
			options.physicalityTolerance > 0,
			options.pencilParameter.map({ $0 > 0 }) ?? true,
			options.functionTolerance.isFinite,
			options.functionTolerance >= 0,
			options.stepTolerance.isFinite,
			options.stepTolerance >= 0,
			options.gradientTolerance.isFinite,
			options.gradientTolerance >= 0,
			options.maximumFunctionEvaluations.map({ $0 > 0 }) ?? true,
			options.maximumPruningCandidatesPerStep > 0
		else {
			throw FittingError.invalidOptions(
				"Invalid correlated-bath fitting options.")
		}
	}

	fileprivate static func prepareSamples(
		domain: Domain,
		axis: [Double],
		values: [Matrix<Complex<Double>>],
		weights suppliedWeights: [Double]?,
		options: Options
	) throws -> Target {
		guard axis.count >= 4, axis.count == values.count else {
			throw FittingError.invalidGrid(
				"At least four grid points and one matrix per point are required."
			)
		}
		guard axis.allSatisfy(\.isFinite),
			zip(axis.dropFirst(), axis).allSatisfy({ $0.0 > $0.1 })
		else {
			throw FittingError.invalidGrid(
				"The sampling grid must be finite and strictly increasing."
			)
		}
		if domain == .bathCorrelation {
			let zeroTolerance = 16 * Double.ulpOfOne * max(1, abs(axis.last ?? 0))
			guard axis[0] >= 0, abs(axis[0]) <= zeroTolerance else {
				throw FittingError.invalidGrid(
					"Positive-lag BCF samples must begin at zero."
				)
			}
		}

		let channelCount = values[0].rows
		guard channelCount > 0, values[0].columns == channelCount else {
			throw FittingError.invalidGrid(
				"Sample matrices must be nonempty and square.")
		}
		for sample in values.indices {
			guard values[sample].rows == channelCount,
				values[sample].columns == channelCount
			else {
				throw FittingError.inconsistentMatrixDimensions(index: sample)
			}
			guard
				values[sample].elements.allSatisfy({
					$0.real.isFinite && $0.imaginary.isFinite
				})
			else {
				throw FittingError.nonFiniteSample(index: sample)
			}
		}

		let weights = suppliedWeights ?? [Double](repeating: 1, count: axis.count)
		guard weights.count == axis.count,
			weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
			weights.contains(where: { $0 > 0 })
		else {
			throw FittingError.invalidWeights
		}

		var sanitizedValues = values
		switch domain {
		case .bathCorrelation:
			sanitizedValues[0] = try projectPositiveSemidefinite(
				values[0],
				index: 0,
				tolerance: options.physicalityTolerance
			)
		case .spectralDensity:
			for sample in sanitizedValues.indices {
				sanitizedValues[sample] = try projectPositiveSemidefinite(
					values[sample],
					index: sample,
					tolerance: options.physicalityTolerance
				)
			}
		}

		let channelScales: [Double]
		if let supplied = options.channelScales {
			guard supplied.count == channelCount,
				supplied.allSatisfy({ $0.isFinite && $0 > 0 })
			else {
				throw FittingError.invalidChannelScales
			}
			channelScales = supplied
		} else {
			var scales = [Double](repeating: 0, count: channelCount)
			for value in sanitizedValues {
				for channel in 0..<channelCount {
					scales[channel] = max(
						scales[channel], value[channel, channel].length)
				}
			}
			let largest = max(scales.max() ?? 0, Double.leastNormalMagnitude)
			channelScales = scales.map { max($0, largest * 1e-12) }
		}

		var targetNormSquared = 0.0
		for sample in sanitizedValues.indices {
			for i in 0..<channelCount {
				for j in 0..<channelCount {
					targetNormSquared +=
						weights[sample]
						* squaredMagnitude(sanitizedValues[sample][i, j])
						/ (channelScales[i] * channelScales[j])
				}
			}
		}
		let complexResidualCount = axis.count * channelCount * channelCount
		let residualNormalization: Double
		if targetNormSquared > Double.leastNormalMagnitude {
			residualNormalization = (Double(complexResidualCount) / targetNormSquared)
				.squareRoot()
		} else {
			residualNormalization = 1
		}

		return Target(
			domain: domain,
			axis: axis,
			values: sanitizedValues,
			weights: weights,
			channelScales: channelScales,
			channelCount: channelCount,
			targetNormSquared: targetNormSquared,
			residualNormalization: residualNormalization
		)
	}

	fileprivate static func projectPositiveSemidefinite(
		_ matrix: Matrix<Complex<Double>>,
		index: Int,
		tolerance relativeTolerance: Double
	) throws -> Matrix<Complex<Double>> {
		let norm = frobeniusNorm(matrix)
		var defectSquared = 0.0
		var hermitian = Matrix<Complex<Double>>.zeros(
			rows: matrix.rows,
			columns: matrix.columns
		)
		for i in 0..<matrix.rows {
			for j in 0..<matrix.columns {
				let adjoint = matrix[j, i].conjugate
				defectSquared += squaredMagnitude(matrix[i, j] - adjoint)
				hermitian[i, j] = 0.5 * (matrix[i, j] + adjoint)
			}
		}
		let relativeDefect =
			defectSquared.squareRoot()
			/ max(norm, Double.leastNormalMagnitude)
		guard relativeDefect <= relativeTolerance else {
			throw FittingError.nonHermitianSample(
				index: index,
				relativeError: relativeDefect
			)
		}

		let decomposition: (eigenValues: [Double], eigenVectors: [Vector<Complex<Double>>])
		do {
			decomposition = try MatrixOperations.diagonalizeHermitian(hermitian)
		} catch {
			throw FittingError.linearAlgebraFailure(String(describing: error))
		}
		let scale = max(
			decomposition.eigenValues.map(abs).max() ?? 0,
			Double.leastNormalMagnitude
		)
		let absoluteTolerance = relativeTolerance * scale
		if let minimum = decomposition.eigenValues.first,
			minimum < -absoluteTolerance
		{
			throw FittingError.nonPositiveSemidefiniteSample(
				index: index,
				minimumEigenvalue: minimum,
				tolerance: absoluteTolerance
			)
		}

		var result = Matrix<Complex<Double>>.zeros(
			rows: matrix.rows,
			columns: matrix.columns
		)
		for eigenvalue in decomposition.eigenValues.indices {
			let value = max(decomposition.eigenValues[eigenvalue], 0)
			guard value > 0 else { continue }
			let vector = decomposition.eigenVectors[eigenvalue]
			for i in 0..<matrix.rows {
				let left = value * vector[i]
				for j in 0..<matrix.columns {
					result[i, j] += left * vector[j].conjugate
				}
			}
		}
		return result
	}

	fileprivate static func makePencilSamples(
		target: Target,
		options: Options
	) throws -> (values: [Matrix<Complex<Double>>], step: Double) {
		switch target.domain {
		case .bathCorrelation:
			let count = max(target.axis.count, 2 * options.maximumPencilPoleCount + 2)
			let step = (target.axis.last! - target.axis[0]) / Double(count - 1)
			guard step.isFinite, step > 0 else {
				throw FittingError.invalidGrid(
					"The BCF time span must be positive.")
			}
			let values = (0..<count).map { sample in
				interpolate(
					axis: target.axis,
					values: target.values,
					at: Double(sample) * step
				)
			}
			return (values, step)

		case .spectralDensity:
			let span = target.axis.last! - target.axis[0]
			guard span.isFinite, span > 0 else {
				throw FittingError.invalidGrid(
					"The spectral frequency span must be positive.")
			}
			let step = 2 * Double.pi / span
			let count = max(
				2 * options.maximumPencilPoleCount + 2,
				min(target.axis.count, 128)
			)
			let quadratureWeights = trapezoidalWeights(target.axis)
			let values = (0..<count).map { sample -> Matrix<Complex<Double>> in
				let lag = Double(sample) * step
				var alpha = Matrix<Complex<Double>>.zeros(
					rows: target.channelCount,
					columns: target.channelCount
				)
				for frequency in target.axis.indices {
					let phase = Complex<Double>.exp(
						Complex(0, -target.axis[frequency] * lag)
					)
					let factor =
						quadratureWeights[frequency] * phase
						/ (2 * Double.pi)
					for element in alpha.elements.indices {
						alpha.elements[element] +=
							factor
							* target.values[frequency].elements[element]
					}
				}
				return alpha
			}
			return (values, step)
		}
	}

	fileprivate static func matrixPencilCandidates(
		samples: [Matrix<Complex<Double>>],
		step: Double,
		options: Options
	) throws -> [PencilCandidate] {
		let pencilParameter = options.pencilParameter ?? samples.count / 2
		guard pencilParameter > 0, pencilParameter < samples.count else {
			throw FittingError.invalidOptions(
				"The pencil parameter must lie between zero and the sample count."
			)
		}
		let estimatedRank = try blockHankelRank(
			samples: samples,
			pencilParameter: pencilParameter,
			relativeTolerance: options.matrixPencilHankelRankTolerance
		)
		let terms = max(1, min(estimatedRank, options.maximumPencilPoleCount))
		let pencil = MatrixPencil.fit(
			samples: samples,
			step: step,
			pencilParameter: pencilParameter,
			terms: terms
		)
		var candidates: [PencilCandidate] = []
		for index in pencil.W.indices {
			let originalPole = pencil.W[index]
			let residue = pencil.G[index]
			guard originalPole.real.isFinite,
				originalPole.imaginary.isFinite,
				residue.elements.allSatisfy({
					$0.real.isFinite && $0.imaginary.isFinite
				})
			else { continue }
			let stablePole = Complex(
				max(abs(originalPole.real), 10 * options.minimumDamping),
				originalPole.imaginary
			)
			let score = frobeniusNorm(residue)
			guard score > 0 else { continue }
			candidates.append(
				PencilCandidate(pole: stablePole, residue: residue, score: score)
			)
		}

		candidates.sort { $0.score > $1.score }
		var merged: [PencilCandidate] = []
		for candidate in candidates {
			if let index = merged.firstIndex(where: {
				poleDistance($0.pole, candidate.pole)
					<= options.duplicatePoleTolerance
			}) {
				let total = merged[index].score + candidate.score
				merged[index].pole =
					(merged[index].score * merged[index].pole
						+ candidate.score * candidate.pole) / total
				for element in merged[index].residue.elements.indices {
					merged[index].residue.elements[element] +=
						candidate.residue.elements[element]
				}
				merged[index].score = frobeniusNorm(merged[index].residue)
			} else {
				merged.append(candidate)
			}
		}
		merged.sort { $0.score > $1.score }
		if merged.count > options.maximumPencilPoleCount {
			merged.removeLast(merged.count - options.maximumPencilPoleCount)
		}

		if merged.isEmpty {
			let channelCount = samples[0].rows
			let scale = max(frobeniusNorm(samples[0]), 1)
			var residue = Matrix<Complex<Double>>.zeros(
				rows: channelCount,
				columns: channelCount
			)
			for channel in 0..<channelCount {
				residue[channel, channel] = Complex(scale / Double(channelCount))
			}
			merged.append(
				PencilCandidate(
					pole: Complex(
						max(
							1 / (step * Double(samples.count)),
							10 * options.minimumDamping
						)
					),
					residue: residue,
					score: scale
				)
			)
		}
		return merged
	}

	fileprivate static func blockHankelRank(
		samples: [Matrix<Complex<Double>>],
		pencilParameter: Int,
		relativeTolerance: Double
	) throws -> Int {
		let blockRows = samples.count - pencilParameter
		let outputCount = samples[0].elements.count
		var gram = Matrix<Complex<Double>>.zeros(
			rows: pencilParameter,
			columns: pencilParameter
		)
		for leftColumn in 0..<pencilParameter {
			for rightColumn in leftColumn..<pencilParameter {
				var value = Complex<Double>.zero
				for blockRow in 0..<blockRows {
					let left = samples[blockRow + leftColumn]
					let right = samples[blockRow + rightColumn]
					for output in 0..<outputCount {
						value +=
							left.elements[output].conjugate
							* right.elements[output]
					}
				}
				gram[leftColumn, rightColumn] = value
				gram[rightColumn, leftColumn] = value.conjugate
			}
		}

		let eigenvalues: [Double]
		do {
			eigenvalues = try MatrixOperations.eigenValuesHermitian(gram)
		} catch {
			throw FittingError.linearAlgebraFailure(String(describing: error))
		}
		let largest = max(eigenvalues.last ?? 0, 0)
		guard largest > 0 else { return 0 }
		let eigenvalueTolerance = max(
			relativeTolerance * relativeTolerance * largest,
			100 * Double.ulpOfOne * Double(pencilParameter) * largest
		)
		return eigenvalues.count { $0 > eigenvalueTolerance }
	}

	fileprivate static func inferLatentBathCount(
		target: Target,
		requested: Int?,
		candidates: [PencilCandidate],
		tolerance: Double
	) throws -> Int {
		if let requested {
			return requested
		}

		var maximumRank = 0
		switch target.domain {
		case .bathCorrelation:
			// alpha(0) can have a larger rank than the number of latent
			// factor columns, so estimate pointwise spectral rank instead.
			let largestDamping = max(
				candidates.map { $0.pole.real }.max() ?? 0,
				Double.leastNormalMagnitude
			)
			let minimumFrequency =
				(candidates.map { $0.pole.imaginary }.min() ?? 0)
				- 4 * largestDamping
			let maximumFrequency =
				(candidates.map { $0.pole.imaginary }.max() ?? 0)
				+ 4 * largestDamping
			let frequencyCount = max(25, 4 * candidates.count + 1)
			for sample in 0..<frequencyCount {
				let fraction = Double(sample) / Double(frequencyCount - 1)
				let frequency =
					minimumFrequency
					+ fraction * (maximumFrequency - minimumFrequency)
				let density = pencilSpectralDensity(
					candidates: candidates,
					frequency: frequency,
					channelCount: target.channelCount
				)
				let eigenvalues: [Double]
				do {
					eigenvalues = try MatrixOperations.eigenValuesHermitian(
						density)
				} catch {
					throw FittingError.linearAlgebraFailure(
						String(describing: error))
				}
				let largest = max(
					eigenvalues.last ?? 0, Double.leastNormalMagnitude)
				maximumRank = max(
					maximumRank,
					eigenvalues.count { $0 > tolerance * largest }
				)
			}
		case .spectralDensity:
			for matrix in target.values {
				let eigenvalues: [Double]
				do {
					eigenvalues = try MatrixOperations.eigenValuesHermitian(
						matrix)
				} catch {
					throw FittingError.linearAlgebraFailure(
						String(describing: error))
				}
				let largest = max(
					eigenvalues.last ?? 0, Double.leastNormalMagnitude)
				maximumRank = max(
					maximumRank,
					eigenvalues.count { $0 > tolerance * largest }
				)
			}
		}
		return max(1, min(maximumRank, target.channelCount))
	}

	fileprivate static func pencilSpectralDensity(
		candidates: [PencilCandidate],
		frequency: Double,
		channelCount: Int
	) -> Matrix<Complex<Double>> {
		let imaginaryFrequency = Complex<Double>(0, frequency)
		var density = Matrix<Complex<Double>>.zeros(
			rows: channelCount,
			columns: channelCount
		)
		for candidate in candidates {
			let positiveDenominator = candidate.pole - imaginaryFrequency
			let negativeDenominator = candidate.pole.conjugate + imaginaryFrequency
			for i in 0..<channelCount {
				for j in 0..<channelCount {
					density[i, j] +=
						candidate.residue[i, j] / positiveDenominator
						+ candidate.residue[j, i].conjugate
						/ negativeDenominator
				}
			}
		}
		return symmetrizedHermitian(density)
	}

	fileprivate static func makeInitialState(
		candidates: [PencilCandidate],
		alphaZero: Matrix<Complex<Double>>,
		channelCount: Int,
		latentBathCount: Int,
		options: Options
	) throws -> ModelState {
		let anchors = try initialAnchors(alphaZero: alphaZero, count: latentBathCount)
		var baths = [BathState](
			repeating: BathState(poles: []),
			count: latentBathCount
		)

		for candidate in candidates {
			let decomposition:
				(
					U: Matrix<Complex<Double>>,
					singularValues: [Double],
					VH: Matrix<Complex<Double>>
				)
			do {
				decomposition = try MatrixOperations.singularValueDecomposition(
					A: candidate.residue
				)
			} catch {
				throw FittingError.linearAlgebraFailure(String(describing: error))
			}
			let largest = max(
				decomposition.singularValues.first ?? 0,
				Double.leastNormalMagnitude
			)
			let residueRank = max(
				1,
				min(
					latentBathCount,
					decomposition.singularValues.count {
						$0 > options.matrixPencilResidueRankTolerance
							* largest
					}
				)
			)
			var usedBaths = Set<Int>()
			for component in 0..<residueRank {
				var direction = [Complex<Double>](
					repeating: .zero, count: channelCount)
				for channel in 0..<channelCount {
					direction[channel] = decomposition.U[channel, component]
				}
				let bath = bestBath(
					for: direction,
					anchors: anchors,
					excluding: usedBaths
				)
				usedBaths.insert(bath)
				let singularValue = decomposition.singularValues[component]
				let amplitude = max(
					2 * candidate.pole.real * singularValue,
					largest * 1e-12
				).squareRoot()
				baths[bath].poles.append(
					PoleState(
						pole: candidate.pole,
						residue: direction.map { amplitude * $0 }
					)
				)
			}
		}

		// HH^dagger has a zero derivative at an exactly zero factor column.
		for bath in baths.indices where baths[bath].poles.isEmpty {
			let candidate = candidates[bath % candidates.count]
			let anchorNorm = vectorNorm(anchors[bath])
			let residue: [Complex<Double>]
			if anchorNorm > 0 {
				residue = anchors[bath].map {
					$0 * (2 * candidate.pole.real).squareRoot() / anchorNorm
				}
			} else {
				residue = (0..<channelCount).map {
					$0 == bath % channelCount
						? Complex((2 * candidate.pole.real).squareRoot())
						: .zero
				}
			}
			baths[bath].poles.append(
				PoleState(pole: candidate.pole, residue: residue)
			)
		}
		return ModelState(channelCount: channelCount, baths: baths)
	}

	fileprivate static func initialAnchors(
		alphaZero: Matrix<Complex<Double>>,
		count: Int
	) throws -> [[Complex<Double>]] {
		let hermitian = symmetrizedHermitian(alphaZero)
		let decomposition: (eigenValues: [Double], eigenVectors: [Vector<Complex<Double>>])
		do {
			decomposition = try MatrixOperations.diagonalizeHermitian(hermitian)
		} catch {
			throw FittingError.linearAlgebraFailure(String(describing: error))
		}
		let largest = max(decomposition.eigenValues.last ?? 0, 1)
		var anchors: [[Complex<Double>]] = []
		anchors.reserveCapacity(count)
		for offset in 0..<count {
			if offset < decomposition.eigenValues.count {
				let index = decomposition.eigenValues.count - 1 - offset
				let amplitude = max(
					decomposition.eigenValues[index],
					largest * 1e-12
				).squareRoot()
				anchors.append(
					decomposition.eigenVectors[index].components.map {
						amplitude * $0
					}
				)
			} else {
				let amplitude = (largest * 1e-12).squareRoot()
				anchors.append(
					(0..<alphaZero.rows).map {
						$0 == offset % alphaZero.rows
							? Complex(amplitude) : .zero
					}
				)
			}
		}
		return anchors
	}

	fileprivate static func bestBath(
		for direction: [Complex<Double>],
		anchors: [[Complex<Double>]],
		excluding: Set<Int>
	) -> Int {
		var best = anchors.indices.first { !excluding.contains($0) } ?? 0
		var bestOverlap = -Double.infinity
		let directionNorm = max(vectorNorm(direction), Double.leastNormalMagnitude)
		for bath in anchors.indices where !excluding.contains(bath) {
			let anchorNorm = max(vectorNorm(anchors[bath]), Double.leastNormalMagnitude)
			var overlap = Complex<Double>.zero
			for channel in direction.indices {
				overlap += anchors[bath][channel].conjugate * direction[channel]
			}
			let normalized = overlap.length / (anchorNorm * directionNorm)
			if normalized > bestOverlap {
				best = bath
				bestOverlap = normalized
			}
		}
		return best
	}

	fileprivate static func optimize(
		_ initial: ModelState,
		target: Target,
		options: Options
	) throws -> OptimizedState {
		guard initial.parameterCount <= target.residualCount else {
			throw FittingError.insufficientResiduals(
				parameters: initial.parameterCount,
				residuals: target.residualCount
			)
		}
		let initialParameters = initial.encoded(minimumDamping: options.minimumDamping)
		let result: Optimize.LevenbergMarquardtResult
		do {
			result = try Optimize.levenbergMarquardt(
				initial: initialParameters,
				functionTolerance: options.functionTolerance,
				stepTolerance: options.stepTolerance,
				gradientTolerance: options.gradientTolerance,
				maxFunctionEvaluations: options.maximumFunctionEvaluations,
				residuals: { parameters in
					let state = initial.decoded(
						from: parameters,
						minimumDamping: options.minimumDamping
					)
					return target.residuals(for: state.model)
				}
			)
		} catch {
			throw FittingError.optimizationFailure(String(describing: error))
		}
		let state = initial.decoded(
			from: result.parameters,
			minimumDamping: options.minimumDamping
		)
		let errors = target.errors(for: state.model)
		return OptimizedState(
			state: state,
			relativeRMS: errors.relativeRMS,
			maximumRelative: errors.maximumRelative,
			functionEvaluations: result.functionEvaluations,
			converged: result.converged,
			info: result.info
		)
	}
}

private func interpolate(
	axis: [Double],
	values: [Matrix<Complex<Double>>],
	at point: Double
) -> Matrix<Complex<Double>> {
	if point <= axis[0] { return values[0] }
	if point >= axis.last! { return values.last! }
	var lower = 0
	var upper = axis.count - 1
	while upper - lower > 1 {
		let middle = (lower + upper) / 2
		if axis[middle] <= point {
			lower = middle
		} else {
			upper = middle
		}
	}
	let fraction = (point - axis[lower]) / (axis[upper] - axis[lower])
	return Matrix(
		elements: zip(values[lower].elements, values[upper].elements).map {
			$0.0 + fraction * ($0.1 - $0.0)
		},
		rows: values[lower].rows,
		columns: values[lower].columns
	)
}

private func trapezoidalWeights(_ axis: [Double]) -> [Double] {
	var weights = [Double](repeating: 0, count: axis.count)
	weights[0] = 0.5 * (axis[1] - axis[0])
	for index in 1..<(axis.count - 1) {
		weights[index] = 0.5 * (axis[index + 1] - axis[index - 1])
	}
	weights[axis.count - 1] = 0.5 * (axis.last! - axis[axis.count - 2])
	return weights
}

private func squaredMagnitude(_ value: Complex<Double>) -> Double {
	value.real * value.real + value.imaginary * value.imaginary
}

private func frobeniusNorm(_ matrix: Matrix<Complex<Double>>) -> Double {
	matrix.elements.reduce(0) { $0 + squaredMagnitude($1) }.squareRoot()
}

private func vectorNorm(_ vector: [Complex<Double>]) -> Double {
	vector.reduce(0) { $0 + squaredMagnitude($1) }.squareRoot()
}

private func symmetrizedHermitian(
	_ matrix: Matrix<Complex<Double>>
) -> Matrix<Complex<Double>> {
	Matrix(rows: matrix.rows, columns: matrix.columns) { elements in
		for i in 0..<matrix.rows {
			for j in 0..<matrix.columns {
				elements[i * matrix.columns + j] =
					0.5
					* (matrix[i, j] + matrix[j, i].conjugate)
			}
		}
	}
}

private func poleDistance(_ lhs: Complex<Double>, _ rhs: Complex<Double>) -> Double {
	(lhs - rhs).length / max(1, max(lhs.length, rhs.length))
}

private func softplus(_ value: Double) -> Double {
	if value > 30 { return value }
	if value < -30 { return Double.exp(value) }
	return Double.log(1 + Double.exp(value))
}

private func inverseSoftplus(_ value: Double) -> Double {
	if value > 30 { return value }
	if value < 1e-8 { return Double.log(value) }
	return Double.log(Double.exp(value) - 1)
}
