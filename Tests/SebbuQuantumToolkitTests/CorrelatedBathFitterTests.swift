// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuQuantumToolkit
import SebbuScience
import Testing

@Suite("Physical correlated-bath fitting")
struct CorrelatedBathFitterTests {
	@Test("Latent model is stationary and spectrally positive")
	func latentModelIdentities() throws {
		let model = twoPoleCorrelatedModel

		for frequency in stride(from: -3.0, through: 3.0, by: 0.25) {
			let density = model.spectralDensity(at: frequency)
			let eigenvalues = try MatrixOperations.eigenValuesHermitian(density)
			#expect((eigenvalues.first ?? 0) > -1e-11)
		}

		let positive = model.bathCorrelation(at: 0.73)
		let negative = model.bathCorrelation(at: -0.73)
		expectMatricesClose(negative, positive.conjugateTranspose, tolerance: 1e-13)

		for lag in [0.0, 0.17, 0.8, 1.7] {
			var reconstructed = Matrix<Complex<Double>>.zeros(rows: 2, columns: 2)
			for term in model.oneSidedExponentialTerms {
				let factor = Complex<Double>.exp(-term.pole * lag)
				for element in reconstructed.elements.indices {
					reconstructed.elements[element] +=
						factor * term.residue.elements[element]
				}
			}
			expectMatricesClose(
				reconstructed,
				model.bathCorrelation(at: lag),
				tolerance: 2e-12
			)
		}
	}

	@Test("BCF samples recover the minimal shared two-pole model")
	func fitBathCorrelation() throws {
		let expected = twoPoleCorrelatedModel
		let step = 0.075
		let times = (0..<52).map { Double($0) * step }
		let weights = times.map { 1 + 2 * $0 / times.last! }

		let result = try CorrelatedBathFitter.fitBathCorrelation(
			times: times,
			values: times.map(expected.bathCorrelation(at:)),
			weights: weights,
			options: fittingOptions(
				maximumPencilPoleCount: 4,
				latentBathCount: nil,
				targetRelativeRMSError: 2e-6
			)
		)

		#expect(result.model.latentBaths.count == 1)
		#expect(result.model.poleCount == 2)
		#expect(result.diagnostics.relativeRMSError < 2e-6)
		#expect(result.diagnostics.finalPoleCount < 3)

		for lag in stride(from: 0.0, through: 4.0, by: 0.11) {
			expectMatricesClose(
				result.model.bathCorrelation(at: lag),
				expected.bathCorrelation(at: lag),
				tolerance: 2e-5
			)
		}
	}

	@Test("Backward elimination removes a negligible hierarchy direction")
	func prunesWeakPole() throws {
		let expected = CorrelatedBathModel(
			channelCount: 1,
			latentBaths: [
				CorrelatedBathModel.LatentBath(
					poles: [Complex(0.32, 0.45), Complex(1.35, -0.30)],
					residues: Matrix(
						elements: [
							Complex(1.1, 0.2), Complex(8e-5, -5e-5),
						],
						rows: 1,
						columns: 2
					)
				)
			]
		)
		let times = (0..<48).map { Double($0) * 0.09 }

		let result = try CorrelatedBathFitter.fitBathCorrelation(
			times: times,
			values: times.map(expected.bathCorrelation(at:)),
			options: fittingOptions(
				maximumPencilPoleCount: 4,
				latentBathCount: 1,
				targetRelativeRMSError: 5e-4
			)
		)

		#expect(result.diagnostics.initialPoleCount >= 2)
		#expect(result.model.poleCount == 1)
		#expect(result.diagnostics.acceptedPoleRemovals >= 1)
		#expect(result.diagnostics.relativeRMSError < 5e-4)
	}

	@Test("Spectral-density samples are fitted through H H-dagger")
	func fitSpectralDensity() throws {
		let expected = CorrelatedBathModel(
			channelCount: 2,
			latentBaths: [
				CorrelatedBathModel.LatentBath(
					poles: [Complex(0.42, 0.85)],
					residues: Matrix(
						elements: [Complex(0.9, 0.2), Complex(-0.35, 0.7)],
						rows: 2,
						columns: 1
					)
				)
			]
		)
		let frequencies = (0..<101).map { -5.0 + 0.1 * Double($0) }

		let result = try CorrelatedBathFitter.fitSpectralDensity(
			frequencies: frequencies,
			values: frequencies.map(expected.spectralDensity(at:)),
			options: fittingOptions(
				maximumPencilPoleCount: 2,
				latentBathCount: nil,
				targetRelativeRMSError: 2e-6
			)
		)

		#expect(result.model.latentBaths.count == 1)
		#expect(result.model.poleCount == 1)
		#expect(result.diagnostics.relativeRMSError < 2e-6)

		for frequency in stride(from: -5.0, through: 5.0, by: 0.17) {
			let fitted = result.model.spectralDensity(at: frequency)
			expectMatricesClose(
				fitted,
				expected.spectralDensity(at: frequency),
				tolerance: 2e-5
			)
			let eigenvalues = try MatrixOperations.eigenValuesHermitian(fitted)
			#expect((eigenvalues.first ?? 0) > -1e-10)
		}
	}

	@Test("Spectral rank infers two independent latent baths")
	func infersMultipleLatentBaths() throws {
		let expected = CorrelatedBathModel(
			channelCount: 2,
			latentBaths: [
				CorrelatedBathModel.LatentBath(
					poles: [Complex(0.34, 0.72)],
					residues: Matrix(
						elements: [
							Complex(0.95, 0.1), Complex(0.18, -0.32),
						],
						rows: 2,
						columns: 1
					)
				),
				CorrelatedBathModel.LatentBath(
					poles: [Complex(0.58, -0.63)],
					residues: Matrix(
						elements: [
							Complex(-0.28, 0.41), Complex(0.76, 0.17),
						],
						rows: 2,
						columns: 1
					)
				),
			]
		)
		let frequencies = (0..<121).map { -6.0 + 0.1 * Double($0) }

		let result = try CorrelatedBathFitter.fitSpectralDensity(
			frequencies: frequencies,
			values: frequencies.map(expected.spectralDensity(at:)),
			options: fittingOptions(
				maximumPencilPoleCount: 2,
				latentBathCount: nil,
				targetRelativeRMSError: 5e-6
			)
		)

		#expect(result.model.latentBaths.count == 2)
		#expect(result.model.poleCount == 2)
		#expect(result.diagnostics.relativeRMSError < 5e-6)
	}
}

private var twoPoleCorrelatedModel: CorrelatedBathModel {
	CorrelatedBathModel(
		channelCount: 2,
		latentBaths: [
			CorrelatedBathModel.LatentBath(
				poles: [Complex(0.28, 0.62), Complex(0.91, -0.48)],
				residues: Matrix(
					elements: [
						Complex(1.05, 0.18), Complex(-0.42, 0.55),
						Complex(0.31, -0.64), Complex(0.88, 0.12),
					],
					rows: 2,
					columns: 2
				)
			)
		]
	)
}

private func fittingOptions(
	maximumPencilPoleCount: Int,
	latentBathCount: Int?,
	targetRelativeRMSError: Double
) -> CorrelatedBathFitter.Options {
	CorrelatedBathFitter.Options(
		maximumPencilPoleCount: maximumPencilPoleCount,
		minimumPoleCount: 1,
		latentBathCount: latentBathCount,
		targetRelativeRMSError: targetRelativeRMSError,
		maximumRelativeRMSEIncrease: 0.02,
		minimumDamping: 1e-7,
		matrixPencilResidueRankTolerance: 1e-8,
		matrixPencilHankelRankTolerance: 1e-8,
		duplicatePoleTolerance: 1e-7,
		physicalityTolerance: 1e-9,
		functionTolerance: 1e-11,
		stepTolerance: 1e-11,
		gradientTolerance: 1e-11,
		maximumFunctionEvaluations: 12_000,
		maximumPruningCandidatesPerStep: 4
	)
}

private func expectMatricesClose(
	_ lhs: Matrix<Complex<Double>>,
	_ rhs: Matrix<Complex<Double>>,
	tolerance: Double,
	sourceLocation: SourceLocation = #_sourceLocation
) {
	#expect(lhs.rows == rhs.rows, sourceLocation: sourceLocation)
	#expect(lhs.columns == rhs.columns, sourceLocation: sourceLocation)
	guard lhs.rows == rhs.rows, lhs.columns == rhs.columns else { return }
	for element in lhs.elements.indices {
		#expect(
			(lhs.elements[element] - rhs.elements[element]).length < tolerance,
			sourceLocation: sourceLocation
		)
	}
}
