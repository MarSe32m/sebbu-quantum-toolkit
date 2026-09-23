// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS sparse physical bath operators")
struct HOPSSparseOperatorTests {
	@Test("CSR operator kernels agree with dense complex non-Hermitian kernels")
	func csrKernels() {
		let dense = sparseTestMatrix(7)
		let matrix = UniqueMatrix(copying: dense)
		let sparse = UniqueCSRMatrix<Complex<Double>>(from: matrix)
		let adjoint = sparse.conjugateTranspose
		var x = UniqueVector<Complex<Double>>.zero(7)
		var denseOutput = UniqueVector<Complex<Double>>.zero(7)
		var sparseOutput = UniqueVector<Complex<Double>>.zero(7)
		for i in 0..<7 {
			x[i] = Complex(0.17 * Double(i + 1), -0.11 * Double(2 * i + 1))
			denseOutput[i] = Complex(-0.3, 0.2)
			sparseOutput[i] = denseOutput[i]
		}
		let coefficient = Complex(0.4, -0.7)

        HOPS.CPUEngine.OperatorApplication.vector(
			matrix, x: x.components, y: denseOutput.components,
			coefficient: coefficient, adding: false)
        HOPS.CPUEngine.OperatorApplication.vector(
			sparse, x: x.components, y: sparseOutput.components,
			coefficient: coefficient, adding: false)
		expectHOPSClose(
			(0..<7).map { denseOutput[$0] },
			(0..<7).map { sparseOutput[$0] }, tolerance: 3e-12)

		for i in 0..<7 {
			denseOutput[i] = Complex(-0.3, 0.2)
			sparseOutput[i] = denseOutput[i]
		}
        HOPS.CPUEngine.OperatorApplication.vector(
			matrix, adjoint: true,
			x: x.components, y: denseOutput.components,
			coefficient: -coefficient, adding: true)
        HOPS.CPUEngine.OperatorApplication.vector(
			adjoint,
			x: x.components, y: sparseOutput.components,
			coefficient: -coefficient, adding: true)
		expectHOPSClose(
			(0..<7).map { denseOutput[$0] },
			(0..<7).map { sparseOutput[$0] }, tolerance: 3e-12)
	}

	@Test("Sparse expectation agrees with dense expectation")
	func sparseExpectation() {
		for d in [3, 6, 9] {
			let dense = sparseTestMatrix(d)
			let matrix = UniqueMatrix(copying: dense)
			let sparse = UniqueCSRMatrix<Complex<Double>>(from: matrix)
			var state = UniqueVector<Complex<Double>>.zero(d)
			for i in 0..<d {
				state[i] = Complex(
					0.2 + 0.03 * Double(i),
					-0.1 + 0.02 * Double(i))
			}
			var expected = Complex<Double>.zero
			for i in 0..<d {
				var row = Complex<Double>.zero
				for j in 0..<d {
					row += matrix[i, j] * state[j]
				}
				expected += state[i].conjugate * row
			}
			let actual = HCT.OperatorApplication.expectation(
				sparse, state: state.components)
			#expect((actual - expected).length < 3e-12)
		}
	}

	@Test("Sparse generator accumulation agrees with dense aL+bLdagger")
	func sparseGeneratorAccumulation() {
		let d = 8
		let op = sparseTestMatrix(d)
		let sparse = UniqueCSRMatrix<Complex<Double>>(
			from: UniqueMatrix(copying: op))
		var expected = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
		var actual = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
		for i in 0..<d {
			for j in 0..<d {
				let seed = Complex(
					0.01 * Double(1 + 2 * i + j),
					-0.02 * Double(1 + i + 3 * j))
				expected[i, j] = seed
				actual[i, j] = seed
			}
		}
		let a = Complex(0.31, -0.27)
		let b = Complex(-0.19, 0.42)
		for i in 0..<d {
			for j in 0..<d {
				expected[i, j] +=
					a * op[i, j] + b * op[j, i].conjugate
			}
		}
		HCT.OperatorApplication.addSparseBathContributions(
			sparse,
			forwardCoefficient: a,
			adjointCoefficient: b,
			into: &actual)
		for i in 0..<d {
			for j in 0..<d {
				#expect((actual[i, j] - expected[i, j]).length < 3e-12)
			}
		}
	}

	@Test("Automatic bath storage selection is localized and conservative")
	func automaticSelection() throws {
		let d = 8
		let denseOperator = Matrix<Complex<Double>>(
			elements: (0..<(d * d)).map {
				Complex(0.01 * Double($0 + 1), 0.02)
			},
			rows: d, columns: d)
		let sparseOperator = sparseTestMatrix(d)

		let densePrep = try preparation(
			dimension: d, operators: [denseOperator], policy: .automatic)
        var isSparse = densePrep.bathChannels[0].op.isSparse
		#expect(!isSparse)

		let sparsePrep = try preparation(
			dimension: d, operators: [sparseOperator], policy: .automatic)
        isSparse = sparsePrep.bathChannels[0].op.isSparse
		#expect(isSparse)

		let tls = hopsMatrix([1, 0, 0, 1])
		let tlsPrep = try preparation(
			dimension: 2, operators: [tls], policy: .sparse)
        #expect({!tlsPrep.bathChannels[0].op.isSparse}())
	}

	@Test(
		"Forced sparse and dense preparations produce identical complete RHS",
		arguments: 0..<6)
	func pathwiseRHSEquivalence(variant: Int) throws {
		let d = 6
		let operators = [sparseTestMatrix(d), sparseTestMatrix(d, offset: 2)]
		let model = hopsFixtureModel()
		var denseConfiguration = hopsConfiguration(
			variant, model: model, operators: operators, depth: 2)
		var sparseConfiguration = denseConfiguration
		denseConfiguration.bathOperatorStoragePolicy = .dense
		sparseConfiguration.bathOperatorStoragePolicy = .sparse
		let h = diagonalHamiltonian(d)
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>(
				(0..<d).map { _ in
					Complex(1 / Double(d).squareRoot(), 0)
				}),
			system: QuantumSystem(h))
		let propagation = hopsPropagation()
		let densePreparation = try HCT.Preparation(
			problem: problem,
			configuration: denseConfiguration,
			propagation: propagation)
		let sparsePreparation = try HCT.Preparation(
			problem: problem,
			configuration: sparseConfiguration,
			propagation: propagation)
		for i in 0..<sparsePreparation.bathChannels.count {
            #expect({sparsePreparation.bathChannels[i].op.isSparse}())
		}

		let hierarchyCount = denseConfiguration.hierarchy.count
		let branches = 3
		var state = HCT.State(
			dimension: d,
			hierarchyCount: branches * hierarchyCount,
			shiftCount: densePreparation.shiftCount)
		var denseDerivative = HCT.State(
			dimension: d,
			hierarchyCount: branches * hierarchyCount,
			shiftCount: densePreparation.shiftCount)
		var sparseDerivative = HCT.State(
			dimension: d,
			hierarchyCount: branches * hierarchyCount,
			shiftCount: sparsePreparation.shiftCount)
		for i in 0..<(hierarchyCount * d) {
			let value = Complex(
				0.15 + 0.003 * Double(i),
				-0.07 + 0.002 * Double(i))
			for branch in 0..<branches {
				state.amplitudes.elements[
					branch * hierarchyCount * d + i] =
					Complex(1 - 0.2 * Double(branch), 0.1 * Double(branch))
					* value
			}
		}
		for i in 0..<state.shifts.count {
			state.shifts[i] = Complex(
				0.01 * Double(i + 1), -0.015 * Double(i + 1))
		}

		var denseRHS = HCT.RightHandSide(
			hamiltonian: problem.system.hamiltonian,
			preparation: densePreparation,
			seed: 17, trajectoryID: 4)
		var sparseRHS = HCT.RightHandSide(
			hamiltonian: problem.system.hamiltonian,
			preparation: sparsePreparation,
			seed: 17, trajectoryID: 4)
		let physicalNoise = [
			Complex(0.22, -0.31),
			Complex(-0.16, 0.27),
		]
		for i in physicalNoise.indices {
			denseRHS.physicalNoise[i] = physicalNoise[i]
			sparseRHS.physicalNoise[i] = physicalNoise[i]
		}
		denseRHS.evaluateWithCurrentNoise(
			t: 0.13, y: state, into: &denseDerivative)
		sparseRHS.evaluateWithCurrentNoise(
			t: 0.13, y: state, into: &sparseDerivative)

		expectHOPSClose(
			(0..<(branches * hierarchyCount * d)).map {
				denseDerivative.amplitudes.elements[$0]
			},
			(0..<(branches * hierarchyCount * d)).map {
				sparseDerivative.amplitudes.elements[$0]
			},
			tolerance: 5e-12)
		expectHOPSClose(
			(0..<denseDerivative.shifts.count).map {
				denseDerivative.shifts[$0]
			},
			(0..<sparseDerivative.shifts.count).map {
				sparseDerivative.shifts[$0]
			},
			tolerance: 5e-12)
	}

	@Test("Time-dependent bath operators remain on the dense dynamic path")
	func dynamicBathRegression() throws {
		let d = 6
		let op = sparseTestMatrix(d)
		let model = CorrelatedBathModel(
			channelCount: 1,
			latentBaths: [
				.init(
					poles: [Complex(0.8, 0.3)],
					residues: Matrix(
						elements: [Complex(0.4, -0.1)],
						rows: 1, columns: 1))
			])
		var configuration = HOPS.Configuration(
			hierarchy: .init(
				environment: .init(
					couplingOperator: .generatedDense(
						.init { t, output in
							output.copyElements(
								from: op,
								multiplied: Complex(1 + t, 0))
						}),
					bath: model),
				truncation: .maximumTier(1)),
			equationType: .linear)
		configuration.bathOperatorStoragePolicy = .sparse
		let problem = PureStateProblem(
			initialState: Vector(
				[Complex<Double>](repeating: 1 / Double(d).squareRoot(), count: d)),
			system: QuantumSystem(diagonalHamiltonian(d)))
		let prep = try HCT.Preparation(
			problem: problem,
			configuration: configuration,
			propagation: hopsPropagation())
        #expect({prep.bathChannels[0].op.isDynamic}())
        #expect({!prep.bathChannels[0].op.isSparse}())
	}
}

private func sparseTestMatrix(
	_ d: Int, offset: Int = 0
) -> Matrix<Complex<Double>> {
	var matrix = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
	for i in 0..<d {
		matrix[i, i] = Complex(
			0.13 * Double(i + 1 + offset),
			-0.07 * Double((i + offset) % 3))
		if i + 2 < d {
			matrix[i, i + 2] = Complex(
				-0.09 * Double(i + 1),
				0.04 * Double(i + offset + 1))
		}
		if i > 1 && (i + offset).isMultiple(of: 2) {
			matrix[i, i - 2] = Complex(
				0.05 * Double(i), -0.03 * Double(i + 1))
		}
	}
	// Deliberately leave irregular empty rows where possible.
	if d > 5 {
		for j in 0..<d { matrix[3, j] = .zero }
	}
	return matrix
}

private func diagonalHamiltonian(_ d: Int) -> Matrix<Complex<Double>> {
	var h = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
	for i in 0..<d { h[i, i] = Complex(0.1 * Double(i)) }
	return h
}

private func preparation(
	dimension d: Int,
	operators: [Matrix<Complex<Double>>],
	policy: HOPS.BathOperatorStoragePolicy
) throws -> HCT.Preparation {
	let residues = Matrix<Complex<Double>>(
		elements: [Complex<Double>](repeating: Complex(0.2), count: operators.count),
		rows: operators.count, columns: 1)
	let model = CorrelatedBathModel(
		channelCount: operators.count,
		latentBaths: [
			.init(poles: [Complex(0.7, 0.4)], residues: residues)
		])
	var configuration = HOPS.Configuration(
		hierarchy: .init(
			environment: .init(
				couplingOperators: operators.map { .constant($0) },
				bath: model),
			truncation: .maximumTier(1)),
		equationType: .linear)
	configuration.bathOperatorStoragePolicy = policy
	let problem = PureStateProblem(
		initialState: Vector(
			[Complex<Double>](repeating: 1 / Double(d).squareRoot(), count: d)),
		system: QuantumSystem(diagonalHamiltonian(d)))
	return try HCT.Preparation(
		problem: problem,
		configuration: configuration,
		propagation: hopsPropagation())
}
