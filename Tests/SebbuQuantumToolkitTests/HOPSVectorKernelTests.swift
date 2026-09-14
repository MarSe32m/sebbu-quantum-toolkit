// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS vector kernels", .serialized)
struct HOPSVectorKernelTests {
	@Test(
		"Normal and adjoint actions preserve complex weights, accumulation and inputs",
		arguments: [1, 2, 3, 4, 5, 9, 32])
	func operatorApplications(dimension d: Int) {
		let op = vectorKernelMatrix(d)
		let matrix = UniqueMatrix(copying: op)
		let input = UniqueMatrix<Complex<Double>>(rows: 7, columns: d) { buffer in
			for i in buffer.indices {
				buffer[i] = Complex(0.1 + 0.01 * Double(i), -0.3)
			}
		}
		let coefficient = Complex(-0.7, 0.3)
		let initial = Complex(0.4, -0.8)
		for adding in [false, true] {
			var output = UniqueMatrix<Complex<Double>>.zeros(rows: 7, columns: d)
			for i in 0..<(7 * d) { output.elements[i] = initial }
			HCT.OperatorApplication.apply(
				matrix, to: input, multiplied: coefficient, adding: adding,
				into: &output)
			for row in 0..<7 {
				for i in 0..<d {
					var expected = Complex<Double>.zero
					for j in 0..<d { expected += op[i, j] * input[row, j] }
					expected =
						coefficient * expected + (adding ? initial : .zero)
					#expect((output[row, i] - expected).length < 3e-12)
				}
			}
			for adjoint in [false, true] {
				for i in 0..<(7 * d) { output.elements[i] = initial }
				for row in 0..<7 {
					HCT.OperatorApplication.vector(
						matrix, adjoint: adjoint,
						x: input.elements + row * d,
						y: output.elements + row * d,
						coefficient: coefficient, adding: adding)
					for i in 0..<d {
						var expected = Complex<Double>.zero
						for j in 0..<d {
							expected +=
								(adjoint
									? op[j, i].conjugate
									: op[i, j]) * input[row, j]
						}
						expected =
							coefficient * expected
							+ (adding ? initial : .zero)
						#expect((output[row, i] - expected).length < 3e-12)
					}
				}
			}
		}
		for i in 0..<(7 * d) {
			#expect(input.elements[i] == Complex(0.1 + 0.01 * Double(i), -0.3))
		}
		for i in 0..<d { for j in 0..<d { #expect(matrix[i, j] == op[i, j]) } }
	}

	@Test(
		"Loss columns form L dagger L with ordinary operator storage",
		arguments: [1, 2, 3, 4, 5, 9, 32])
	func lossMatrix(dimension d: Int) {
		let op = vectorKernelMatrix(d)
		let matrix = UniqueMatrix(copying: op)
		var loss = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
		HCT.OperatorApplication.loss(matrix, into: &loss)
		let reference = op.conjugateTranspose.dot(op)
		for i in 0..<d {
			for j in 0..<d { #expect((loss[i, j] - reference[i, j]).length < 3e-12) }
		}
	}

	@Test(
		"Fused hierarchy actions agree with the independent latent equations beyond TLS",
		arguments: 0..<6, [1, 3, 5, 9])
	func hierarchyAgainstReference(variant: Int, dimension d: Int) throws {
		try checkVectorHierarchy(variant: variant, dimension: d, dynamic: false)
	}

	@Test(
		"Generated and expansion channels exercise the larger-system GEMV path",
		arguments: 0..<6)
	func dynamicFiveLevel(variant: Int) throws {
		try checkVectorHierarchy(variant: variant, dimension: 5, dynamic: true)
	}
}

private func vectorKernelMatrix(_ d: Int, offset: Int = 0) -> Matrix<Complex<Double>> {
	var result = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
	for i in 0..<d {
		for j in 0..<d {
			let real: Double = 0.03 * Double((7 * i + 3 * j + offset) % 11 - 5)
			let imaginary: Double = 0.02 * Double((5 * i - 2 * j + offset) % 7)
			result[i, j] = Complex<Double>(real, imaginary)
		}
	}
	return result
}

private func checkVectorHierarchy(variant: Int, dimension d: Int, dynamic: Bool) throws {
	let matrices = [vectorKernelMatrix(d), vectorKernelMatrix(d, offset: 3)]
	let h = Complex<Double>(0.5) * (matrices[0] + matrices[0].conjugateTranspose)
	let time = 0.17
	let factor = Complex(1.17, 0.034)
	let operators: [TimeDependentOperator]
	if dynamic {
		operators = [
			.generatedDense(
				.init { t, output in
					output.copyElements(
						from: matrices[0],
						multiplied: Complex(1 + t, 0.2 * t))
				}),
			.linearCombination(
				.init(
					coefficients: [.generated { Complex(1 + $0, 0.2 * $0) }],
					operators: [.init(matrices[1])])),
		]
	} else {
		operators = matrices.map { .constant($0) }
	}
	let instantaneous = dynamic ? matrices.map { factor * $0 } : matrices
	let config = HOPS.Configuration(
		hierarchy: .init(
			environment: .init(couplingOperators: operators, bath: hopsFixtureModel()),
			truncation: .maximumTier(2)),
		equationType: hopsEquationTypes[variant / 2],
		shiftType: variant % 2 == 0 ? .none : .meanField, noiseStepSize: 0.01)
	let problem = PureStateProblem(
		initialState: Vector<Complex<Double>>(
			[Complex<Double>](repeating: Complex(1 / Double(d).squareRoot()), count: d)),
		system: QuantumSystem(h),
		markovianChannels: [.init(rate: .constant(0.4), collapseOperator: operators[0])])
	let prep = try HCT.Preparation(
		problem: problem, configuration: config, propagation: hopsPropagation())
	let size = config.hierarchy.count * d
	var state = HCT.State(
		dimension: d, hierarchyCount: 3 * config.hierarchy.count,
		shiftCount: prep.shiftCount)
	var derivative = HCT.State(
		dimension: d, hierarchyCount: 3 * config.hierarchy.count,
		shiftCount: prep.shiftCount)
	let values: [Complex<Double>] = (0..<size).map { i in
		let real: Double = 0.2 + 0.007 * Double(i)
		let imaginary: Double = -0.1 + 0.003 * Double(i)
		return Complex<Double>(real, imaginary)
	}
	let shifts = (0..<prep.shiftCount).map { Complex(0.01 * Double($0 + 1), -0.02) }
	let coefficients = [Complex<Double>.one, Complex(0.6, 0.3), Complex(-0.4, 0.5)]
	for branch in 0..<3 {
		for i in 0..<size {
			state.amplitudes.elements[branch * size + i] =
				coefficients[branch] * values[i]
		}
	}
	for i in shifts.indices { state.shifts[i] = shifts[i] }
	let noise = [Complex(0.2, -0.7), Complex(-0.4, 0.3)]
	var rhs = HCT.RightHandSide(
		hamiltonian: problem.system.hamiltonian, preparation: prep, seed: 1,
		trajectoryID: 3)
	for i in noise.indices { rhs.physicalNoise[i] = noise[i] }
	let reference = hopsReferenceDrift(
		configuration: config, hamiltonian: h, operators: instantaneous,
		noise: noise, state: values, shifts: shifts, markovian: [(0.4, instantaneous[0])])
	rhs.evaluateWithCurrentNoise(t: time, y: state, into: &derivative)
	for branch in 0..<3 {
		expectHOPSClose(
			(0..<size).map { derivative.amplitudes.elements[branch * size + $0] },
			reference.0.map { coefficients[branch] * $0 }, tolerance: 3e-12)
	}
	expectHOPSClose(shifts.indices.map { derivative.shifts[$0] }, reference.1, tolerance: 3e-12)
	rhs.diffusion(t: time, y: state, channel: 0, into: &derivative)
	let norm = values.prefix(d).reduce(0.0) { $0 + $1.lengthSquared }
	let rootAction = instantaneous[0].dot(Vector(Array(values.prefix(d))))
	var mean = Complex<Double>.zero
	for i in 0..<d { mean += values[i].conjugate * rootAction[i] / norm }
	for branch in 0..<3 {
		for tier in 0..<config.hierarchy.count {
			for i in 0..<d {
				var expected = Complex<Double>.zero
				for j in 0..<d {
					expected += instantaneous[0][i, j] * values[tier * d + j]
				}
				if variant / 2 == 2 { expected -= mean * values[tier * d + i] }
				expected *= Complex(0.2.squareRoot()) * coefficients[branch]
				#expect(
					(derivative.amplitudes.elements[
						branch * size + tier * d + i] - expected).length
						< 3e-12)
			}
		}
	}
	for i in shifts.indices { #expect(derivative.shifts[i] == .zero) }
}
