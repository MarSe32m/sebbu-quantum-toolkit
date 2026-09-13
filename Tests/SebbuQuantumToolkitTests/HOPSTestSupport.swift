// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

let hopsEquationTypes: [HOPS.EquationType] = [.linear, .nonLinear, .nonLinearNormalized]

func hopsConfiguration(
	_ variant: Int, model: CorrelatedBathModel, operators: [Matrix<Complex<Double>>],
	depth: Int = 3, noiseStep: Double = 0.01
) -> HOPS.Configuration {
	.init(
		hierarchy: .init(
			environment: .init(
				couplingOperators: operators.map { .constant($0) }, bath: model),
			truncation: .maximumTier(depth)),
		equationType: hopsEquationTypes[variant / 2],
		shiftType: variant % 2 == 0 ? .none : .meanField,
		noiseStepSize: noiseStep)
}

func hopsPropagation(
	end: Double = 0.3, start: Double = 0, maximumStep: Double = 0.02,
	tolerance: Double = 1e-8, output: OutputSchedule = .final
) -> PropagationOptions<IntegrationOptions> {
	.init(
		timeSpan: .init(start: start, end: end), output: output,
		integration: .init(
			minimumStepSize: 0, maximumStepSize: maximumStep,
			absoluteTolerance: tolerance, relativeTolerance: tolerance))
}

func hopsMatrix(_ elements: [Complex<Double>], _ dimension: Int = 2) -> Matrix<Complex<Double>> {
	.init(elements: elements, rows: dimension, columns: dimension)
}

func hopsProblem(
	_ h: Matrix<Complex<Double>> = hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)]),
	initial: [Complex<Double>] = [Complex(0.6), Complex(0, 0.8)],
	markovian: [MarkovianChannel] = []
) -> PureStateProblem<ConstantHamiltonian> {
	.init(
		initialState: Vector<Complex<Double>>(initial), system: .init(h),
		markovianChannels: markovian)
}

func hopsFinalState<Hamiltonian>(
	problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
	propagation: PropagationOptions<IntegrationOptions> = hopsPropagation(),
	seed: UInt64 = 71, id: UInt64 = 3
) throws -> [Complex<Double>] {
	var result: [Complex<Double>] = []
	try HOPS.solveTrajectory(
		problem: problem, configuration: configuration, propagation: propagation,
		seed: seed, trajectoryID: id
	) { _, state in
		result = (0..<state.count).map { state[$0] }
		return .proceed
	}
	return result
}

func hopsProjector(_ state: [Complex<Double>], normalized: Bool = true) -> [Complex<Double>] {
	let norm = normalized ? state.reduce(0.0) { $0 + $1.lengthSquared } : 1
	return state.flatMap { a in state.map { a * $0.conjugate / norm } }
}

func expectHOPSClose(
	_ lhs: [Complex<Double>], _ rhs: [Complex<Double>], tolerance: Double = 1e-11,
	sourceLocation: SourceLocation = #_sourceLocation
) {
	#expect(lhs.count == rhs.count, sourceLocation: sourceLocation)
	for i in lhs.indices {
		#expect(
			(lhs[i] - rhs[i]).length <= tolerance,
			"component \(i): \(lhs[i]) vs \(rhs[i])", sourceLocation: sourceLocation)
	}
}

func hopsOccupations(_ hierarchy: HOPS.Hierarchy) -> [[Int]] {
	(0..<hierarchy.count).map { h in
		var result: [Int] = []
		hierarchy.multiIndex(of: h) { s in for p in 0..<s.count { result.append(s[p]) } }
		return result
	}
}

func hopsFixtureModel() -> CorrelatedBathModel {
	.init(
		channelCount: 2,
		latentBaths: [
			noiseTestBath(
				[Complex(0.7, 1.1), Complex(1.3, -0.4)],
				[
					[Complex(0.8, 0.2), Complex(-0.2, 0.4)],
					[Complex(0.3, -0.5), Complex(0.6, 0.1)],
				]),
			noiseTestBath(
				[Complex(0.9, 0.3)], [[Complex(0.2, -0.1)], [Complex(-0.4, 0.3)]]),
		])
}

let hopsFixtureOperators = [
	hopsMatrix([Complex(0.3, 0.2), Complex(0.8, -0.1), Complex(-0.4, 0.3), Complex(0.6, 0.1)]),
	hopsMatrix([Complex(-0.2, 0.1), Complex(0.1, 0.5), Complex(0.7, -0.2), Complex(0.1, -0.3)]),
]

// Literal latent-basis reference. This deliberately constructs Lambda_p and M_p
// and applies each operator to each neighbour separately, unlike the CPU kernel.
func hopsReferenceDrift(
	configuration: HOPS.Configuration, hamiltonian: Matrix<Complex<Double>>,
	operators: [Matrix<Complex<Double>>], noise: [Complex<Double>],
	state: [Complex<Double>], shifts: [Complex<Double>],
	markovian: [(Double, Matrix<Complex<Double>>)] = []
) -> ([Complex<Double>], [Complex<Double>]) {
	let model = configuration.hierarchy.environment.bath
	let d = hamiltonian.rows
	let ns = hopsOccupations(configuration.hierarchy)
	let ids = Dictionary(uniqueKeysWithValues: ns.enumerated().map { ($0.element, $0.offset) })
	let poles = model.latentBaths.flatMap(\.poles)
	var lambda = Array(
		repeating: Array(repeating: Complex<Double>.zero, count: d * d), count: poles.count)
	var memory = lambda
	var offset = 0
	for bath in model.latentBaths {
		for p in 0..<bath.poleCount {
			for i in operators.indices {
				for j in 0..<(d * d) {
					lambda[offset + p][j] +=
						bath.residues[i, p].conjugate
						* operators[i].elements[j]
				}
			}
		}
		for p in 0..<bath.poleCount {
			for q in 0..<bath.poleCount {
				let k =
					Complex<Double>.one
					/ (bath.poles[p] + bath.poles[q].conjugate)
				for j in 0..<(d * d) {
					memory[offset + p][j] += k * lambda[offset + q][j]
				}
			}
		}
		offset += bath.poleCount
	}
	func apply(_ op: [Complex<Double>], _ h: Int, adjoint: Bool = false) -> [Complex<Double>] {
		(0..<d).map { i in
			(0..<d).reduce(Complex<Double>.zero) { sum, j in
				sum + (adjoint ? op[j * d + i].conjugate : op[i * d + j])
					* state[h * d + j]
			}
		}
	}
	let norm = state.prefix(d).reduce(0.0) { $0 + $1.lengthSquared }
	func expectation(_ op: [Complex<Double>]) -> Complex<Double> {
		let applied = apply(op, 0)
		return (0..<d).reduce(Complex<Double>.zero) {
			$0 + state[$1].conjugate * applied[$1]
		} / norm
	}
	let meanLambda = lambda.map(expectation)
	let meanMemory = memory.map(expectation)
	let nonlinear = configuration.equationType != .linear
	let displaced = configuration.shiftType == .meanField
	var result = Array(repeating: Complex<Double>.zero, count: state.count)
	for h in ns.indices {
		let hy = apply(hamiltonian.elements, h)
		for j in 0..<d { result[h * d + j] -= .i * hy[j] }
		for i in operators.indices {
			let ly = apply(operators[i].elements, h)
			for j in 0..<d { result[h * d + j] += noise[i].conjugate * ly[j] }
		}
		for p in poles.indices {
			let ly = apply(lambda[p], h)
			let adjoint = apply(lambda[p], h, adjoint: true)
			for j in 0..<d {
				result[h * d + j] -= Double(ns[h][p]) * poles[p] * state[h * d + j]
				if nonlinear { result[h * d + j] += shifts[p].conjugate * ly[j] }
				if displaced {
					result[h * d + j] -=
						shifts[p]
						* (adjoint[j]
							- (nonlinear
								? meanLambda[p].conjugate
									* state[h * d + j] : .zero))
				}
			}
			var n = ns[h]
			n[p] -= 1
			if let parent = ids[n] {
				let down = apply(memory[p], parent)
				for j in 0..<d {
					result[h * d + j] +=
						Double(ns[h][p]).squareRoot()
						* (down[j]
							- (displaced
								? meanMemory[p]
									* state[parent * d + j]
								: .zero))
				}
			}
			n = ns[h]
			n[p] += 1
			if let child = ids[n] {
				let up = apply(lambda[p], child, adjoint: true)
				for j in 0..<d {
					result[h * d + j] -=
						Double(ns[h][p] + 1).squareRoot()
						* (up[j]
							- (nonlinear
								? meanLambda[p].conjugate
									* state[child * d + j]
								: .zero))
				}
			}
		}
		for (rate, c) in markovian {
			let loss = c.conjugateTranspose.dot(c)
			let mean = expectation(c.elements)
			let lossMean = expectation(loss.elements).real
			let cy = apply(c.elements, h)
			let lossY = apply(loss.elements, h)
			for j in 0..<d {
				result[h * d + j] -= 0.5 * rate * lossY[j]
				if nonlinear { result[h * d + j] += rate * mean.conjugate * cy[j] }
				if configuration.equationType == .nonLinearNormalized {
					result[h * d + j] +=
						rate * (0.5 * lossMean - mean.lengthSquared)
						* state[h * d + j]
				}
			}
		}
	}
	if configuration.equationType == .nonLinearNormalized {
		let gamma =
			(0..<d).reduce(Complex<Double>.zero) {
				$0 + state[$1].conjugate * result[$1]
			}.real / norm
		for i in result.indices { result[i] -= gamma * state[i] }
	}
	let shiftDerivative = shifts.indices.map { -poles[$0] * shifts[$0] + meanMemory[$0] }
	return (result, shiftDerivative)
}
