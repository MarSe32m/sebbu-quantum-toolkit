// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS physical validation", .serialized)
struct HOPSPhysicalValidationTests {
    //TODO: Find a better way to express these arguments... Can we use enums?
	@Test(
		"Single, independent, shared and complex multipole baths reproduce analytic pure dephasing",
		arguments: 0..<30)
	func dephasing(caseIndex: Int) throws {
		let kind = caseIndex / 6
		let variant = caseIndex % 6
		let w = Complex(0.8, 0.9)
		let model: CorrelatedBathModel
		let diagonal: [[Double]]
		if kind == 0 || kind == 4 {
			model = .init(channelCount: 1, latentBaths: [noiseTestBath([w], [[1]])])
			diagonal = [[0, 1]]
		} else {
			diagonal = [[0, 0, 1, 1], [0, 1, 0, 1]]
			if kind == 1 {
				model = .init(
					channelCount: 2,
					latentBaths: [
						noiseTestBath([w], [[1], [0]]),
						noiseTestBath([w], [[0], [Complex(0.8)]]),
					])
			} else if kind == 2 {
				model = .init(
					channelCount: 2,
					latentBaths: [noiseTestBath([w], [[1], [Complex(0.8)]])])
			} else {
				model = hopsFixtureModel()
			}
		}
		let d = diagonal[0].count
		let operators = diagonal.map { values in
			Matrix<Complex<Double>>(
				elements: (0..<(d * d)).map { i in
					i / d == i % d ? Complex(values[i / d]) : .zero
				}, rows: d, columns: d)
		}
		let zero = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
		let initial = Array(repeating: Complex(1 / Double(d).squareRoot()), count: d)
		let markovianRate = kind == 4 ? 0.6 : 0
		let problem = hopsProblem(
			zero, initial: initial,
			markovian: kind == 4
				? [.init(rate: markovianRate, collapseOperator: operators[0])] : [])
		let t = 0.5
		let config = hopsConfiguration(
			variant, model: model, operators: operators, depth: 5, noiseStep: 0.01)
		let samples = 512
		let moments = Mutex(
			(
				sum: Array(repeating: Complex<Double>.zero, count: d * d),
				squared: Array(repeating: 0.0, count: d * d)
			))
		try HOPS.solveTrajectories(
			problem: problem, configuration: config,
			propagation: hopsPropagation(
				end: t, maximumStep: kind == 4 ? 0.002 : 0.04, tolerance: 3e-6),
			execution: .init(
				trajectories: samples, seed: 0x941_2026,
				parallelism: .maximumConcurrentTasks(2))
		) { _, _, state in
			let norm = variant < 2 ? 1 : state.normSquared
			moments.withLock { m in
				for i in 0..<d {
					for j in 0..<d {
						let value = state[i] * state[j].conjugate / norm
						m.sum[i * d + j] += value
						m.squared[i * d + j] += value.lengthSquared
					}
				}
			}
		}
		// F_ij(t) = integral_0^t (t-s) alpha_ij(s) ds, evaluated analytically.
		var f = Matrix<Complex<Double>>.zeros(
			rows: model.channelCount, columns: model.channelCount)
		for term in model.oneSidedExponentialTerms {
			let pole = term.pole
			let integrated: Complex<Double> =
				Complex(t) / pole
				- (Complex<Double>.one - Complex<Double>.exp(-pole * t))
				/ (pole * pole)
			f.add(term.residue, multiplied: integrated)
		}
		let m = moments.withLock { $0 }
		for a in 0..<d {
			for b in 0..<d {
				var exponent = Complex<Double>.zero
				for i in 0..<model.channelCount {
					for j in 0..<model.channelCount {
						exponent -=
							diagonal[i][a] * diagonal[j][a] * f[i, j]
						exponent -=
							diagonal[i][b] * diagonal[j][b]
							* f[i, j].conjugate
						exponent +=
							diagonal[i][b] * diagonal[j][a]
							* (f[i, j] + f[j, i].conjugate)
					}
				}
				let difference = diagonal[0][a] - diagonal[0][b]
				exponent -= Complex(
					0.5 * markovianRate * t * difference * difference)
				let expected = Complex<Double>.exp(exponent) / Double(d)
				let index = a * d + b
				let mean = m.sum[index] / Double(samples)
				let variance = max(
					0,
					(m.squared[index] - Double(samples) * mean.lengthSquared)
						/ Double(samples - 1))
				let bound = 6 * (variance / Double(samples)).squareRoot() + 0.0015
				#expect(
					(mean - expected).length < bound,
					"variant \(variant), bath \(kind), rho[\(a),\(b)]: \(mean) vs \(expected), bound \(bound)"
				)
			}
		}
	}

	@Test("Combined colored memory and Markovian decay match an exact survival amplitude")
	func hybridDamping() throws {
		let w = Complex(0.8, 0.6)
		let g = 0.45
		let rate = 0.4
		let model = CorrelatedBathModel(
			channelCount: 1,
			latentBaths: [
				noiseTestBath([w], [[Complex((2 * w.real * g).squareRoot())]])
			])
		let l = hopsMatrix([0, 1, 0, 0])
		let config = hopsConfiguration(
			0, model: model, operators: [l], depth: 1, noiseStep: 0.002)
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, 0]), initial: [0, 1],
			markovian: [.init(rate: rate, collapseOperator: l)])
		let t = 0.8
		let a: Complex<Double> = w - Complex(rate / 2)
		let disc = Complex<Double>.sqrt(a * a - Complex(4 * g))
		let expected =
			Complex<Double>.exp(-0.5 * (w + Complex(rate / 2)) * t)
			* (Complex<Double>.cosh(0.5 * disc * t) + a / disc * .sinh(0.5 * disc * t))
		for id in 0..<3 {
			let result = try hopsFinalState(
				problem: problem, configuration: config,
				propagation: hopsPropagation(end: t, maximumStep: 0.001),
				id: UInt64(id))
			#expect((result[1] - expected).length < 3e-7)
		}
	}

	@Test(
		"Adaptive retries preserve one noise path and convergence is independent of the output grid"
	)
	func adaptiveNoiseReplay() throws {
		let config = hopsConfiguration(
			5, model: noiseTestModel(0), operators: [hopsMatrix([1, 0, 0, -1])],
			depth: 5, noiseStep: 0.05)
		let h = hopsMatrix([0, 20, 20, 0])
		let problem = hopsProblem(h)
		let evaluations = Mutex((previous: -Double.infinity, rewound: false))
		let tracedSystem = QuantumSystem(dimension: 2) { t, into in
			evaluations.withLock {
				if t < $0.previous - 1e-12 { $0.rewound = true }
				$0.previous = t
			}
			into.copyElements(from: h)
		}
		let tracedProblem = PureStateProblem(
			initialState: problem.initialState, system: tracedSystem)
		let fine = try hopsFinalState(
			problem: problem, configuration: config,
			propagation: hopsPropagation(
				end: 0.25, maximumStep: 0.0025, tolerance: 1e-11))
		let retried = try hopsFinalState(
			problem: tracedProblem, configuration: config,
			propagation: hopsPropagation(end: 0.25, maximumStep: 0.25, tolerance: 1e-10)
		)
		#expect(
			evaluations.withLock { $0.rewound },
			"The large initial trial must exercise OU backtracking")
		let observed = try hopsFinalState(
			problem: problem, configuration: config,
			propagation: hopsPropagation(
				end: 0.25, maximumStep: 0.25, tolerance: 1e-10,
				output: .times([0, 0.017, 0.111, 0.197, 0.25])))
		expectHOPSClose(hopsProjector(fine), hopsProjector(retried), tolerance: 4e-7)
		expectHOPSClose(hopsProjector(fine), hopsProjector(observed), tolerance: 4e-7)
	}
}
