// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS trajectory propagation")
struct HOPSTrajectoryTests {
	@Test("Zero bath gives the analytic unitary solution in every variant", arguments: 0..<6)
	func unitary(variant: Int) throws {
		let h = hopsMatrix([0, Complex(0.5), Complex(0.5), 0])
		let problem = hopsProblem(h, initial: [1, 0])
		let config = hopsConfiguration(
			variant, model: .zero(channelCount: 1),
			operators: [hopsMatrix([1, 0, 0, -1])])
		let result = try hopsFinalState(
			problem: problem, configuration: config,
			propagation: hopsPropagation(end: 1.3, maximumStep: 0.4, tolerance: 1e-11))
		expectHOPSClose(
			result, [Complex(Double.cos(0.65)), Complex(0, -Double.sin(0.65))],
			tolerance: 2e-10)
	}

	@Test(
		"Generated Hamiltonians follow an exactly integrable time-dependent drive",
		arguments: 0..<6)
	func generatedHamiltonian(variant: Int) throws {
		let system = QuantumSystem(dimension: 2) { t, into in
			into.zeroElements()
			into[0, 1] = Complex(0.3 + t)
			into[1, 0] = Complex(0.3 + t)
		}
		let problem = PureStateProblem(
			initialState: Vector<Complex<Double>>([1, 0]), system: system)
		let config = hopsConfiguration(
			variant, model: .zero(channelCount: 1),
			operators: [hopsMatrix([1, 0, 0, -1])])
		let t = 1.1
		let angle = 0.3 * t + 0.5 * t * t
		let result = try hopsFinalState(
			problem: problem, configuration: config,
			propagation: hopsPropagation(end: t, maximumStep: 0.3, tolerance: 1e-11))
		expectHOPSClose(
			result, [Complex(Double.cos(angle)), Complex(0, -Double.sin(angle))],
			tolerance: 2e-10)
	}

	@Test(
		"Zero colored bath agrees pathwise with QSD, including dynamic white channels",
		arguments: 0..<6)
	func qsdLimit(variant: Int) throws {
		let c = hopsMatrix([Complex(0.1, 0.2), Complex(0.7), 0, Complex(-0.1, 0.3)])
		let channels: [MarkovianChannel] = [
			.init(rate: 0.4, collapseOperator: c),
			.init(
				rate: .generated { 0.3 + $0 },
				collapseOperator: .generatedDense(
					.init { t, into in
						into.copyElements(
							from: c, multiplied: Complex(0.2 + t, 0.1))
					})),
		]
		let problem = hopsProblem(markovian: channels)
		let config = hopsConfiguration(
			variant, model: .zero(channelCount: 1), operators: [c])
		let propagation = hopsPropagation(maximumStep: 0.003)
		let hops = try hopsFinalState(
			problem: problem, configuration: config, propagation: propagation)
		var qsd: [Complex<Double>] = []
		let types: [QSD.EquationType] = [.linear, .nonLinear, .nonLinearNormalized]
		try QSD.solveTrajectory(
			problem: problem, configuration: .init(equationType: types[variant / 2]),
			propagation: propagation, seed: 71, trajectoryID: 3
		) { _, state in
			qsd = [state[0], state[1]]
			return .proceed
		}
		expectHOPSClose(hops, qsd, tolerance: 3e-12)
	}

	@Test("Constant operator expansions match their generated equivalent", arguments: 0..<6)
	func constantExpansion(variant: Int) throws {
		let l = hopsMatrix([Complex(0.7), Complex(0.2, -0.1), 0, Complex(-0.4)])
		let model = noiseTestModel(0)
		let constant = hopsConfiguration(variant, model: model, operators: [l], depth: 3)
		let expansion = HOPS.Configuration(
			hierarchy: .init(
				environment: .init(
					couplingOperator:
						.linearCombination(
							.init(
								coefficients: [
									.constant(Complex(0.25)),
									.constant(Complex(0.75)),
								], operators: [.init(l), .init(l)])),
					bath: model), truncation: .maximumTier(3)),
			equationType: constant.equationType,
			shiftType: constant.shiftType, noiseStepSize: 0.01)
		let a = try hopsFinalState(problem: hopsProblem(), configuration: constant)
		let b = try hopsFinalState(problem: hopsProblem(), configuration: expansion)
		expectHOPSClose(a, b, tolerance: 3e-10)
	}

	@Test("Linear mean-field displacement is exact at tier zero for a scalar system")
	func scalarLinearDisplacement() throws {
		let w = Complex(0.8, 1.1)
		let r = Complex(0.6, -0.3)
		let l = Complex(0.7, 0.2)
		let model = CorrelatedBathModel(
			channelCount: 1, latentBaths: [noiseTestBath([w], [[r]])])
		let problem = hopsProblem(hopsMatrix([Complex(0.4)], 1), initial: [1])
		let t = 0.8
		let step = 0.005
		let config = hopsConfiguration(
			1, model: model, operators: [hopsMatrix([l], 1)], depth: 0, noiseStep: step)
		let result = try hopsFinalState(
			problem: problem, configuration: config,
			propagation: hopsPropagation(end: t, maximumStep: 0.02, tolerance: 1e-11))
		var rng = TrajectoryRandomNumberGenerator(
			seed: 71, trajectoryID: 3, purpose: .coloredNoiseGeneration)
		var noise = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
			model: model, windowDuration: 0.1,
			step: step, generator: &rng)
		var output = UniqueVector<Complex<Double>>.zero(1)
		noise.sample(0, into: &output.mutableSpan, generator: &rng)
		var previous = output[0]
		var integral = Complex<Double>.zero
		for k in 1...160 {
			noise.sample(Double(k) * step, into: &output.mutableSpan, generator: &rng)
			integral += 0.5 * step * (previous + output[0])
			previous = output[0]
		}
		let g = r.lengthSquared / (2 * w.real)
		let f: Complex<Double> =
			g
			* (Complex(t) / w - (Complex<Double>.one - Complex<Double>.exp(-w * t))
				/ (w * w))
		let expected = Complex<Double>.exp(
			Complex(0, -0.4 * t) + l * integral.conjugate - l.lengthSquared * f)
		expectHOPSClose(result, [expected], tolerance: 3e-8)
	}

	@Test("Single exponential amplitude damping matches its exact survival amplitude")
	func exactDamping() throws {
		let w = Complex(0.9, 0.7)
		let g = 0.6
		let model = CorrelatedBathModel(
			channelCount: 1,
			latentBaths: [
				noiseTestBath([w], [[Complex((2 * w.real * g).squareRoot())]])
			])
		let config = hopsConfiguration(
			0, model: model, operators: [hopsMatrix([0, 1, 0, 0])], depth: 1)
		let problem = hopsProblem(hopsMatrix([0, 0, 0, 0]), initial: [0, 1])
		let t = 1.2
		let discriminant = Complex<Double>.sqrt(w * w - 4 * g)
		let expected =
			Complex<Double>.exp(-0.5 * w * t)
			* (Complex<Double>.cosh(0.5 * discriminant * t) + w / discriminant
				* .sinh(0.5 * discriminant * t))
		for id in 0..<4 {
			let state = try hopsFinalState(
				problem: problem, configuration: config,
				propagation: hopsPropagation(
					end: t, maximumStep: 0.04, tolerance: 1e-10), id: UInt64(id)
			)
			#expect((state[1] - expected).length < 2e-9)
		}
	}

	@Test(
		"Displacement and normalization gauges converge to the same root projectors",
		arguments: [0, 2])
	func gaugeConvergence(baseVariant: Int) throws {
		let model = CorrelatedBathModel(
			channelCount: 1,
			latentBaths: [
				noiseTestBath([Complex(0.8, 0.4)], [[Complex(0.6, 0.2)]])
			])
		let l = hopsMatrix([Complex(0.2), Complex(0.8), 0, Complex(-0.1)])
		let problem = hopsProblem()
		let propagation = hopsPropagation(end: 0.5, maximumStep: 0.02, tolerance: 2e-10)
		let unshifted = try hopsFinalState(
			problem: problem,
			configuration: hopsConfiguration(
				baseVariant, model: model, operators: [l], depth: 7),
			propagation: propagation)
		let shifted = try hopsFinalState(
			problem: problem,
			configuration: hopsConfiguration(
				baseVariant + 1, model: model, operators: [l], depth: 7),
			propagation: propagation)
		expectHOPSClose(
			hopsProjector(unshifted, normalized: false),
			hopsProjector(shifted, normalized: false), tolerance: 2e-7)
		if baseVariant == 2 {
			let normalized = try hopsFinalState(
				problem: problem,
				configuration: hopsConfiguration(
					5, model: model, operators: [l], depth: 7),
				propagation: propagation)
			expectHOPSClose(
				hopsProjector(shifted), hopsProjector(normalized), tolerance: 3e-7)
		}
	}

	@Test(
		"Continuously normalized propagation keeps the root norm with colored and white noise",
		arguments: [4, 5])
	func normalizedNorm(variant: Int) throws {
		let l = hopsMatrix([Complex(0.2), Complex(0.7), 0, Complex(-0.3)])
		let config = hopsConfiguration(
			variant, model: noiseTestModel(0), operators: [l], depth: 5)
		let problem = hopsProblem(
			initial: [Complex(1.2), Complex(0, 1.6)],
			markovian: [.init(rate: 0.6, collapseOperator: l)])
		var samples = 0
		try HOPS.solveTrajectory(
			problem: problem, configuration: config,
			propagation: hopsPropagation(
				end: 0.4, maximumStep: 0.002, output: .uniform(step: 0.01)),
			seed: 12, trajectoryID: 5
		) { _, state in
			#expect(abs(state.normSquared - 1) < 2e-14)
			samples += 1
			return .proceed
		}
		#expect(samples == 41)
	}

	@Test(
		"Full hierarchy observations expose the same physical state and initialize auxiliaries to zero"
	)
	func hierarchyView() throws {
		let config = hopsConfiguration(
			5, model: noiseTestModel(0),
			operators: [hopsMatrix([Complex(0.3), Complex(0.8), 0, Complex(-0.2)])])
		let problem = hopsProblem()
		let propagation = hopsPropagation(output: .times([0, 0.3]))
		let expected = try hopsFinalState(
			problem: problem, configuration: config, propagation: propagation)
		var final: [Complex<Double>] = []
		try HOPS.solveWithHierarchy(
			problem: problem, configuration: config, propagation: propagation,
			seed: 71, trajectoryID: 3
		) { t, hierarchy in
			#expect(hierarchy.count == config.hierarchy.count)
			for h in 0..<hierarchy.count {
				hierarchy.withState(at: h) { state in
					if t == 0 && h > 0 {
						#expect(state[0] == .zero && state[1] == .zero)
					}
				}
			}
			hierarchy.withPhysicalState { state in final = [state[0], state[1]] }
		}
		expectHOPSClose(final, expected)
	}

	@Test("Output boundaries, translated starts, initial stops and zero-duration runs")
	func schedules() throws {
		let problem = hopsProblem()
		let config = hopsConfiguration(
			0, model: .zero(channelCount: 1), operators: [hopsMatrix([1, 0, 0, -1])])
		for withWhite in [false, true] {
			let p =
				withWhite
				? hopsProblem(markovian: [
					.init(rate: 0, collapseOperator: hopsMatrix([0, 1, 0, 0]))
				]) : problem
			let times = [2.0, 2.013, 2.07, 2.11]
			var actual: [Double] = []
			let summary = try HOPS.solveTrajectory(
				problem: p, configuration: config,
				propagation: hopsPropagation(
					end: 2.11, start: 2, maximumStep: 0.04,
					output: .times(times)), seed: 1, trajectoryID: 0
			) { t, _ in
				actual.append(t)
				return .proceed
			}
			#expect(actual == times)
			#expect(summary.propagation.finalTime == 2.11)
		}
		var calls = 0
		let stopped = try HOPS.solveTrajectory(
			problem: problem, configuration: config,
			propagation: hopsPropagation(output: .uniform(step: 0.01)), seed: 2,
			trajectoryID: 9
		) { _, _ in
			calls += 1
			return .stop
		}
		#expect(
			calls == 1 && stopped.propagation.finalTime == 0
				&& stopped.propagation.endReason == .stoppedByObserver)
		let empty = try HOPS.solveTrajectory(
			problem: problem, configuration: config,
			propagation: hopsPropagation(end: 2, start: 2), seed: 2, trajectoryID: 9
		) { t, _ in
			#expect(t == 2)
			return .proceed
		}
		#expect(empty.propagation.finalTime == 2)
		let mid = try HOPS.solveTrajectory(
			problem: problem, configuration: config,
			propagation: hopsPropagation(output: .everyAcceptedStep), seed: 2,
			trajectoryID: 9
		) { t, _ in t >= 0.04 ? .stop : .proceed }
		#expect(mid.propagation.endReason == .stoppedByObserver)
		#expect(mid.propagation.finalTime < 0.3)
	}

	@Test("RNG overload returns replayable seeds and handles the maximum random ID")
	func rngOverload() throws {
		struct MaxRNG: RandomNumberGenerator {
			var calls = 0
			mutating func next() -> UInt64 {
				calls += 1
				return .max
			}
		}
		let config = hopsConfiguration(
			3, model: noiseTestModel(0),
			operators: [hopsMatrix([Complex(0.2), Complex(0.6), 0, Complex(-0.3)])])
		var rng = MaxRNG()
		var sampled: [Complex<Double>] = []
		let summary = try HOPS.solveTrajectory(
			problem: hopsProblem(), configuration: config,
			propagation: hopsPropagation(), rng: &rng
		) { _, state in
			sampled = [state[0], state[1]]
			return .proceed
		}
		#expect(rng.calls == 2)
		#expect(summary.trajectoryIDs == 0..<1)
		let replay = try hopsFinalState(
			problem: hopsProblem(), configuration: config, seed: summary.masterSeed,
			id: summary.trajectoryIDs.lowerBound)
		expectHOPSClose(sampled, replay, tolerance: 0)
	}

	@Test("Invalid norms, operator dimensions and jump unravellings are reported")
	func validation() throws {
		var config = hopsConfiguration(
			0, model: noiseTestModel(0), operators: [hopsMatrix([0, 1, 0, 0])])
		config.unravelling = .jump
		#expect(throws: HOPS.CPUEngine.SolverError.unsupportedUnravelling) {
			try hopsFinalState(problem: hopsProblem(), configuration: config)
		}
		config.unravelling = .diffusive
		#expect(throws: HOPS.CPUEngine.SolverError.invalidStateNorm(time: 0)) {
			try hopsFinalState(
				problem: hopsProblem(initial: [0, 0]), configuration: config)
		}
		let mismatch = hopsConfiguration(
			2, model: noiseTestModel(0), operators: [hopsMatrix([1], 1)])
		#expect(throws: HOPS.CPUEngine.SolverError.operatorDimensionMismatch) {
			try hopsFinalState(problem: hopsProblem(), configuration: mismatch)
		}
	}
}
