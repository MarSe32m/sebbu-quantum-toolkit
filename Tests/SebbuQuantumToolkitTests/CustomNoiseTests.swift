// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

// Noncopyable types exercise the full ownership path through the public APIs.
private struct ConstantBathNoise: ~Copyable, MultiChannelNoiseRealization {
	let channelCount: Int
	let value: Complex<Double>
	mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>) {
		for i in values.indices { values[i] = value }
	}
}

extension CustomNoiseTests {
	@Test("FFT realizations supply the physical HOPS noise")
	func fftBathNoise() throws {
		let generator = GaussianFFTMultiNoiseProcessGenerator(
			tMax: 0.3, dtMax: 0.02, deltaOmegaMax: 1, omegaMax: 8,
			spectralDensity: { _ in hopsMatrix([1, 0, 0, Complex(0.5)]) })
		var rng = SplitMix64(seed: 31)
		let noise = generator.generateProcess(generator: &rng)
		var values = UniqueVector<Complex<Double>>.zero(2)
		noise.sample(0.13, into: &values.mutableSpan)
		let problem = hopsProblem()
		let config = hopsConfiguration(
			0, model: hopsFixtureModel(), operators: hopsFixtureOperators, depth: 1)
		let preparation = try HOPS.CPUEngine.Preparation(
			problem: problem, configuration: config, propagation: hopsPropagation(),
			preparingStandardNoise: false)
		var rhs = HOPS.CPUEngine.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: preparation,
			bathNoise: noise, markovianNoise: ZeroDiffusiveNoise(channelCount: 0))
		let state = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count, shiftCount: 0)
		for i in 0..<2 { state.amplitudes.elements[i] = problem.initialState[i] }
		var derivative = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count, shiftCount: 0)
		rhs.drift(t: 0.13, y: state, into: &derivative)
		#expect(rhs.physicalNoise[0] == values[0] && rhs.physicalNoise[1] == values[1])
		let h = hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)])
		let initial = (0..<(config.hierarchy.count * 2)).map {
			state.amplitudes.elements[$0]
		}
		let expected = hopsReferenceDrift(
			configuration: config, hamiltonian: h, operators: hopsFixtureOperators,
			noise: [values[0], values[1]], state: initial, shifts: [])
		expectHOPSClose(
			initial.indices.map { derivative.amplitudes.elements[$0] }, expected.0,
			tolerance: 2e-13)
	}

	@Test("Zero-noise HOPS drift equals the literal equations", arguments: 0..<6)
	func zeroDriftReference(variant: Int) throws {
		let h = hopsMatrix([
			Complex(0.1), Complex(0.2, 0.1), Complex(0.2, -0.1), Complex(-0.3),
		])
		let collapse = hopsMatrix([Complex(0.1), Complex(0.7, -0.3), 0, Complex(0.2)])
		let problem = hopsProblem(
			h, markovian: [.init(rate: 0.6, collapseOperator: collapse)])
		let config = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2)
		let preparation = try HOPS.CPUEngine.Preparation(
			problem: problem, configuration: config, propagation: hopsPropagation(),
			preparingStandardNoise: false)
		var rhs = HOPS.CPUEngine.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: preparation,
			bathNoise: ZeroMultiChannelNoise(channelCount: 2),
			markovianNoise: ZeroDiffusiveNoise(channelCount: 1))
		var state = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: preparation.shiftCount)
		var derivative = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: preparation.shiftCount)
		let values: [Complex<Double>] = (0..<(config.hierarchy.count * 2)).map { i in
			Complex(0.3 + 0.02 * Double(i), -0.1 + 0.01 * Double(i))
		}
		let shifts: [Complex<Double>] = (0..<preparation.shiftCount).map { p in
			Complex(0.03 * Double(p + 1), -0.02)
		}
		for i in values.indices { state.amplitudes.elements[i] = values[i] }
		for p in shifts.indices { state.shifts[p] = shifts[p] }
		rhs.drift(t: 0.13, y: state, into: &derivative)
		let expected = hopsReferenceDrift(
			configuration: config, hamiltonian: h, operators: hopsFixtureOperators,
			noise: [0, 0], state: values, shifts: shifts, markovian: [(0.6, collapse)])
		expectHOPSClose(
			values.indices.map { derivative.amplitudes.elements[$0] }, expected.0,
			tolerance: 3e-13)
		expectHOPSClose(
			shifts.indices.map { derivative.shifts[$0] }, expected.1, tolerance: 3e-13)
	}
}

private final class NoiseCalls {
	var times: [Double] = []
	var steps: [Double] = []
}

private struct PrescribedDiffusiveNoise: ~Copyable, DiffusiveNoiseRealization {
	let channelCount = 1
	let calls: NoiseCalls
	static func value(at t: Double, step h: Double) -> Complex<Double> {
		Complex(0.2 + t, -0.3 + h)
	}
	mutating func sampleNormalized(
		at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>
	) {
		// Test-only call recording, not a production sampling implementation.
		calls.times.append(time)
		calls.steps.append(stepSize)
		values[0] = Self.value(at: time, step: stepSize)
	}
}

private struct NeverSampledZeroNoise: ~Copyable, DiffusiveNoiseRealization {
	static var isIdenticallyZero: Bool { true }
	let channelCount = 1
	mutating func sampleNormalized(
		at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>
	) {
		Issue.record("Zero HOPS noise must use the ODE solver without SDE sampling")
		for i in values.indices { values[i] = .zero }
	}
}

@Suite("Custom noise realizations")
struct CustomNoiseTests {
	@Test(
		"Gaussian realization preserves the original RNG stream", arguments: [0, 1, 3],
		[false, true])
	func gaussianStream(count: Int, antithetic: Bool) {
		let sampling: EnsembleSampling = antithetic ? .antithetic : .independent
		for id in [UInt64(0), 6, 7] {
			var noise = GaussianDiffusiveNoise(
				channelCount: count, seed: 217, trajectoryID: id,
				ensembleSampling: sampling)
			var oldRNG = TrajectoryRandomNumberGenerator(
				seed: 217, trajectoryID: id,
				ensembleSampling: sampling, purpose: .gaussianWhiteNoise)
			var values = UniqueVector<Complex<Double>>.zero(count)
			for step in 0..<30 {
				noise.sampleNormalized(
					at: Double(step) * 0.01, stepSize: 0.01,
					into: &values.mutableSpan)
				for i in 0..<count {
					let expected: Complex<Double> = oldRNG.nextNormal()
					#expect(values[i] == expected)
				}
			}
		}
	}

	@Test("Zero realizations overwrite the entire span", arguments: [0, 1, 5])
	func zeroFill(count: Int) {
		var bath = ZeroMultiChannelNoise(channelCount: count)
		var white = ZeroDiffusiveNoise(channelCount: count)
		var values = UniqueVector<Complex<Double>>.zero(count)
		for t in [0.0, 1.0, 0.4, 0.4] {
			for i in 0..<count { values[i] = Complex(17, -3) }
			bath.sample(t, into: &values.mutableSpan)
			#expect((0..<count).allSatisfy { values[$0] == .zero })
			for i in 0..<count { values[i] = Complex(-8, 2) }
			white.sampleNormalized(at: t, stepSize: 0.01, into: &values.mutableSpan)
			#expect((0..<count).allSatisfy { values[$0] == .zero })
		}
		#expect(ZeroDiffusiveNoise.isIdenticallyZero)
		#expect(!GaussianDiffusiveNoise.isIdenticallyZero)
	}

	@Test("QSD explicit Gaussian equals seeded propagation", arguments: 0..<3, [false, true])
	func qsdSeeded(variant: Int, antithetic: Bool) throws {
		let equations: [QSD.EquationType] = [.linear, .nonLinear, .nonLinearNormalized]
		let problem = hopsProblem(markovian: [
			.init(rate: 0.7, collapseOperator: hopsMatrix([0, 1, 0, 0])),
			.init(rate: 0.2, collapseOperator: hopsMatrix([1, 0, 0, -1])),
		])
		let configuration = QSD.Configuration(equationType: equations[variant])
		let propagation = hopsPropagation(
			end: 0.08, maximumStep: 0.007, output: .times([0, 0.023, 0.08]))
		let sampling: EnsembleSampling = antithetic ? .antithetic : .independent
		var expected: [Complex<Double>] = []
		let seeded = try QSD.solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: 273, trajectoryID: 7, ensembleSampling: sampling
		) { _, state in
			for i in 0..<state.count { expected.append(state[i]) }
			return .proceed
		}
		for _ in 0..<2 {
			var actual: [Complex<Double>] = []
			let explicit = try QSD.solveTrajectory(
				problem: problem, configuration: configuration,
				propagation: propagation,
				noise: GaussianDiffusiveNoise(
					channelCount: 2, seed: 273, trajectoryID: 7,
					ensembleSampling: sampling)
			) { _, state in
				for i in 0..<state.count { actual.append(state[i]) }
				return .proceed
			}
			#expect(actual == expected)
			#expect(explicit.finalTime == seeded.propagation.finalTime)
			#expect(explicit.endReason == seeded.propagation.endReason)
		}
	}

	@Test("Zero QSD retains non-Hermitian drift and is repeatable")
	func qsdZeroDrift() throws {
		let gamma = 0.8
		let omega = 0.3
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, Complex(omega)]), initial: [0, 1],
			markovian: [.init(rate: gamma, collapseOperator: hopsMatrix([0, 1, 0, 0]))])
		let propagation = hopsPropagation(
			end: 0.4, maximumStep: 0.0005, output: .times([0.1, 0.4]))
		func run() throws -> [Complex<Double>] {
			var result: [Complex<Double>] = []
			try QSD.CPUEngine().solveTrajectory(
				problem: problem, configuration: .init(equationType: .linear),
				propagation: propagation, noise: ZeroDiffusiveNoise(channelCount: 1)
			) { t, state in
				#expect(state[0] == .zero)
				#expect(
					(state[1] - Complex.exp(Complex(-gamma / 2, -omega) * t))
						.length < 3e-9)
				result.append(state[1])
				return .proceed
			}
			return result
		}
		let first = try run()
		#expect(first == (try run()))
		#expect(first.last!.length < 0.9)
	}

	@Test(
		"Both solvers consume custom diffusive values once per actual step",
		arguments: [false, true])
	func prescribedWhiteNoise(hops: Bool) throws {
		let gamma = 0.6
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, 0]), initial: [1, 0],
			markovian: [.init(rate: gamma, collapseOperator: hopsMatrix([1, 0, 0, 1]))])
		let propagation = hopsPropagation(
			end: 0.08, maximumStep: 0.03, output: .times([0.02, 0.08]))
		let calls = NoiseCalls()
		var actual: [Complex<Double>] = []
		if hops {
			let config = hopsConfiguration(
				0, model: .zero(channelCount: 1),
				operators: [hopsMatrix([1, 0, 0, -1])])
			try HOPS.solveTrajectory(
				problem: problem, configuration: config, propagation: propagation,
				bathNoise: ZeroMultiChannelNoise(channelCount: 1),
				markovianNoise: PrescribedDiffusiveNoise(calls: calls)
			) { _, state in
				actual.append(state[0])
				return .proceed
			}
		} else {
			try QSD.solveTrajectory(
				problem: problem, configuration: .init(equationType: .linear),
				propagation: propagation,
				noise: PrescribedDiffusiveNoise(calls: calls)
			) { _, state in
				actual.append(state[0])
				return .proceed
			}
		}
		#expect(calls.times.count == 3)
		#expect(calls.times[0] == 0 && calls.times[1] == 0.02 && calls.times[2] == 0.05)
		var reference = Complex<Double>.one
		for i in calls.times.indices {
			let h = calls.steps[i]
			let increment =
				Complex(-gamma * h / 2)
				+ (gamma * h / 2).squareRoot()
				* PrescribedDiffusiveNoise.value(at: calls.times[i], step: h)
			reference *= 1 + increment + 0.5 * increment * increment
			if i == 0 { #expect((actual[0] - reference).length < 3e-15) }
		}
		#expect((actual.last! - reference).length < 3e-15)
	}

	@Test("HOPS replay plus explicit Gaussian equals seeded paths", arguments: 0..<6, 0..<4)
	func hopsSeeded(variant: Int, mode: Int) throws {
		let channels: [MarkovianChannel] =
			mode % 2 == 0
			? [] : [.init(rate: 0.3, collapseOperator: hopsMatrix([0, 1, 0, 0]))]
		let sampling: EnsembleSampling = mode < 2 ? .independent : .antithetic
		let problem = hopsProblem(markovian: channels)
		let config = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2, noiseStep: 0.009)
		let propagation = hopsPropagation(
			end: 0.081, start: 0.003, maximumStep: 0.017,
			output: .times([0.003, 0.026, 0.081]))
		var expected: [Complex<Double>] = []
		let run = try HOPS.solveTrajectory(
			problem: problem, configuration: config, propagation: propagation,
			seed: 43, trajectoryID: 7, ensembleSampling: sampling
		) { _, state in
			for i in 0..<state.count { expected.append(state[i]) }
			return .proceed
		}
		for _ in 0..<2 {
			var actual: [Complex<Double>] = []
			let summary = try HOPS.solveTrajectory(
				problem: problem, configuration: config, propagation: propagation,
				bathNoise: run.bathNoise.makeSampler(
					windowDuration: propagation.integration.maximumStepSize),
				markovianNoise: GaussianDiffusiveNoise(
					channelCount: channels.count, seed: 43,
					trajectoryID: 7, ensembleSampling: sampling)
			) { _, state in
				for i in 0..<state.count { actual.append(state[i]) }
				return .proceed
			}
			#expect(actual == expected)
			#expect(summary.finalTime == run.propagation.finalTime)
			#expect(summary.endReason == run.propagation.endReason)
		}
	}

	@Test("Zero HOPS noise retains memory and dissipative drift, using adaptive ODE")
	func hopsZeroMemory() throws {
		let w = Complex(0.8, 0.5)
		let strength = 0.35
		let gamma = 0.7
		let residue = (2 * w.real * strength).squareRoot()
		let model = CorrelatedBathModel(
			channelCount: 1, latentBaths: [noiseTestBath([w], [[Complex(residue)]])])
		let config = hopsConfiguration(
			0, model: model, operators: [hopsMatrix([0, 0, 0, 1])], depth: 8)
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, 0]), initial: [0, 1],
			markovian: [.init(rate: gamma, collapseOperator: hopsMatrix([0, 1, 0, 0]))])
		let propagation = hopsPropagation(
			end: 0.6, maximumStep: 0.3, tolerance: 1e-12, output: .times([0.2, 0.6]))
		func run() throws -> [Complex<Double>] {
			var result: [Complex<Double>] = []
			try HOPS.CPUEngine().solveTrajectory(
				problem: problem, configuration: config, propagation: propagation,
				bathNoise: ZeroMultiChannelNoise(channelCount: 1),
				markovianNoise: NeverSampledZeroNoise()
			) { t, state in
				let memory =
					strength * (t / w - (1 - Complex.exp(-w * t)) / (w * w))
				let reference = Complex.exp(-memory - Complex(gamma * t / 2))
				#expect(state[0] == .zero)
				#expect((state[1] - reference).length < 2e-10)
				#expect(
					(state[1] - Complex(Double.exp(-gamma * t / 2))).length
						> 0.001)
				result.append(state[1])
				return .proceed
			}
			return result
		}
		let first = try run()
		#expect(first == (try run()))
	}

	@Test("Zero HOPS realizations work in all variants", arguments: 0..<6)
	func hopsZeroVariants(variant: Int) throws {
		let problem = hopsProblem(markovian: [
			.init(rate: 0.4, collapseOperator: hopsMatrix([0, 1, 0, 0]))
		])
		let config = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2)
		func run() throws -> [Complex<Double>] {
			var result: [Complex<Double>] = []
			try HOPS.solveTrajectory(
				problem: problem, configuration: config,
				propagation: hopsPropagation(end: 0.1),
				bathNoise: ZeroMultiChannelNoise(channelCount: 2),
				markovianNoise: ZeroDiffusiveNoise(channelCount: 1)
			) { _, state in
				if variant >= 4 { #expect(abs(state.normSquared - 1) < 1e-12) }
				result = (0..<state.count).map { state[$0] }
				return .proceed
			}
			return result
		}
		let first = try run()
		#expect(first == (try run()))
		#expect(first != [Complex(0.6), Complex(0, 0.8)])
	}

	@Test(
		"Noncopyable colored paths are used, even without bath poles",
		arguments: [false, true])
	func prescribedBathNoise(zeroModel: Bool) throws {
		let value = Complex(0.25, -0.6)
		let model =
			zeroModel ? CorrelatedBathModel.zero(channelCount: 1) : noiseTestModel(0)
		let config = hopsConfiguration(
			0, model: model, operators: [hopsMatrix([0, 0, 0, 1])], depth: 0)
		let problem = hopsProblem(hopsMatrix([0, 0, 0, 0]), initial: [0, 1])
		try HOPS.solveTrajectory(
			problem: problem, configuration: config,
			propagation: hopsPropagation(end: 0.4, maximumStep: 0.15, tolerance: 1e-12),
			bathNoise: ConstantBathNoise(channelCount: 1, value: value),
			markovianNoise: ZeroDiffusiveNoise(channelCount: 0)
		) { t, state in
			#expect((state[1] - Complex.exp(value.conjugate * t)).length < 1e-11)
			return .proceed
		}
	}

	@Test("Channel mismatch is rejected before callbacks, even at zero duration")
	func invalidChannels() throws {
		let problem = hopsProblem(markovian: [
			.init(rate: 0.4, collapseOperator: hopsMatrix([0, 1, 0, 0]))
		])
		let config = hopsConfiguration(
			0, model: hopsFixtureModel(), operators: hopsFixtureOperators, depth: 1)
		let propagation = hopsPropagation(end: 0, output: .times([0]))
		for count in [0, 2] {
			#expect(
				throws: QSD.CPUEngine.SolverError.noiseChannelCountMismatch(
					expected: 1, actual: count)
			) {
				try QSD.solveTrajectory(
					problem: problem, configuration: .init(),
					propagation: propagation,
					noise: ZeroDiffusiveNoise(channelCount: count)
				) { _, _ in
					Issue.record("Invalid noise reached the observer")
					return .stop
				}
			}
			#expect(
				throws: HOPS.CPUEngine.SolverError
					.markovianNoiseChannelCountMismatch(
						expected: 1, actual: count)
			) {
				try HOPS.solveTrajectory(
					problem: problem, configuration: config,
					propagation: propagation,
					bathNoise: ZeroMultiChannelNoise(channelCount: 2),
					markovianNoise: ZeroDiffusiveNoise(channelCount: count)
				) { _, _ in
					Issue.record("Invalid noise reached the observer")
					return .stop
				}
			}
		}
		#expect(
			throws: HOPS.CPUEngine.SolverError.bathNoiseChannelCountMismatch(
				expected: 2, actual: 1)
		) {
			try HOPS.solveTrajectory(
				problem: problem, configuration: config, propagation: propagation,
				bathNoise: ZeroMultiChannelNoise(channelCount: 1),
				markovianNoise: ZeroDiffusiveNoise(channelCount: 1)
			) { _, _ in
				Issue.record("Invalid noise reached the observer")
				return .stop
			}
		}
	}

	@Test("Custom paths preserve initial stopping and empty duration")
	func observerControl() throws {
		let problem = hopsProblem()
		let config = hopsConfiguration(
			0, model: .zero(channelCount: 1), operators: [hopsMatrix([1, 0, 0, -1])])
		for end in [0.0, 0.2] {
			let propagation = hopsPropagation(end: end, output: .times([0]))
			var count = 0
			let h = try HOPS.solveTrajectory(
				problem: problem, configuration: config, propagation: propagation,
				bathNoise: ZeroMultiChannelNoise(channelCount: 1),
				markovianNoise: ZeroDiffusiveNoise(channelCount: 0)
			) { _, _ in
				count += 1
				return .stop
			}
			let q = try QSD.solveTrajectory(
				problem: problem, configuration: .init(), propagation: propagation,
				noise: ZeroDiffusiveNoise(channelCount: 0)
			) { _, _ in
				count += 1
				return .stop
			}
			#expect(count == 2)
			#expect(h.finalTime == 0 && q.finalTime == 0)
			#expect(
				h.endReason == .stoppedByObserver
					&& q.endReason == .stoppedByObserver)
		}
	}
}
