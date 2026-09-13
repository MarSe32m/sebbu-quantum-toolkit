// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS right-hand-side equations")
struct HOPSRightHandSideTests {
	@Test(
		"Batched physical-operator kernel equals the literal latent equations",
		arguments: 0..<6)
	func latentEquation(variant: Int) throws {
		let model = hopsFixtureModel()
		let c = hopsMatrix([Complex(0.1, 0.2), Complex(0.7, -0.3), 0, Complex(-0.2, 0.1)])
		let h = hopsMatrix([
			Complex(0.3), Complex(0.2, 0.1), Complex(0.2, -0.1), Complex(-0.4),
		])
		let problem = hopsProblem(h, markovian: [.init(rate: 0.6, collapseOperator: c)])
		let config = hopsConfiguration(
			variant, model: model, operators: hopsFixtureOperators, depth: 2)
		let preparation = try HOPS.CPUEngine.Preparation(
			problem: problem, configuration: config, propagation: hopsPropagation())
		var rhs = HOPS.CPUEngine.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: preparation, seed: 1,
			trajectoryID: 0)
		var y = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: preparation.shiftCount)
		var dy = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: preparation.shiftCount)
		let values: [Complex<Double>] = (0..<(config.hierarchy.count * 2)).map { i in
			let real: Double = 0.3 + 0.02 * Double(i)
			let imaginary: Double = -0.1 + 0.01 * Double(i)
			return Complex<Double>(real, imaginary)
		}
		for i in values.indices { y.amplitudes.elements[i] = values[i] }
		let shifts: [Complex<Double>] = (0..<preparation.shiftCount).map { p in
			let real: Double = 0.03 * Double(p + 1)
			let imaginary: Double = -0.02 * Double(p + 2)
			return Complex<Double>(real, imaginary)
		}
		for p in shifts.indices { y.shifts[p] = shifts[p] }
		let noise = [Complex(0.2, -0.7), Complex(-0.4, 0.3)]
		for i in noise.indices { rhs.physicalNoise[i] = noise[i] }
		rhs.evaluateWithCurrentNoise(t: 0.17, y: y, into: &dy)
		let expected = hopsReferenceDrift(
			configuration: config, hamiltonian: h, operators: hopsFixtureOperators,
			noise: noise, state: values, shifts: shifts, markovian: [(0.6, c)])
		expectHOPSClose(
			values.indices.map { dy.amplitudes.elements[$0] }, expected.0,
			tolerance: 3e-13)
		expectHOPSClose(shifts.indices.map { dy.shifts[$0] }, expected.1, tolerance: 3e-13)
		if variant / 2 == 2 {
			let tangent =
				values[0].conjugate * dy.amplitudes.elements[0] + values[1]
				.conjugate * dy.amplitudes.elements[1]
			#expect(abs(tangent.real) < 2e-14)
		}

		// A diffusion call must overwrite all amplitudes and erase shift drift.
		rhs.diffusion(t: 0.17, y: y, channel: 0, into: &dy)
		let norm = values[0].lengthSquared + values[1].lengthSquared
		var mean = Complex<Double>.zero
		for i in 0..<2 {
			for j in 0..<2 { mean += values[i].conjugate * c[i, j] * values[j] / norm }
		}
		for n in 0..<config.hierarchy.count {
			for i in 0..<2 {
				var expected = Complex<Double>.zero
				for j in 0..<2 { expected += c[i, j] * values[n * 2 + j] }
				if variant / 2 == 2 { expected -= mean * values[n * 2 + i] }
				#expect(
					(dy.amplitudes.elements[n * 2 + i] - 0.3.squareRoot()
						* expected).length < 1e-13)
			}
		}
		for p in shifts.indices { #expect(dy.shifts[p] == .zero) }
	}

	@Test(
		"Dynamic physical and Markovian operators agree with their instantaneous constants",
		arguments: 0..<6)
	func dynamicOperators(variant: Int) throws {
		let model = hopsFixtureModel()
		let t = 0.37
		let factor = Complex(1 + t, 0.2 * t)
		let instantaneous = hopsFixtureOperators.map { factor * $0 }
		let config = hopsConfiguration(
			variant, model: model, operators: instantaneous, depth: 2)
		let dynamic = HOPS.Configuration(
			hierarchy: .init(
				environment: .init(
					couplingOperators: hopsFixtureOperators.map { matrix in
						.generatedDense(
							.init { time, buffer in
								buffer.copyElements(
									from: matrix,
									multiplied: Complex(
										1 + time, 0.2 * time
									))
							})
					}, bath: model), truncation: .maximumTier(2)),
			equationType: config.equationType,
			shiftType: config.shiftType, noiseStepSize: 0.01)
		let c = hopsMatrix([0, Complex(0.8, 0.2), 0, 0])
		let staticProblem = hopsProblem(markovian: [
			.init(rate: 0.4 + t, collapseOperator: factor * c)
		])
		let dynamicProblem = hopsProblem(markovian: [
			.init(
				rate: .generated { 0.4 + $0 },
				collapseOperator:
					.linearCombination(
						.init(
							coefficients: [
								.generated {
									Complex(1 + $0, 0.2 * $0)
								}
							], operators: [.init(c)])))
		])
		let a = try HOPS.CPUEngine.Preparation(
			problem: staticProblem, configuration: config,
			propagation: hopsPropagation())
		let b = try HOPS.CPUEngine.Preparation(
			problem: dynamicProblem, configuration: dynamic,
			propagation: hopsPropagation())
		var rhsA = HOPS.CPUEngine.RightHandSide(
			hamiltonian: staticProblem.system.hamiltonian, preparation: a, seed: 1,
			trajectoryID: 2)
		var rhsB = HOPS.CPUEngine.RightHandSide(
			hamiltonian: dynamicProblem.system.hamiltonian, preparation: b, seed: 1,
			trajectoryID: 2)
		var y = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: a.shiftCount)
		var da = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: a.shiftCount)
		var db = HOPS.CPUEngine.State(
			dimension: 2, hierarchyCount: config.hierarchy.count,
			shiftCount: a.shiftCount)
		for i in 0..<(2 * config.hierarchy.count) {
			y.amplitudes.elements[i] = Complex(0.2 + 0.01 * Double(i), 0.3)
		}
		for p in 0..<a.shiftCount { y.shifts[p] = Complex(0.03, 0.02 * Double(p)) }
		rhsA.evaluateWithCurrentNoise(t: t, y: y, into: &da)
		rhsB.evaluateWithCurrentNoise(t: t, y: y, into: &db)
		expectHOPSClose(
			(0..<(2 * config.hierarchy.count)).map { da.amplitudes.elements[$0] },
			(0..<(2 * config.hierarchy.count)).map { db.amplitudes.elements[$0] })
		expectHOPSClose(
			(0..<a.shiftCount).map { da.shifts[$0] },
			(0..<a.shiftCount).map { db.shifts[$0] })
		rhsA.diffusion(t: t, y: y, channel: 0, into: &da)
		rhsB.diffusion(t: t, y: y, channel: 0, into: &db)
		expectHOPSClose(
			(0..<(2 * config.hierarchy.count)).map { da.amplitudes.elements[$0] },
			(0..<(2 * config.hierarchy.count)).map { db.amplitudes.elements[$0] })
	}
}
