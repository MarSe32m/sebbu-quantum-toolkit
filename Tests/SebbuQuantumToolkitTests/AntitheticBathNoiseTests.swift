// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("Antithetic bath-noise replay")
struct AntitheticBathNoiseTests {
	private let times = [0.0, 0.003, 0.01, 0.027, 0.19, 0.411, 0.9, 1.0]
	private func family(_ kind: Int, sampling: EnsembleSampling) -> HOPS.EnsembleNoisePaths {
		.init(
			definition: .init(
				model: noiseTestModel(kind), timeSpan: .init(start: 0, end: 1),
				stepSize: 0.01),
			masterSeed: 813, trajectoryIDs: 0..<17, ensembleSampling: sampling)
	}
	private func samples(_ path: HOPS.BathNoisePath) -> [[Complex<Double>]] {
		var result: [[Complex<Double>]] = []
		path.forEachSample(at: times) { _, values in
			result.append((0..<values.count).map { values[$0] })
		}
		return result
	}
	@Test(
		"Physical and latent paths negate completely, including stationary initialization",
		arguments: 0..<4)
	func pairedPaths(kind: Int) {
		let paths = family(kind, sampling: .antithetic)
		for pair: UInt64 in [0, 3, 7] {
			var a = paths.path(for: 2 * pair).makeSampler(windowDuration: 0.04)
			var b = paths.path(for: 2 * pair + 1).makeSampler(windowDuration: 0.04)
			var physicalA = UniqueVector<Complex<Double>>.zero(paths.channelCount)
			var physicalB = UniqueVector<Complex<Double>>.zero(paths.channelCount)
			var latentA = UniqueVector<Complex<Double>>.zero(paths.latentCount)
			var latentB = UniqueVector<Complex<Double>>.zero(paths.latentCount)
			for time in times {
				a.sample(
					time, physical: &physicalA.mutableSpan,
					latent: &latentA.mutableSpan)
				b.sample(
					time, physical: &physicalB.mutableSpan,
					latent: &latentB.mutableSpan)
				for i in 0..<paths.channelCount {
					#expect((physicalA[i] + physicalB[i]).length < 2e-14)
					for j in 0..<paths.channelCount {
						let original = physicalA[i] * physicalA[j].conjugate
						let reflected =
							physicalB[i] * physicalB[j].conjugate
						#expect((original - reflected).length < 2e-13)
					}
				}
				for i in 0..<paths.latentCount {
					#expect((latentA[i] + latentB[i]).length < 2e-14)
				}
			}
			#expect(
				samples(family(kind, sampling: .independent).path(for: pair))
					== samples(paths.path(for: 2 * pair)))
		}
	}
	@Test(
		"Reconstruction, request order and partial pairs preserve replay",
		arguments: [EnsembleSampling.independent, .antithetic], 0..<4)
	func replay(sampling: EnsembleSampling, kind: Int) {
		let paths = family(kind, sampling: sampling)
		let reconstructed = family(kind, sampling: sampling)
		var first: [UInt64: [[Complex<Double>]]] = [:]
		for id: UInt64 in [12, 4, 13, 5, 0, 1, 16] {
			let path = paths.path(for: id)
			#expect(path.ensembleSampling == sampling)
			first[id] = samples(path)
			#expect(first[id] == samples(path))
		}
		for id: UInt64 in [16, 1, 0, 5, 13, 4, 12] {
			#expect(first[id] == samples(reconstructed.path(for: id)))
		}
		#expect(first[12] != first[4])
		let partial = HOPS.EnsembleNoisePaths(
			definition: paths.definition, masterSeed: paths.masterSeed,
			trajectoryIDs: 5..<6, ensembleSampling: sampling)
		#expect(samples(partial.path(for: 5)) == first[5])
		if sampling == .antithetic {
			#expect(
				first[16]
					== samples(
						family(kind, sampling: .independent).path(for: 8)))
		}
	}
	@Test(
		"Window changes, repeated queries and physical/latent access preserve each realization",
		arguments: [EnsembleSampling.independent, .antithetic])
	func samplerQueries(sampling: EnsembleSampling) {
		let path = family(3, sampling: sampling).path(for: 5)
		var cursor = path.makeSampler(windowDuration: 0.3)
		var physical = UniqueVector<Complex<Double>>.zero(path.channelCount)
		var latent = UniqueVector<Complex<Double>>.zero(path.latentCount)
		for time in [0.0, 0.11, 0.03, 0.11, 0.2, 0.15, 0.6, 0.49] {
			cursor.sampleLatent(time, into: &latent.mutableSpan)
			cursor.sample(time, into: &physical.mutableSpan)
			var fresh = path.makeSampler(windowDuration: 0)
			var expected = UniqueVector<Complex<Double>>.zero(path.channelCount)
			var expectedLatent = UniqueVector<Complex<Double>>.zero(path.latentCount)
			fresh.sample(
				time, physical: &expected.mutableSpan,
				latent: &expectedLatent.mutableSpan)
			for i in 0..<path.channelCount { #expect(physical[i] == expected[i]) }
			for i in 0..<path.latentCount { #expect(latent[i] == expectedLatent[i]) }
		}
	}
	@Test(
		"Live trajectory and hierarchy noise replay every equation/shift variant",
		arguments: 0..<6)
	func liveReplay(variant: Int) throws {
		let configuration = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2)
		let propagation = hopsPropagation(
			end: 0.04, maximumStep: 0.005, output: .times([0, 0.013, 0.04]))
		var observed: [[Complex<Double>]] = []
		var observedTimes: [Double] = []
		let result = try HOPS.solveTrajectory(
			problem: hopsProblem(), configuration: configuration,
			propagation: propagation,
			seed: 913, trajectoryID: 5, ensembleSampling: .antithetic,
			observingWithNoise: { time, _, noise in
				observedTimes.append(time)
				observed.append((0..<noise.count).map { noise[$0] })
				return .proceed
			})
		#expect(
			result.ensembleSampling == .antithetic
				&& result.bathNoise.ensembleSampling == .antithetic)
		var replay: [[Complex<Double>]] = []
		result.bathNoise.forEachSample(at: observedTimes) { _, values in
			replay.append((0..<values.count).map { values[$0] })
		}
		#expect(observed == replay)
		var hierarchyNoise: [[Complex<Double>]] = []
		let hierarchy = try HOPS.solveWithHierarchy(
			problem: hopsProblem(), configuration: configuration,
			propagation: propagation,
			seed: 913, trajectoryID: 5, ensembleSampling: .antithetic,
			observingWithNoise: { _, _, noise in
				hierarchyNoise.append((0..<noise.count).map { noise[$0] })
				return .proceed
			})
		#expect(hierarchy.ensembleSampling == .antithetic && hierarchyNoise == replay)
		let stateOnly = try HOPS.solveWithHierarchy(
			problem: hopsProblem(), configuration: configuration,
			propagation: propagation,
			seed: 913, trajectoryID: 5, ensembleSampling: .antithetic
		) { _, _ in }
		#expect(stateOnly.bathNoise.ensembleSampling == .antithetic)
	}
}
