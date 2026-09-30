// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS bath-noise result APIs")
struct HOPSBathNoiseResultAPITests {
	@Test("Hierarchy solves return replay provenance and live noise agrees with replay")
	func hierarchyNoiseResult() throws {
		let model = noiseTestModel(0)
		let configuration = hopsConfiguration(
			0,
			model: model,
			operators: [hopsMatrix([1, 0, 0, -1])],
			depth: 2,
			noiseStep: 0.005
		)
		let propagation = hopsPropagation(
			end: 0.04,
			maximumStep: 0.01,
			output: .times([0, 0.013, 0.027, 0.04])
		)
		let seed: UInt64 = 0xBEEF
		let trajectoryID: UInt64 = 12

		let ordinary: HOPS.TrajectoryRunResult = try HOPS.solveWithHierarchy(
			problem: hopsProblem(),
			configuration: configuration,
			propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID
		) { _, _ in }

		var observedTimes: [Double] = []
		var observedNoise: [[Complex<Double>]] = []
		let live: HOPS.TrajectoryRunResult = try HOPS.solveWithHierarchy(
			problem: hopsProblem(),
			configuration: configuration,
			propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID,
			observingWithNoise: { time, _, noise in
				observedTimes.append(time)
				observedNoise.append((0..<noise.count).map { noise[$0] })
				return .proceed
			}
		)

		#expect(ordinary.masterSeed == seed)
		#expect(ordinary.trajectoryIDs == trajectoryID..<(trajectoryID + 1))
		#expect(live.masterSeed == seed)
		#expect(live.bathNoise.trajectoryID == trajectoryID)
		#expect(live.bathNoise.stepSize == 0.005)

		var replayed: [[Complex<Double>]] = []
		live.bathNoise.forEachSample(at: observedTimes) { _, noise in
			replayed.append((0..<noise.count).map { noise[$0] })
		}
		#expect(replayed == observedNoise)

		var ordinarySampler = ordinary.bathNoise.makeSampler(windowDuration: 0)
		var liveSampler = live.bathNoise.makeSampler(windowDuration: 0)
		var ordinaryNoise = [Complex<Double>](repeating: .zero, count: model.channelCount)
		var liveNoise = ordinaryNoise
		var ordinarySpan = ordinaryNoise.mutableSpan
		var liveSpan = liveNoise.mutableSpan
		ordinarySampler.sample(0.031, into: &ordinarySpan)
		liveSampler.sample(0.031, into: &liveSpan)
		#expect(ordinaryNoise == liveNoise)
	}

	@Test("Two-time and multi-time correlation solves return ensemble replay provenance")
	func correlationNoiseResults() throws {
		let model = noiseTestModel(0)
		let coupling = hopsMatrix([1, 0, 0, -1])
		let identity = hopsMatrix([1, 0, 0, 1])
		let configuration = hopsConfiguration(
			0,
			model: model,
			operators: [coupling],
			depth: 2,
			noiseStep: 0.005
		)
		let propagation = hopsPropagation(
			end: 0.04,
			maximumStep: 0.01,
			output: .final
		)
		let ids: Range<UInt64> = 21..<24
		let execution = TrajectoryExecution(
			trajectoryIDs: ids,
			randomness: .seeded(0xCAFE),
			parallelism: .serial
		)

		let multiRequest = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.02, insertion: .left(.constant(identity)))
			],
			observable: .constant(identity)
		)
		let multi: HOPS.EnsembleRunResult = try HOPS.solveMultiTimeOrderedCorrelation(
			problem: hopsProblem(),
			configuration: configuration,
			request: multiRequest,
			propagation: propagation,
			execution: execution
		) { _, _ in .proceed }

		let twoRequest = TwoTimeCorrelationRequest(
			insertionTime: 0.02,
			insertion: .left(.constant(identity)),
			observable: .constant(identity)
		)
		let two: HOPS.EnsembleRunResult = try HOPS.solveTwoTimeCorrelation(
			problem: hopsProblem(),
			configuration: configuration,
			request: twoRequest,
			propagation: propagation,
			execution: execution
		) { _, _ in .proceed }

		for result in [multi, two] {
			#expect(result.masterSeed == 0xCAFE)
			#expect(result.trajectoryIDs == ids)
			#expect(result.bathNoise.masterSeed == 0xCAFE)
			#expect(result.bathNoise.trajectoryIDs == ids)
			#expect(result.bathNoise.stepSize == 0.005)
		}

		let multiPath = multi.bathNoise.path(for: ids.lowerBound)
		let twoPath = two.bathNoise.path(for: ids.lowerBound)
		var multiSampler = multiPath.makeSampler(windowDuration: 0)
		var twoSampler = twoPath.makeSampler(windowDuration: 0)
		var multiNoise = [Complex<Double>](repeating: .zero, count: model.channelCount)
		var twoNoise = multiNoise
		var multiSpan = multiNoise.mutableSpan
		var twoSpan = twoNoise.mutableSpan
		multiSampler.sample(0.037, into: &multiSpan)
		twoSampler.sample(0.037, into: &twoSpan)
		#expect(multiNoise == twoNoise)
	}
}
