// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS replayable bath-noise paths")
struct HOPSBathNoisePathTests {
    private func result(
        seed: UInt64 = 91,
        id: UInt64 = 7,
        variant: Int = 0,
        model: CorrelatedBathModel = hopsFixtureModel(),
        end: Double = 0.12
    ) throws -> HOPS.TrajectoryRunResult {
        let operators = model.channelCount == 2
            ? hopsFixtureOperators
            : [hopsMatrix([1, 0, 0, -1])]
        return try HOPS.solveTrajectory(
            problem: hopsProblem(),
            configuration: hopsConfiguration(
                variant, model: model, operators: operators, depth: 1,
                noiseStep: 0.01),
            propagation: hopsPropagation(
                end: end, maximumStep: 0.02, output: .times([])),
            seed: seed,
            trajectoryID: id, observing: { _, _ in .proceed })
    }

    @Test("A path records stable replay metadata without storing samples")
    func metadata() throws {
        let run = try result()
        let path = run.bathNoise
        #expect(run.masterSeed == 91)
        #expect(run.trajectoryIDs == 7..<8)
        #expect(path.masterSeed == 91)
        #expect(path.trajectoryID == 7)
        #expect(path.channelCount == 2)
        #expect(path.latentCount == hopsFixtureModel().poleCount)
        #expect(path.stepSize == 0.01)
        #expect(path.timeSpan.start == 0 && path.timeSpan.end == 0.12)
        #expect(path.meshOrigin == 0)
        #expect(path.generationAlgorithm == .correlatedOUV1)
        #expect(path.generationAlgorithm.rawValue == 1)
    }

    @Test("Fresh samplers reproduce exactly; seed and trajectory ID select different paths")
    func deterministicIdentity() throws {
        let a = try result(seed: 123, id: 4)
        let otherID = try result(seed: 123, id: 5)
        let otherSeed = try result(seed: 124, id: 4)
        let times = [0.0, 0.013, 0.04, 0.099, 0.12]

        func samples(_ path: HOPS.BathNoisePath) -> [[Complex<Double>]] {
            var values: [[Complex<Double>]] = []
            path.forEachSample(at: times) { _, noise in
                values.append((0..<noise.count).map { noise[$0] })
            }
            return values
        }

        let first = samples(a.bathNoise)
        #expect(first == samples(a.bathNoise))
        #expect(first != samples(otherID.bathNoise))
        #expect(first != samples(otherSeed.bathNoise))
    }

    @Test("Public replay is the existing correlated OU algorithm, including interpolation")
    func lowerLevelAgreement() throws {
        let model = hopsFixtureModel()
        let run = try result(seed: 222, id: 13, model: model)
        let path = run.bathNoise
        var sampler = path.makeSampler(windowDuration: 0.04)
        var actual = [Complex<Double>](repeating: .zero, count: path.channelCount)

        var rng = TrajectoryRandomNumberGenerator(
            seed: 222, trajectoryID: 13, purpose: .coloredNoiseGeneration)
        var reference = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: model, windowDuration: 0.04, start: path.meshOrigin,
            step: path.stepSize, generator: &rng)
        var expected = actual

        for time in [0.0, 0.006, 0.027, 0.031, 0.089, 0.12] {
            do {
                var span = actual.mutableSpan
                sampler.sample(time, into: &span)
            }
            do {
                var span = expected.mutableSpan
                reference.sample(time, into: &span, generator: &rng)
            }
            #expect(actual == expected)
        }
    }

    @Test("Sequential replay and direct sampling agree exactly")
    func sequentialReplay() throws {
        let path = try result().bathNoise
        let times = [0.0, 0.007, 0.01, 0.031, 0.075, 0.12]
        var sequential: [[Complex<Double>]] = []
        path.forEachSample(at: times) { _, noise in
            sequential.append((0..<noise.count).map { noise[$0] })
        }

        var sampler = path.makeSampler(windowDuration: 0.04)
        var noise = [Complex<Double>](repeating: .zero, count: path.channelCount)
        for (index, time) in times.enumerated() {
            var span = noise.mutableSpan
            sampler.sample(time, into: &span)
            #expect(noise == sequential[index])
        }
    }

    @Test("Physical and latent coordinates describe the same correlated realization")
    func physicalLatentMixing() throws {
        let model = hopsFixtureModel()
        let path = try result(model: model).bathNoise
        var sampler = path.makeSampler(windowDuration: 0.03)
        var physical = [Complex<Double>](repeating: .zero, count: path.channelCount)
        var latent = [Complex<Double>](repeating: .zero, count: path.latentCount)
        do {
            var z = physical.mutableSpan
            var x = latent.mutableSpan
            sampler.sample(0.073, physical: &z, latent: &x)
        }

        for i in 0..<model.channelCount {
            var expected = Complex<Double>.zero
            var offset = 0
            for bath in model.latentBaths {
                for p in bath.poles.indices {
                    expected += bath.residues[i, p] * latent[offset + p]
                }
                offset += bath.poleCount
            }
            #expect((physical[i] - expected).length < 2e-14)
        }
    }

    @Test("Returned replay path is the colored noise that drives the HOPS equation")
    func solverPathwiseAgreement() throws {
        let pole = Complex(0.8, 0.6)
        let residue = Complex(0.7, -0.2)
        let model = CorrelatedBathModel(
            channelCount: 1,
            latentBaths: [noiseTestBath([pole], [[residue]])]
        )
        let coupling = Complex(0.3, 0.15)
        let config = hopsConfiguration(
            0, model: model, operators: [hopsMatrix([coupling], 1)],
            depth: 0, noiseStep: 0.005)
        let problem = hopsProblem(hopsMatrix([0], 1), initial: [1])
        let propagation = hopsPropagation(
            end: 0.1, maximumStep: 0.01, tolerance: 1e-11,
            output: .final)
        var final = Complex<Double>.zero
        let run = try HOPS.solveTrajectory(
            problem: problem, configuration: config, propagation: propagation,
            seed: 313, trajectoryID: 11
        ) { _, state in
            final = state[0]
            return .proceed
        }

        var integral = Complex<Double>.zero
        var previous = Complex<Double>.zero
        var first = true
        let times = (0...20).map { Double($0) * 0.005 }
        run.bathNoise.forEachSample(at: times) { time, noise in
            if first {
                previous = noise[0]
                first = false
            } else {
                integral += 0.5 * 0.005 * (previous + noise[0])
                previous = noise[0]
            }
            _ = time
        }
        let expected = Complex<Double>.exp(coupling * integral.conjugate)
        #expect((final - expected).length < 2e-9)
    }

    @Test("Live observations are the exact path returned by the trajectory", arguments: 0..<6)
    func liveObservation(variant: Int) throws {
        let model = noiseTestModel(0)
        let config = hopsConfiguration(
            variant, model: model,
            operators: [hopsMatrix([Complex(0.2), Complex(0.8), 0, Complex(-0.3)])],
            depth: 2, noiseStep: 0.01)
        let propagation = hopsPropagation(
            end: 0.08, maximumStep: 0.02,
            output: .times([0, 0.013, 0.04, 0.071, 0.08]))
        var observedTimes: [Double] = []
        var observedNoise: [[Complex<Double>]] = []
        let run = try HOPS.solveTrajectory(
            problem: hopsProblem(), configuration: config, propagation: propagation,
            seed: 77, trajectoryID: 9,
            observingWithNoise: { time, state, noise in
                #expect(state.count == 2)
                observedTimes.append(time)
                observedNoise.append(
                    (0..<noise.count).map { noise[$0] })
                return .proceed
            })

        var replayed: [[Complex<Double>]] = []
        run.bathNoise.forEachSample(at: observedTimes) { _, noise in
            replayed.append((0..<noise.count).map { noise[$0] })
        }
        #expect(replayed == observedNoise)
    }

    @Test("Ensemble provenance is independent of scheduling and resolves nondeterministic seeds")
    func ensemblePaths() throws {
        let model = noiseTestModel(0)
        let config = hopsConfiguration(
            0, model: model, operators: [hopsMatrix([1, 0, 0, -1])], depth: 1)
        let propagation = hopsPropagation(
            end: 0.04, maximumStep: 0.02, output: .times([]))
        let ids: Range<UInt64> = 20..<24

        func run(_ parallelism: TrajectoryParallelism) throws -> HOPS.EnsembleRunResult {
            try HOPS.solveEnsemble(
                problem: hopsProblem(), configuration: config, propagation: propagation,
                execution: .init(
                    trajectoryIDs: ids, randomness: .seeded(456),
                    parallelism: parallelism)
            ) { _, _ in }
        }

        let serial = try run(.serial)
        let parallel = try run(.maximumConcurrentTasks(3))
        #expect(serial.bathNoise.trajectoryIDs == ids)
        #expect(serial.bathNoise.masterSeed == 456)
        for id in ids {
            let a = serial.bathNoise.path(for: id)
            let b = parallel.bathNoise.path(for: id)
            var sa = a.makeSampler(windowDuration: 0)
            var sb = b.makeSampler(windowDuration: 0)
            var za = [Complex<Double>](repeating: .zero, count: a.channelCount)
            var zb = za
            var zaSpan = za.mutableSpan
            var zbSpan = zb.mutableSpan
            sa.sample(0.037, into: &zaSpan)
            sb.sample(0.037, into: &zbSpan)
            #expect(za == zb)
        }

        let nondeterministic = try HOPS.solveEnsemble(
            problem: hopsProblem(), configuration: config, propagation: propagation,
            execution: .init(
                trajectoryIDs: ids, randomness: .nondeterministic,
                parallelism: .serial)
        ) { _, _ in }
        #expect(nondeterministic.masterSeed == nondeterministic.bathNoise.masterSeed)
        let path = nondeterministic.bathNoise.path(for: ids.lowerBound)
        #expect(path.masterSeed == nondeterministic.masterSeed)
    }

    @Test("Public sampler keeps bounded lookback and a fresh sampler replays from the origin")
    func boundedWindow() throws {
        let path = try result(end: 0.5).bathNoise
        var sampler = path.makeSampler(windowDuration: 0.03)
        var values = [Complex<Double>](repeating: .zero, count: path.channelCount)
        var original = values
        do {
            var span = values.mutableSpan
            sampler.sample(0, into: &span)
            original = values
        }
        for n in 1...50 {
            var span = values.mutableSpan
            sampler.sample(Double(n) * 0.01, into: &span)
        }
        #expect(sampler.latestGeneratedTime >= 0.5)
        #expect(sampler.earliestAvailableTime > 0.4)

        var fresh = path.makeSampler(windowDuration: 0.03)
        var replay = [Complex<Double>](repeating: .zero, count: path.channelCount)
        var replaySpan = replay.mutableSpan
        fresh.sample(0, into: &replaySpan)
        #expect(replay == original)
    }

    @Test("Zero-bath paths replay exact zero without latent coordinates")
    func zeroBath() throws {
        let model = CorrelatedBathModel.zero(channelCount: 1)
        let path = try result(model: model).bathNoise
        #expect(path.latentCount == 0)
        var sampler = path.makeSampler(windowDuration: 0)
        var physical: [Complex<Double>] = [.one]
        var latent: [Complex<Double>] = []
        var z = physical.mutableSpan
        var x = latent.mutableSpan
        sampler.sample(0.09, physical: &z, latent: &x)
        #expect(physical == [.zero])
    }
}
