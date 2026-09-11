// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

@Suite("Correlated OU noise")
struct CorrelatedOrnsteinUhlenbeckNoiseTests {
    @Test("Single, independent, shared and partially shared baths reproduce their BCFs",
          arguments: [0, 1, 2, 3])
    func ensembleCovariances(kind: Int) {
        checkOUStatistics(noiseTestModel(kind), seed: UInt64(211 + kind))
    }

    @Test("A single OU path agrees with the exact scalar recurrence")
    func scalarRecurrence() {
        let w = Complex(0.7, 1.2)
        let g = 1.3
        let dt = 0.05
        var rng = SplitMix64(seed: 109)
        var referenceRNG = SplitMix64(seed: 109)
        var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: noiseTestModel(0), windowDuration: 0, step: dt, generator: &rng
        )
        let first: Complex<Double> = referenceRNG.nextNormal(stdev: Double(0.5).squareRoot())
        var expected = g.squareRoot() * first
        let decay = Complex<Double>.exp(-w * dt)
        let amplitude = (-g * Double.expMinusOne(-2 * w.real * dt)).squareRoot()
        var output = UniqueVector<Complex<Double>>.zero(1)
        for n in 0..<100 {
            if n > 0 {
                let gaussian: Complex<Double> = referenceRNG.nextNormal(stdev: Double(0.5).squareRoot())
                expected = decay * expected + amplitude * gaussian
            }
            process.sample(Double(n) * dt, into: &output.mutableSpan, generator: &rng)
            #expect((output[0] - expected).length < 4e-14)
        }
        #expect(rng.next() == referenceRNG.next())
    }

    @Test("Factors satisfy stationarity, including small steps and repeated poles",
          arguments: [1e-14, 1e-7, 0.05, 3.0])
    func covarianceFactors(step: Double) {
        let poles = [Complex(0.7, 1.1), Complex(0.7, 1.1), Complex(1.6, -0.8)]
        let model = CorrelatedBathModel(channelCount: 1, latentBaths: [
            noiseTestBath(poles, [[1, Complex(-0.2, 0.5), Complex(0.4, 0.1)]])
        ])
        let factory = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
            model: model, windowDuration: 0, step: step
        )
        let block = factory.blocks[0]
        let k = model.latentBaths[0].stationaryCovariance
        let reconstructedK = block.stationaryFactor.dot(block.stationaryFactor.conjugateTranspose)
        let q = block.innovationFactor.dot(block.innovationFactor.conjugateTranspose)
        expectNoiseMatricesClose(reconstructedK, k, tolerance: 3e-14)
        for p in poles.indices {
            for r in poles.indices {
                let retained = block.decay[p] * k[p, r] * block.decay[r].conjugate
                #expect((retained + q[p, r] - k[p, r]).length < 5e-14)
                if step <= 1e-14 {
                    // Subtraction-based Q loses relative accuracy at this scale.
                    #expect((q[p, r] / step - .one).length < 2e-12)
                }
            }
        }
    }

    @Test("Duplicate poles share a path without artificial diagonal jitter")
    func duplicatePoles() {
        let w = Complex(0.6, 1.2)
        let model = CorrelatedBathModel(channelCount: 2, latentBaths: [
            noiseTestBath([w, w], [[1, -1], [Complex(0.2, 0.4), Complex(0.7, -0.1)]])
        ])
        var rng = SplitMix64(seed: 12)
        var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: model, windowDuration: 0.2, step: 0.01, generator: &rng
        )
        var x = [Complex<Double>](repeating: .zero, count: 2)
        var z = x
        for n in 0..<300 {
            do {
                var zSpan = z.mutableSpan
                var xSpan = x.mutableSpan
                process.sample(Double(n) * 0.01, physical: &zSpan, latent: &xSpan, generator: &rng)
            }
            #expect((x[0] - x[1]).length < 1e-7)
            #expect(z[0].length < 1e-7)
        }
    }

    @Test("Physical/latent sampling, interpolation and solver retries use the same path")
    func cachedQueriesAndMixing() {
        let factory = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
            model: noiseTestModel(3), windowDuration: 0.4, start: -0.7, step: 0.1
        )
        var rng = NoiseCountingRNG(seed: 733)
        var process = factory.generate(generator: &rng)
        var z = [Complex<Double>](repeating: .zero, count: factory.channelCount)
        var x = [Complex<Double>](repeating: .zero, count: factory.latentCount)
        #expect(process.latentBathRanges == [0..<2, 2..<4])
        let t0 = factory.start.addingProduct(4, factory.step)
        let t1 = factory.start.addingProduct(5, factory.step)
        do {
            var xSpan = x.mutableSpan
            process.sampleLatent(t0, into: &xSpan, generator: &rng)
        }
        let left = x
        do {
            var xSpan = x.mutableSpan
            process.sampleLatent(t1, into: &xSpan, generator: &rng)
        }
        let right = x
        let draws = rng.count
        let midpoint = (t0 + t1) / 2
        do {
            var zSpan = z.mutableSpan
            var xSpan = x.mutableSpan
            process.sample(midpoint, physical: &zSpan, latent: &xSpan, generator: &rng)
        }
        for p in x.indices {
            let expected: Complex<Double> = (left[p] + right[p]) / 2.0
            #expect((x[p] - expected).length < 2e-14)
        }
        for i in z.indices {
            var expected = Complex<Double>.zero
            for (a, bath) in factory.model.latentBaths.enumerated() {
                for p in bath.poles.indices {
                    expected += bath.residues[i, p] * x[factory.latentBathRanges[a].lowerBound + p]
                }
            }
            #expect((z[i] - expected).length < 2e-14)
        }
        let savedX = x
        let savedZ = z
        for time in [t1, t0, midpoint, t1, midpoint] {
            do {
                var zSpan = z.mutableSpan
                process.sample(time, into: &zSpan, generator: &rng)
            }
            do {
                var xSpan = x.mutableSpan
                process.sampleLatent(time, into: &xSpan, generator: &rng)
            }
        }
        #expect(x == savedX && z == savedZ)
        #expect(rng.count == draws)
    }

    @Test("Window wraparound and large forward requests preserve a seeded path")
    func ringBufferAndQueryOrder() {
        let model = noiseTestModel(2)
        var aRNG = NoiseCountingRNG(seed: 91)
        var bRNG = NoiseCountingRNG(seed: 91)
        var small = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: model, windowDuration: 0.03, start: 0.13, step: 0.01, generator: &aRNG
        )
        var large = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: model, windowDuration: 4, start: 0.13, step: 0.01, generator: &bRNG
        )
        var a = [Complex<Double>](repeating: .zero, count: model.poleCount)
        var b = a
        let samplesAddress = small.samples.components
        let gaussianAddress = small.gaussians.components
        let scratchAddress = small.interpolated.components
        for n in 0...300 {
            let time = 0.13.addingProduct(Double(n), 0.01)
            do {
                var aSpan = a.mutableSpan
                small.sampleLatent(time, into: &aSpan, generator: &aRNG)
            }
            if n.isMultiple(of: 37) || n == 300 {
                do {
                    var bSpan = b.mutableSpan
                    large.sampleLatent(time, into: &bSpan, generator: &bRNG)
                }
                #expect(a == b)
                #expect(aRNG.count == bRNG.count)
                #expect(time - small.earliestAvailableTime >= min(0.03, time - 0.13) - 1e-13)
            }
        }
        #expect(small.samples.components == samplesAddress)
        #expect(small.gaussians.components == gaussianAddress)
        #expect(small.interpolated.components == scratchAddress)
        #expect(small.samples.count == 5 * model.poleCount)
        let retained = small.earliestAvailableTime
        do {
            var aSpan = a.mutableSpan
            small.sampleLatent(retained, into: &aSpan, generator: &aRNG)
        }
        do {
            var bSpan = b.mutableSpan
            large.sampleLatent(retained, into: &bSpan, generator: &bRNG)
        }
        #expect(a == b)
        // Reset reuses every buffer and reproduces fresh construction with the same RNG.
        aRNG = NoiseCountingRNG(seed: 91)
        bRNG = NoiseCountingRNG(seed: 91)
        small.reset(generator: &aRNG)
        large.reset(generator: &bRNG)
        do {
            var aSpan = a.mutableSpan
            small.sampleLatent(0.33, into: &aSpan, generator: &aRNG)
        }
        do {
            var bSpan = b.mutableSpan
            large.sampleLatent(0.33, into: &bSpan, generator: &bRNG)
        }
        #expect(a == b && aRNG.count == bRNG.count)
        #expect(small.samples.components == samplesAddress)
    }

    @Test("Zero models overwrite outputs and consume no random numbers")
    func zeroModel() {
        var rng = NoiseCountingRNG(seed: 1)
        var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: .zero(channelCount: 3), windowDuration: 0, start: -2, step: 0.01, generator: &rng
        )
        var z = [Complex<Double>](repeating: .one, count: 3)
        var x: [Complex<Double>] = []
        for t in [-2.0, 0, 10.001, 10000] {
            do {
                var zSpan = z.mutableSpan
                var xSpan = x.mutableSpan
                process.sample(t, physical: &zSpan, latent: &xSpan, generator: &rng)
            }
            #expect(z == [.zero, .zero, .zero])
        }
        process.reset(generator: &rng)
        #expect(process.earliestAvailableTime == -2)
        #expect(rng.count == 0)
    }

    @Test("A zero residue row remains exactly zero while latent noise evolves")
    func zeroResidues() {
        let model = CorrelatedBathModel(channelCount: 1, latentBaths: [
            noiseTestBath([Complex(0.6, 0.8)], [[0]])
        ])
        var rng = SplitMix64(seed: 18)
        var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            model: model, windowDuration: 0, step: 0.1, generator: &rng
        )
        var z: [Complex<Double>] = [1]
        var x: [Complex<Double>] = [0]
        do {
            var zSpan = z.mutableSpan
            var xSpan = x.mutableSpan
            process.sample(2, physical: &zSpan, latent: &xSpan, generator: &rng)
        }
        #expect(z[0] == .zero)
        #expect(x[0].length > 0)
    }

    @Test("Fitter results feed the sampler without residue conversion")
    func fittedModel() throws {
        let target = noiseTestModel(0)
        let times = (0..<30).map { Double($0) * 0.1 }
        let fit = try CorrelatedBathFitter.fitBathCorrelation(
            times: times, values: times.map { target.bathCorrelation(at: $0)[0, 0] },
            options: .init(maximumPencilPoleCount: 1, latentBathCount: 1)
        )
        #expect(fit.diagnostics.relativeRMSError < 1e-5)
        checkOUStatistics(fit.model, seed: 58)
    }
}

private func checkOUStatistics(_ model: CorrelatedBathModel, seed: UInt64) {
    let factory = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
        model: model, windowDuration: 0.1, start: -1, step: 0.05
    )
    var rng = SplitMix64(seed: seed)
    var process = factory.generate(generator: &rng)
    var z0 = [Complex<Double>](repeating: .zero, count: model.channelCount)
    var z1 = z0
    var z2 = z0
    var z3 = z0
    var x0 = [Complex<Double>](repeating: .zero, count: model.poleCount)
    var x1 = x0
    var physical = NoiseTestMoments(dimension: model.channelCount)
    var shifted = NoiseTestMoments(dimension: model.channelCount)
    var latent = NoiseTestMoments(dimension: model.poleCount)
    for _ in 0..<10000 {
        process.reset(generator: &rng)
        do {
            var z0Span = z0.mutableSpan
            var x0Span = x0.mutableSpan
            process.sample(-1, physical: &z0Span, latent: &x0Span, generator: &rng)
        }
        do {
            var z1Span = z1.mutableSpan
            var x1Span = x1.mutableSpan
            process.sample(-0.65, physical: &z1Span, latent: &x1Span, generator: &rng)
        }
        do {
            var z2Span = z2.mutableSpan
            process.sample(-0.2, into: &z2Span, generator: &rng)
        }
        do {
            var z3Span = z3.mutableSpan
            process.sample(0.15, into: &z3Span, generator: &rng)
        }
        physical.record(z0, z1)
        shifted.record(z2, z3)
        latent.record(x0, x1)
    }
    let alpha0 = model.bathCorrelation(at: 0)
    let alphaLag = model.bathCorrelation(at: 0.35)
    physical.check(equal: alpha0, lag: alphaLag)
    shifted.check(equal: alpha0, lag: alphaLag)
    var k = Matrix<Complex<Double>>.zeros(rows: model.poleCount, columns: model.poleCount)
    var lag = k
    for (a, bath) in model.latentBaths.enumerated() {
        let offset = factory.latentBathRanges[a].lowerBound
        for p in bath.poles.indices {
            for q in bath.poles.indices {
                let value = Complex<Double>.one / (bath.poles[p] + bath.poles[q].conjugate)
                k[offset + p, offset + q] = value
                lag[offset + p, offset + q] = Complex<Double>.exp(-bath.poles[p] * 0.35) * value
            }
        }
    }
    latent.check(equal: k, lag: lag)
}
