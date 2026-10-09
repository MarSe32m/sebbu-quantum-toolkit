// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing
@testable import SebbuQuantumToolkit

func potentialTestModel() -> CorrelatedBathModel {
    .independentExponentials(amplitudes: [0.4, 0.13], poles: [Complex(0.7, 1.2), Complex(1.8, -0.3)])
}

func potentialTestDefinition(_ potential: StochasticPotential, start: Double = 0, end: Double = 1)
    -> HOPS.PotentialNoiseDefinition {
    .init(potential, propagation: hopsPropagation(end: end, start: start, maximumStep: 0.3), fallbackStep: 0.025)
}

@Suite("HOPS potential noise", .serialized)
struct HOPSPotentialNoiseTests {
    @Test("Independent exponentials reproduce amplitudes and negative lags")
    func exponentialNormalization() {
        let model = potentialTestModel()
        #expect(model.channelCount == 1 && model.latentBaths.count == 2 && model.poleCount == 2)
        for t in [-1.0, -0.25, 0, 0.25, 1] {
            let positive = 0.4 * Complex<Double>.exp(-Complex(0.7, 1.2) * abs(t))
                + 0.13 * Complex<Double>.exp(-Complex(1.8, -0.3) * abs(t))
            let expected = t < 0 ? positive.conjugate : positive
            #expect((model.bathCorrelation(at: t)[0, 0] - expected).length < 2e-15)
        }
        let zero = CorrelatedBathModel.independentExponentials(amplitudes: [0], poles: [Complex(1, 2)])
        #expect(zero.poleCount == 0 && zero.channelCount == 1)
        #expect(CorrelatedBathModel.independentExponentials(amplitudes: [], poles: []).poleCount == 0)
    }

    @Test("OU stationary covariance, properness and complex cross-channel mixing")
    func stationaryStatistics() {
        // A single complex mixing vector fixes the orientation of K_01 as well
        // as its magnitude, and exercises correlated physical channels.
        let b = [Complex<Double>(0.7, 0.2), Complex(-0.3, 0.6)]
        let pole = Complex<Double>(0.8, 1.3)
        let model = CorrelatedBathModel(channelCount: 2, latentBaths: [.init(
            poles: [pole], residues: .init(elements: b.map { (2 * pole.real).squareRoot() * $0 },
                                           rows: 2, columns: 1))])
        let definition = potentialTestDefinition(.init(
            couplingOperators: [.constant(hcL), .constant(hcR)], covarianceModel: model))
        let count = 6000
        var covariance = [Complex<Double>](repeating: .zero, count: 4)
        var lagged = covariance
        var properness = covariance
        var mean = [Complex<Double>](repeating: .zero, count: 2)
        var atZero = UniqueVector<Complex<Double>>.zero(2)
        var atLag = UniqueVector<Complex<Double>>.zero(2)
        for id in 0..<count {
            var sampler = HOPS.PotentialNoiseSampler(
                definition: definition, seed: 4321, trajectoryID: UInt64(id), ensembleSampling: .independent)
            sampler.sample(0, into: &atZero.mutableSpan)
            sampler.sample(0.25, into: &atLag.mutableSpan)
            for i in 0..<2 {
                mean[i] += atZero[i] / Double(count)
                for j in 0..<2 {
                    covariance[2 * i + j] += atZero[i] * atZero[j].conjugate / Double(count)
                    lagged[2 * i + j] += atLag[i] * atZero[j].conjugate / Double(count)
                    properness[2 * i + j] += atZero[i] * atZero[j] / Double(count)
                }
            }
        }
        for i in 0..<2 {
            #expect(mean[i].length < 6 * b[i].length / Double(count).squareRoot())
            for j in 0..<2 {
                // Six standard errors, including the sqrt(2) properness bound.
                let tolerance = 6 * Double(2).squareRoot() * b[i].length * b[j].length / Double(count).squareRoot()
                #expect((covariance[2 * i + j] - model.bathCorrelation(at: 0)[i, j]).length < tolerance)
                #expect((lagged[2 * i + j] - model.bathCorrelation(at: 0.25)[i, j]).length < tolerance)
                #expect(properness[2 * i + j].length < tolerance)
            }
        }
    }

    @Test("OU replay, retries, antithetic coefficients and purpose separation")
    func referencePaths() {
        let definition = potentialTestDefinition(.init(couplingOperator: .constant(hcL), covarianceModel: potentialTestModel()))
        let evenPath = HOPS.PotentialNoisePath(definition: definition, masterSeed: 5, trajectoryID: 8, ensembleSampling: .antithetic)
        let oddPath = HOPS.PotentialNoisePath(definition: definition, masterSeed: 5, trajectoryID: 9, ensembleSampling: .antithetic)
        var even = evenPath.makeSampler()
        var odd = oddPath.makeSampler()
        var replay = evenPath.makeSampler(windowDuration: 0.4)
        var a = UniqueVector<Complex<Double>>.zero(1)
        var b = UniqueVector<Complex<Double>>.zero(1)
        var c = UniqueVector<Complex<Double>>.zero(1)
        for time in [0.0, 0.075, 0.21, 0.075, 0.1, 0.21, 0.4, 0.31, 0.4, 0.6] {
            even.sample(time, into: &a.mutableSpan)
            odd.sample(time, into: &b.mutableSpan)
            replay.sample(time, into: &c.mutableSpan)
            #expect(a[0] == -b[0] && a[0] == c[0])
        }
        #expect(evenPath.meshOrigin == 0 && evenPath.generationAlgorithm == .correlatedOUV1)
        let purposes: [TrajectoryRandomNumberGenerator.Purpose] = [
            .unspecified, .gaussianWhiteNoise, .coloredNoiseGeneration, .mcwfJumps, .stochasticPotentialGeneration]
        #expect(purposes.map(\.rawValue) == [0, 1, 2, 3, 4])
        let words = purposes.map { purpose in
            var rng = TrajectoryRandomNumberGenerator(seed: 5, trajectoryID: 8, purpose: purpose)
            return rng.next()
        }
        #expect(Set(words).count == purposes.count)
    }

    @Test("Zero covariance and zero FFT power allocate no realized source or RNG")
    func zeroSources() {
        for source: StochasticPotential.Source in [
            .covarianceModel(.zero(channelCount: 1)),
            .fft(options: .init(dtMax: 0.1, deltaOmegaMax: 1, omegaMax: 2), powerDensity: { _ in
                .zeros(rows: 1, columns: 1)
            })
        ] {
            let definition = potentialTestDefinition(.init(couplingOperator: .constant(hcL), source: source))
            var sampler = HOPS.PotentialNoiseSampler(
                definition: definition, seed: 3, trajectoryID: 1, ensembleSampling: .antithetic)
            let hasOU = sampler.ou != nil
            #expect(sampler.rng == nil && !hasOU && sampler.fft == nil)
            #expect(sampler.integrationGrid == nil)
            var output = UniqueVector<Complex<Double>>([Complex(3, 4)])
            sampler.sample(0.5, into: &output.mutableSpan)
            #expect(output[0] == .zero && sampler.rng == nil)
        }
    }

    @Test("FFT weights are frozen once, replay respects origin and antithetic sampling")
    func fftPreparationAndOrigin() throws {
        let calls = Mutex(0)
        let potential = StochasticPotential(
            couplingOperator: .constant(hcL), fft: .init(dtMax: 0.125, deltaOmegaMax: 0.5, omegaMax: 4)
        ) { _ in calls.withLock { $0 += 1 }; return 0.3 }
        let definition = potentialTestDefinition(potential, start: 8, end: 8.5)
        let callCount = calls.withLock { $0 }
        #expect(callCount == 8)
        let shiftedPath = HOPS.PotentialNoisePath(definition: definition, masterSeed: 11, trajectoryID: 0, ensembleSampling: .antithetic)
        var shifted = shiftedPath.makeSampler()
        var repeated = shiftedPath.makeSampler()
        var odd = HOPS.PotentialNoisePath(definition: definition, masterSeed: 11, trajectoryID: 1, ensembleSampling: .antithetic).makeSampler()
        var rng = TrajectoryRandomNumberGenerator(seed: 11, trajectoryID: 0, ensembleSampling: .antithetic,
                                                purpose: .stochasticPotentialGeneration)
        guard case .fft(let generator, _) = definition.source else { Issue.record("Expected FFT source"); return }
        let direct = generator.generateProcess(generator: &rng.gaussian)
        var a = UniqueVector<Complex<Double>>.zero(1)
        var b = UniqueVector<Complex<Double>>.zero(1)
        var c = UniqueVector<Complex<Double>>.zero(1)
        for time in [8.0, 8.375, 8.125, 8.5, 8.125] {
            shifted.sample(time, into: &a.mutableSpan)
            direct.sample(time - 8, into: &b.mutableSpan)
            repeated.sample(time, into: &c.mutableSpan)
            #expect(a[0] == b[0] && a[0] == c[0])
            odd.sample(time, into: &c.mutableSpan)
            #expect(a[0] == -c[0])
        }
        #expect(calls.withLock { $0 } == callCount)
        #expect(shifted.integrationGrid == nil && shiftedPath.meshOrigin == 8)
        #expect(shiftedPath.generationAlgorithm == .positiveFrequencyFFTV1)

        var config = hcConfiguration(0)
        config.stochasticPotential = potential
        _ = try HOPS.solveEnsemble(
            problem: hopsProblem(), configuration: config, propagation: hopsPropagation(end: 0.125),
            execution: .init(trajectories: 4, seed: 1, parallelism: .serial)
        ) { _, _ in }
        #expect(calls.withLock { $0 } == 2 * callCount)
    }

    @Test("FFT positive-frequency normalization, lag sign and channel mixing")
    func fftCovariance() {
        let b = [Complex<Double>(1), Complex(0.3, 0.7)]
        let potential = StochasticPotential(
            couplingOperators: [.constant(hcL), .constant(hcR)],
            fft: .init(dtMax: 0.05, deltaOmegaMax: 0.4, omegaMax: 4)
        ) { _ in .init(elements: b.flatMap { x in b.map { x * $0.conjugate } }, rows: 2, columns: 2) }
        let definition = potentialTestDefinition(potential, end: 0.5)
        let count = 3000
        var sum = Complex<Double>.zero
        var variance = 0.0
        var properness = Complex<Double>.zero
        var a = UniqueVector<Complex<Double>>.zero(2)
        var c = UniqueVector<Complex<Double>>.zero(2)
        for id in 0..<count {
            var sampler = HOPS.PotentialNoiseSampler(definition: definition, seed: 935, trajectoryID: UInt64(id), ensembleSampling: .independent)
            sampler.sample(0, into: &a.mutableSpan)
            sampler.sample(0.4, into: &c.mutableSpan)
            variance += a[0].lengthSquared / Double(count)
            properness += a[0] * a[0] / Double(count)
            sum += c[0] * a[1].conjugate / Double(count)
        }
        let expected = (Complex<Double>.one - .exp(Complex(0, -1.6))) / Complex(0, 0.4) * b[1].conjugate
        let tolerance = 6 * Double(2).squareRoot() * 4 / Double(count).squareRoot()
        #expect(abs(variance - 4) < tolerance)
        #expect(properness.length < tolerance)
        #expect((sum - expected).length < tolerance * b[1].length + 0.006)
    }
}
