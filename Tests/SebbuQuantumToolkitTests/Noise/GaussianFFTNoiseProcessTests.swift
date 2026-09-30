// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing
@testable import SebbuQuantumToolkit

@Suite("Gaussian FFT noise")
struct GaussianFFTNoiseProcessTests {
    @Test("The requested endpoint is covered, including sub-step durations",
          arguments: [1e-12, 0.001, 0.37, 2.0])
    func endpointCoverage(duration: Double) {
        var rng = SplitMix64(seed: 42)
        let noise = GaussianFFTNoiseProcess(
            tMax: duration, dtMax: 0.1, deltaOmegaMax: 0.7, omegaMax: 3,
            generator: &rng, spectralDensity: { Double.exp(-$0) }
        )
        #expect(noise.spline.x.count >= 2)
        #expect(noise.spline.x.first == 0)
        #expect(noise.tMax >= duration)
        #expect(noise.spline.x[noise.spline.x.count - 2] < duration)
        for index in 1..<noise.spline.x.count {
            let step = noise.spline.x[index] - noise.spline.x[index - 1]
            #expect(step > 0 && step <= 0.1 * (1 + 1e-13))
        }
        for time in [0, duration / 2, duration, noise.tMax] {
            let value = noise.sample(time)
            #expect(value.real.isFinite && value.imaginary.isFinite)
        }
    }

    @Test("FFT knots agree with a direct positive-frequency Fourier sum")
    func directFourierReference() {
        // Four midpoint bins on [0, 4], each of width one. This independent
        // finite Fourier sum detects the exponent sign, normalization, frequency
        // offset, and the N versus N-1 time-grid error.
        let frequencies = [0.5, 1.5, 2.5, 3.5]
        var referenceRNG = SplitMix64(seed: 0x1234)
        let coefficients: [Complex<Double>] = frequencies.map { omega in
            let gaussian: Complex<Double> =
                referenceRNG.nextNormal(stdev: Double(0.5).squareRoot())
            return Double.exp(-omega / 2) * gaussian
        }
        func reference(_ time: Double) -> Complex<Double> {
            var result = Complex<Double>.zero
            for index in frequencies.indices {
                result += coefficients[index]
                    * Complex<Double>(length: 1, phase: -frequencies[index] * time)
            }
            return result
        }

        var rng = SplitMix64(seed: 0x1234)
        let noise = GaussianFFTNoiseProcess(
            tMax: 0.7, dtMax: 0.01, deltaOmegaMax: 1, omegaMax: 4,
            generator: &rng, spectralDensity: { Double.exp(-$0) }
        )
        for time in noise.spline.x {
            expectNoiseClose(noise.sample(time), reference(time), tolerance: 2e-12)
        }
        // Include both endpoint intervals and off-grid solver stage times.
        for time in [0.001, 0.017, 0.213, 0.699, 0.7] {
            expectNoiseClose(noise.sample(time), reference(time), tolerance: 2e-9)
        }
        #expect(rng.next() == referenceRNG.next())
    }

    @Test("A single frequency rotates with the correct sign between knots")
    func singleFrequencyInterpolation() {
        var rng = SplitMix64(seed: 981)
        let noise = GaussianFFTNoiseProcess(
            tMax: 1, dtMax: 0.02, deltaOmegaMax: 2, omegaMax: 2,
            generator: &rng, spectralDensity: { _ in 1 }
        )
        // The only midpoint frequency is one; its random amplitude cancels.
        let initial = noise.sample(0)
        for time in [0.003, 0.27, 0.51, 0.999, 1.0] {
            let expected = initial * Complex<Double>(length: 1, phase: -time)
            expectNoiseClose(noise.sample(time), expected,
                             tolerance: 1e-9 * max(1, initial.length))
        }
    }

    @Test("Preparation evaluates only the requested band, once")
    func preparesSpectrumOnce() {
        let evaluations = NoiseSpectralEvaluations()
        let factory = GaussianFFTNoiseProcessGenerator(
            tMax: 0.6, dtMax: 0.01, deltaOmegaMax: 0.4, omegaMax: 1.1
        ) { omega in
            evaluations.record(omega)
            return 1
        }
        let frequencies = evaluations.values
        #expect(frequencies.count == 3)
        #expect(frequencies.allSatisfy { $0 > 0 && $0 < 1.1 })
        var rng = SplitMix64(seed: 24)
        _ = factory.generate(generator: &rng)
        _ = factory.generate(generator: &rng)
        #expect(evaluations.values == frequencies)
    }

    @Test("Midpoint integration avoids singular spectral endpoints")
    func integrableEndpoint() {
        let evaluations = NoiseSpectralEvaluations()
        let factory = GaussianFFTNoiseProcessGenerator(
            tMax: 0.5, dtMax: 0.05, deltaOmegaMax: 0.1, omegaMax: 1
        ) { omega in
            evaluations.record(omega)
            return 1 / omega.squareRoot()
        }
        var rng = SplitMix64(seed: 9)
        let noise = factory.generate(generator: &rng)
        #expect(evaluations.values.allSatisfy { $0 > 0 && $0 < 1 })
        #expect(noise.sample(0.5).length.isFinite)
    }

    @Test("Time refinement preserves the spectrum and random coefficients")
    func timeRefinementPreservesSpectrum() {
        let coarse = GaussianFFTNoiseProcessGenerator(
            tMax: 0.9, dtMax: 0.1, deltaOmegaMax: 0.4, omegaMax: 2
        ) { Double.exp(-$0) }
        let fine = GaussianFFTNoiseProcessGenerator(
            tMax: 0.9, dtMax: 0.01, deltaOmegaMax: 0.4, omegaMax: 2
        ) { Double.exp(-$0) }
        var coarseRNG = SplitMix64(seed: 193)
        var fineRNG = SplitMix64(seed: 193)
        let coarseNoise = coarse.generate(generator: &coarseRNG)
        let fineNoise = fine.generate(generator: &fineRNG)
        for time in coarseNoise.spline.x where time <= 0.9 {
            expectNoiseClose(coarseNoise.sample(time), fineNoise.sample(time),
                             tolerance: 2e-12)
        }
        #expect(coarseRNG.next() == fineRNG.next())
    }

    @Test("One-shot and prepared sampling agree for an identical seed")
    func preparedMatchesOneShot() {
        let factory = GaussianFFTNoiseProcessGenerator(
            tMax: 0.75, dtMax: 0.05, deltaOmegaMax: 0.2, omegaMax: 3
        ) { Double.exp(-$0) }
        var firstRNG = SplitMix64(seed: 77)
        var secondRNG = SplitMix64(seed: 77)
        let prepared = factory.generate(generator: &firstRNG)
        let oneShot = GaussianFFTNoiseProcess(
            tMax: 0.75, dtMax: 0.05, deltaOmegaMax: 0.2, omegaMax: 3,
            generator: &secondRNG, spectralDensity: { Double.exp(-$0) }
        )
        for time in [0, 0.123, 0.74, 0.75] {
            #expect(prepared.sample(time) == oneShot.sample(time))
        }
        #expect(firstRNG.next() == secondRNG.next())
        let next = factory.generate(generator: &firstRNG)
        #expect(next.sample(0) != prepared.sample(0))
    }

    @Test("The default cutoff is pi divided by dtMax")
    func defaultCutoff() {
        let implicit = GaussianFFTNoiseProcessGenerator(
            tMax: 0.7, dtMax: 0.1, deltaOmegaMax: 0.5
        ) { Double.exp(-$0) }
        let explicit = GaussianFFTNoiseProcessGenerator(
            tMax: 0.7, dtMax: 0.1, deltaOmegaMax: 0.5, omegaMax: .pi / 0.1
        ) { Double.exp(-$0) }
        var firstRNG = SplitMix64(seed: 14)
        var secondRNG = SplitMix64(seed: 14)
        let first = implicit.generate(generator: &firstRNG)
        let second = explicit.generate(generator: &secondRNG)
        for time in [0, 0.123, 0.7] {
            #expect(first.sample(time) == second.sample(time))
        }
        #expect(firstRNG.next() == secondRNG.next())
    }

    @Test("Zero spectral density produces exactly zero noise")
    func zeroSpectrum() {
        var rng = SplitMix64(seed: 104)
        let noise = GaussianFFTNoiseProcess(
            tMax: 0.3, dtMax: 0.05, deltaOmegaMax: 0.5, omegaMax: 2,
            generator: &rng, spectralDensity: { _ in 0 }
        )
        for time in [0, 0.017, 0.3, noise.tMax] {
            #expect(noise.sample(time) == .zero)
            #expect(noise.conjugate().sample(time) == .zero)
            #expect(noise.antithetic().sample(time) == .zero)
        }
    }

    @Test("Conjugation, antithetics, and repeated queries preserve the path")
    func pathTransformations() {
        var rng = SplitMix64(seed: 181)
        let noise = GaussianFFTNoiseProcess(
            tMax: 1, dtMax: 0.05, deltaOmegaMax: 0.5, omegaMax: 4,
            generator: &rng, spectralDensity: { Double.exp(-$0) }
        )
        let conjugate = noise.conjugate()
        let antithetic = noise.antithetic()
        let first = noise.sample(0.413)
        for time in [1, 0, 0.413, 0.19, 0.997, 0.413] {
            let value = noise.sample(time)
            #expect(conjugate.sample(time) == value.conjugate)
            #expect(antithetic.sample(time) == -value)
            #expect(conjugate.conjugate().sample(time) == value)
            #expect(antithetic.antithetic().sample(time) == value)
            #expect(conjugate.antithetic().sample(time) == -value.conjugate)
        }
        #expect(noise.sample(0.413) == first)
    }

    @Test("A shared preparation is reproducible across concurrent trajectories")
    func concurrentSampling() async {
        let factory = GaussianFFTNoiseProcessGenerator(
            tMax: 0.5, dtMax: 0.05, deltaOmegaMax: 0.5, omegaMax: 3
        ) { Double.exp(-$0) }
        let expected = (0..<12).map { index in
            var rng = SplitMix64(seed: UInt64(index))
            return factory.generate(generator: &rng).sample(0.213)
        }
        var actual = [Complex<Double>](repeating: .zero, count: expected.count)
        await withTaskGroup(of: (Int, Complex<Double>).self) { group in
            for index in expected.indices {
                group.addTask {
                    var rng = SplitMix64(seed: UInt64(index))
                    return (index, factory.generate(generator: &rng).sample(0.213))
                }
            }
            for await (index, sample) in group { actual[index] = sample }
        }
        #expect(actual == expected)
    }

    @Test("Ensemble covariance matches an analytic BCF and is proper complex")
    func analyticCovariance() {
        // J(w) = exp(-w) gives alpha(t) = 1 / (1 + i*t). The omitted tail
        // beyond eight is < 0.00034; midpoint and interpolation errors are small
        // compared with the 8192-realization Monte Carlo uncertainty.
        let factory = GaussianFFTNoiseProcessGenerator(
            tMax: 1.1, dtMax: 0.25, deltaOmegaMax: 0.25, omegaMax: 8
        ) { Double.exp(-$0) }
        let count = 8192
        var rng = SplitMix64(seed: 0xFA87_71C2)
        var mean = Complex<Double>.zero
        var variance = 0.0
        var shiftedVariance = 0.0
        var covariance = Complex<Double>.zero
        var shiftedCovariance = Complex<Double>.zero
        var pseudoCovariance = Complex<Double>.zero
        var forceVariance = 0.0
        for _ in 0..<count {
            let noise = factory.generate(generator: &rng)
            let z0 = noise.sample(0)
            let z1 = noise.sample(0.4)
            let z2 = noise.sample(0.7)
            let z3 = noise.sample(1.1)
            mean += z0
            variance += z0.lengthSquared
            shiftedVariance += z2.lengthSquared
            covariance += z1 * z0.conjugate
            shiftedCovariance += z3 * z2.conjugate
            pseudoCovariance += z1 * z0
            forceVariance += 4 * z0.real * z0.real
        }
        let scale = 1 / Double(count)
        let expected = Complex<Double>.one / Complex<Double>(1, 0.4)
        #expect((mean * scale).length < 0.05)
        #expect(abs(variance * scale - 1) < 0.06)
        #expect(abs(shiftedVariance * scale - 1) < 0.06)
        expectNoiseClose(covariance * scale, expected, tolerance: 0.07)
        expectNoiseClose(shiftedCovariance * scale, expected, tolerance: 0.07)
        #expect((pseudoCovariance * scale).length < 0.07)
        // f = 2 Re(z) must have variance 2 alpha(0), not alpha(0).
        #expect(abs(forceVariance * scale - 2) < 0.12)
    }

    @Test("The multichannel factory returns scalar paths covering its interval",
          arguments: [0.0001, 0.37])
    func multichannelEndpointCoverage(duration: Double) {
        let factory = GaussianFFTMultiNoiseProcessGenerator(
            tMax: duration, dtMax: 0.1, deltaOmegaMax: 0.5, omegaMax: 3
        ) { omega in
            Matrix<Complex<Double>>(
                elements: [Complex(Double.exp(-omega))], rows: 1, columns: 1
            )
        }
        var rng = SplitMix64(seed: 83)
        let noises = factory.generate(generator: &rng)
        #expect(noises.count == 1)
        let noise = noises[0]
        #expect(noise.tMax >= duration)
        #expect(noise.sample(duration).length.isFinite)
        #expect(noise.spline.x.count >= 2)
        // Padding refines the time mesh without extending the spectral cutoff.
        #expect(noise.spline.x[1] > 0 && noise.spline.x[1] <= 0.1)
    }

    @Test("Thermal wrappers retain endpoint coverage after scalar sampling")
    func thermalWrapperCompatibility() {
        var rng = SplitMix64(seed: 552)
        let potential = GaussianFFTThermalNoiseProcess(
            temperature: 0.7, tMax: 0.37, dtMax: 0.03,
            deltaOmegaMax: 0.3, omegaMax: 4, generator: &rng,
            spectralDensity: { $0 * Double.exp(-$0) }
        )
        let fullBCF = GaussianFFTThermalBCFNoiseProcess(
            temperature: 0.7, tMax: 0.37, dtMax: 0.03,
            deltaOmegaMax: 0.3, omegaMax: 4, generator: &rng,
            spectralDensity: { $0 * Double.exp(-$0) }
        )
        #expect(potential.tMax >= 0.37)
        #expect(potential.sample(0.37).length.isFinite)
        #expect(fullBCF.spline.x.last! >= 0.37)
        #expect(fullBCF.sample(0.37).length.isFinite)
    }
}

private final class NoiseSpectralEvaluations: Sendable {
    private let frequencies = Mutex<[Double]>([])

    func record(_ omega: Double) {
        frequencies.withLock { $0.append(omega) }
    }

    var values: [Double] { frequencies.withLock { $0 } }
}

private func expectNoiseClose(
    _ actual: Complex<Double>, _ expected: Complex<Double>,
    tolerance: Double, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect((actual - expected).length < tolerance, sourceLocation: sourceLocation)
}
