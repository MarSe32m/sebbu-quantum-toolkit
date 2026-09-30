// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing
@testable import SebbuQuantumToolkit

@Suite("Multichannel Gaussian FFT noise")
struct GaussianFFTMultiNoiseProcessTests {
    @Test("One channel matches the scalar generator for an identical seed")
    func scalarAgreement() {
        let scalar = GaussianFFTNoiseProcessGenerator(
            tMax: 0.7, dtMax: 0.02, deltaOmegaMax: 0.5, omegaMax: 3
        ) { Double.exp(-$0) }
        let multi = GaussianFFTMultiNoiseProcessGenerator(
            tMax: 0.7, dtMax: 0.02, deltaOmegaMax: 0.5, omegaMax: 3
        ) { Matrix(elements: [Complex(Double.exp(-$0))], rows: 1, columns: 1) }
        var aRNG = SplitMix64(seed: 71)
        var bRNG = SplitMix64(seed: 71)
        let a = scalar.generate(generator: &aRNG)
        let b = multi.generate(generator: &bRNG)[0]
        #expect(a.spline.x == b.spline.x)
        for time in [0, 0.003, 0.23, 0.691, 0.7, a.tMax] {
            #expect((a.sample(time) - b.sample(time)).length < 1e-12)
        }
        #expect(aRNG.next() == bRNG.next())
    }

    @Test("Complex spectral factors reproduce the full Hermitian spectrum")
    func factors() {
        let factory = fftTestFactory(kind: 2)
        for k in factory.grid.frequencies.indices {
            var expected = fftTestSpectrum(factory.grid.frequencies[k], kind: 2)
            for i in expected.elements.indices { expected.elements[i] *= factory.grid.frequencyStep }
            let factor = factory.factors[k]
            expectNoiseMatricesClose(factor.dot(factor.conjugateTranspose), expected, tolerance: 4e-14)
        }
    }

    @Test("FFT nodes and spectral tangents agree with a direct Fourier sum")
    func directFourierSum() {
        let factory = fftTestFactory(kind: 2, dt: 0.01)
        var referenceRNG = SplitMix64(seed: 274)
        var coefficients = [[Complex<Double>]]()
        for k in factory.grid.frequencies.indices {
            let white: [Complex<Double>] = (0..<factory.channelCount).map { _ in
                referenceRNG.nextNormal(stdev: Double(0.5).squareRoot())
            }
            coefficients.append(factory.factors[k].dot(Vector(white)).components)
        }
        var rng = SplitMix64(seed: 274)
        let noise = factory.generateProcess(generator: &rng)
        var output = [Complex<Double>](repeating: .zero, count: noise.channelCount)
        let draws = rng.next()
        #expect(draws == referenceRNG.next())
        let times = factory.grid.times + [0.001, 0.197, 0.619, 1.199, 1.2]
        for time in times {
            do {
                var outputSpan = output.mutableSpan
                noise.sample(time, into: &outputSpan)
            }
            for i in output.indices {
                var expected = Complex<Double>.zero
                for k in coefficients.indices {
                    expected += coefficients[k][i]
                        * Complex<Double>(length: 1, phase: -factory.grid.frequencies[k] * time)
                }
                #expect((output[i] - expected).length < 2e-8)
            }
        }
    }

    @Test("Independent, shared and partially shared spectra have correct ensemble moments",
          arguments: [0, 1, 2])
    func ensembleMoments(kind: Int) {
        let factory = fftTestFactory(kind: kind)
        var rng = SplitMix64(seed: UInt64(450 + kind))
        var a = [Complex<Double>](repeating: .zero, count: 3)
        var b = a
        var c = a
        var d = a
        var initial = NoiseTestMoments(dimension: 3)
        var shifted = NoiseTestMoments(dimension: 3)
        for _ in 0..<5000 {
            let noise = factory.generateProcess(generator: &rng)
            do {
                var aSpan = a.mutableSpan
                noise.sample(0, into: &aSpan, generator: &rng)
            }
            do {
                var bSpan = b.mutableSpan
                noise.sample(0.35, into: &bSpan, generator: &rng)
            }
            do {
                var cSpan = c.mutableSpan
                noise.sample(0.7, into: &cSpan, generator: &rng)
            }
            do {
                var dSpan = d.mutableSpan
                noise.sample(1.05, into: &dSpan, generator: &rng)
            }
            initial.record(a, b)
            shifted.record(c, d)
        }
        var equal = Matrix<Complex<Double>>.zeros(rows: 3, columns: 3)
        var lag = equal
        for omega in factory.grid.frequencies {
            let density = fftTestSpectrum(omega, kind: kind)
            for i in equal.elements.indices {
                equal.elements[i] += density.elements[i] * factory.grid.frequencyStep
                lag.elements[i] += density.elements[i] * factory.grid.frequencyStep
                    * Complex<Double>(length: 1, phase: -omega * 0.35)
            }
        }
        initial.check(equal: equal, lag: lag)
        shifted.check(equal: equal, lag: lag)
    }

    @Test("Preparation evaluates only midpoint frequencies inside the requested cutoff")
    func preparesOnce() {
        let evaluations = Mutex<[Double]>([])
        let factory = GaussianFFTMultiNoiseProcessGenerator(
            tMax: 0.3, dtMax: 0.02, deltaOmegaMax: 0.4, omegaMax: 1.1
        ) { omega in
            evaluations.withLock { $0.append(omega) }
            return Matrix(elements: [Complex(1 / omega.squareRoot())], rows: 1, columns: 1)
        }
        let original = evaluations.withLock { $0 }
        #expect(original.count == 3)
        #expect(original.allSatisfy { $0 > 0 && $0 < 1.1 })
        var rng = SplitMix64(seed: 5)
        _ = factory.generate(generator: &rng)
        _ = factory.generateProcess(generator: &rng)
        #expect(evaluations.withLock { $0 } == original)
    }

    @Test("Changing FFT padding leaves the spectrum and seeded path unchanged")
    func paddingIndependence() {
        let coarse = fftTestFactory(kind: 2, dt: 0.1)
        let fine = fftTestFactory(kind: 2, dt: 0.01)
        var aRNG = SplitMix64(seed: 31)
        var bRNG = SplitMix64(seed: 31)
        let a = coarse.generate(generator: &aRNG)
        let b = fine.generate(generator: &bRNG)
        #expect(coarse.grid.frequencies == fine.grid.frequencies)
        for time in a[0].spline.x where time <= 1.2 {
            for i in a.indices { #expect((a[i].sample(time) - b[i].sample(time)).length < 2e-12) }
        }
        #expect(aRNG.next() == bRNG.next())
    }

    @Test("Rank-deficient spectra preserve deterministic channel relations")
    func rankDeficiency() {
        // All integer entries are exactly representable; eigenvalues include zero.
        let v: [Complex<Double>] = [1, Complex(0, 1), 2]
        let factory = GaussianFFTMultiNoiseProcessGenerator(
            tMax: 0.7, dtMax: 0.02, deltaOmegaMax: 0.5, omegaMax: 3
        ) { omega in
            var result = Matrix<Complex<Double>>.zeros(rows: 3, columns: 3)
            for i in 0..<3 {
                for j in 0..<3 { result[i, j] = Double.exp(-omega) * v[i] * v[j].conjugate }
            }
            return result
        }
        var rng = SplitMix64(seed: 71)
        for _ in 0..<20 {
            let z = factory.generate(generator: &rng)
            for time in [0, 0.013, 0.23, 0.7] {
                #expect((z[1].sample(time) - Complex<Double>(0, 1) * z[0].sample(time)).length < 2e-7)
                #expect((z[2].sample(time) - 2 * z[0].sample(time)).length < 2e-7)
            }
        }
    }

    @Test("PSD clipping scales with the spectrum and never introduces jitter",
          arguments: [1e-160, 1.0, 1e160])
    func semidefiniteRoundoff(scale: Double) {
        let covariance = Matrix<Complex<Double>>(
            elements: [Complex(scale), 0, 0, Complex(-Double.ulpOfOne * scale)], rows: 2, columns: 2
        )
        let factor = _noiseCovarianceFactor(covariance)
        let actual = factor.dot(factor.conjugateTranspose)
        #expect(abs(actual[0, 0].real / scale - 1) < 2e-14)
        #expect(actual[1, 1] == .zero)
    }

    @Test("Zero spectra and span sampling preserve output buffers and RNG state")
    func spanAndZeroSpectrum() {
        let factory = GaussianFFTMultiNoiseProcessGenerator(
            tMax: 0.00001, dtMax: 0.1, deltaOmegaMax: 0.5, omegaMax: 3
        ) { _ in .zeros(rows: 2, columns: 2) }
        var rng = NoiseCountingRNG(seed: 14)
        let noise = factory.generateProcess(generator: &rng)
        #expect(noise.channelCount == 2 && noise.tMax >= 0.00001)
        var output = UniqueVector<Complex<Double>>.zero(2)
        let address = output.components
        let draws = rng.count
        for time in [noise.tMax, 0, 0.000003, 0.00001, 0] {
            output[0] = .one
            output[1] = .one
            noise.sample(time, into: &output.mutableSpan, generator: &rng)
            #expect(output[0] == .zero && output[1] == .zero)
        }
        #expect(rng.count == draws)
        #expect(output.components == address)
    }

    @Test("A shared preparation generates reproducible paths concurrently")
    func concurrentGeneration() async {
        let factory = fftTestFactory(kind: 2)
        let seeds: [UInt64] = [42, 51, 76, 93]
        let expected = seeds.map { seed -> [Complex<Double>] in
            var rng = SplitMix64(seed: seed)
            return factory.generate(generator: &rng).map { $0.sample(0.413) }
        }
        await withTaskGroup(of: (Int, [Complex<Double>]).self) { group in
            for (index, seed) in seeds.enumerated() {
                group.addTask {
                    var rng = SplitMix64(seed: seed)
                    return (index, factory.generate(generator: &rng).map { $0.sample(0.413) })
                }
            }
            for await (index, values) in group { #expect(values == expected[index]) }
        }
    }
}

private func fftTestFactory(kind: Int, dt: Double = 0.04) -> GaussianFFTMultiNoiseProcessGenerator {
    .init(tMax: 1.2, dtMax: dt, deltaOmegaMax: 0.5, omegaMax: 4) {
        fftTestSpectrum($0, kind: kind)
    }
}

private func fftTestSpectrum(_ omega: Double, kind: Int) -> Matrix<Complex<Double>> {
    var result = Matrix<Complex<Double>>.zeros(rows: 3, columns: 3)
    if kind == 0 {
        result[0, 0] = Complex(Double.exp(-omega))
        result[1, 1] = Complex(0.6 * Double.exp(-0.7 * omega))
        return result // Third channel identically zero.
    }
    let v: [Complex<Double>] = [1, Complex(0.3, -0.7), Complex(0.4, 0.2)]
    let u: [Complex<Double>] = [0, Complex(0.5, 0.1), Complex(-0.3, 0.9)]
    for i in 0..<3 {
        for j in 0..<3 {
            result[i, j] = Double.exp(-omega) * v[i] * v[j].conjugate
            if kind == 2 { result[i, j] += Double.exp(-0.5 * omega) * u[i] * u[j].conjugate }
        }
    }
    return result
}
