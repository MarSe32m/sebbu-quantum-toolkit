// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

@Suite("Noise input validation")
struct NoiseValidationTests {
    @Test("Invalid meshes fail before generating a path")
    func invalidMesh() async {
        await #expect(processExitsWith: .failure) {
            _ = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
                model: .zero(channelCount: 1), windowDuration: 1, step: .nan
            )
        }
        await #expect(processExitsWith: .failure) {
            _ = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
                model: .zero(channelCount: 1), windowDuration: -1, step: 0.1
            )
        }
        await #expect(processExitsWith: .failure) {
            _ = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
                model: .zero(channelCount: 1), windowDuration: 1, start: 1e20, step: 1e-10
            )
        }
    }

    @Test("Expired times, non-finite times and incorrect spans fail")
    func invalidOUQueries() async {
        await #expect(processExitsWith: .failure) {
            var rng = SplitMix64(seed: 1)
            var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
                model: .zero(channelCount: 1), windowDuration: 0.1, step: 0.1, generator: &rng
            )
            var output: [Complex<Double>] = [0]
            do {
                var outputSpan = output.mutableSpan
                process.sample(1, into: &outputSpan, generator: &rng)
            }
            do {
                var outputSpan = output.mutableSpan
                process.sample(0, into: &outputSpan, generator: &rng)
            }
        }
        await #expect(processExitsWith: .failure) {
            var rng = SplitMix64(seed: 1)
            var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
                model: .zero(channelCount: 1), windowDuration: 0, step: 0.1, generator: &rng
            )
            var output: [Complex<Double>] = [0]
            do {
                var outputSpan = output.mutableSpan
                process.sample(.nan, into: &outputSpan, generator: &rng)
            }
        }
        await #expect(processExitsWith: .failure) {
            var rng = SplitMix64(seed: 1)
            var process = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
                model: .zero(channelCount: 1), windowDuration: 0, step: 0.1, generator: &rng
            )
            var output: [Complex<Double>] = []
            do {
                var outputSpan = output.mutableSpan
                process.sample(0, into: &outputSpan, generator: &rng)
            }
        }
    }

    @Test("Non-Hermitian, indefinite and non-finite spectra fail during preparation")
    func invalidSpectra() async {
        await #expect(processExitsWith: .failure) {
            _ = _noiseCovarianceFactor(Matrix<Complex<Double>>(
                elements: [1, Complex(0, 1), Complex(0, 1), 1], rows: 2, columns: 2
            ))
        }
        await #expect(processExitsWith: .failure) {
            _ = _noiseCovarianceFactor(Matrix<Complex<Double>>(
                elements: [1, 2, 2, 1], rows: 2, columns: 2
            ))
        }
        await #expect(processExitsWith: .failure) {
            _ = _noiseCovarianceFactor(Matrix<Complex<Double>>(
                elements: [Complex(.nan)], rows: 1, columns: 1
            ))
        }
        await #expect(processExitsWith: .failure) {
            _ = GaussianFFTMultiNoiseProcessGenerator(
                tMax: 0.3, dtMax: 0.1, deltaOmegaMax: 1, omegaMax: 2
            ) { omega in
                omega < 1 ? Matrix<Complex<Double>>.identity(rows: 1) : .identity(rows: 2)
            }
        }
    }
}
