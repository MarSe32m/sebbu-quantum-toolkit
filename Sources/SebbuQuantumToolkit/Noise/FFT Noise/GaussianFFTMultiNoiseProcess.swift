// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Prepares a stationary, proper complex multichannel Gaussian process.
///
/// The convention matches `GaussianFFTNoiseProcessGenerator`:
/// `E[z_i(t) conj(z_j(s))] = integral_0^omegaMax J_ij(w) exp(-iw(t-s)) dw`,
/// with no implicit `1 / pi` or `1 / (2 pi)` factor. `E[z_i(t) z_j(s)] = 0`.
/// Every spectral matrix must be finite, Hermitian and positive semidefinite,
/// and must have the same nonzero dimension. Singular spectra are supported.
///
/// Uses midpoint quadrature, an explicit cutoff independent of FFT padding,
/// and cubic interpolation with spectral derivatives. Spectral evaluation and
/// factorization occur once, during initialization. Each generated realization
/// owns its FFT workspace, so this immutable preparation can be shared safely.
///
/// A `CorrelatedBathModel` uses a two-sided rational spectrum with a `1/(2 pi)`
/// inverse-transform convention. Use the correlated OU generator to reproduce
/// that model's BCF directly; passing its spectrum here changes the convention.
public struct GaussianFFTMultiNoiseProcessGenerator: Sendable {
    @usableFromInline internal let grid: _GaussianFFTNoiseGrid
    @usableFromInline internal let factors: [Matrix<Complex<Double>>]
    public let channelCount: Int

    /// Parameters have the same meanings and defaults as the scalar FFT factory.
    /// The spectral closure is not called at zero or at the integration cutoff.
    @inlinable
    public init(
        tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01,
        omegaMax: Double? = nil,
        spectralDensity: @Sendable @escaping (_ omega: Double) -> Matrix<Complex<Double>>
    ) {
        let grid = _GaussianFFTNoiseGrid(
            tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax
        )
        var factors: [Matrix<Complex<Double>>] = []
        factors.reserveCapacity(grid.frequencies.count)
        var channelCount = 0
        let rootFrequencyStep = grid.frequencyStep.squareRoot()
        for omega in grid.frequencies {
            let density = spectralDensity(omega)
            if factors.isEmpty { channelCount = density.rows }
            precondition(density.rows == channelCount && density.columns == channelCount,
                         "Every spectral matrix must have the same square shape.")
            var factor = _noiseCovarianceFactor(density)
            for index in factor.elements.indices {
                // Scale after taking the root to avoid premature underflow.
                factor.elements[index] *= rootFrequencyStep
                precondition(factor.elements[index].real.isFinite
                             && factor.elements[index].imaginary.isFinite,
                             "A spectral factor overflowed.")
            }
            factors.append(factor)
        }
        self.grid = grid
        self.factors = factors
        self.channelCount = channelCount
    }

    /// Generates correlated scalar paths, preserving the existing array API.
    /// Constructing a realization allocates; sampling it does not.
    @inlinable
    public func generate<Generator: RandomNumberGenerator>(
        generator: inout Generator
    ) -> [GaussianFFTNoiseProcess] {
        var coefficients = [[Complex<Double>]](
            repeating: .init(repeating: .zero, count: grid.fftCount), count: channelCount
        )
        var gaussians = [Complex<Double>](repeating: .zero, count: channelCount)
        for frequency in grid.frequencies.indices {
            for j in 0..<channelCount {
                gaussians[j] = generator.nextNormal(stdev: Double(0.5).squareRoot())
            }
            let factor = factors[frequency]
            for i in 0..<channelCount {
                var value = Complex<Double>.zero
                for j in 0..<channelCount { value += factor[i, j] * gaussians[j] }
                coefficients[i][frequency] = value
            }
        }

        let plan = FFT.defaultFFTPlan(sampleSize: grid.fftCount)
        var noises: [GaussianFFTNoiseProcess] = []
        noises.reserveCapacity(channelCount)
        for channel in 0..<channelCount {
            let values = plan.execute(coefficients[channel], spacing: 1).y
            for frequency in grid.frequencies.indices {
                coefficients[channel][frequency] *= Complex<Double>(0, -grid.frequencies[frequency])
            }
            let derivatives = plan.execute(coefficients[channel], spacing: 1).y
            let samples = grid.times.indices.map { values[$0] * grid.midpointPhases[$0] }
            let tangents = grid.times.indices.map { derivatives[$0] * grid.midpointPhases[$0] }
            noises.append(GaussianFFTNoiseProcess(spline: CubicHermiteSpline(
                x: grid.times, y: samples, tangents: tangents
            )))
        }
        return noises
    }

    /// Generates a multichannel wrapper for allocation-free `MutableSpan` sampling.
    @inlinable
    public func generateProcess<Generator: RandomNumberGenerator>(
        generator: inout Generator
    ) -> GaussianFFTMultiNoiseProcess {
        GaussianFFTMultiNoiseProcess(noises: generate(generator: &generator))
    }
}

/// One precomputed multichannel FFT realization. Samples can be requested in any order.
public struct GaussianFFTMultiNoiseProcess: Sendable {
    @usableFromInline internal let noises: [GaussianFFTNoiseProcess]

    public var channelCount: Int { noises.count }
    public var tMax: Double { noises[0].tMax }

    @usableFromInline
    internal init(noises: [GaussianFFTNoiseProcess]) { self.noises = noises }

    /// A scalar view of a channel of this same realization.
    public subscript(channel: Int) -> GaussianFFTNoiseProcess { noises[channel] }

    @inlinable
    public func sample(_ t: Double, into output: inout MutableSpan<Complex<Double>>) {
        precondition(output.count == noises.count, "Expected one output per physical channel.")
        for i in noises.indices { output[i] = noises[i].sample(t) }
    }

    /// Matches the streaming sampler's calling convention. No random numbers
    /// are consumed: this realization was completely generated by the factory.
    @inlinable
    public func sample<Generator: RandomNumberGenerator>(
        _ t: Double, into output: inout MutableSpan<Complex<Double>>,
        generator: inout Generator
    ) {
        sample(t, into: &output)
    }
}
