// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// One realization of stationary, proper complex Gaussian noise.
///
/// The target covariance is `E[z(t) * conj(z(s))] = integral J(w) exp(-iw(t-s)) dw`
/// over positive angular frequencies; `E[z(t) * z(s)] = 0`. There is no implicit
/// factor of `1 / pi` in the supplied spectral density.
///
/// Frequency integration uses midpoint quadrature on `0..<omegaMax`. The FFT
/// samples and their spectral derivatives define a cubic Hermite interpolant.
/// Converge the frequency cutoff, frequency spacing, and time spacing separately.
/// Sampling an existing realization never advances a random-number generator.
public struct GaussianFFTNoiseProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let spline: CubicHermiteSpline<Complex<Double>>
    // Keep transformations outside the spline so its exact spectral tangents
    // are preserved, without rebuilding or copying its coefficient arrays.
    @usableFromInline
    internal let isConjugated: Bool
    @usableFromInline
    internal let isAntithetic: Bool

    /// Last supported time, including the knot bracketing the requested `tMax`.
    public var tMax: Double { spline.x.last! }

    /// Creates one realization. Use ``GaussianFFTNoiseProcessGenerator`` to
    /// reuse spectral preparation when drawing an ensemble.
    ///
    /// - Parameters:
    ///   - tMax: Finite, positive duration that must be covered by the noise.
    ///   - dtMax: Finite, positive upper bound on the interpolation knot spacing.
    ///   - deltaOmegaMax: Finite, positive upper bound on the quadrature spacing.
    ///   - omegaMax: Finite, positive integration cutoff. `nil` uses `pi / dtMax`.
    ///     A supplied cutoff is independent of the FFT padding and time spacing.
    ///   - generator: Source of independent Gaussian quadrature coefficients.
    ///   - spectralDensity: Finite, nonnegative density at positive frequencies.
    ///     Endpoint values at zero and at the cutoff are not evaluated.
    @inlinable
    public init<Generator: RandomNumberGenerator>(
        tMax: Double, dtMax: Double, deltaOmegaMax: Double, omegaMax: Double?,
        generator: inout Generator, spectralDensity: (_ omega: Double) -> Double
    ) {
        let preparation = _GaussianFFTNoisePreparation(
            tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax,
            omegaMax: omegaMax, spectralDensity: spectralDensity
        )
        self = preparation.generate(generator: &generator)
    }

    @inlinable
    internal init(
        spline: CubicHermiteSpline<Complex<Double>>,
        isConjugated: Bool = false, isAntithetic: Bool = false
    ) {
        self.spline = spline
        self.isConjugated = isConjugated
        self.isAntithetic = isAntithetic
    }

    /// Evaluates this fixed realization, in any query order, on `0...tMax`.
    /// Queries outside this interval fail instead of silently clamping the noise.
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        precondition(t.isFinite && t >= 0 && t <= spline.x.last!,
                     "The requested time must be within the sampled noise interval.")
        let value = spline.sample(t)
        let transformed = isConjugated ? value.conjugate : value
        return isAntithetic ? -transformed : transformed
    }

    /// Returns the conjugate of the same realization, including between knots.
    @inlinable
    public func conjugate() -> GaussianFFTNoiseProcess {
        .init(spline: spline, isConjugated: !isConjugated, isAntithetic: isAntithetic)
    }

    /// Returns the negative of the same realization without drawing new noise.
    @inlinable
    public func antithetic() -> GaussianFFTNoiseProcess {
        .init(spline: spline, isConjugated: isConjugated, isAntithetic: !isAntithetic)
    }
}

/// Immutable spectral preparation shared by independently generated realizations.
///
/// The spectral-density closure is evaluated only during initialization. Each
/// `generate` call owns its Gaussian coefficients, FFT workspace and interpolant.
/// Concurrent callers should provide separate random-number generators!
public struct GaussianFFTNoiseProcessGenerator: Sendable {
    @usableFromInline
    internal let preparation: _GaussianFFTNoisePreparation

    /// Uses the same quadrature and parameter conventions as
    /// ``GaussianFFTNoiseProcess``. Changing the supplied density after this
    /// initializer returns does not change the prepared spectrum.
    @inlinable
    public init(
        tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01,
        omegaMax: Double? = nil,
        spectralDensity: @Sendable @escaping (_ omega: Double) -> Double
    ) {
        self.preparation = _GaussianFFTNoisePreparation(
            tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax,
            omegaMax: omegaMax, spectralDensity: spectralDensity
        )
    }

    @inlinable
    public func generate<Generator: RandomNumberGenerator>(
        generator: inout Generator
    ) -> GaussianFFTNoiseProcess {
        preparation.generate(generator: &generator)
    }
}

@usableFromInline
internal struct _GaussianFFTNoisePreparation: Sendable {
    @usableFromInline internal let fftCount: Int
    @usableFromInline internal let frequencyStep: Double
    @usableFromInline internal let frequencies: [Double]
    @usableFromInline internal let amplitudes: [Double]
    @usableFromInline internal let times: [Double]
    @usableFromInline internal let midpointPhases: [Complex<Double>]

    @inlinable
    internal init(
        tMax: Double, dtMax: Double, deltaOmegaMax: Double, omegaMax: Double?,
        spectralDensity: (Double) -> Double
    ) {
        let grid = _GaussianFFTNoiseGrid(
            tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax
        )
        let fftCount = grid.fftCount
        let frequencyStep = grid.frequencyStep
        let frequencies = grid.frequencies
        let times = grid.times
        let rootFrequencyStep = frequencyStep.squareRoot()
        let amplitudes = frequencies.map { omega in
            precondition(omega.isFinite && omega > 0,
                         "The quadrature frequency is not representable.")
            let density = spectralDensity(omega)
            precondition(density.isFinite && density >= 0,
                         "The spectral density must be finite and nonnegative.")
            // Taking the square roots separately avoids premature overflow or
            // underflow of density * frequencyStep.
            let amplitude = density.squareRoot() * rootFrequencyStep
            precondition(amplitude.isFinite, "A spectral amplitude overflowed.")
            return amplitude
        }

        self.fftCount = fftCount
        self.frequencyStep = frequencyStep
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.times = times
        self.midpointPhases = grid.midpointPhases
    }

    @inlinable
    internal func generate<Generator: RandomNumberGenerator>(
        generator: inout Generator
    ) -> GaussianFFTNoiseProcess {
        var coefficients = [Complex<Double>](repeating: .zero, count: fftCount)
        for index in frequencies.indices {
            // Unit complex variance: each Cartesian component has variance 1/2.
            let gaussian: Complex<Double> = generator.nextNormal(stdev: Double(0.5).squareRoot())
            coefficients[index] = amplitudes[index] * gaussian
        }

        // The unnormalized forward FFT supplies exp(-i * k * dw * t).
        // The common phase supplies the half-bin offset of midpoint quadrature.
        // A plan is local to this call; native FFT workspaces are never shared.
        let plan = FFT.defaultFFTPlan(sampleSize: fftCount)
        let values = plan.execute(coefficients, spacing: 1).y
        for index in frequencies.indices {
            coefficients[index] *= Complex<Double>(0, -frequencies[index])
        }
        let derivatives = plan.execute(coefficients, spacing: 1).y
        let samples = times.indices.map { values[$0] * midpointPhases[$0] }
        let tangents = times.indices.map { derivatives[$0] * midpointPhases[$0] }
        let spline = CubicHermiteSpline(x: times, y: samples, tangents: tangents)
        return GaussianFFTNoiseProcess(spline: spline)
    }
}
