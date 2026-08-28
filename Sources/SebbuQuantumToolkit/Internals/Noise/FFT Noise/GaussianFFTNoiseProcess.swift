// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct GaussianFFTNoiseProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let spline: CubicHermiteSpline<Complex<Double>>
    
    public var tMax: Double { spline.x.last! }
    
    @inlinable
    public init(tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, seed: UInt32 = .random(in: .min ... .max), spectralDensity: (_ omega: Double) -> Double) {
        var generator = NumPyRandom(seed: seed)
        self.init(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax, generator: &generator, spectralDensity: spectralDensity)
    }
    
    @inlinable
    public init<RNG: RandomNumberGenerator>(tMax: Double, dtMax: Double, deltaOmegaMax: Double, omegaMax: Double?, generator: inout RNG, spectralDensity: (_ omega: Double) -> Double) {
        precondition(tMax > 0)
        precondition(dtMax > 0)
        precondition(deltaOmegaMax > 0)
        
        // Set frequency resoluation based on tMax
        let deltaOmega = min(deltaOmegaMax, .pi / tMax)
        
        // Compute minimum N so that dt <= dtMax and optionally omegaMax is covered
        var N = omegaMax != nil ? Int(omegaMax! / deltaOmega).nextPowerOf2 : 1024
        N = max(1024, N)
        var dt = 2.0 * .pi / (Double(N) * deltaOmega)
        while dt > dtMax {
            N <<= 1
            dt = 2.0 * .pi / (Double(N) * deltaOmega)
        }
        
        let omegaMax = Double(N - 1) * deltaOmega
        let omegaSpace = [Double].linearSpace(0, omegaMax, N)
        
        // Generate complex Guassian coefficients
        let randomCoefficients: [Complex<Double>] = generator.nextNormal(count: N, stdev: .sqrt(0.5))
        
        // Evaluate J(omega) and build sqrt(J * deltaOmega) * xi
        let J = omegaSpace.map(spectralDensity)
        let coefficients = zip(J, randomCoefficients).map { Double.sqrt(deltaOmega * $0.0) * $0.1 }
        
        // FFT
        let noise = FFT.fft(coefficients).spectrum
        
        // Create valid time grid
        let tMaxFFT = 2.0 * .pi / deltaOmega
        let tSpace = [Double].linearSpace(0, tMaxFFT, N)
        
        // Trim the noise up to desired tMax
        let validIndex = tSpace.lastIndex(where: {$0 <= tMax}) ?? 0
        let trimmedTime = Array(tSpace[0...validIndex])
        let trimmedNoise = Array(noise[0...validIndex])
        
        // Interpolate
        self.spline = CubicHermiteSpline(x: trimmedTime, y: trimmedNoise)
    }
    
    @inlinable
    internal init(spline: CubicHermiteSpline<Complex<Double>>) {
        self.spline = spline
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        spline.sample(t)
    }
    
    @inlinable
    public func conjugate() -> GaussianFFTNoiseProcess {
        let spline = CubicHermiteSpline(x: spline.x, y: spline.y.map { $0.conjugate })
        return GaussianFFTNoiseProcess(spline: spline)
    }
    
    @inlinable
    public func antithetic() -> GaussianFFTNoiseProcess {
        let newSpline = CubicHermiteSpline(x: spline.x, y: spline.y.map { -$0 })
        return .init(spline: newSpline)
    }
}

public struct GaussianFFTNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let tMax: Double
    @usableFromInline
    internal let dtMax: Double
    @usableFromInline
    internal let deltaOmegaMax: Double
    @usableFromInline
    internal let omegaMax: Double?
    @usableFromInline
    internal let spectralDensity: @Sendable (Double) -> Double
    
    @inlinable
    public init(tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, spectralDensity: @Sendable @escaping (_ omega: Double) -> Double) {
        self.tMax = tMax
        self.spectralDensity = spectralDensity
        self.dtMax = dtMax
        self.deltaOmegaMax = deltaOmegaMax
        self.omegaMax = omegaMax
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> GaussianFFTNoiseProcess {
        GaussianFFTNoiseProcess(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax, seed: seed, spectralDensity: spectralDensity)
    }
}
