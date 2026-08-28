// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct PreSampledOrnsteinUhlenbeckProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let interpolator: UniformLinearInterpolator<Complex<Double>>
    
    @inlinable
    public init(G: [Double], W: [Complex<Double>], start: Double, end: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        precondition(G.count == W.count, "The count of G and W must be equal.")
        var random = NumPyRandom(seed: seed)
        var samples: [Complex<Double>] = .init(repeating: .zero, count: Int((end - start) / step))
        for(g, w) in zip(G, W) {
            precondition(g >= 0, "The coefficients must be non-negative.")
            let randomNumbers: [Complex<Double>] = random.nextNormal(count: samples.count, stdev: .sqrt(0.5))
            var x = .sqrt(g) * randomNumbers[0]
            samples[0] += x
            let r: Complex<Double> = .exp(-step * w)
            let exponent: Double = step * 2 * w.real
            let B: Double = .sqrt(g * .oneMinusExpMinus(exponent))
            for i in 1..<samples.count {
                x = r * x + B * randomNumbers[i]
                samples[i] += x
            }
        }
        self.interpolator = UniformLinearInterpolator(start: start, step: step, y: samples)
    }
    
    @inlinable
    public init(G: Double, W: Complex<Double>, start: Double, end: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        self.init(G: [G], W: [W], start: start, end: end, step: step, seed: seed)
    }
    
    @inlinable
    internal init(_ interpolator: UniformLinearInterpolator<Complex<Double>>) {
        self.interpolator = interpolator
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        interpolator.sample(t)
    }
    
    @inlinable
    @inline(always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> PreSampledOrnsteinUhlenbeckProcess {
        PreSampledOrnsteinUhlenbeckProcess(UniformLinearInterpolator(start: interpolator.start, step: interpolator.step, y: interpolator.y.map { -$0 }))
    }
}

public struct PreSampledOrnsteinUhlenbeckProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let G: [Double]
    
    @usableFromInline
    internal let W: [Complex<Double>]
    
    @usableFromInline
    internal let start: Double
    
    @usableFromInline
    internal let end: Double
    
    @usableFromInline
    internal let step: Double
    
    @inlinable
    public init(G: Double, W: Complex<Double>, start: Double, end: Double, step: Double) {
        self.init(G: [G], W: [W], start: start, end: end, step: step)
    }
    
    @inlinable
    public init(G: [Double], W: [Complex<Double>], start: Double, end: Double, step: Double) {
        self.G = G
        self.W = W
        self.start = start
        self.end = end
        self.step = step
    }
    
    @inlinable
    @inline(always)
    public func generate() -> sending PreSampledOrnsteinUhlenbeckProcess {
        PreSampledOrnsteinUhlenbeckProcess(G: G, W: W, start: start, end: end, step: step)
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> PreSampledOrnsteinUhlenbeckProcess {
        PreSampledOrnsteinUhlenbeckProcess(G: G, W: W, start: start, end: end, step: step, seed: seed)
    }
    
}
