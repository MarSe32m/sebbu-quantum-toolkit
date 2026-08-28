// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience
import DequeModule

public final class UniformSlidingWindowOrnsteinUhlenbeckProcess: ComplexNoiseProcess {
    @usableFromInline
    internal var interpolator: UniformDequeLinearInterpolator<Complex<Double>>
    
    @usableFromInline
    internal var generator: Philox4x64
    
    @usableFromInline
    internal let r: [Complex<Double>]
    
    @usableFromInline
    internal let B: [Double]
    
    @usableFromInline
    internal var x: [Complex<Double>]
    
    @usableFromInline
    internal let isAntithetic: Bool
    
    @usableFromInline
    internal let windowSize: Int
    
    @inlinable
    public init(G: [Double], W: [Complex<Double>], windowDuration: Double, start: Double, step: Double, generator: consuming Philox4x64) {
        precondition(G.count == W.count, "The count of G and W must be equal.")
        let windowSize = Int(windowDuration / step) + 1
        var samples: [Complex<Double>] = .init(repeating: .zero, count: windowSize)
        var _x: [Complex<Double>] = []
        var _r: [Complex<Double>] = []
        var _B: [Double] = []
        for(g, w) in zip(G, W) {
            precondition(g >= 0, "The coefficients must be non-negative.")
            let randomNumbers: [Complex<Double>] = generator.nextNormal(count: samples.count, stdev: .sqrt(0.5))
            var x = .sqrt(g) * randomNumbers[0]
            samples[0] += x
            let r: Complex<Double> = .exp(-step * w)
            let exponent: Double = step * 2 * w.real
            let B: Double = .sqrt(g * .oneMinusExpMinus(exponent))
            for i in 1..<samples.count {
                x = r * x + B * randomNumbers[i]
                samples[i] += x
            }
            _x.append(x)
            _r.append(r)
            _B.append(B)
        }
        self.interpolator = UniformDequeLinearInterpolator(start: start, step: step, y: samples)
        self.generator = generator
        self.x = _x
        self.r = _r
        self.B = _B
        self.windowSize = windowSize
        self.isAntithetic = false
    }
    
    @inlinable
    public convenience init(G: [Double], W: [Complex<Double>], windowDuration: Double, start: Double, step: Double, seed: UInt64) {
        let generator = Philox4x64(seed: seed)
        self.init(G: G, W: W, windowDuration: windowDuration, start: start, step: step, generator: generator)
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        if t < interpolator.start { preconditionFailure("Asking for a sample before the start.") }
        while true {
            if interpolator.count > windowSize { _ = interpolator.popFirst() }
            let sample = interpolator.sample(t)
            if let sample { return isAntithetic ? -sample : sample }
            var newSample: Complex<Double> = .zero
            for i in x.indices {
                let randomNumber: Complex<Double> = generator.nextNormal(stdev: .sqrt(0.5))
                x[i] = r[i] * x[i] + B[i] * randomNumber
                newSample += x[i]
            }
            interpolator.prepend(newSample)
        }
    }
    
    @inlinable
    internal init(athithetic of: UniformSlidingWindowOrnsteinUhlenbeckProcess) {
        self.generator = of.generator
        self.interpolator = of.interpolator
        self.x = of.x
        self.r = of.r
        self.B = of.B
        self.windowSize = of.windowSize
        self.isAntithetic = !of.isAntithetic
    }
    
    @inlinable
    public func antithetic() -> UniformSlidingWindowOrnsteinUhlenbeckProcess {
        UniformSlidingWindowOrnsteinUhlenbeckProcess(athithetic: self)
    }
}

public struct UniformSlidingWindowOrnsteinUhlenbeckProcessGenerator: Sendable {
    @usableFromInline
    internal let G: [Double]
    
    @usableFromInline
    internal let W: [Complex<Double>]
    
    @usableFromInline
    internal let windowDuration: Double
    
    @usableFromInline
    internal let start: Double
    
    @usableFromInline
    internal let step: Double
    
    @inlinable
    public init(G: Double, W: Complex<Double>, windowDuration: Double, start: Double, step: Double) {
        self.init(G: [G], W: [W], windowDuration: windowDuration, start: start, step: step)
    }
    
    @inlinable
    public init(G: [Double], W: [Complex<Double>], windowDuration: Double, start: Double, step: Double) {
        self.G = G
        self.W = W
        self.windowDuration = windowDuration
        self.start = start
        self.step = step
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt64) -> UniformSlidingWindowOrnsteinUhlenbeckProcess {
        UniformSlidingWindowOrnsteinUhlenbeckProcess(G: G, W: W, windowDuration: windowDuration, start: start, step: step, seed: seed)
    }
    
    @inlinable
    @inline(always)
    public func generate(generator: consuming Philox4x64) -> UniformSlidingWindowOrnsteinUhlenbeckProcess {
        UniformSlidingWindowOrnsteinUhlenbeckProcess(G: G, W: W, windowDuration: windowDuration, start: start, step: step, generator: generator)
    }
    
}
