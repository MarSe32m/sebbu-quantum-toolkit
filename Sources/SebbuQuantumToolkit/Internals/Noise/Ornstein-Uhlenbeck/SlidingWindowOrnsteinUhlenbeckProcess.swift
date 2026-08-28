// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience
import DequeModule

@frozen
public struct UniformDequeLinearInterpolator<Element> {
    @usableFromInline
    internal var _start: Double
        
    public let step: Double
    
    @usableFromInline
    internal let inverseStep: Double
    
    @usableFromInline
    internal var _y: Deque<Element>
    
    @inlinable
    public var start: Double { _start }
    
    @inlinable
    public var y: Deque<Element> {
        _read { yield _y }
    }
    
    @inlinable
    public var count: Int {
        _y.count
    }
    
    @inlinable
    public init(start: Double, step: Double, y: [Element]) {
        precondition(!y.isEmpty, "`y` must not be empty")
        self._start = start
        self.step = step
        self.inverseStep = 1.0 / step
        self._y = .init(y)
    }
    
    @inlinable
    public mutating func popFirst() -> Element? {
        if y.count == 1 { return nil }
        _start += step
        return _y.popFirst()
    }
    
    @inlinable
    public mutating func popLast() -> Element? {
        if y.count == 1 { return nil }
        _start -= step
        return _y.popLast()
    }
    
    @inlinable
    public mutating func prepend(_ element: Element) {
        _y.prepend(element)
        _start -= step
    }
    
    @inlinable
    public mutating func append(_ element: Element) {
        _y.append(element)
        _start += step
    }
}

public extension UniformDequeLinearInterpolator<Double> {
    @inlinable
    @inline(always)
    func callAsFunction(_ t: Double) -> Double? {
        sample(t)
    }
    
    @inlinable
    func sample(_ t: Double) -> Double? {
        if t < _start { return nil }
        if t == _start { return y[0] }
        let u = (t - _start) * inverseStep
        var k = Int(u)
        if k > _y.count - 1 { return nil }
        if k == _y.count - 1 { return y.last! }
        if k < 0 { k = 0 }
        let theta = u - Double(k)
        return (1.0 - theta) * _y[k] + theta * _y[k + 1]
    }
}

public extension UniformDequeLinearInterpolator<Complex<Double>> {
    @inlinable
    @inline(always)
    func callAsFunction(_ t: Double) -> Complex<Double>? {
        sample(t)
    }
    
    @inlinable
    func sample(_ t: Double) -> Complex<Double>? {
        if t < _start { return nil }
        if t == _start { return y[0] }
        let u = (t - _start) * inverseStep
        var k = Int(u)
        if k > _y.count - 1 { return nil }
        if k == _y.count - 1 { return y.last! }
        if k < 0 { k = 0 }
        let theta = u - Double(k)
        return (1.0 - theta) * _y[k] + theta * _y[k + 1]
    }
}

public final class UniformSlidingWindowOrnsteinUhlenbeckProcess: ComplexNoiseProcess {
    @usableFromInline
    internal var interpolator: UniformDequeLinearInterpolator<Complex<Double>>
    
    @usableFromInline
    internal var generator: NumPyRandom
    
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
    public init(G: [Double], W: [Complex<Double>], windowDuration: Double, start: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        precondition(G.count == W.count, "The count of G and W must be equal.")
        let windowSize = Int(windowDuration / step) + 1
        var generator = NumPyRandom(seed: seed)
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
    public convenience init(G: Double, W: Complex<Double>, windowDuration: Double, start: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        self.init(G: [G], W: [W], windowDuration: windowDuration, start: start, step: step, seed: seed)
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        if t < interpolator.start { preconditionFailure("Asking for a sample before the start.") }
        if interpolator.count > windowSize { _ = interpolator.popFirst() }
        while true {
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

public struct UniformSlidingWindowOrnsteinUhlenbeckProcessGenerator: NoiseProcessGenerator, Sendable {
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
    public func generate(seed: UInt32) -> UniformSlidingWindowOrnsteinUhlenbeckProcess {
        UniformSlidingWindowOrnsteinUhlenbeckProcess(G: G, W: W, windowDuration: windowDuration, start: start, step: step, seed: seed)
    }
    
}
