// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct PreSampledGaussianWhiteNoiseProcess: ComplexWhiteNoiseProcess, @unchecked Sendable {
    @usableFromInline
    internal let interpolator: NearestNeighbourInterpolator<Complex<Double>>
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: ComplexTimeFunction, deviation: ScalarTimeFunction, tSpace: [Double]) {
        let generatingProcess = GaussianWhiteNoiseProcess(seed: seed, mean: mean, deviation: deviation)
        let samples = tSpace.map { generatingProcess.sample($0)}
        self.interpolator = NearestNeighbourInterpolator(x: tSpace, y: samples)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .constant(mean), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping @Sendable (Double) -> Double, deviation: Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping @Sendable (Double) -> Complex<Double>, deviation: Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .generated({ mean($0) }), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: @escaping @Sendable (Double) -> Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: @escaping @Sendable (Double) -> Double, tSpace: [Double]) {
        self.init(seed: seed, mean: .constant(mean), deviation: .generated({ deviation($0) }), tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: ComplexTimeFunction, deviation: ScalarTimeFunction, start: Double, end: Double, step: Double) {
        var tSpace: [Double] = []
        tSpace.reserveCapacity(Int((end - start) / step) + 1)
        var t = start
        while t <= end {
            tSpace.append(t)
            t += step
        }
        self.init(seed: seed, mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .constant(mean), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping @Sendable (Double) -> Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping @Sendable (Double) -> Complex<Double>, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .generated({ mean($0) }), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: @escaping @Sendable (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: @escaping @Sendable (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: .constant(mean), deviation: .generated({ deviation($0) }), start: start, end: end, step: step)
    }
    
    @inlinable
    internal init(interpolator: NearestNeighbourInterpolator<Complex<Double>>) {
        self.interpolator = interpolator
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        interpolator(t)
    }
    
    @inlinable
    @inline(__always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> PreSampledGaussianWhiteNoiseProcess {
        let newInterpolator = NearestNeighbourInterpolator(x: interpolator.x, y: interpolator.y.map { -$0 })
        return PreSampledGaussianWhiteNoiseProcess(interpolator: newInterpolator)
    }
}

public struct PreSampledGaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, @unchecked Sendable {
    @usableFromInline
    internal let mean: ComplexTimeFunction
    @usableFromInline
    internal let deviation: ScalarTimeFunction
    @usableFromInline
    internal let tSpace: [Double]
    
    @inlinable
    public init(mean: ComplexTimeFunction, deviation: ScalarTimeFunction, tSpace: [Double]) {
        self.mean = mean
        self.deviation = deviation
        self.tSpace = tSpace
    }
    
    @inlinable
    public init(mean: Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: .constant(Complex(mean)), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: Complex<Double>, deviation: Double, tSpace: [Double]) {
        self.init(mean: .constant(mean), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: @escaping @Sendable (Double) -> Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: @escaping @Sendable (Double) -> Complex<Double>, deviation: Double, tSpace: [Double]) {
        self.init(mean: .generated({ mean($0) }), deviation: .constant(deviation), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: Double, deviation: @escaping @Sendable (Double) -> Double, tSpace: [Double]) {
        self.init(mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: Complex<Double>, deviation: @escaping @Sendable (Double) -> Double, tSpace: [Double]) {
        self.init(mean: .constant(mean), deviation: .generated({ deviation($0) }), tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: ComplexTimeFunction, deviation: ScalarTimeFunction, start: Double, end: Double, step: Double) {
        var tSpace: [Double] = []
        tSpace.reserveCapacity(Int((end - start) / step) + 1)
        var t = start
        while t <= end {
            tSpace.append(t)
            t += step
        }
        self.init(mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: .constant(Complex(mean)), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: Complex<Double>, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: .constant(mean), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: @escaping @Sendable (Double) -> Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: @escaping  @Sendable (Double) -> Complex<Double>, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: .generated({ mean($0) }), deviation: .constant(deviation), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: Double, deviation: @escaping @Sendable (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }), start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: Complex<Double>, deviation: @escaping @Sendable (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(mean: .constant(mean), deviation: .generated({ deviation($0) }), start: start, end: end, step: step)
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> sending PreSampledGaussianWhiteNoiseProcess {
        PreSampledGaussianWhiteNoiseProcess(mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> PreSampledGaussianWhiteNoiseProcess {
        PreSampledGaussianWhiteNoiseProcess(seed: seed, mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
}
