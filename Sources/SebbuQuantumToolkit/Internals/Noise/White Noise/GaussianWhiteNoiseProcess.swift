// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

//TODO: Maybe this should be generic over RandomNumberGenerator?
public class GaussianWhiteNoiseProcess: ComplexWhiteNoiseProcess {
    @usableFromInline
    internal let mean: ComplexTimeFunction
    @usableFromInline
    internal let deviation: ScalarTimeFunction
    @usableFromInline
    internal var generator: NumPyRandom
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: ComplexTimeFunction, deviation: ScalarTimeFunction) {
        self.mean = mean
        self.deviation = deviation
        self.generator = NumPyRandom(seed: seed)
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .constant(deviation))
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: Double) {
        self.init(seed: seed, mean: .constant(mean), deviation: .constant(deviation))
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: @Sendable @escaping (Double) -> Double, deviation: Double) {
        self.init(seed: seed, mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation))
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: @Sendable @escaping (Double) -> Complex<Double>, deviation: Double) {
        self.init(seed: seed, mean: .generated({ mean($0) }), deviation: .constant(deviation))
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: @Sendable @escaping (Double) -> Double) {
        self.init(seed: seed, mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }))
    }
    
    @inlinable
    public convenience init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: @Sendable @escaping (Double) -> Double) {
        self.init(seed: seed, mean: .constant(mean), deviation: .generated({ deviation($0) }))
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        generator.nextNormal(mean: mean(t), stdev: deviation(t))
    }
    
    @inlinable
    @inline(__always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> Self {
        fatalError("Cannot define antithetic for GaussianWhiteNoiseProcess")
    }
}


public struct GaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, @unchecked Sendable {
    @usableFromInline
    internal let mean: ComplexTimeFunction
    @usableFromInline
    internal let deviation: ScalarTimeFunction
    
    @inlinable
    public init(mean: ComplexTimeFunction, deviation: ScalarTimeFunction) {
        self.mean = mean
        self.deviation = deviation
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double) {
        self.init(mean: .constant(Complex(mean)), deviation: .constant(deviation))
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Complex<Double>, deviation: Double) {
        self.init(mean: .constant(mean), deviation: .constant(deviation))
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @Sendable @escaping (Double) -> Double, deviation: Double) {
        self.init(mean: .generated({ Complex(mean($0)) }), deviation: .constant(deviation))
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @Sendable @escaping (Double) -> Complex<Double>, deviation: Double) {
        self.init(mean: .generated({ mean($0) }), deviation: .constant(deviation))
    }
    
    @inlinable
    public init(mean: Double, deviation: @Sendable @escaping (Double) -> Double) {
        self.init(mean: .constant(Complex(mean)), deviation: .generated({ deviation($0) }))
    }
    
    @inlinable
    public init(mean: Complex<Double>, deviation: @Sendable @escaping (Double) -> Double) {
        self.init(mean: .constant(mean), deviation: .generated({ deviation($0) }))
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> GaussianWhiteNoiseProcess {
        GaussianWhiteNoiseProcess(seed: seed, mean: mean, deviation: deviation)
    }
}
