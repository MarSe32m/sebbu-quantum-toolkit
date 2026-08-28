// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public final class StandardGaussianWhiteNoiseProcess: ComplexWhiteNoiseProcess, @unchecked Sendable {
    @usableFromInline
    internal var generator: NumPyRandom
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max)) {
        self.generator = NumPyRandom(seed: seed)
    }

    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        generator.nextNormal()
    }
    
    @inlinable
    @inline(always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> Self {
        fatalError("Cannot define antithetic for StandardGaussianWhiteNoiseProcess")
    }
}


public struct StandardGaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, Sendable {
    @inlinable
    public init() { }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> StandardGaussianWhiteNoiseProcess {
        StandardGaussianWhiteNoiseProcess(seed: seed)
    }
}
