// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

public struct ZeroNoiseProcess: Sendable {
    public typealias Element = Complex<Double>
    
    @inlinable
    public init() {}
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        .zero
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Double {
        .zero
    }
}

extension ZeroNoiseProcess: ComplexNoiseProcess {
    public func antithetic() -> ZeroNoiseProcess {
        self
    }
}
extension ZeroNoiseProcess: ComplexWhiteNoiseProcess {}

public struct ZeroNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
    
    @inlinable
    public init() {}
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> ZeroNoiseProcess {
        ZeroNoiseProcess()
    }
}
