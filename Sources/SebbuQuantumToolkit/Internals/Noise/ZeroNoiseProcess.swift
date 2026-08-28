// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

public struct RealZeroNoiseProcess: Sendable {
    @inlinable
    public init() {}
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Double {
        .zero
    }
}

public struct ComplexZeroNoiseProcess: Sendable {
    @inlinable
    public init() {}
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        .zero
    }
}

extension RealZeroNoiseProcess: RealNoiseProcess {
    public func antithetic() -> RealZeroNoiseProcess {
        self
    }
}

extension ComplexZeroNoiseProcess: ComplexNoiseProcess {
    public func antithetic() -> ComplexZeroNoiseProcess {
        self
    }
}

extension RealZeroNoiseProcess: RealWhiteNoiseProcess {}
extension ComplexZeroNoiseProcess: ComplexWhiteNoiseProcess {}
