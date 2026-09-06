// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public protocol NoiseProcess {
    associatedtype Element
    
    func callAsFunction(_ t: Double) -> Element
    func sample(_ t: Double) -> Element
    func antithetic() -> Self
}

public extension NoiseProcess {
    @inlinable
    @inline(always)
    func callAsFunction(_ t: Double) -> Element {
        sample(t)
    }
}

public protocol RealNoiseProcess: NoiseProcess where Element == Double {}
public protocol ComplexNoiseProcess: NoiseProcess where Element == Complex<Double> {}

public protocol WhiteNoiseProcess: NoiseProcess {}
public protocol RealWhiteNoiseProcess: WhiteNoiseProcess where Element == Double {}
public protocol ComplexWhiteNoiseProcess: WhiteNoiseProcess where Element == Complex<Double> {}
