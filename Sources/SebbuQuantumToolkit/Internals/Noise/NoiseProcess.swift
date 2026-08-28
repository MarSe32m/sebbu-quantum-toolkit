// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

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

public protocol NoiseProcessGenerator {
    associatedtype Process: NoiseProcess
    func generate() -> Process
    func generate(seed: UInt32) -> Process
}

public extension NoiseProcessGenerator {
    @inlinable
    @inline(always)
    func generate() -> Process {
        generate(seed: .random(in: .min ... .max))
    }
    
    @inlinable
    @inline(always)
    func generate(count: Int) -> [Process] {
        (0..<count).map { _ in self.generate() }
    }
    
    @inlinable
    @inline(always)
    func generate(seeds: [UInt32]) -> [Process] {
        seeds.map { self.generate(seed: $0) }
    }
}

public protocol MultiNoiseProcessGenerator {
    associatedtype Process: NoiseProcess
    func generate() -> [Process]
    func generate(seed: UInt32) -> [Process]
}

public extension MultiNoiseProcessGenerator {
    @inlinable
    @inline(always)
    func generate() -> [Process] {
        generate(seed: .random(in: .min ... .max))
    }
    
    @inlinable
    @inline(always)
    func generate(count: Int) -> [[Process]] {
        (0..<count).map { _ in self.generate() }
    }
    
    @inlinable
    @inline(always)
    func generate(seeds: [UInt32]) -> [[Process]] {
        seeds.map { self.generate(seed: $0) }
    }
}

public protocol RealNoiseProcess: NoiseProcess where Element == Double {}
public protocol ComplexNoiseProcess: NoiseProcess where Element == Complex<Double> {}

public protocol WhiteNoiseProcess: NoiseProcess {}
public protocol RealWhiteNoiseProcess: WhiteNoiseProcess where Element == Double {}
public protocol ComplexWhiteNoiseProcess: WhiteNoiseProcess where Element == Complex<Double> {}

public protocol WhiteNoiseProcessGenerator: NoiseProcessGenerator where Process: WhiteNoiseProcess {}
