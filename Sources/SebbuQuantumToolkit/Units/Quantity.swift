// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Marker protocol for the semantic dimension carried by a `Quantity`.
///
/// Some of the marker types below have the same SI dimensions but intentionally
/// remain distinct. For example, a cyclic frequency must acquire a factor of
/// `2 * pi` before it is used as an angular phase rate, whereas a decay rate
/// must not.
@_marker
public protocol QuantityDimension: Sendable {}

public enum EnergyDimension: QuantityDimension {}
public enum TimeDimension: QuantityDimension {}
public enum AngularFrequencyDimension: QuantityDimension {}
public enum CyclicFrequencyDimension: QuantityDimension {}
public enum DecayRateDimension: QuantityDimension {}
public enum TemperatureDimension: QuantityDimension {}
public enum SpectroscopicWavenumberDimension: QuantityDimension {}
public enum EnergySquaredDimension: QuantityDimension {}
public enum InverseTimeSquaredDimension: QuantityDimension {}

/// A strongly typed physical quantity stored in coherent SI units.
///
/// SI storage makes a quantity independent of the eventual simulation unit
/// system. Conversion to raw solver values is performed explicitly by a
/// `QuantumUnitSystem`.
public struct Quantity<Dimension: QuantityDimension>:
    Sendable,
    Hashable,
    Codable,
    Comparable,
    AdditiveArithmetic
{
    /// Value in the coherent SI unit associated with `Dimension`.
    ///
    /// Examples include J for energy, s for time, s^-1 for rates, K for
    /// temperature, and J^2 for squared energy.
    public let valueInSIUnits: Double

    public init(valueInSIUnits: Double) {
        self.valueInSIUnits = valueInSIUnits
    }

    public static var zero: Self {
        Self(valueInSIUnits: 0)
    }

    public var isFinite: Bool {
        valueInSIUnits.isFinite
    }

    public var isNaN: Bool {
        valueInSIUnits.isNaN
    }

    public var magnitude: Self {
        Self(valueInSIUnits: Swift.abs(valueInSIUnits))
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.valueInSIUnits < rhs.valueInSIUnits
    }

    public static func + (lhs: Self, rhs: Self) -> Self {
        Self(valueInSIUnits: lhs.valueInSIUnits + rhs.valueInSIUnits)
    }

    public static func - (lhs: Self, rhs: Self) -> Self {
        Self(valueInSIUnits: lhs.valueInSIUnits - rhs.valueInSIUnits)
    }

    public static prefix func - (value: Self) -> Self {
        Self(valueInSIUnits: -value.valueInSIUnits)
    }

    public static func * (lhs: Self, rhs: Double) -> Self {
        Self(valueInSIUnits: lhs.valueInSIUnits * rhs)
    }

    public static func * (lhs: Double, rhs: Self) -> Self {
        rhs * lhs
    }

    public static func / (lhs: Self, rhs: Double) -> Self {
        Self(valueInSIUnits: lhs.valueInSIUnits / rhs)
    }

    /// Ratio of two quantities having the same semantic dimension.
    public static func / (lhs: Self, rhs: Self) -> Double {
        lhs.valueInSIUnits / rhs.valueInSIUnits
    }
}

public typealias Energy = Quantity<EnergyDimension>
public typealias Time = Quantity<TimeDimension>
public typealias AngularFrequency = Quantity<AngularFrequencyDimension>
public typealias CyclicFrequency = Quantity<CyclicFrequencyDimension>
public typealias DecayRate = Quantity<DecayRateDimension>
public typealias Temperature = Quantity<TemperatureDimension>
public typealias SpectroscopicWavenumber = Quantity<SpectroscopicWavenumberDimension>
public typealias EnergySquared = Quantity<EnergySquaredDimension>
public typealias InverseTimeSquared = Quantity<InverseTimeSquaredDimension>

public extension Quantity where Dimension == EnergyDimension {
    init(_ value: Double, _ unit: EnergyUnit) {
        self.init(valueInSIUnits: value * unit.joulesPerUnit)
    }

    func value(in unit: EnergyUnit) -> Double {
        valueInSIUnits / unit.joulesPerUnit
    }
}

public extension Quantity where Dimension == TimeDimension {
    init(_ value: Double, _ unit: TimeUnit) {
        self.init(valueInSIUnits: value * unit.secondsPerUnit)
    }

    func value(in unit: TimeUnit) -> Double {
        valueInSIUnits / unit.secondsPerUnit
    }
}

public extension Quantity where Dimension == AngularFrequencyDimension {
    init(_ value: Double, _ unit: AngularFrequencyUnit) {
        self.init(valueInSIUnits: value * unit.radiansPerSecondPerUnit)
    }

    func value(in unit: AngularFrequencyUnit) -> Double {
        valueInSIUnits / unit.radiansPerSecondPerUnit
    }
}

public extension Quantity where Dimension == CyclicFrequencyDimension {
    init(_ value: Double, _ unit: CyclicFrequencyUnit) {
        self.init(valueInSIUnits: value * unit.hertzPerUnit)
    }

    func value(in unit: CyclicFrequencyUnit) -> Double {
        valueInSIUnits / unit.hertzPerUnit
    }
}

public extension Quantity where Dimension == DecayRateDimension {
    init(_ value: Double, _ unit: DecayRateUnit) {
        self.init(valueInSIUnits: value * unit.perSecondPerUnit)
    }

    func value(in unit: DecayRateUnit) -> Double {
        valueInSIUnits / unit.perSecondPerUnit
    }
}

public extension Quantity where Dimension == TemperatureDimension {
    init(_ value: Double, _ unit: TemperatureUnit) {
        self.init(valueInSIUnits: value * unit.kelvinPerUnit)
    }

    func value(in unit: TemperatureUnit) -> Double {
        valueInSIUnits / unit.kelvinPerUnit
    }
}

public extension Quantity where Dimension == SpectroscopicWavenumberDimension {
    init(_ value: Double, _ unit: SpectroscopicWavenumberUnit) {
        self.init(valueInSIUnits: value * unit.reciprocalMetersPerUnit)
    }

    func value(in unit: SpectroscopicWavenumberUnit) -> Double {
        valueInSIUnits / unit.reciprocalMetersPerUnit
    }
}

public extension Quantity where Dimension == EnergySquaredDimension {
    /// Creates a squared-energy quantity. For example,
    /// `EnergySquared(0.01, squared: .electronVolt)` represents `0.01 eV^2`.
    init(_ value: Double, squared unit: EnergyUnit) {
        let scale = unit.joulesPerUnit
        self.init(valueInSIUnits: value * scale * scale)
    }

    func value(squared unit: EnergyUnit) -> Double {
        let scale = unit.joulesPerUnit
        return valueInSIUnits / (scale * scale)
    }
}

public extension Quantity where Dimension == InverseTimeSquaredDimension {
    /// Creates an inverse-time-squared quantity. For example,
    /// `InverseTimeSquared(2, perSquared: .picosecond)` represents `2 ps^-2`.
    init(_ value: Double, perSquared unit: TimeUnit) {
        let scale = unit.secondsPerUnit
        self.init(valueInSIUnits: value / (scale * scale))
    }

    func value(perSquared unit: TimeUnit) -> Double {
        let scale = unit.secondsPerUnit
        return valueInSIUnits * scale * scale
    }
}
