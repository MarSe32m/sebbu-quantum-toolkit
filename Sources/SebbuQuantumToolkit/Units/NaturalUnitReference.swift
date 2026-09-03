// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// The dimensional reference that fixes an hbar = 1 natural-unit system.
///
/// Setting hbar to one only imposes `E0 * t0 = hbar`; it does not choose a
/// numerical scale. Supplying either `E0` or `t0` determines the other.
public enum NaturalUnitReference: Sendable, Hashable, Codable {
    /// One internal energy unit represents `value` of `unit`.
    case energy(value: Double, unit: EnergyUnit)

    /// One internal time unit represents `value` of `unit`.
    case time(value: Double, unit: TimeUnit)

    public var physicalValue: Double {
        switch self {
        case let .energy(value, _), let .time(value, _):
            value
        }
    }

    public var energy: Energy? {
        guard case let .energy(value, unit) = self else {
            return nil
        }
        return Energy(value, unit)
    }

    public var time: Time? {
        guard case let .time(value, unit) = self else {
            return nil
        }
        return Time(value, unit)
    }
}
