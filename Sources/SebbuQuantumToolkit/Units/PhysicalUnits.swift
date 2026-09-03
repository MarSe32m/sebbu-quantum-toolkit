// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Supported units of energy.
public enum EnergyUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case joule
    case electronVolt
    case milliElectronVolt
    case microElectronVolt
    case nanoElectronVolt
    case hartree

    /// Number of joules represented by one unit.
    public var joulesPerUnit: Double {
        switch self {
        case .joule:
            1
        case .electronVolt:
            PhysicalConstants.electronVolt
        case .milliElectronVolt:
            1e-3 * PhysicalConstants.electronVolt
        case .microElectronVolt:
            1e-6 * PhysicalConstants.electronVolt
        case .nanoElectronVolt:
            1e-9 * PhysicalConstants.electronVolt
        case .hartree:
            PhysicalConstants.hartreeEnergy
        }
    }

    public var symbol: String {
        switch self {
        case .joule: "J"
        case .electronVolt: "eV"
        case .milliElectronVolt: "meV"
        case .microElectronVolt: "ueV"
        case .nanoElectronVolt: "neV"
        case .hartree: "E_h"
        }
    }
}

/// Supported units of time.
public enum TimeUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case second
    case millisecond
    case microsecond
    case nanosecond
    case picosecond
    case femtosecond
    case attosecond
    case atomicUnit

    /// Number of seconds represented by one unit.
    public var secondsPerUnit: Double {
        switch self {
        case .second: 1
        case .millisecond: 1e-3
        case .microsecond: 1e-6
        case .nanosecond: 1e-9
        case .picosecond: 1e-12
        case .femtosecond: 1e-15
        case .attosecond: 1e-18
        case .atomicUnit: PhysicalConstants.atomicUnitOfTime
        }
    }

    public var symbol: String {
        switch self {
        case .second: "s"
        case .millisecond: "ms"
        case .microsecond: "us"
        case .nanosecond: "ns"
        case .picosecond: "ps"
        case .femtosecond: "fs"
        case .attosecond: "as"
        case .atomicUnit: "t_au"
        }
    }
}

/// Angular-frequency units. Values are angular phase rates, not cycles per
/// second, so no factor of 2 pi is introduced when converting them to a
/// Hamiltonian coefficient.
public enum AngularFrequencyUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case radiansPerSecond
    case kiloRadiansPerSecond
    case megaRadiansPerSecond
    case gigaRadiansPerSecond
    case teraRadiansPerSecond
    case radiansPerNanosecond
    case radiansPerPicosecond
    case radiansPerFemtosecond

    /// Number of radians per second represented by one unit. Radians are
    /// dimensionless in SI, but retained in the name for semantic clarity.
    public var radiansPerSecondPerUnit: Double {
        switch self {
        case .radiansPerSecond: 1
        case .kiloRadiansPerSecond: 1e3
        case .megaRadiansPerSecond: 1e6
        case .gigaRadiansPerSecond: 1e9
        case .teraRadiansPerSecond: 1e12
        case .radiansPerNanosecond: 1e9
        case .radiansPerPicosecond: 1e12
        case .radiansPerFemtosecond: 1e15
        }
    }

    public var symbol: String {
        switch self {
        case .radiansPerSecond: "rad/s"
        case .kiloRadiansPerSecond: "krad/s"
        case .megaRadiansPerSecond: "Mrad/s"
        case .gigaRadiansPerSecond: "Grad/s"
        case .teraRadiansPerSecond: "Trad/s"
        case .radiansPerNanosecond: "rad/ns"
        case .radiansPerPicosecond: "rad/ps"
        case .radiansPerFemtosecond: "rad/fs"
        }
    }
}

/// Cyclic-frequency units. A cyclic frequency `f` contributes the angular
/// phase rate `2 * pi * f` to a Hamiltonian.
public enum CyclicFrequencyUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case hertz
    case kilohertz
    case megahertz
    case gigahertz
    case terahertz
    case petahertz

    /// Number of cycles per second represented by one unit.
    public var hertzPerUnit: Double {
        switch self {
        case .hertz: 1
        case .kilohertz: 1e3
        case .megahertz: 1e6
        case .gigahertz: 1e9
        case .terahertz: 1e12
        case .petahertz: 1e15
        }
    }

    public var symbol: String {
        switch self {
        case .hertz: "Hz"
        case .kilohertz: "kHz"
        case .megahertz: "MHz"
        case .gigahertz: "GHz"
        case .terahertz: "THz"
        case .petahertz: "PHz"
        }
    }
}

/// Units for exponential decay and jump rates. These are inverse times, not
/// cyclic frequencies, and therefore never introduce a factor of 2 pi.
public enum DecayRateUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case perSecond
    case kiloPerSecond
    case megaPerSecond
    case gigaPerSecond
    case teraPerSecond
    case perMillisecond
    case perMicrosecond
    case perNanosecond
    case perPicosecond
    case perFemtosecond

    /// Number of inverse seconds represented by one unit.
    public var perSecondPerUnit: Double {
        switch self {
        case .perSecond: 1
        case .kiloPerSecond: 1e3
        case .megaPerSecond: 1e6
        case .gigaPerSecond: 1e9
        case .teraPerSecond: 1e12
        case .perMillisecond: 1e3
        case .perMicrosecond: 1e6
        case .perNanosecond: 1e9
        case .perPicosecond: 1e12
        case .perFemtosecond: 1e15
        }
    }

    public var symbol: String {
        switch self {
        case .perSecond: "s^-1"
        case .kiloPerSecond: "10^3 s^-1"
        case .megaPerSecond: "10^6 s^-1"
        case .gigaPerSecond: "10^9 s^-1"
        case .teraPerSecond: "10^12 s^-1"
        case .perMillisecond: "ms^-1"
        case .perMicrosecond: "us^-1"
        case .perNanosecond: "ns^-1"
        case .perPicosecond: "ps^-1"
        case .perFemtosecond: "fs^-1"
        }
    }
}

/// Absolute-temperature units.
public enum TemperatureUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case kelvin
    case atomicUnit

    /// Number of kelvin represented by one unit.
    public var kelvinPerUnit: Double {
        switch self {
        case .kelvin: 1
        case .atomicUnit: PhysicalConstants.atomicUnitOfTemperature
        }
    }

    public var symbol: String {
        switch self {
        case .kelvin: "K"
        case .atomicUnit: "T_au"
        }
    }
}

/// Spectroscopic wavenumber units. The conversion uses
/// `E = h * c * wavenumber`.
public enum SpectroscopicWavenumberUnit: String, CaseIterable, Sendable, Hashable, Codable {
    case reciprocalMeter
    case reciprocalCentimeter

    /// Number of inverse metres represented by one unit.
    public var reciprocalMetersPerUnit: Double {
        switch self {
        case .reciprocalMeter: 1
        case .reciprocalCentimeter: 100
        }
    }

    public var symbol: String {
        switch self {
        case .reciprocalMeter: "m^-1"
        case .reciprocalCentimeter: "cm^-1"
        }
    }
}
