// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// A coherent energy-time unit system for quantum dynamics.
///
/// The system chooses an internal energy scale `E0` and derives the internal
/// time scale as `t0 = hbar / E0`. Consequently, the internal numerical value
/// of hbar is one and a Schrodinger equation can be propagated as
/// `d(psi)/d(tau) = -i H_bar psi`, where `tau = t/t0` and `H_bar = H/E0`.
///
/// This type intentionally models the energy-time sector needed by open
/// quantum-system solvers. The natural-unit convention does not silently set
/// the speed of light to one or define units of mass, charge, or length.
public struct QuantumUnitSystem: Sendable, Hashable, Codable {
    public enum Convention: Sendable, Hashable, Codable {
        /// Hartree atomic energy-time units: E0 = E_h and t0 = hbar / E_h.
        case hartreeAtomic

        /// An hbar = 1 system fixed by an explicit energy or time reference.
        case natural(reference: NaturalUnitReference)
    }

    public let convention: Convention

    private let storedJoulesPerInternalEnergyUnit: Double

    /// Hartree atomic units in the energy-time sector.
    public static let hartreeAtomic = QuantumUnitSystem(
        uncheckedConvention: .hartreeAtomic,
        joulesPerInternalEnergyUnit: PhysicalConstants.hartreeEnergy
    )

    /// Creates and validates a unit system from a convention.
    public init(convention: Convention) throws {
        switch convention {
        case .hartreeAtomic:
            self.init(
                uncheckedConvention: convention,
                joulesPerInternalEnergyUnit: PhysicalConstants.hartreeEnergy
            )

        case let .natural(reference):
            let energyScale = try Self.energyScale(for: reference)
            self.init(
                uncheckedConvention: convention,
                joulesPerInternalEnergyUnit: energyScale
            )
        }
    }

    /// Creates an hbar = 1 system with an explicitly stated reference.
    public static func natural(
        reference: NaturalUnitReference
    ) throws -> QuantumUnitSystem {
        try QuantumUnitSystem(convention: .natural(reference: reference))
    }

    /// Creates an hbar = 1 system while preserving the stated energy unit in
    /// the system's serialised convention and description.
    public static func natural(
        energyScale value: Double,
        unit: EnergyUnit
    ) throws -> QuantumUnitSystem {
        try natural(reference: .energy(value: value, unit: unit))
    }

    /// Creates an hbar = 1 system while preserving the stated time unit in the
    /// system's serialised convention and description.
    public static func natural(
        timeScale value: Double,
        unit: TimeUnit
    ) throws -> QuantumUnitSystem {
        try natural(reference: .time(value: value, unit: unit))
    }

    /// Creates an hbar = 1 system in which one internal energy unit equals
    /// `energyScale`.
    public static func natural(
        energyScale: Energy
    ) throws -> QuantumUnitSystem {
        try natural(
            reference: .energy(
                value: energyScale.value(in: .joule),
                unit: .joule
            )
        )
    }

    /// Creates an hbar = 1 system in which one internal time unit equals
    /// `timeScale`.
    public static func natural(
        timeScale: Time
    ) throws -> QuantumUnitSystem {
        try natural(
            reference: .time(
                value: timeScale.value(in: .second),
                unit: .second
            )
        )
    }

    private init(
        uncheckedConvention: Convention,
        joulesPerInternalEnergyUnit: Double
    ) {
        convention = uncheckedConvention
        storedJoulesPerInternalEnergyUnit = joulesPerInternalEnergyUnit
    }

    /// Number of joules represented by one internal Hamiltonian-energy unit.
    public var joulesPerInternalEnergyUnit: Double {
        storedJoulesPerInternalEnergyUnit
    }

    /// Number of seconds represented by one internal time unit.
    public var secondsPerInternalTimeUnit: Double {
        PhysicalConstants.reducedPlanckConstant
            / storedJoulesPerInternalEnergyUnit
    }

    /// Energy represented by one internal Hamiltonian-energy unit.
    public var internalEnergyUnit: Energy {
        Energy(valueInSIUnits: joulesPerInternalEnergyUnit)
    }

    /// Time represented by one internal time unit.
    public var internalTimeUnit: Time {
        Time(valueInSIUnits: secondsPerInternalTimeUnit)
    }

    /// Numerical value of hbar in this coherent energy-time system.
    public var reducedPlanckConstantInInternalUnits: Double {
        1
    }

    /// Converts a physical time to the raw time used by a solver.
    public func value(of time: Time) -> Double {
        time.valueInSIUnits / secondsPerInternalTimeUnit
    }

    /// Converts an internal solver time back to a physical quantity.
    public func time(fromInternalValue value: Double) -> Time {
        Time(valueInSIUnits: value * secondsPerInternalTimeUnit)
    }

    /// Converts an energy to a raw Hamiltonian coefficient.
    public func hamiltonianValue(of energy: Energy) -> Double {
        energy.valueInSIUnits / joulesPerInternalEnergyUnit
    }

    /// Converts an angular frequency to a raw Hamiltonian coefficient.
    public func hamiltonianValue(
        of angularFrequency: AngularFrequency
    ) -> Double {
        angularFrequency.valueInSIUnits * secondsPerInternalTimeUnit
    }

    /// Converts a cyclic frequency to a raw Hamiltonian coefficient. This is
    /// the conversion `2 * pi * f * t0`, not `f * t0`.
    public func hamiltonianValue(
        of cyclicFrequency: CyclicFrequency
    ) -> Double {
        2 * Double.pi
            * cyclicFrequency.valueInSIUnits
            * secondsPerInternalTimeUnit
    }

    /// Converts a spectroscopic wavenumber to a raw Hamiltonian coefficient
    /// using `E = h c nu_tilde`.
    public func hamiltonianValue(
        of wavenumber: SpectroscopicWavenumber
    ) -> Double {
        2 * Double.pi
            * PhysicalConstants.speedOfLight
            * wavenumber.valueInSIUnits
            * secondsPerInternalTimeUnit
    }

    /// Converts an internal Hamiltonian coefficient back to energy.
    public func energy(fromInternalHamiltonianValue value: Double) -> Energy {
        Energy(valueInSIUnits: value * joulesPerInternalEnergyUnit)
    }

    /// Converts an internal Hamiltonian coefficient back to angular frequency.
    public func angularFrequency(
        fromInternalHamiltonianValue value: Double
    ) -> AngularFrequency {
        AngularFrequency(
            valueInSIUnits: value / secondsPerInternalTimeUnit
        )
    }

    /// Converts an internal Hamiltonian coefficient back to cyclic frequency.
    public func cyclicFrequency(
        fromInternalHamiltonianValue value: Double
    ) -> CyclicFrequency {
        CyclicFrequency(
            valueInSIUnits:
                value / (2 * Double.pi * secondsPerInternalTimeUnit)
        )
    }

    /// Converts an internal Hamiltonian coefficient back to spectroscopic
    /// wavenumber.
    public func spectroscopicWavenumber(
        fromInternalHamiltonianValue value: Double
    ) -> SpectroscopicWavenumber {
        SpectroscopicWavenumber(
            valueInSIUnits:
                value
                / (
                    2 * Double.pi
                        * PhysicalConstants.speedOfLight
                        * secondsPerInternalTimeUnit
                )
        )
    }

    /// Converts an exponential decay or jump rate to a raw inverse-time value.
    /// No factor of 2 pi is introduced.
    public func rateValue(of rate: DecayRate) -> Double {
        rate.valueInSIUnits * secondsPerInternalTimeUnit
    }

    /// Converts an internal exponential decay or jump rate back to a physical
    /// rate.
    public func decayRate(fromInternalValue value: Double) -> DecayRate {
        DecayRate(valueInSIUnits: value / secondsPerInternalTimeUnit)
    }

    /// Converts an absolute temperature to the internal thermal energy k_B T.
    public func thermalEnergyValue(at temperature: Temperature) -> Double {
        PhysicalConstants.boltzmannConstant
            * temperature.valueInSIUnits
            / joulesPerInternalEnergyUnit
    }

    /// Converts an internal thermal-energy value back to absolute temperature.
    public func temperature(
        fromInternalThermalEnergy value: Double
    ) -> Temperature {
        Temperature(
            valueInSIUnits:
                value
                * joulesPerInternalEnergyUnit
                / PhysicalConstants.boltzmannConstant
        )
    }

    /// Converts a bath-correlation coefficient expressed as energy squared.
    /// This is appropriate when the coupling operator is dimensionless and the
    /// physical coefficient has units such as eV^2.
    public func bathCorrelationValue(
        of coefficient: EnergySquared
    ) -> Double {
        let scale = joulesPerInternalEnergyUnit
        return coefficient.valueInSIUnits / (scale * scale)
    }

    /// Converts a bath-correlation coefficient expressed as inverse time
    /// squared, for example ps^-2.
    public func bathCorrelationValue(
        of coefficient: InverseTimeSquared
    ) -> Double {
        let scale = secondsPerInternalTimeUnit
        return coefficient.valueInSIUnits * scale * scale
    }

    /// Converts an internal bath-correlation coefficient back to energy
    /// squared.
    public func energySquared(
        fromInternalBathCorrelationValue value: Double
    ) -> EnergySquared {
        let scale = joulesPerInternalEnergyUnit
        return EnergySquared(valueInSIUnits: value * scale * scale)
    }

    /// Converts an internal bath-correlation coefficient back to inverse time
    /// squared.
    public func inverseTimeSquared(
        fromInternalBathCorrelationValue value: Double
    ) -> InverseTimeSquared {
        let scale = secondsPerInternalTimeUnit
        return InverseTimeSquared(valueInSIUnits: value / (scale * scale))
    }

    // MARK: - Multiplicative conversion factors

    /// Factor that converts a value in `unit` to an internal Hamiltonian value.
    /// Useful for matrices and complex-valued coefficients.
    public func hamiltonianScale(from unit: EnergyUnit) -> Double {
        unit.joulesPerUnit / joulesPerInternalEnergyUnit
    }

    /// Factor that converts a value in `unit` to internal time.
    public func timeScale(from unit: TimeUnit) -> Double {
        unit.secondsPerUnit / secondsPerInternalTimeUnit
    }

    /// Factor that converts an angular-frequency value in `unit` to an
    /// internal Hamiltonian value.
    public func angularFrequencyScale(
        from unit: AngularFrequencyUnit
    ) -> Double {
        unit.radiansPerSecondPerUnit * secondsPerInternalTimeUnit
    }

    /// Factor that converts a cyclic-frequency value in `unit` to an internal
    /// Hamiltonian value, including the required factor of 2 pi.
    public func cyclicFrequencyHamiltonianScale(
        from unit: CyclicFrequencyUnit
    ) -> Double {
        2 * Double.pi * unit.hertzPerUnit * secondsPerInternalTimeUnit
    }

    /// Factor that converts a decay-rate value in `unit` to an internal rate.
    public func decayRateScale(from unit: DecayRateUnit) -> Double {
        unit.perSecondPerUnit * secondsPerInternalTimeUnit
    }

    /// Factor that converts a value in `unit^-1` to internal inverse time.
    /// The value is interpreted as an angular/exponential rate, not a cyclic
    /// frequency.
    public func inverseTimeScale(from unit: TimeUnit) -> Double {
        secondsPerInternalTimeUnit / unit.secondsPerUnit
    }

    /// Factor that converts a value in `unit^2` to internal energy squared.
    public func energySquaredScale(from unit: EnergyUnit) -> Double {
        let scale = hamiltonianScale(from: unit)
        return scale * scale
    }

    /// Factor that converts a value in `unit^-2` to internal inverse time
    /// squared.
    public func inverseTimeSquaredScale(from unit: TimeUnit) -> Double {
        let scale = inverseTimeScale(from: unit)
        return scale * scale
    }

    private static func energyScale(
        for reference: NaturalUnitReference
    ) throws -> Double {
        let value = reference.physicalValue
        guard value.isFinite else {
            throw UnitSystemError.referenceMustBeFinite
        }
        guard value > 0 else {
            throw UnitSystemError.referenceMustBePositive
        }

        let result: Double
        switch reference {
        case let .energy(value, unit):
            result = value * unit.joulesPerUnit

        case let .time(value, unit):
            let seconds = value * unit.secondsPerUnit
            guard seconds.isFinite, seconds > 0 else {
                throw UnitSystemError.derivedScaleIsNotRepresentable
            }
            result = PhysicalConstants.reducedPlanckConstant / seconds
        }

        guard result.isFinite, result > 0 else {
            throw UnitSystemError.derivedScaleIsNotRepresentable
        }
        return result
    }

    private enum CodingKeys: String, CodingKey {
        case convention
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let convention = try container.decode(Convention.self, forKey: .convention)

        do {
            try self.init(convention: convention)
        } catch {
            throw DecodingError.dataCorruptedError(
                forKey: .convention,
                in: container,
                debugDescription: "Invalid quantum unit system: \(error)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(convention, forKey: .convention)
    }
}

extension QuantumUnitSystem: CustomStringConvertible {
    public var description: String {
        switch convention {
        case .hartreeAtomic:
            "Hartree atomic units (E0 = E_h, t0 = hbar/E_h)"

        case let .natural(reference):
            switch reference {
            case let .energy(value, unit):
                "Natural units (hbar = 1, E0 = \(value) \(unit.symbol))"
            case let .time(value, unit):
                "Natural units (hbar = 1, t0 = \(value) \(unit.symbol))"
            }
        }
    }
}
