// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Physical constants used by the unit conversions.
///
/// The SI defining constants are exact. The Hartree energy is the 2022 CODATA
/// recommended value and therefore carries a (negligible here) measurement
/// uncertainty. `atomicUnitOfTime` is deliberately derived from the same
/// `reducedPlanckConstant` and `hartreeEnergy` values used by
/// `QuantumUnitSystem`, so the numerical identity `E_h * t_au / hbar == 1`
/// is preserved to floating-point accuracy.
public enum PhysicalConstants {
    /// Version of CODATA used for non-exact constants.
    public static let codataRevision = "2022"

    /// Speed of light in vacuum, in m s^-1 (exact).
    public static let speedOfLight = 299_792_458.0

    /// Planck constant, in J s (exact).
    public static let planckConstant = 6.626_070_15e-34

    /// Reduced Planck constant, in J s.
    public static let reducedPlanckConstant = planckConstant / (2 * Double.pi)

    /// Elementary charge, in C (exact).
    public static let elementaryCharge = 1.602_176_634e-19

    /// Boltzmann constant, in J K^-1 (exact).
    public static let boltzmannConstant = 1.380_649e-23

    /// One electronvolt, in J (exact).
    public static let electronVolt = elementaryCharge

    /// Hartree energy E_h, in J (2022 CODATA).
    public static let hartreeEnergy = 4.359_744_722_206_0e-18

    /// Atomic unit of time hbar / E_h, in s.
    public static let atomicUnitOfTime = reducedPlanckConstant / hartreeEnergy

    /// Atomic unit of temperature E_h / k_B, in K.
    public static let atomicUnitOfTemperature = hartreeEnergy / boltzmannConstant
}
