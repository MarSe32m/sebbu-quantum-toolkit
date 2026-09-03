// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

public extension QuantumUnitSystem {
    /// Scales a complex energy, expressed in `unit`, to a complex internal
    /// Hamiltonian coefficient. This is useful for non-Hermitian effective
    /// Hamiltonians and arrays that cannot carry `Quantity` wrappers.
    func hamiltonianValue<RealType>(
        of energy: Complex<RealType>,
        expressedIn unit: EnergyUnit
    ) -> Complex<RealType>
    where RealType: Real, RealType: BinaryFloatingPoint {
        scale(energy, by: hamiltonianScale(from: unit))
    }

    /// Scales a complex angular/exponential inverse-time quantity expressed in
    /// `unit^-1`. No factor of 2 pi is introduced.
    func inverseTimeValue<RealType>(
        of value: Complex<RealType>,
        per unit: TimeUnit
    ) -> Complex<RealType>
    where RealType: Real, RealType: BinaryFloatingPoint {
        scale(value, by: inverseTimeScale(from: unit))
    }

    /// Scales a complex bath-correlation coefficient expressed in `unit^2`.
    func bathCorrelationValue<RealType>(
        of coefficient: Complex<RealType>,
        squared unit: EnergyUnit
    ) -> Complex<RealType>
    where RealType: Real, RealType: BinaryFloatingPoint {
        scale(coefficient, by: energySquaredScale(from: unit))
    }

    /// Scales a complex bath-correlation coefficient expressed in `unit^-2`.
    func bathCorrelationValue<RealType>(
        of coefficient: Complex<RealType>,
        perSquared unit: TimeUnit
    ) -> Complex<RealType>
    where RealType: Real, RealType: BinaryFloatingPoint {
        scale(coefficient, by: inverseTimeSquaredScale(from: unit))
    }

    /// Constructs the common HOPS exponent `W = gamma + i Omega` from a decay
    /// rate and an angular frequency.
    func bathExponentValue(
        decayRate: DecayRate,
        angularFrequency: AngularFrequency
    ) -> Complex<Double> {
        Complex(
            rateValue(of: decayRate),
            hamiltonianValue(of: angularFrequency)
        )
    }

    /// Constructs `W = gamma + i 2 pi f` when the oscillation is supplied as a
    /// cyclic frequency.
    func bathExponentValue(
        decayRate: DecayRate,
        cyclicFrequency: CyclicFrequency
    ) -> Complex<Double> {
        Complex(
            rateValue(of: decayRate),
            hamiltonianValue(of: cyclicFrequency)
        )
    }
}

private func scale<RealType>(
    _ value: Complex<RealType>,
    by doubleScale: Double
) -> Complex<RealType>
where RealType: Real, RealType: BinaryFloatingPoint {
    let factor = RealType(doubleScale)
    return Complex(value.real * factor, value.imaginary * factor)
}
