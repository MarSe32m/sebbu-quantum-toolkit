// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum GKSL: Sendable {}

public extension GKSL {
    protocol Implementation {
        static func solve<
            Hamiltonian: HamiltonianFunction & ~Copyable
        >(
            start: Double,
            end: Double,
            samplingTimes: [Double]?,
            initialState: borrowing UniqueMatrix<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            markovianChannels: [MarkovianChannel],
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
        )
    }
}

extension GKSL: GKSL.Implementation {
    @inlinable
    public static func solve<Hamiltonian>(
        start: Double, end: Double, samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueMatrix<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel],
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}
