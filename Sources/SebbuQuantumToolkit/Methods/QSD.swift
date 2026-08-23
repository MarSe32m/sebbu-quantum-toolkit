// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum QSD: Sendable {}

public extension QSD {
    protocol Implementation {
        static func solve<
            Hamiltonian: HamiltonianFunction & ~Copyable,
            RNG: RandomNumberGenerator
        >(
            start: Double,
            end: Double,
            on: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            linbladChannels: [LindbladChannel],
            rng: inout RNG,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
        )
        
        static func solveEnsemble<
            Hamiltonian: HamiltonianFunction & ~Copyable
        >(
            start: Double,
            end: Double,
            on: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            linbladChannels: [LindbladChannel],
            seed: UInt64,
            trajectories: Int,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
        )
    }
}

public extension QSD.Implementation {
    static func solveEnsemble<
        Hamiltonian: HamiltonianFunction & ~Copyable
    >(
        start: Double,
        end: Double,
        on: [Double]?,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        linbladChannels: [LindbladChannel],
        seed: UInt64,
        trajectories: Int,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) {
        fatalError("TODO: Default implementation")
    }
}

extension QSD: QSD.Implementation {
    @inlinable
    public static func solve<Hamiltonian, RNG>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        linbladChannels: [LindbladChannel],
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}
