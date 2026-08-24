// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum MCWF: Sendable {}

public extension MCWF {
    protocol Implementation {
        static func solve<
            Hamiltonian: HamiltonianFunction & ~Copyable,
            RNG: RandomNumberGenerator
        >(
            start: Double,
            end: Double,
            samplingTimes: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            markovianChannels: [MarkovianChannel],
            rng: inout RNG,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
        )
        
        static func solveEnsemble<
            Hamiltonian: HamiltonianFunction & ~Copyable
        >(
            start: Double,
            end: Double,
            samplingTimes: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            markovianChannels: [MarkovianChannel],
            seed: UInt64,
            trajectories: Int,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
        )
    }
}

public extension MCWF.Implementation {
    @inlinable
    static func solveEnsemble<
        Hamiltonian: HamiltonianFunction & ~Copyable
    >(
        start: Double,
        end: Double,
        samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel],
        seed: UInt64,
        trajectories: Int,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) {
        fatalError("TODO: Default implementation")
    }
    
    @inlinable
    static func solve<Hamiltonian>(
        start: Double, end: Double, samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel],
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, Hamiltonian : ~Copyable {
        var rng = NumPyRandom()
        solve(start: start, end: end, samplingTimes: samplingTimes, initialState: initialState, system: system, markovianChannels: markovianChannels, rng: &rng, intergation: intergation, forEach)
    }
}

extension MCWF: MCWF.Implementation {
    @inlinable
    public static func solve<Hamiltonian, RNG>(
        start: Double, end: Double, samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel],
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}

