// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum NMQSD {}

public extension NMQSD {
    struct BathCorrelationFunction: Sendable {
        
    }
    
    struct CouplingOperator: Sendable {
        
    }
    
    enum EquationType: Sendable {
        case linear
        case nonLinear
        case nonLinearNormalized
    }
}

public extension NMQSD {
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
            unravelling: MarkovianUnravelling,
            baths: Matrix<BathCorrelationFunction>,
            couplingOperators: [CouplingOperator],
            equationType: EquationType,
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
            unravelling: MarkovianUnravelling,
            baths: Matrix<BathCorrelationFunction>,
            couplingOperators: [CouplingOperator],
            equationType: EquationType,
            seed: UInt64,
            trajectories: Int,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
        )
    }
}

public extension NMQSD.Implementation {
    static func solveEnsemble<
        Hamiltonian: HamiltonianFunction & ~Copyable
    >(
        start: Double,
        end: Double,
        samplingTimes: [Double]?,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel],
        unravelling: MarkovianUnravelling,
        baths: Matrix<NMQSD.BathCorrelationFunction>,
        couplingOperators: [NMQSD.CouplingOperator],
        equationType: NMQSD.EquationType,
        seed: UInt64,
        trajectories: Int,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) {
        fatalError("TODO: Default implementation")
    }
}
