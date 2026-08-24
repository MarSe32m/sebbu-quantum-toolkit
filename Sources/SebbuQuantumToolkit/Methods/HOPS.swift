// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum HOPS: Sendable {}

public extension HOPS {
    enum EquationType: Sendable {
        case linear
        case nonLinear
        case nonLinearNormalized
    }
    
    enum ShiftType: Sendable {
        case none
        case meanField
    }
    
    struct CouplingOperator: Sendable {
        @usableFromInline
        internal var L: Matrix<Complex<Double>>
    }
    
    struct BathCorrelationFunction: Sendable {
        @usableFromInline
        internal var G: [Complex<Double>]
        @usableFromInline
        internal var W: [Complex<Double>]
        @usableFromInline
        internal var r: [Complex<Double>]
    }
    
}

public extension HOPS {
    protocol Implementation {
        associatedtype HierarchySpecification
        
        static func solveWithAuxiliaries<
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
            hierarchy: HierarchySpecification,
            equationType: EquationType,
            shiftType: ShiftType,
            rng: inout RNG,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing Span<UniqueVector<Complex<Double>>>) -> Void
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
            hierarchy: HierarchySpecification,
            equationType: EquationType,
            shiftType: ShiftType,
            seed: UInt64,
            trajectories: Int,
            integration: IntegrationOptions,
            _ forEach: (Double, Int, borrowing UniqueVector<Complex<Double>>) -> Void
        )
    }
}

public extension HOPS.Implementation {
    @inlinable
    static func solve<Hamiltonian, RNG>(
        start: Double, end: Double, samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel] = [],
        unravelling: MarkovianUnravelling = .diffusive,
        hierarchy: HierarchySpecification,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator, Hamiltonian: ~Copyable {
        solveWithAuxiliaries(start: start, end: end, samplingTimes: samplingTimes, initialState: initialState, system: system, markovianChannels: markovianChannels, unravelling: unravelling, hierarchy: hierarchy, equationType: equationType, shiftType: shiftType, rng: &rng, intergation: intergation) { t, totalStateSpan in
            forEach(t, totalStateSpan[unchecked: 0])
        }
    }
    
    @inlinable
    static func solve<Hamiltonian>(
        start: Double, end: Double, samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel] = [],
        unravelling: MarkovianUnravelling = .diffusive,
        hierarchy: HierarchySpecification,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
        var rng = NumPyRandom()
        solve(start: start, end: end, samplingTimes: samplingTimes, initialState: initialState, system: system, hierarchy: hierarchy, equationType: equationType, shiftType: shiftType, rng: &rng, intergation: intergation, forEach)
    }
    
    @inlinable
    static func solveEnsemble<
        Hamiltonian: HamiltonianFunction & ~Copyable
    >(
        start: Double,
        end: Double,
        samplingTimes: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel] = [],
        unravelling: MarkovianUnravelling = .diffusive,
        hierarchy: HierarchySpecification,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        seed: UInt64,
        trajectories: Int,
        integration: IntegrationOptions,
        _ forEach: (Double, Int, borrowing UniqueVector<Complex<Double>>) -> Void
    ) {
        fatalError("TODO: Default implementation")
    }
}

public extension HOPS {
    final class HierarchySpecification {
        public let bathCorrelationFunctions: SebbuScience.Matrix<HOPS.BathCorrelationFunction>
        
        public let couplingOperators: [HOPS.CouplingOperator]
        
        @inlinable
        public init(bathCorrelationFunctions: Matrix<HOPS.BathCorrelationFunction>, couplingOperators: [HOPS.CouplingOperator], truncationCondition: (borrowing Span<Int>) -> Bool) {
            self.bathCorrelationFunctions = bathCorrelationFunctions
            self.couplingOperators = couplingOperators
            fatalError("TODO: Implement")
        }
    }
}

extension HOPS: HOPS.Implementation {
    @inlinable
    public static func solveWithAuxiliaries<
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
        hierarchy: HierarchySpecification,
        equationType: EquationType,
        shiftType: ShiftType,
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing Span<UniqueVector<Complex<Double>>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}
