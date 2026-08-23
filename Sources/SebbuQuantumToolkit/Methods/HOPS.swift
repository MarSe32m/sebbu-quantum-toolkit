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
    
    protocol HierarchySpecification: Sendable {
        var bathCorrelationFunctions: Matrix<BathCorrelationFunction> { get }
        var couplingOperators: [CouplingOperator] { get }
        
        init(bathCorrelationFunctions: Matrix<BathCorrelationFunction>, couplingOperators: [CouplingOperator], truncationCondition: (borrowing Span<Int>) -> Bool)
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
    
    enum ShiftType: Sendable {
        case none
        case meanField
    }
    
    enum LindbladMethod: Sendable {
        case qsd
        case mcwf
    }
}

public extension HOPS {
    protocol Implementation {
        associatedtype Hierarchy: HierarchySpecification
        
        static func solveWithAuxiliaries<
            Hamiltonian: HamiltonianFunction & ~Copyable,
            RNG: RandomNumberGenerator
        >(
            start: Double,
            end: Double,
            on: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            lindbladChannels: [LindbladChannel],
            lindbladMethod: LindbladMethod,
            hierarchy: Hierarchy,
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
            on: [Double]?,
            initialState: borrowing UniqueVector<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            lindbladChannels: [LindbladChannel],
            lindbladMethod: LindbladMethod,
            hierarchy: Hierarchy,
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
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        lindbladChannels: [LindbladChannel] = [],
        lindbladMethod: HOPS.LindbladMethod = .qsd,
        hierarchy: Hierarchy,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator, Hamiltonian: ~Copyable {
        solveWithAuxiliaries(start: start, end: end, on: on, initialState: initialState, system: system, lindbladChannels: lindbladChannels, lindbladMethod: lindbladMethod, hierarchy: hierarchy, equationType: equationType, shiftType: shiftType, rng: &rng, intergation: intergation) { t, totalStateSpan in
            forEach(t, totalStateSpan[unchecked: 0])
        }
    }
    
    @inlinable
    static func solve<Hamiltonian>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        lindbladChannels: [LindbladChannel] = [],
        lindbladMethod: HOPS.LindbladMethod = .qsd,
        hierarchy: Hierarchy,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
        var rng = NumPyRandom()
        solve(start: start, end: end, on: on, initialState: initialState, system: system, hierarchy: hierarchy, equationType: equationType, shiftType: shiftType, rng: &rng, intergation: intergation, forEach)
    }
    
    @inlinable
    static func solveEnsemble<
        Hamiltonian: HamiltonianFunction & ~Copyable
    >(
        start: Double,
        end: Double,
        on: [Double]?,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        lindbladChannels: [LindbladChannel] = [],
        lindbladMethod: HOPS.LindbladMethod = .qsd,
        hierarchy: Hierarchy,
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
    final class Hierarchy: HierarchySpecification {
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
    public static func solveWithAuxiliaries<Hamiltonian, RNG>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        lindbladChannels: [LindbladChannel],
        lindbladMethod: LindbladMethod,
        hierarchy: Hierarchy,
        equationType: EquationType,
        shiftType: ShiftType,
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing Span<UniqueVector<Complex<Double>>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}
