// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum HEOM: Sendable {}

public extension HEOM {
    protocol HierarchySpecification: Sendable {
        var bathCorrelationFunctions: Matrix<BathCorrelationFunction> { get }
        var couplingOperators: [CouplingOperator] { get }
    }
    
    struct BathCorrelationFunction: Sendable {
        
    }
    
    struct CouplingOperator: Sendable {
        
    }
    
    enum ShiftType {
        case none
        case meanField
    }
}

public extension HEOM {
    protocol Implementation {
        associatedtype Hierarchy: HierarchySpecification
        
        static func solveWithAuxiliaries<
            Hamiltonian: HamiltonianFunction & ~Copyable
        >(
            start: Double,
            end: Double,
            on: [Double]?,
            initialState: borrowing UniqueMatrix<Complex<Double>>,
            system: borrowing QuantumSystem<Hamiltonian>,
            linbladChannels: [LindbladChannel],
            hierarchy: Hierarchy,
            shiftType: ShiftType,
            intergation: IntegrationOptions,
            _ forEach: (Double, borrowing Span<UniqueMatrix<Complex<Double>>>) -> Void
        )
    }
}

public extension HEOM.Implementation {
    @inlinable
    static func solve<Hamiltonian>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueMatrix<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        linbladChannels: [LindbladChannel],
        hierarchy: Hierarchy,
        shiftType: HEOM.ShiftType,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, Hierarchy : HEOM.HierarchySpecification, Hamiltonian : ~Copyable {
        solveWithAuxiliaries(start: start, end: end, on: on, initialState: initialState, system: system, linbladChannels: linbladChannels, hierarchy: hierarchy, shiftType: shiftType, intergation: intergation) { t, totalState in
            assert(totalState.count > 0, "Expected at least one state")
            forEach(t, totalState[unchecked: 0])
        }
    }
}

public extension HEOM {
    final class Hierarchy: HierarchySpecification {
        public let bathCorrelationFunctions: SebbuScience.Matrix<BathCorrelationFunction>
        
        public let couplingOperators: [CouplingOperator]
        
        @inlinable
        public init(bathCorrelationFunctions: Matrix<BathCorrelationFunction>, couplingOperators: [CouplingOperator], truncationCondition: (borrowing Span<Int>) -> Bool) {
            self.bathCorrelationFunctions = bathCorrelationFunctions
            self.couplingOperators = couplingOperators
            fatalError("TODO: Implement")
        }
    }
}

extension HEOM: HEOM.Implementation {
    @inlinable
    public static func solveWithAuxiliaries<Hamiltonian>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueMatrix<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        linbladChannels: [LindbladChannel],
        hierarchy: Hierarchy,
        shiftType: ShiftType,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing Span<UniqueMatrix<Complex<Double>>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
}

