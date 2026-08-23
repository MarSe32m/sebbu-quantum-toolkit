// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuScience
import SebbuQuantumToolkit

public struct GPUHOPS: HOPS.Implementation {
    public static func solveWithAuxiliaries<Hamiltonian, RNG>(
        start: Double, end: Double, on: [Double]? = nil,
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: borrowing QuantumSystem<Hamiltonian>,
        lindbladChannels: [LindbladChannel],
        lindbladMethod: HOPS.LindbladMethod,
        hierarchy: Hierarchy,
        equationType: HOPS.EquationType,
        shiftType: HOPS.ShiftType,
        rng: inout RNG,
        intergation: IntegrationOptions,
        _ forEach: (Double, borrowing Span<UniqueVector<Complex<Double>>>) -> Void
    ) where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator, Hamiltonian : ~Copyable {
        fatalError("TODO: Implement")
    }
    
    final public class Hierarchy: HOPS.HierarchySpecification {
        public let bathCorrelationFunctions: SebbuScience.Matrix<SebbuQuantumToolkit.HOPS.BathCorrelationFunction>
        
        public let couplingOperators: [SebbuQuantumToolkit.HOPS.CouplingOperator]
        
        public init(bathCorrelationFunctions: SebbuScience.Matrix<SebbuQuantumToolkit.HOPS.BathCorrelationFunction>, couplingOperators: [SebbuQuantumToolkit.HOPS.CouplingOperator], truncationCondition: (borrowing Span<Int>) -> Bool) {
            fatalError()
        }
        
        
    }
}
