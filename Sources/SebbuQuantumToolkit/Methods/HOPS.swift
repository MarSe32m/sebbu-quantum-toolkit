//
//  HOPS.swift
//  sebbu-quantum-toolkit
//
//  Created by Sebastian Toivonen on 24.8.2026.
//

import Numerics
import SebbuScience

public enum HOPS {}

public extension HOPS {
    struct Hierarchy: Sendable {
        
    }
    
    enum EquationType: Sendable {
        case linear
        case nonLinear
        case nonLinearNormalized
    }
    
    enum ShiftType: Sendable {
        case none
        case meanField
    }
    
    struct Configuration: Sendable {
        public let hierarchy: Hierarchy
        public var equationType: EquationType
        public var shiftType: ShiftType
        public var unravelling: MarkovianUnravelling

        public init(
            hierarchy: Hierarchy,
            equationType: EquationType,
            shiftType: ShiftType = .none,
            unravelling: MarkovianUnravelling = .diffusive
        ) {
            self.hierarchy = hierarchy
            self.equationType = equationType
            self.shiftType = shiftType
            self.unravelling = unravelling
        }
    }
}

public extension HOPS {
    protocol Implementation {
        static func solve<Hamiltonian, RNG>(
            problem: borrowing PureStateProblem<Hamiltonian>,
            configuration: HOPS.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            rng: inout RNG,
            _ forEach: (
                Double,
                borrowing UniqueVector<Complex<Double>>
            ) -> Void
        )
        where
            Hamiltonian: HamiltonianFunction & ~Copyable,
            RNG: RandomNumberGenerator

        static func solve<Hamiltonian>(
            problem: borrowing PureStateProblem<Hamiltonian>,
            configuration: HOPS.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            seed: UInt64,
            trajectoryID: UInt64,
            _ forEach: (
                Double,
                borrowing UniqueVector<Complex<Double>>
            ) -> Void
        )
        where Hamiltonian: HamiltonianFunction & ~Copyable

        @discardableResult
        static func solveEnsemble<Hamiltonian>(
            problem: borrowing PureStateProblem<Hamiltonian>,
            configuration: HOPS.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            execution: TrajectoryExecution,
            _ forEach: (
                Double,
                borrowing UniqueMatrix<Complex<Double>>
            ) -> Void
        ) -> TrajectoryRunSummary
        where Hamiltonian: HamiltonianFunction & ~Copyable
        
        static func solveTrajectories<Hamiltonian>(
            problem: borrowing PureStateProblem<Hamiltonian>,
            configuration: HOPS.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            execution: TrajectoryExecution,
            _ forEach: @Sendable ( // Can be called from multiple threads for each trajectory
                UInt64,
                Double,
                borrowing UniqueVector<Complex<Double>>
            ) -> Void
        )
    }
}
