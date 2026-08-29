// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUHOPSEngine: Sendable {
    @inlinable
    public init() {}
}

extension CPUHOPSEngine: HOPS.Implementation {
    @inlinable
    public func solveWithHierarchy<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG, observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solveWithHierarchy<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, seed: UInt64, trajectoryID: UInt64, observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solve<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG, observing observer: (Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solve<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, seed: UInt64, trajectoryID: UInt64, observing observer: (Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solveEnsemble<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, execution: TrajectoryExecution, _ forEach: (Double, borrowing SebbuScience.UniqueMatrix<ComplexModule.Complex<Double>>) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solveTrajectories<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<IntegrationOptions>, execution: TrajectoryExecution, _ forEach: @Sendable (UInt64, Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension CPUHOPSEngine: HOPS.TwoTimeCorrelationImplementation {
    @inlinable
    public func solveTwoTimeCorrelation<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, request: TwoTimeCorrelationRequest, propagation: PropagationOptions<IntegrationOptions>, execution: TrajectoryExecution, observing observer: (Double, ComplexModule.Complex<Double>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension HOPS {
    @inlinable
    @inline(always)
    @discardableResult
    public func solveWithHierarchy<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, rng: inout RNG, observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        let engine = CPUHOPSEngine()
        return try engine.solveWithHierarchy(problem: problem, configuration: configuration, propagation: propagation, rng: &rng, observing: observer)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public func solveWithHierarchy<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, seed: UInt64, trajectoryID: UInt64, observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        let engine = CPUHOPSEngine()
        return try engine.solveWithHierarchy(problem: problem, configuration: configuration, propagation: propagation, seed: seed, trajectoryID: trajectoryID, observing: observer)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public func solve<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, rng: inout RNG, observing observer: (Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        let engine = CPUHOPSEngine()
        return try engine.solve(problem: problem, configuration: configuration, propagation: propagation, rng: &rng, observing: observer)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public func solve<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, seed: UInt64, trajectoryID: UInt64, observing observer: (Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        let engine = CPUHOPSEngine()
        return try engine.solve(problem: problem, configuration: configuration, propagation: propagation, seed: seed, trajectoryID: trajectoryID, observing: observer)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public func solveEnsemble<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, execution: TrajectoryExecution, _ forEach: (Double, borrowing SebbuScience.UniqueMatrix<ComplexModule.Complex<Double>>) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        let engine = CPUHOPSEngine()
        return try engine.solveEnsemble(problem: problem, configuration: configuration, propagation: propagation, execution: execution, forEach)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public func solveTrajectories<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration, propagation: PropagationOptions<CPUHOPSEngine.IntegratorConfiguration>, execution: TrajectoryExecution, _ forEach: @Sendable (UInt64, Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>) -> Void) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        let engine = CPUHOPSEngine()
        return try engine.solveTrajectories(problem: problem, configuration: configuration, propagation: propagation, execution: execution, forEach)
    }
}
