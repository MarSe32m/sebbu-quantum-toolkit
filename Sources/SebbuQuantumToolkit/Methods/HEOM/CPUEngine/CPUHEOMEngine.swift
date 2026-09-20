// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public extension HEOM {
    /// Accepts `PropagationOptions.progress` through the common solver API.
    /// When propagation is implemented, use its continuous reporter after each
    /// accepted time step, as GKSL does. The current stubs throw without output.
    struct CPUEngine: Sendable {
        @inlinable
        public init() {}
    }
}

extension HEOM.CPUEngine: HEOM.Implementation {
    @inlinable
    @discardableResult
    public func solve<Hamiltonian>(
        problem: DensityMatrixProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension HEOM.CPUEngine: HEOM.HierarchyProvidingImplementation {
    @inlinable
    @discardableResult
    public func solveWithHierarchy<Hamiltonian>(
        problem: DensityMatrixProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension HEOM {
    @inlinable
    @discardableResult
    public static func solve<Hamiltonian>(
        problem: DensityMatrixProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    @discardableResult
    public static func solve<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    @discardableResult
    public static func solveWithHierarchy<Hamiltonian>(
        problem: DensityMatrixProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    @discardableResult
    public static func solveWithHierarchy<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}
