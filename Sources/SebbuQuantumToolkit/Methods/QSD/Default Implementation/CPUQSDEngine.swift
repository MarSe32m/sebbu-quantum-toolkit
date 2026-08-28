// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUQSDEngine: Sendable {
    @inlinable
    public init() {}
}

extension CPUQSDEngine: QSD.RandomNumberGeneratorDrivenImplementation {
    public enum SolverError: Error {
        case forbiddenOutputSchedule
    }
    @inlinable
    public func solve<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: QSD.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
        trajectoryID: UInt64,
        observing observer: (
            Double,
            borrowing UniqueVector<Complex<Double>>
        ) -> PropagationControl
    ) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        switch propagation.output {
        case .everyAcceptedStep: break // Allowed
        case .final: break // Allowed
        case .times(_): throw SolverError.forbiddenOutputSchedule // Not allowed? Since we can't faithfully interpolate a stochastic process
        case .uniform(step: let step): break // Allowed? How do we allow this?
        }
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    @discardableResult
    public func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: QSD.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        _ forEach: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> Void
    ) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }

    @inlinable
    public func solveTrajectories<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: QSD.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        _ forEach:
            @Sendable (  // Will potentially be called from multiple threads for each trajectory
                UInt64,
                Double,
                borrowing UniqueVector<Complex<Double>>
            ) -> Void
    ) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    public func solve<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: QSD.Configuration, propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG, observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        throw ImplementationError.notImplemented
    }
    
	@inlinable
	public func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
    where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
        throw ImplementationError.notImplemented
    }
}

extension QSD {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
        propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
        let implementation = CPUQSDEngine()
        try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUQSDEngine()
        return try implementation.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        let implementation = CPUQSDEngine()
        return try implementation.solveTrajectories(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}
}
