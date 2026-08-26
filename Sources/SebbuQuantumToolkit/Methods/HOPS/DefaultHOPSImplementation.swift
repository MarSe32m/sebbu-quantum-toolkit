// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum DefaultHOPSImplementation: Sendable {}

extension DefaultHOPSImplementation: HOPS.RandomNumberGeneratorDrivenImplementation {
	@inlinable
	public static func solve<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public static func solve<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public static func solveEnsemble<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public static func solveTrajectories<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension HOPS: HOPS.RandomNumberGeneratorDrivenImplementation {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		try DefaultHOPSImplementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, forEach)
	}

	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		try DefaultHOPSImplementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveEnsemble<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		try DefaultHOPSImplementation.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
		try DefaultHOPSImplementation.solveTrajectories(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}
}
