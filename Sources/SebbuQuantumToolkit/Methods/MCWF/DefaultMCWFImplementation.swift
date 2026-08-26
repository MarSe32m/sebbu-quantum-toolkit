// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum DefaultMCWFImplementation: Sendable {}

extension DefaultMCWFImplementation: MCWF.RandomNumberGeneratorDrivenImplementation {
	@inlinable
	public static func solve<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
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
		configuration: MCWF.Configuration,
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
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public static func solveTrajectories<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension MCWF: MCWF.RandomNumberGeneratorDrivenImplementation {
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
		try DefaultMCWFImplementation.solve(
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
		try DefaultMCWFImplementation.solve(
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
		try DefaultMCWFImplementation.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
		try DefaultMCWFImplementation.solveTrajectories(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

}
