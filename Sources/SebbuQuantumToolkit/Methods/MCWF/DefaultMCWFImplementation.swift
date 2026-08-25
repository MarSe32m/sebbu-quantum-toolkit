// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum DefaultMCWFImplementation: Sendable {}

extension DefaultMCWFImplementation: MCWF.Implementation {
	@inlinable
	public static func solve<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	)
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator, Hamiltonian: ~Copyable {
		fatalError("TODO: Implement")
	}

	@inlinable
	public static func solve<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
		fatalError("TODO: Implement")
	}

	@inlinable
	public static func solveEnsemble<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
		fatalError("TODO: Implementation")
	}
}

extension MCWF: MCWF.Implementation {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	)
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator, Hamiltonian: ~Copyable {
		DefaultMCWFImplementation.solve(
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
	) where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
		DefaultMCWFImplementation.solve(
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
	) -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction, Hamiltonian: ~Copyable {
		DefaultMCWFImplementation.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}
}
