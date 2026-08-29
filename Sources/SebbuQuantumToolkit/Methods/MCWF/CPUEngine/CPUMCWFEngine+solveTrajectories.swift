// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

extension CPUMCWFEngine {
	@inlinable
	public func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		throw ImplementationError.notImplemented
	}
}

extension MCWF {
	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64,
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let implementation = CPUMCWFEngine()
		return try implementation.solveTrajectories(
			problem: problem,
			configuration: configuration,
			propagation: propagation,
			execution: execution,
			forEach
		)
	}
}
