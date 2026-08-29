// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
    @inlinable
    public func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: MCWF.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        _ forEach: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> Void
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension MCWF {
    @inlinable
    @inline(always)
    public static func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: Configuration,
        propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
        execution: TrajectoryExecution,
        _ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        return try implementation.solveEnsemble(
            problem: problem,
            configuration: configuration,
            propagation: propagation,
            execution: execution,
            forEach
        )
    }
}
