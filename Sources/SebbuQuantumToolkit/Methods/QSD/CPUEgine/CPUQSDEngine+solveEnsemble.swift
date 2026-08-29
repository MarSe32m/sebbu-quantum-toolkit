// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
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
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}


extension QSD {
    @inlinable
    @inline(always)
    @discardableResult
    public static func solveEnsemble<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: Configuration,
        propagation: PropagationOptions<CPUQSDEngine.IntegratorConfiguration>,
        execution: TrajectoryExecution,
        _ forEach: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> Void
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUQSDEngine()
        return try implementation.solveEnsemble(
            problem: problem,
            configuration: configuration,
            propagation: propagation,
            execution: execution,
            forEach
        )
    }
}
