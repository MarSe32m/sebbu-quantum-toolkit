// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
    @inlinable
    public func solveTwoTimeCorrelation<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: MCWF.Configuration,
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        observing observer: (Double, ComplexModule.Complex<Double>) -> PropagationControl
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
}

extension MCWF {
    @inlinable
    public static func solveTwoTimeCorrelation<Hamiltonian>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: MCWF.Configuration,
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        execution: TrajectoryExecution,
        observing observer: (Double, ComplexModule.Complex<Double>) -> PropagationControl
    ) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let engine = CPUMCWFEngine()
        return try engine.solveTwoTimeCorrelation(
            problem: problem,
            configuration: configuration,
            request: request,
            propagation: propagation,
            execution: execution,
            observing: observer
        )
    }
}
