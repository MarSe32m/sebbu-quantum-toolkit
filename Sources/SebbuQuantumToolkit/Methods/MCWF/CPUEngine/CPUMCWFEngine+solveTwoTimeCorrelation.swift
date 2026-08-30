// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	@discardableResult
	public func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		try _validateTwoTimeCorrelationRequest(
			request,
			timeSpan: propagation.timeSpan,
			dimension: problem.system.dimension
		)
		do {
			return try solveMultiTimeOrderedCorrelation(
				problem: problem,
				configuration: configuration,
				request: _multiTimeOrderedRequest(request),
				propagation: propagation,
				execution: execution,
				observing: observer
			)
		} catch let error as MultiTimeOrderedCorrelationError {
			throw _mapMultiTimeOrderedErrorToTwoTime(error)
		}
	}
}

extension MCWF {
	@discardableResult
	public static func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		try CPUMCWFEngine().solveTwoTimeCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			execution: execution,
			observing: observer
		)
	}
}
