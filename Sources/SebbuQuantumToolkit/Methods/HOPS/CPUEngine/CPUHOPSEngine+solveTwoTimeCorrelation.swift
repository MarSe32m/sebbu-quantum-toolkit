// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine: HOPS.TwoTimeCorrelationImplementation {
	@discardableResult
	public func solveTwoTimeCorrelation(
		problem: PureStateProblem,
		configuration: HOPS.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> HOPS.EnsembleRunResult {
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

extension HOPS {
	@discardableResult
	public static func solveTwoTimeCorrelation(
		problem: PureStateProblem,
		configuration: HOPS.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> HOPS.EnsembleRunResult {
		try CPUEngine().solveTwoTimeCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			execution: execution,
			observing: observer
		)
	}
}
