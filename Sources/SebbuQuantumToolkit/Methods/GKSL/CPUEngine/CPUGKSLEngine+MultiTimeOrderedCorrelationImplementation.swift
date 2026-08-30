// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUGKSLEngine: GKSL.MultiTimeOrderedCorrelationImplementation {
	public func solveMultiTimeOrderedCorrelation<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		_ = configuration
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let dimension = problem.system.dimension
		try _validateMultiTimeOrderedCorrelationRequest(
			request,
			timeSpan: propagation.timeSpan,
			dimension: dimension
		)
		let lastInsertionTime = request.insertions.last!.time
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		outputCursor.discardTimes(before: lastInsertionTime)
		var state = DensityMatrixState(problem.initialState)
		var scratchState = DensityMatrixState(dimension: dimension)
		var operatorStorage = UniqueMatrix<Complex<Double>>.zeros(
			rows: dimension,
			columns: dimension
		)

		if start == end {
			for index in request.insertions.indices {
				try Self.applyMultiTimeInsertion(
					request.insertions[index],
					index: index,
					dimension: dimension,
					to: &state,
					using: &scratchState,
					operatorStorage: &operatorStorage
				)
			}
			if case .everyAcceptedStep = propagation.output {
				return .init(finalTime: end, endReason: .reachedEndTime)
			}
			while let time = outputCursor.nextTime(through: end) {
				let value = try Self.multiTimeCorrelation(
					observable: request.observable,
					at: time,
					dimension: dimension,
					state: state,
					operatorStorage: &operatorStorage
				)
				if observer(time, value) == .stop {
					return .init(finalTime: time, endReason: .stoppedByObserver)
				}
			}
			return .init(finalTime: end, endReason: .reachedEndTime)
		}

		let rhs = GKSLRightHandSide(problem)
		var solver = UniqueVerner76Solver(
			t: start,
			dt: Swift.min(propagation.integration.maximumStepSize, end - start),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y6: .init(dimension: dimension),
			k1: .init(dimension: dimension),
			k2: .init(dimension: dimension),
			k3: .init(dimension: dimension),
			k4: .init(dimension: dimension),
			k5: .init(dimension: dimension),
			k6: .init(dimension: dimension),
			k7: .init(dimension: dimension),
			k8: .init(dimension: dimension),
			k9: .init(dimension: dimension),
			k10: .init(dimension: dimension),
			temporary: .init(dimension: dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)

		for index in request.insertions.indices {
			let event = request.insertions[index]
			while solver.t < event.time {
				_ = try solver.step(y: &state, upTo: event.time)
			}
			try Self.applyMultiTimeInsertion(
				event,
				index: index,
				dimension: dimension,
				to: &state,
				using: &scratchState,
				operatorStorage: &operatorStorage
			)
			solver.restart(at: event.time)
		}

		if case .everyAcceptedStep = propagation.output {
			// Equal-time output is intentionally omitted for this schedule.
		} else {
			while let time = outputCursor.nextTime(through: lastInsertionTime) {
				let value = try Self.multiTimeCorrelation(
					observable: request.observable,
					at: time,
					dimension: dimension,
					state: state,
					operatorStorage: &operatorStorage
				)
				if observer(time, value) == .stop {
					return .init(finalTime: time, endReason: .stoppedByObserver)
				}
			}
		}

		while solver.t < end {
			let step = try solver.step(y: &state, upTo: end)
			while let time = outputCursor.nextTime(through: step.endTime) {
				let value: Complex<Double>
				if time == step.endTime {
					value = try Self.multiTimeCorrelation(
						observable: request.observable,
						at: time,
						dimension: dimension,
						state: state,
						operatorStorage: &operatorStorage
					)
				} else {
					solver.interpolateLastStep(at: time, into: &scratchState)
					value = try Self.multiTimeCorrelation(
						observable: request.observable,
						at: time,
						dimension: dimension,
						state: scratchState,
						operatorStorage: &operatorStorage
					)
				}
				if observer(time, value) == .stop {
					return .init(finalTime: time, endReason: .stoppedByObserver)
				}
			}
		}
		return .init(finalTime: end, endReason: .reachedEndTime)
	}
}

extension CPUGKSLEngine {
	private static func applyMultiTimeInsertion(
		_ event: TimedCorrelationInsertion,
		index: Int,
		dimension: Int,
		to state: inout DensityMatrixState,
		using scratchState: inout DensityMatrixState,
		operatorStorage: inout UniqueMatrix<Complex<Double>>
	) throws {
		_correlationInsertionOperator(event.insertion).insert(
			t: event.time,
			into: &operatorStorage
		)
		guard operatorStorage.rows == dimension && operatorStorage.columns == dimension else {
			throw MultiTimeOrderedCorrelationError.insertionOperatorDimensionMismatch(
				index: index,
				expected: dimension,
				rows: operatorStorage.rows,
				columns: operatorStorage.columns
			)
		}
		switch event.insertion {
		case .left:
			operatorStorage.dotBLAS(state.densityMatrix, into: &scratchState.densityMatrix)
		case .right:
			state.densityMatrix.dotBLAS(operatorStorage, into: &scratchState.densityMatrix)
		}
		Swift.swap(&state, &scratchState)
	}

	private static func multiTimeCorrelation(
		observable: TimeDependentOperator,
		at time: Double,
		dimension: Int,
		state: borrowing DensityMatrixState,
		operatorStorage: inout UniqueMatrix<Complex<Double>>
	) throws -> Complex<Double> {
		observable.insert(t: time, into: &operatorStorage)
		guard operatorStorage.rows == dimension && operatorStorage.columns == dimension else {
			throw MultiTimeOrderedCorrelationError.observableDimensionMismatch(
				expected: dimension,
				rows: operatorStorage.rows,
				columns: operatorStorage.columns
			)
		}
		return MatrixOperations.traceOfProduct(state.densityMatrix, operatorStorage)
	}
}

extension GKSL {
	@discardableResult
	public static func solveMultiTimeOrderedCorrelation<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		try CPUGKSLEngine().solveMultiTimeOrderedCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			observing: observer
		)
	}

	@discardableResult
	public static func solveMultiTimeOrderedCorrelation<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		try CPUGKSLEngine().solveMultiTimeOrderedCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			observing: observer
		)
	}
}
