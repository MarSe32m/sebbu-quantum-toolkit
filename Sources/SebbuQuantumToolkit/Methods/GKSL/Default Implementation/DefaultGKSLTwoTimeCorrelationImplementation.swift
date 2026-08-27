// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension DefaultGKSLImplementation: GKSL.TwoTimeCorrelationImplementation {
    @inlinable
	public func solveTwoTimeCorrelation<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, Complex<Double>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		_ = configuration

		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let insertionTime = request.insertionTime
		let dimension = problem.system.dimension

		try Self.validate(
			request: request,
			timeSpan: propagation.timeSpan,
			dimension: dimension
		)

		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		outputCursor.discardTimes(before: insertionTime)

		var state = DensityMatrixState(problem.initialState)
		var scratchState = DensityMatrixState(dimension: dimension)
		var operatorStorage = UniqueMatrix<Complex<Double>>.zeros(
			rows: dimension,
			columns: dimension
		)

		if start == end {
			try Self.apply(
				request.insertion,
				at: insertionTime,
				dimension: dimension,
				to: &state,
				using: &scratchState,
				operatorStorage: &operatorStorage
			)

			if case .everyAcceptedStep = propagation.output {
				return
			}
			while let outputTime = outputCursor.nextTime(
				through: insertionTime
			) {
				let value = try Self.correlation(
					observable: request.observable,
					at: outputTime,
					dimension: dimension,
					state: state,
					operatorStorage: &operatorStorage
				)
				forEach(outputTime, value)
			}
			return
		}

		let rhs = GKSLRightHandSide(problem)
		var solver = UniqueDOPRISolver(
			t: start,
			dt: Swift.min(
				propagation.integration.maximumStepSize,
				end - start
			),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y4: .init(dimension: dimension),
			k1: .init(dimension: dimension),
			k2: .init(dimension: dimension),
			k3: .init(dimension: dimension),
			k4: .init(dimension: dimension),
			k5: .init(dimension: dimension),
			k6: .init(dimension: dimension),
			k7: .init(dimension: dimension),
			temporary: .init(dimension: dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)

		// First obtain the unconditioned density matrix rho(s). Output times
		// are intentionally not allowed to constrain these adaptive steps.
		while solver.t < insertionTime {
			_ = try solver.step(y: &state, upTo: insertionTime)
		}

		// Apply B rho(s) or rho(s) B into the preallocated scratch state.
		try Self.apply(
			request.insertion,
			at: insertionTime,
			dimension: dimension,
			to: &state,
			using: &scratchState,
			operatorStorage: &operatorStorage
		)

		// The insertion is a discontinuous state change. In particular, the
		// FSAL derivative from the unconditioned propagation cannot be reused.
		solver.restart(at: insertionTime)

		// Explicit, uniform, and final schedules may request the equal-time
		// value. An accepted-step schedule begins with the first conditioned
		// step strictly after the insertion, as for a fresh propagation.
		if case .everyAcceptedStep = propagation.output {
			// No equal-time output.
		} else {
			while let outputTime = outputCursor.nextTime(
				through: insertionTime
			) {
				let value = try Self.correlation(
					observable: request.observable,
					at: outputTime,
					dimension: dimension,
					state: state,
					operatorStorage: &operatorStorage
				)
				forEach(outputTime, value)
			}
		}

		while solver.t < end {
			let step = try solver.step(y: &state, upTo: end)
			while let outputTime = outputCursor.nextTime(
				through: step.endTime
			) {
				let value: Complex<Double>
				if outputTime == step.endTime {
					value = try Self.correlation(
						observable: request.observable,
						at: outputTime,
						dimension: dimension,
						state: state,
						operatorStorage: &operatorStorage
					)
				} else {
					solver.interpolateLastStep(
						at: outputTime,
						into: &scratchState
					)
					value = try Self.correlation(
						observable: request.observable,
						at: outputTime,
						dimension: dimension,
						state: scratchState,
						operatorStorage: &operatorStorage
					)
				}
				forEach(outputTime, value)
			}
		}
	}
}

extension DefaultGKSLImplementation {
	@inlinable
    internal static func validate(
		request: TwoTimeCorrelationRequest,
		timeSpan: SimulationTimeSpan,
		dimension: Int
	) throws {
		guard request.insertionTime.isFinite else {
			throw TwoTimeCorrelationError.nonFiniteInsertionTime
		}
		guard
			request.insertionTime >= timeSpan.start
				&& request.insertionTime <= timeSpan.end
		else {
			throw TwoTimeCorrelationError.insertionTimeOutsideTimeSpan(
				insertionTime: request.insertionTime,
				start: timeSpan.start,
				end: timeSpan.end
			)
		}

		let insertionOperator: TimeDependentOperator
		switch request.insertion {
		case .left(let value), .right(let value):
			insertionOperator = value
		}

		if let mismatch = insertionOperator.firstDimensionMismatch(
			expected: dimension
		) {
			throw TwoTimeCorrelationError.insertionOperatorDimensionMismatch(
				expected: dimension,
				rows: mismatch.rows,
				columns: mismatch.columns
			)
		}
		if let mismatch = request.observable.firstDimensionMismatch(
			expected: dimension
		) {
			throw TwoTimeCorrelationError.observableDimensionMismatch(
				expected: dimension,
				rows: mismatch.rows,
				columns: mismatch.columns
			)
		}
	}

	@inlinable
    internal static func apply(
		_ insertion: CorrelationInsertion,
		at time: Double,
		dimension: Int,
		to state: inout DensityMatrixState,
		using scratchState: inout DensityMatrixState,
		operatorStorage: inout UniqueMatrix<Complex<Double>>
	) throws {
		let insertionOperator: TimeDependentOperator
		switch insertion {
		case .left(let value), .right(let value):
			insertionOperator = value
		}
		insertionOperator.insert(t: time, into: &operatorStorage)

		guard
			operatorStorage.rows == dimension
				&& operatorStorage.columns == dimension
		else {
			throw TwoTimeCorrelationError.insertionOperatorDimensionMismatch(
				expected: dimension,
				rows: operatorStorage.rows,
				columns: operatorStorage.columns
			)
		}

		switch insertion {
		case .left:
			operatorStorage.dotBLAS(
				state.densityMatrix,
				into: &scratchState.densityMatrix
			)
		case .right:
			state.densityMatrix.dotBLAS(
				operatorStorage,
				into: &scratchState.densityMatrix
			)
		}
		Swift.swap(&state, &scratchState)
	}

    @inlinable
	internal static func correlation(
		observable: TimeDependentOperator,
		at time: Double,
		dimension: Int,
		state: borrowing DensityMatrixState,
		operatorStorage: inout UniqueMatrix<Complex<Double>>
	) throws -> Complex<Double> {
		observable.insert(t: time, into: &operatorStorage)
		guard
			operatorStorage.rows == dimension
				&& operatorStorage.columns == dimension
		else {
			throw TwoTimeCorrelationError.observableDimensionMismatch(
				expected: dimension,
				rows: operatorStorage.rows,
				columns: operatorStorage.columns
			)
		}

		return MatrixOperations.traceOfProduct(
			state.densityMatrix,
			operatorStorage
		)
	}
}

extension TimeDependentOperator {
	@inlinable
    internal func firstDimensionMismatch(
		expected dimension: Int
	) -> (rows: Int, columns: Int)? {
		switch self {
		case .constant(let constantOperator):
			let matrix = constantOperator.matrix
			guard matrix.rows != dimension || matrix.columns != dimension else {
				return nil
			}
			return (matrix.rows, matrix.columns)

		case .linearCombination(let expansion):
			for component in expansion.operators {
				let matrix = component.matrix
				if matrix.rows != dimension || matrix.columns != dimension {
					return (matrix.rows, matrix.columns)
				}
			}
			return nil

		case .generatedDense:
			// Generated operators materialize into dimensioned caller-owned
			// storage. Validate that storage after invoking the generator.
			return nil
		}
	}
}

extension GKSL {
	@inlinable
	@inline(always)
	public static func solveTwoTimeCorrelation<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, Complex<Double>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		let implementation = DefaultGKSLImplementation()
		try implementation.solveTwoTimeCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			forEach
		)
	}

	@inlinable
	@inline(always)
	public static func solveTwoTimeCorrelation<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, Complex<Double>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		let implementation = DefaultGKSLImplementation()
		try implementation.solveTwoTimeCorrelation(
			problem: problem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			forEach
		)
	}
}
