// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

extension CPUQSDEngine {
	internal func solveMultiTimeCorrelationTrajectory<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, Complex<Double>) -> Void
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let dimension = problem.system.dimension
		var state = CorrelationState(guide: problem.initialState)
		try Self.prepareMultiTimeCorrelationState(
			&state,
			equationType: configuration.equationType,
			at: start
		)
		var operatorStorage = UniqueMatrix<Complex<Double>>.zeros(
			rows: dimension,
			columns: dimension
		)
		var actionStorage = UniqueVector<Complex<Double>>.zero(dimension)
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		outputCursor.discardTimes(before: request.insertions.last!.time)
		var insertionIndex = 0

		func emit(_ time: Double, _ sample: borrowing CorrelationState) throws {
			observer(
				time,
				try Self.multiTimeCorrelationSample(
					observable: request.observable,
					at: time,
					dimension: dimension,
					state: sample,
					equationType: configuration.equationType,
					operatorStorage: &operatorStorage,
					actionStorage: &actionStorage
				)
			)
		}

		if request.insertions[0].time == start {
			state.initializeDyadFromGuide()
			let event = request.insertions[0]
			try _applyMultiTimeCorrelationInsertion(
				event.insertion,
				index: 0,
				at: event.time,
				dimension: dimension,
				ket: &state.ket,
				bra: &state.bra,
				operatorStorage: &operatorStorage,
				actionStorage: &actionStorage
			)
			insertionIndex = 1
			if insertionIndex == request.insertions.count {
				while let time = outputCursor.nextTime(through: start) {
					try emit(time, state)
				}
			}
		}
		guard start < end else {
			return .init(finalTime: end, endReason: .reachedEndTime)
		}

		var noiseStorage = [Complex<Double>](
			repeating: .zero,
			count: problem.markovianChannels.count
		)
		let noises = noiseStorage.mutableSpan
		let rhs = CorrelationRightHandSide(
			problem,
			equationType: configuration.equationType,
			seed: seed,
			trajectoryID: trajectoryID
		)
		var solver = UniqueSRK2Solver(
			t: start,
			dt: Swift.min(propagation.integration.maximumStepSize, end - start),
			rhs: rhs,
			drift0: CorrelationState(dimension: dimension),
			drift1: CorrelationState(dimension: dimension),
			noise0: CorrelationState(dimension: dimension),
			noise1: CorrelationState(dimension: dimension),
			temporary: CorrelationState(dimension: dimension),
			noises: noises
		)

		while solver.t < end {
			let limit: Double
			if insertionIndex < request.insertions.count {
				limit = request.insertions[insertionIndex].time
			} else {
				limit = Swift.min(outputCursor.nextRequiredStepBoundary ?? end, end)
			}
			precondition(limit > solver.t)
			let step = solver.step(y: &state, upTo: limit)
			try Self.prepareMultiTimeCorrelationState(
				&state,
				equationType: configuration.equationType,
				at: step.endTime
			)
			if case .nonLinearNormalized = configuration.equationType {
				solver.stateDidChange()
			}

			if insertionIndex < request.insertions.count,
				step.endTime == request.insertions[insertionIndex].time
			{
				if insertionIndex == 0 { state.initializeDyadFromGuide() }
				let event = request.insertions[insertionIndex]
				try _applyMultiTimeCorrelationInsertion(
					event.insertion,
					index: insertionIndex,
					at: event.time,
					dimension: dimension,
					ket: &state.ket,
					bra: &state.bra,
					operatorStorage: &operatorStorage,
					actionStorage: &actionStorage
				)
				insertionIndex += 1
				solver.stateDidChange()
				if insertionIndex == request.insertions.count {
					while let time = outputCursor.nextTime(through: event.time) {
						try emit(time, state)
					}
				}
				continue
			}

			if insertionIndex == request.insertions.count {
				while let time = outputCursor.nextTime(through: step.endTime) {
					precondition(time == step.endTime)
					try emit(time, state)
				}
			}
		}
		return .init(finalTime: end, endReason: .reachedEndTime)
	}

	internal static func prepareMultiTimeCorrelationState(
		_ state: inout CorrelationState,
		equationType: QSD.EquationType,
		at time: Double
	) throws {
		switch equationType {
		case .linear:
			return
		case .nonLinear:
			let normSquared = state.guide.normSquared
			guard normSquared.isFinite && normSquared > .zero else {
				throw SolverError.invalidStateNorm(time: time)
			}
		case .nonLinearNormalized:
			let normSquared = state.guide.normSquared
			guard normSquared.isFinite && normSquared > .zero else {
				throw SolverError.invalidStateNorm(time: time)
			}
			let norm = normSquared.squareRoot()
			state.guide.divideBLAS(by: norm)
			state.ket.divideBLAS(by: norm)
			state.bra.divideBLAS(by: norm)
		}
	}

	internal static func multiTimeCorrelationSample(
		observable: TimeDependentOperator,
		at time: Double,
		dimension: Int,
		state: borrowing CorrelationState,
		equationType: QSD.EquationType,
		operatorStorage: inout UniqueMatrix<Complex<Double>>,
		actionStorage: inout UniqueVector<Complex<Double>>
	) throws -> Complex<Double> {
		let normalization: Double
		switch equationType {
		case .linear:
			normalization = 1
		case .nonLinear, .nonLinearNormalized:
			normalization = state.guide.normSquared
			guard normalization.isFinite && normalization > .zero else {
				throw SolverError.invalidStateNorm(time: time)
			}
		}
		return try _multiTimeCorrelationSample(
			observable: observable,
			at: time,
			dimension: dimension,
			ket: state.ket,
			bra: state.bra,
			normalization: normalization,
			operatorStorage: &operatorStorage,
			actionStorage: &actionStorage
		)
	}
}
