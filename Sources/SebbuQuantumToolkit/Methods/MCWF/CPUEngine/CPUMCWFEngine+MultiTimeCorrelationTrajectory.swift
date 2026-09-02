// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUMCWFEngine {
	internal func solveMultiTimeCorrelationTrajectory<Hamiltonian, RNG>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		observing observer: (Double, Complex<Double>) -> Void
	) throws -> PropagationRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let firstTime = request.insertions[0].time
		let dimension = problem.system.dimension

		var guide: Vector<Complex<Double>>
		if start < firstTime {
			var propagated: Vector<Complex<Double>>?
			let summary = try solveTrajectory(
				problem: problem,
				configuration: configuration,
				propagation: PropagationOptions(
					timeSpan: .init(start: start, end: firstTime),
					output: .final,
					integration: propagation.integration
				),
				rng: &rng
			) { _, state in
				propagated = Vector(copying: state)
				return .proceed
			}
			precondition(summary.finalTime == firstTime && summary.endReason == .reachedEndTime)
			guard let propagated else {
				preconditionFailure("The MCWF guide did not emit its final state")
			}
			guide = propagated
		} else {
			var initial = TrajectoryState(problem.initialState)
			try Self.normalize(&initial, at: start)
			guide = Vector(copying: initial.wavefunction)
		}

		var state = CorrelationState(guideAndDyad: guide)
		var operatorStorage = UniqueMatrix<Complex<Double>>.zeros(
			rows: dimension,
			columns: dimension
		)
		var actionStorage = UniqueVector<Complex<Double>>.zero(dimension)
		try _applyMultiTimeCorrelationInsertion(
			request.insertions[0].insertion,
			index: 0,
			at: firstTime,
			dimension: dimension,
			ket: &state.ket,
			bra: &state.bra,
			operatorStorage: &operatorStorage,
			actionStorage: &actionStorage
		)
		var insertionIndex = 1
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		outputCursor.discardTimes(before: request.insertions.last!.time)

		func emit(_ time: Double, _ sample: borrowing CorrelationState) throws {
			observer(
				time,
				try _multiTimeCorrelationSample(
					observable: request.observable,
					at: time,
					dimension: dimension,
					ket: sample.ket,
					bra: sample.bra,
					normalization: sample.guide.normSquared,
					operatorStorage: &operatorStorage,
					actionStorage: &actionStorage
				)
			)
		}

		if insertionIndex == request.insertions.count {
			while let time = outputCursor.nextTime(through: firstTime) {
				try emit(time, state)
			}
		}
		guard firstTime < end else {
			return .init(finalTime: end, endReason: .reachedEndTime)
		}

		let channels = Self.prepareChannels(
			problem.markovianChannels,
			dimension: dimension
		)
		let rhs = CorrelationRightHandSide(
			hamiltonian: problem.system.hamiltonian,
			channels: channels,
			dimension: dimension
		)
		var solver = UniqueDOPRISolver(
			t: firstTime,
			dt: Swift.min(propagation.integration.maximumStepSize, end - firstTime),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y4: CorrelationState(dimension: dimension),
			k1: CorrelationState(dimension: dimension),
			k2: CorrelationState(dimension: dimension),
			k3: CorrelationState(dimension: dimension),
			k4: CorrelationState(dimension: dimension),
			k5: CorrelationState(dimension: dimension),
			k6: CorrelationState(dimension: dimension),
			k7: CorrelationState(dimension: dimension),
			temporary: CorrelationState(dimension: dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)
		var outputState = CorrelationState(dimension: dimension)
		var jumpWorkspace = CorrelationJumpWorkspace(
			channels: channels,
			dimension: dimension
		)
		switch configuration.jumpAlgorithm {
		case .waitingTime(let eventTolerance, let maximumEventIterations):
			var pdpSolver = UniquePDPSolver(
				deterministicSolver: solver,
				jumpKernel: jumpWorkspace,
				eventTimeTolerance: eventTolerance,
				maximumEventIterations: maximumEventIterations,
				using: &rng
			)

			while pdpSolver.t < end {
				let limit = insertionIndex < request.insertions.count
					? request.insertions[insertionIndex].time : end
				let pdpStep: PDPStep
				do {
					pdpStep = try pdpSolver.step(y: &state, upTo: limit)
				} catch let error {
					throw Self.mapPDPSolverError(error)
				}

				switch pdpStep {
				case .accepted(let step):
					if insertionIndex == request.insertions.count {
						while let time = outputCursor.nextTime(
							through: step.endTime
						) {
							if time == step.endTime {
								try emit(time, state)
							} else {
								pdpSolver.interpolateLastStep(
									at: time,
									into: &outputState
								)
								try emit(time, outputState)
							}
						}
					}

				case .jumpPending(let jump):
					if insertionIndex == request.insertions.count {
						while let time = outputCursor.nextTime(
							before: jump.time
						) {
							pdpSolver.interpolateLastStep(
								at: time,
								into: &outputState
							)
							try emit(time, outputState)
						}
					}
					try pdpSolver.applyPendingJump(
						y: &state,
						using: &rng
					)
					if insertionIndex == request.insertions.count {
						while let time = outputCursor.nextTime(
							through: jump.time
						) {
							precondition(time == jump.time)
							try emit(time, state)
						}
					}
				}

				if insertionIndex < request.insertions.count,
					pdpSolver.t == request.insertions[insertionIndex].time
				{
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
					pdpSolver.stateDidChange()
					if insertionIndex == request.insertions.count {
						while let time = outputCursor.nextTime(
							through: event.time
						) {
							try emit(time, state)
						}
					}
				}
			}

		case .discreteTime:
			while solver.t < end {
				let limit = insertionIndex < request.insertions.count
					? request.insertions[insertionIndex].time : end
				let step = try solver.step(y: &state, upTo: limit)
				if insertionIndex == request.insertions.count {
					while let time = outputCursor.nextTime(before: step.endTime) {
						solver.interpolateLastStep(at: time, into: &outputState)
						try emit(time, outputState)
					}
				}
				let hazardTolerance =
					128 * Double.ulpOfOne * Swift.max(1, abs(state.cumulativeHazard))
				guard state.cumulativeHazard.isFinite, state.cumulativeHazard >= -hazardTolerance else {
					throw SolverError.invalidHazard(time: step.endTime)
				}
				let probability = 1 - Double.exp(-Swift.max(.zero, state.cumulativeHazard))
				if probability > .zero, rng.nextUnitDouble() < probability {
					try jumpWorkspace.applyJump(
						at: step.endTime,
						state: &state,
						using: &rng
					)
				}
				state.cumulativeHazard = .zero
				solver.stateDidChange()
				if insertionIndex == request.insertions.count {
					while let time = outputCursor.nextTime(through: step.endTime) {
						precondition(time == step.endTime)
						try emit(time, state)
					}
				}

				if insertionIndex < request.insertions.count,
					solver.t == request.insertions[insertionIndex].time
				{
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
						while let time = outputCursor.nextTime(
							through: event.time
						) {
							try emit(time, state)
						}
					}
				}
			}
		}
		return .init(finalTime: end, endReason: .reachedEndTime)
	}
}
