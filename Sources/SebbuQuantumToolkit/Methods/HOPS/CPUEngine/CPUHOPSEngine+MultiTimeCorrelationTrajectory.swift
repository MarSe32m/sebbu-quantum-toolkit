// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// One guide-controlled generator, colored path, white-noise stream and
	/// scalar gauge act on all branches at every integration stage.
	internal func solveMultiTimeCorrelationTrajectory<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>, preparation: Preparation,
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>, seed: UInt64,
		trajectoryID: UInt64, observing observer: (Double, Complex<Double>) -> Void
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let d = preparation.dimension
		let count = preparation.configuration.hierarchy.count
		let equation = preparation.configuration.equationType
		var workspace = CorrelationWorkspace(
			request: request, hierarchyCount: count, dimension: d)
		let rows = count * workspace.branchCount
		let shifts = preparation.shiftCount
		var state = State(dimension: d, hierarchyCount: rows, shiftCount: shifts)
		for j in 0..<d { state.amplitudes.elements[j] = problem.initialState[j] }
		try Self.validate(state, at: start)
		if equation == .nonLinearNormalized { Self.normalize(&state) }
		var cursor = OutputCursor(
			timeSpan: propagation.timeSpan, schedule: propagation.output)
		cursor.discardTimes(before: request.insertions.last!.time)
		var insertionIndex = 0
		if start == end {
			_ = try workspace.process(
				at: start, state: &state, insertionIndex: &insertionIndex,
				cursor: &cursor, request: request, equationType: equation,
				observing: observer)
			return .init(finalTime: end, endReason: .reachedEndTime)
		}
		let rhs = RightHandSide(
			hamiltonian: problem.system.hamiltonian,
			preparation: preparation, seed: seed, trajectoryID: trajectoryID,
			branchCount: workspace.branchCount)
		if preparation.markovianOperators.isEmpty {
			var solver = UniqueDOPRISolver(
				t: start,
				dt: min(propagation.integration.maximumStepSize, end - start),
				maxStep: propagation.integration.maximumStepSize, rhs: rhs,
				y4: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k1: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k2: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k3: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k4: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k5: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k6: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				k7: State(dimension: d, hierarchyCount: rows, shiftCount: shifts),
				temporary: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				absoluteTolerance: propagation.integration.absoluteTolerance,
				relativeTolerance: propagation.integration.relativeTolerance,
				minimumStep: propagation.integration.minimumStepSize)
			_ = try workspace.process(
				at: start, state: &state, insertionIndex: &insertionIndex,
				cursor: &cursor, request: request, equationType: equation,
				observing: observer)
			var noiseIndex = 1
			while solver.t < end {
				var limit =
					insertionIndex < request.insertions.count
					? request.insertions[insertionIndex].time
					: min(cursor.nextRequiredStepBoundary ?? end, end)
				if preparation.noise.latentCount > 0 {
					var boundary = start.addingProduct(
						Double(noiseIndex), preparation.noise.step)
					while boundary <= solver.t {
						noiseIndex += 1
						boundary = start.addingProduct(
							Double(noiseIndex), preparation.noise.step)
					}
					limit = min(limit, boundary)
				}
				let step = try solver.step(y: &state, upTo: limit)
				try Self.validate(state, at: step.endTime)
				if equation == .nonLinearNormalized { Self.normalize(&state) }
				let inserted = try workspace.process(
					at: step.endTime, state: &state,
					insertionIndex: &insertionIndex,
					cursor: &cursor, request: request, equationType: equation,
					observing: observer)
				if inserted || equation == .nonLinearNormalized {
					solver.stateDidChange()
				}
			}
		} else {
			var noiseStorage = [Complex<Double>](
				repeating: .zero, count: preparation.markovianOperators.count)
			let noises = noiseStorage.mutableSpan
			var solver = UniqueSRK2Solver(
				t: start,
				dt: min(propagation.integration.maximumStepSize, end - start),
				rhs: rhs,
				drift0: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				drift1: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				noise0: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				noise1: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				temporary: State(
					dimension: d, hierarchyCount: rows, shiftCount: shifts),
				noises: noises)
			_ = try workspace.process(
				at: start, state: &state, insertionIndex: &insertionIndex,
				cursor: &cursor, request: request, equationType: equation,
				observing: observer)
			while solver.t < end {
				let limit =
					insertionIndex < request.insertions.count
					? request.insertions[insertionIndex].time
					: min(cursor.nextRequiredStepBoundary ?? end, end)
				let step = solver.step(y: &state, upTo: limit)
				try Self.validate(state, at: step.endTime)
				if equation == .nonLinearNormalized { Self.normalize(&state) }
				let inserted = try workspace.process(
					at: step.endTime, state: &state,
					insertionIndex: &insertionIndex,
					cursor: &cursor, request: request, equationType: equation,
					observing: observer)
				if inserted || equation == .nonLinearNormalized {
					solver.stateDidChange()
				}
			}
		}
		return .init(finalTime: end, endReason: .reachedEndTime)
	}
}
