// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    @inlinable
	@discardableResult
	public func solveTrajectory<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		precondition(
			trajectoryID < UInt64.max,
			"The trajectory ID must fit in a half-open range.")
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let summary = try _solveTrajectory(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID, observing: observer)
		return .init(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1), masterSeed: seed,
			propagation: summary)
	}

	/// Consumes two words to select the reproducible master seed and trajectory
	/// ID, following the other CPU trajectory engines. Propagation uses private,
	/// purpose-separated Philox streams. UInt64.max maps to the valid ID zero.
	@inlinable
    @discardableResult
	public func solveTrajectory<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> TrajectoryRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		try solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: rng.next(), trajectoryID: rng.next() % UInt64.max, observing: observer
		)
	}

    @inlinable
	internal func _solveTrajectory<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>, preparation: Preparation,
		propagation: PropagationOptions<IntegrationOptions>, seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		var root = UniqueVector<Complex<Double>>.zero(preparation.dimension)
		return try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID
		) { t, state in
			for j in 0..<root.count { root[j] = state.amplitudes.elements[j] }
			return observer(t, root)
		}
	}

    @inlinable
	@discardableResult
	public func solveWithHierarchy<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		precondition(
			trajectoryID < UInt64.max,
			"The trajectory ID must fit in a half-open range.")
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let summary = try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID
		) { t, state in
			return Self.withHierarchyView(state) { view in
				observer(t, view)
				return .proceed
			}
		}
		return .init(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1), masterSeed: seed,
			propagation: summary)
	}

    @inlinable
	@discardableResult
	public func solveWithHierarchy<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) ->
			PropagationControl
	) throws -> TrajectoryRunSummary
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let seed = rng.next()
		let id = rng.next() % UInt64.max
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let summary = try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: id
		) { t, state in
			return Self.withHierarchyView(state) { observer(t, $0) }
		}
		return .init(trajectoryIDs: id..<(id + 1), masterSeed: seed, propagation: summary)
	}

    @inlinable
	internal func propagate<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>, preparation: Preparation,
		propagation: PropagationOptions<IntegrationOptions>, seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing State) -> PropagationControl
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let configuration = preparation.configuration
		let d = preparation.dimension
		let count = configuration.hierarchy.count
		precondition(
			count <= Int.max / d
				&& count * d <= Int.max / MemoryLayout<Complex<Double>>.stride,
			"The hierarchy state buffer is too large.")
		let shiftCount = preparation.shiftCount
		var state = State(dimension: d, hierarchyCount: count, shiftCount: shiftCount)
		for j in 0..<d { state.amplitudes.elements[j] = problem.initialState[j] }
		try Self.validate(state, at: start)
		if configuration.equationType == .nonLinearNormalized { Self.normalize(&state) }
		var cursor = OutputCursor(
			timeSpan: propagation.timeSpan, schedule: propagation.output)
		if start == end {
			if let time = cursor.takeInitialTime(), observer(time, state) == .stop {
				return .init(finalTime: time, endReason: .stoppedByObserver)
			}
			return .init(finalTime: end, endReason: .reachedEndTime)
		}

		let rhs = RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: preparation,
			seed: seed, trajectoryID: trajectoryID)
		if preparation.markovianOperators.isEmpty {
			// DOPRI uses fewer full-hierarchy stage buffers than the higher-order
			// Verner solver. OU interpolation error is controlled separately.
			var solver = UniqueDOPRISolver(
				t: start,
				dt: min(propagation.integration.maximumStepSize, end - start),
				maxStep: propagation.integration.maximumStepSize, rhs: rhs,
				y4: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k1: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k2: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k3: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k4: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k5: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k6: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				k7: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				temporary: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				absoluteTolerance: propagation.integration.absoluteTolerance,
				relativeTolerance: propagation.integration.relativeTolerance,
				minimumStep: propagation.integration.minimumStepSize)
			// Every owned propagation buffer exists before the first callback.
			if let time = cursor.takeInitialTime(), observer(time, state) == .stop {
				return .init(finalTime: time, endReason: .stoppedByObserver)
			}
			var noiseIndex = 1
			while solver.t < end {
				var limit = min(cursor.nextRequiredStepBoundary ?? end, end)
				if preparation.noise.latentCount > 0 {
					// Interpolated OU paths have derivative jumps at mesh nodes.
					// Ending steps there avoids unreliable embedded error estimates
					// across kinks and repeated tiny rejected steps around them.
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
				//TODO: Do we need to normalize? There is a term responsible for that in the propagation?
                if configuration.equationType == .nonLinearNormalized {
					Self.normalize(&state)
					solver.stateDidChange()
				}
				while let time = cursor.nextTime(through: step.endTime) {
					if observer(time, state) == .stop {
						return .init(
							finalTime: time,
							endReason: .stoppedByObserver)
					}
				}
			}
		} else {
			// The existing Heun solver advances all tiers and finite-variation
			// shifts together, sharing each Wiener increment between stages.
			var storage = [Complex<Double>](
				repeating: .zero, count: preparation.markovianOperators.count)
			let noises = storage.mutableSpan
			var solver = UniqueSRK2Solver(
				t: start,
				dt: min(propagation.integration.maximumStepSize, end - start),
				rhs: rhs,
				drift0: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				drift1: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				noise0: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				noise1: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				temporary: State(
					dimension: d, hierarchyCount: count, shiftCount: shiftCount),
				noises: noises)
			if let time = cursor.takeInitialTime(), observer(time, state) == .stop {
				return .init(finalTime: time, endReason: .stoppedByObserver)
			}
			while solver.t < end {
				let limit = min(cursor.nextRequiredStepBoundary ?? end, end)
				let step = solver.step(y: &state, upTo: limit)
				try Self.validate(state, at: step.endTime)
                //TODO: Do we need to normalize? There is a term responsible for that in the propagation?
                if configuration.equationType == .nonLinearNormalized {
                    Self.normalize(&state)
                    solver.stateDidChange()
                }
				while let time = cursor.nextTime(through: step.endTime) {
					precondition(
						time == step.endTime,
						"Stochastic outputs must lie at step boundaries.")
					if observer(time, state) == .stop {
						return .init(
							finalTime: time,
							endReason: .stoppedByObserver)
					}
				}
			}
		}
		return .init(finalTime: end, endReason: .reachedEndTime)
	}

    @inlinable
	internal static func withHierarchyView<Result>(
		_ state: borrowing State, _ body: (borrowing HOPS.HierarchyStateView) -> Result
	) -> Result {
		// The callback cannot outlive this explicit borrow. Constructing the
		// span here avoids consuming UniqueMatrix's borrowed span accessor.
		let span = Span(
			_unsafeStart: state.amplitudes.elements,
			count: state.amplitudes.rows * state.amplitudes.columns)
		let view = HOPS.HierarchyStateView(
			systemDimension: state.amplitudes.columns, states: span)
		return body(view)
	}

    @inlinable
    @inline(always)
	internal static func normalize(_ state: inout State) {
		state.amplitudes.divide(by: state.rootNormSquared.squareRoot())
	}

    @inlinable
    @inline(always)
	internal static func validate(_ state: borrowing State, at time: Double) throws {
		let norm = state.rootNormSquared
		guard norm.isFinite && norm > 0 else {
			throw SolverError.invalidStateNorm(time: time)
		}
		for i in 0..<(state.amplitudes.rows * state.amplitudes.columns) {
			let value = state.amplitudes.elements[i]
			guard value.real.isFinite && value.imaginary.isFinite else {
				throw SolverError.nonFiniteState(time: time)
			}
		}
		for i in 0..<state.shifts.count {
			guard state.shifts[i].real.isFinite && state.shifts[i].imaginary.isFinite
			else {
				throw SolverError.nonFiniteState(time: time)
			}
		}
	}
}
