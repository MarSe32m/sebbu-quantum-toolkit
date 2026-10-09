// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    /// Propagates caller-supplied colored and diffusive realizations. No seed or
    /// bath-path provenance is invented for custom inputs. Zero noise retains
    /// hierarchy memory and Markovian drift; it is not an ensemble average.
    @inlinable
    @discardableResult
    public func solveTrajectory<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        guard configuration.stochasticPotential == nil else {
            throw SolverError.missingPotentialNoise
        }
        return try solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise,
            potentialNoise: ZeroMultiChannelNoise(channelCount: 0), observing: observer)
    }

    @inlinable
	@discardableResult
	public func solveTrajectory(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult {
		try solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: .independent,
			observing: observer)
	}

	@inlinable
	@discardableResult
	public func solveTrajectory(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult {
		precondition(
			trajectoryID < UInt64.max,
			"The trajectory ID must fit in a half-open range.")
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let summary = try _solveTrajectory(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, observing: observer)
		let runSummary = TrajectoryRunSummary(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1), masterSeed: seed,
			ensembleSampling: ensembleSampling,
			propagation: summary)
		let definition = HOPS.BathNoiseDefinition(
			model: configuration.hierarchy.environment.bath,
			timeSpan: propagation.timeSpan,
			stepSize: preparation.noise.step)
		return .init(
			summary: runSummary,
			bathNoise: .init(
				definition: definition, masterSeed: seed, trajectoryID: trajectoryID,
				ensembleSampling: ensembleSampling),
            potentialNoise: preparation.potentialPath(
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling))
	}

	/// Consumes two words to select the reproducible master seed and trajectory
	/// ID, following the other CPU trajectory engines. Propagation uses private,
	/// purpose-separated Philox streams. UInt64.max maps to the valid ID zero.
	@inlinable
    @discardableResult
	public func solveTrajectory<RNG>(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where RNG: RandomNumberGenerator {
		try solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: rng.next(), trajectoryID: rng.next() % UInt64.max, observing: observer
		)
	}

	/// Solves one HOPS trajectory while exposing the exact colored bath noise
	/// evaluated at each output time. The observation values are borrowed and no
	/// samples are retained after the callback returns.
	@discardableResult
	public func solveTrajectory(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observingWithNoise observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing Span<Complex<Double>>) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult {
        try solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling,
            observingWithNoises: { time, state, noise in observer(time, state, noise.bath) })
    }

	/// RNG convenience matching the state-only trajectory observer.
	@discardableResult
	public func solveTrajectory<RNG>(
		problem: PureStateProblem, configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG,
		observingWithNoise observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing Span<Complex<Double>>) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where RNG: RandomNumberGenerator {
		let seed = rng.next()
		let trajectoryID = rng.next() % UInt64.max
		return try solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID, observingWithNoise: observer)
	}

    @inlinable
	internal func _solveTrajectory(
		problem: borrowing PureStateProblem,
        preparation: Preparation,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observing observer: (Double, borrowing UniqueVector<Complex<Double>>) ->
			PropagationControl
	) throws -> PropagationRunSummary {
		var root = UniqueVector<Complex<Double>>.zero(preparation.dimension)
		return try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling
		) { t, state in
			for j in 0..<root.count { root[j] = state.amplitudes.elements[j] }
			return observer(t, root)
		}
	}

    @inlinable
	@discardableResult
	public func solveWithHierarchy(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void
	) throws -> HOPS.TrajectoryRunResult {
		try solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: .independent,
			observing: observer)
	}

	@inlinable
	@discardableResult
	public func solveWithHierarchy(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void
	) throws -> HOPS.TrajectoryRunResult {
		precondition(
			trajectoryID < UInt64.max,
			"The trajectory ID must fit in a half-open range.")
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let propagationSummary = try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling
		) { t, state in
			return Self.withHierarchyView(state) { view in
				observer(t, view)
				return .proceed
			}
		}
		let runSummary = TrajectoryRunSummary(
			trajectoryIDs: trajectoryID..<(trajectoryID + 1), masterSeed: seed,
			ensembleSampling: ensembleSampling,
			propagation: propagationSummary)
		let definition = HOPS.BathNoiseDefinition(
			model: configuration.hierarchy.environment.bath,
			timeSpan: propagation.timeSpan,
			stepSize: preparation.noise.step)
		return .init(
			summary: runSummary,
			bathNoise: .init(
				definition: definition, masterSeed: seed, trajectoryID: trajectoryID,
				ensembleSampling: ensembleSampling),
            potentialNoise: preparation.potentialPath(
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling))
	}

    @inlinable
	@discardableResult
	public func solveWithHierarchy<RNG>(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where RNG: RandomNumberGenerator {
		let seed = rng.next()
		let id = rng.next() % UInt64.max
		let preparation = try Preparation(
			problem: problem, configuration: configuration, propagation: propagation)
		let propagationSummary = try propagate(
			problem: problem, preparation: preparation, propagation: propagation,
            seed: seed, trajectoryID: id, ensembleSampling: .independent
		) { t, state in
			return Self.withHierarchyView(state) { observer(t, $0) }
		}
		let runSummary = TrajectoryRunSummary(
			trajectoryIDs: id..<(id + 1), masterSeed: seed,
			propagation: propagationSummary)
		let definition = HOPS.BathNoiseDefinition(
			model: configuration.hierarchy.environment.bath,
			timeSpan: propagation.timeSpan,
			stepSize: preparation.noise.step)
		return .init(
			summary: runSummary,
			bathNoise: .init(
				definition: definition, masterSeed: seed, trajectoryID: id),
            potentialNoise: preparation.potentialPath(
                seed: seed, trajectoryID: id, ensembleSampling: .independent))
	}

	/// Solves one HOPS trajectory while exposing both the complete hierarchy and
	/// the exact colored bath noise evaluated at each output time.
	@discardableResult
	public func solveWithHierarchy(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observingWithNoise observer: (
			Double, borrowing HOPS.HierarchyStateView, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult {
        try solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling,
            observingWithNoises: { time, state, noise in observer(time, state, noise.bath) })
    }

	/// RNG convenience matching the hierarchy-only observer.
	@discardableResult
	public func solveWithHierarchy<RNG>(
		problem: PureStateProblem,
        configuration: HOPS.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
		observingWithNoise observer: (
			Double, borrowing HOPS.HierarchyStateView, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where RNG: RandomNumberGenerator {
		let seed = rng.next()
		let trajectoryID = rng.next() % UInt64.max
		return try solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID, observingWithNoise: observer)
	}

    @inlinable
	internal func propagate(
		problem: borrowing PureStateProblem,
        preparation: Preparation,
		propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling,
		observing observer: (Double, borrowing State) -> PropagationControl
	) throws -> PropagationRunSummary {
        try propagate(
            problem: problem, preparation: preparation, propagation: propagation,
            rhs: Self.makeSeededRightHandSide(
                hamiltonian: problem.system.hamiltonian, preparation: preparation,
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling)
        ) { time, state, _ in observer(time, state) }
    }

    @inlinable
    internal func propagate<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: borrowing PureStateProblem, preparation: Preparation,
        propagation: PropagationOptions<IntegrationOptions>,
        rhs: consuming RightHandSide<BathNoise, MarkovianNoise, PotentialNoise>,
        observing observer: (Double, borrowing State, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
		let progress = propagation.progress.incrementing(total: 1)
		defer { progress?.finish() }

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
        var rhs = rhs
        var bathGrid = NoiseGridCursor(rhs.bathNoise.integrationGrid, start: start)
        var potentialGrid = NoiseGridCursor(rhs.potentialNoise.integrationGrid, start: start)
        rhs.sampleColoredNoises(at: start)
        // These addresses are owned by rhs, which is moved into exactly one
        // solver below. Each solver is kept alive through the last callback.
        // Accepted DOPRI/Heun steps evaluate their final drift at step.endTime;
        // prescribed outputs also lie at these boundaries. Thus these are the
        // retained path values at the observation time, without a replay sampler.
        let noise = NoiseObservation(
            bath: rhs.physicalNoise.components, bathCount: rhs.physicalNoise.count,
            potential: rhs.potentialCoefficients.components,
            potentialCount: rhs.potentialCoefficients.count)
        if start == end {
            defer { _fixLifetime(rhs) }
            if let time = cursor.takeInitialTime(), noise.observe(time, state, observer) == .stop {
                return .init(finalTime: time, endReason: .stoppedByObserver)
            }
            progress?.increment()
            return .init(finalTime: end, endReason: .reachedEndTime)
        }
        if preparation.markovianChannels.isEmpty || MarkovianNoise.isIdenticallyZero {
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
            defer { _fixLifetime(solver) }
			// Every owned propagation buffer exists before the first callback.
			if let time = cursor.takeInitialTime(), noise.observe(time, state, observer) == .stop {
				return .init(finalTime: time, endReason: .stoppedByObserver)
			}
			while solver.t < end {
				var limit = min(cursor.nextRequiredStepBoundary ?? end, end)
                limit = bathGrid.limit(limit, after: solver.t)
                limit = potentialGrid.limit(limit, after: solver.t)
				let step = try solver.step(y: &state, upTo: limit)
				try Self.validate(state, at: step.endTime)
                if configuration.equationType == .nonLinearNormalized {
                    let normSquared = state.rootNormSquared
                    let norm = normSquared.squareRoot()
                    
                    let normalizationTolerance = propagation.integration.normalizationDriftTolerance
                    ?? 100 * (
                        propagation.integration.absoluteTolerance
                        + propagation.integration.relativeTolerance
                    )
                    
                    if Swift.abs(norm - 1) > normalizationTolerance {
                        Self.normalize(&state)
                        solver.stateDidChange()
                    }
                }
				while let time = cursor.nextTime(through: step.endTime) {
					if noise.observe(time, state, observer) == .stop {
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
				repeating: .zero, count: preparation.markovianChannels.count)
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
            defer { _fixLifetime(solver) }
			if let time = cursor.takeInitialTime(), noise.observe(time, state, observer) == .stop {
				return .init(finalTime: time, endReason: .stoppedByObserver)
			}
			while solver.t < end {
                var limit = min(cursor.nextRequiredStepBoundary ?? end, end)
                limit = bathGrid.limit(limit, after: solver.t)
                limit = potentialGrid.limit(limit, after: solver.t)
				let step = solver.step(y: &state, upTo: limit)
				try Self.validate(state, at: step.endTime)
                if configuration.equationType == .nonLinearNormalized {
                    let normSquared = state.rootNormSquared
                    let norm = normSquared.squareRoot()
                    
                    let normalizationTolerance = propagation.integration.normalizationDriftTolerance
                    ?? 100 * (
                        propagation.integration.absoluteTolerance
                        + propagation.integration.relativeTolerance
                    )
                    
                    if Swift.abs(norm - 1) > normalizationTolerance {
                        Self.normalize(&state)
                        solver.stateDidChange()
                    }
                }
				while let time = cursor.nextTime(through: step.endTime) {
					precondition(
						time == step.endTime,
						"Stochastic outputs must lie at step boundaries.")
					if noise.observe(time, state, observer) == .stop {
						return .init(
							finalTime: time,
							endReason: .stoppedByObserver)
					}
				}
			}
		}
		progress?.increment()
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
