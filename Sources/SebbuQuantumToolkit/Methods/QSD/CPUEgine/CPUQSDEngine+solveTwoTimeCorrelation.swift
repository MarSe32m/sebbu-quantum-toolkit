// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience
import Synchronization

extension CPUQSDEngine {
	/// Computes an ensemble two-time correlation with a shared-noise
	/// guide--companion construction.
	///
	/// The guide determines all nonlinear QSD expectations and noise shifts.
	/// The companion follows the resulting linear stochastic propagator with
	/// the same Wiener increments. Scheduled output must be fixed across
	/// trajectories.
	@discardableResult
    @inlinable
	public func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
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

		let outputTimes = try _fixedEnsembleOutputTimes(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		).filter { $0 >= request.insertionTime }
		let masterSeed = execution.resolvedMasterSeed()
		let trajectoryCount = execution.trajectoryIDs.count
		let maximumConcurrentTasks = Swift.min(
			execution.resolvedMaximumConcurrentTasks,
			trajectoryCount
		)

		let currentThreadCount = BLAS.getNumThreads()
		BLAS.setNumThreads(1)
		defer { BLAS.setNumThreads(currentThreadCount) }

		let ensembleSums = Mutex(
			[Complex<Double>](repeating: .zero, count: outputTimes.count)
		)
		let currentTrajectoryID = Atomic(execution.trajectoryIDs.lowerBound)
		let results: [_TrajectoryEnsembleBatchResult] = FixedWorkerPool.with(
			workers: maximumConcurrentTasks
		) { _ in
			var localSums = [Complex<Double>](
				repeating: .zero,
				count: outputTimes.count
			)
			var trajectoryID = currentTrajectoryID.wrappingAdd(
				1,
				ordering: .relaxed
			).oldValue
			var completedTrajectoryCount = 0

			while trajectoryID < execution.trajectoryIDs.upperBound {
				var sampleIndex = 0
				do {
					let summary = try solveCorrelationTrajectory(
						problem: problem,
						configuration: configuration,
						request: request,
						propagation: propagation,
						seed: masterSeed,
						trajectoryID: trajectoryID
					) { time, value in
						precondition(
							sampleIndex < outputTimes.count
								&& time == outputTimes[sampleIndex],
							"A QSD correlation trajectory produced an unexpected output time"
						)
						localSums[sampleIndex] += value
						sampleIndex += 1
					}
					precondition(
						sampleIndex == outputTimes.count,
						"A QSD correlation trajectory omitted an output sample"
					)
					precondition(
						summary.finalTime == propagation.timeSpan.end
							&& summary.endReason == .reachedEndTime,
						"A QSD correlation trajectory must reach the end time"
					)
				} catch {
					return _TrajectoryEnsembleBatchResult(
						trajectoryCount: completedTrajectoryCount,
						failure: _TrajectoryFailure(
							trajectoryID: trajectoryID,
							error: error
						)
					)
				}

				completedTrajectoryCount += 1
				trajectoryID =
					currentTrajectoryID.add(
						1,
						ordering: .relaxed
					).oldValue
			}

			ensembleSums.withLock { sums in
				for index in sums.indices {
					sums[index] += localSums[index]
				}
			}
			return _TrajectoryEnsembleBatchResult(
				trajectoryCount: completedTrajectoryCount,
				failure: nil
			)
		}

		if let failure = results.compactMap(\.failure).first {
			throw failure.error
		}
		let completedTrajectories = results.reduce(into: 0) {
			$0 += $1.trajectoryCount
		}
		precondition(
			completedTrajectories == trajectoryCount,
			"The QSD correlation reduction omitted one or more trajectories"
		)

		let inverseTrajectoryCount = 1 / Double(trajectoryCount)
		var propagationSummary = PropagationRunSummary(
			finalTime: propagation.timeSpan.end,
			endReason: .reachedEndTime
		)
		ensembleSums.withLock { sums in
			for index in outputTimes.indices {
				guard propagationSummary.endReason == .reachedEndTime else {
					break
				}
				let value = inverseTrajectoryCount * sums[index]
				if observer(outputTimes[index], value) == .stop {
					propagationSummary = PropagationRunSummary(
						finalTime: outputTimes[index],
						endReason: .stoppedByObserver
					)
				}
			}
		}

		return TrajectoryRunSummary(
			trajectoryIDs: execution.trajectoryIDs,
			masterSeed: masterSeed,
			propagation: propagationSummary
		)
	}
}

extension CPUQSDEngine {
    @inlinable
	internal func solveCorrelationTrajectory<Hamiltonian>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration,
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		observing observer: (Double, Complex<Double>) -> Void
	) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		let insertionTime = request.insertionTime
		let dimension = problem.system.dimension

		var state = CorrelationState(guide: problem.initialState)
		try Self.prepareCorrelationState(
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
		outputCursor.discardTimes(before: insertionTime)

		var insertionApplied = false
		if insertionTime == start {
			try _applyCorrelationInsertion(
				request.insertion,
				at: insertionTime,
				dimension: dimension,
				guide: state.guide,
				companion: &state.companion,
				operatorStorage: &operatorStorage
			)
			insertionApplied = true
			while let outputTime = outputCursor.nextTime(through: insertionTime) {
				observer(
					outputTime,
					try Self.correlationSample(
						request: request,
						at: outputTime,
						dimension: dimension,
						state: state,
						equationType: configuration.equationType,
						operatorStorage: &operatorStorage,
						actionStorage: &actionStorage
					)
				)
			}
		}

		guard start < end else {
			return PropagationRunSummary(
				finalTime: end,
				endReason: .reachedEndTime
			)
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
			dt: Swift.min(
				propagation.integration.maximumStepSize,
				end - start
			),
			rhs: rhs,
			drift0: CorrelationState(dimension: dimension),
			drift1: CorrelationState(dimension: dimension),
			noise0: CorrelationState(dimension: dimension),
			noise1: CorrelationState(dimension: dimension),
			temporary: CorrelationState(dimension: dimension),
			noises: noises
		)

		while solver.t < end {
			let stepLimit: Double
			if insertionApplied {
				stepLimit = Swift.min(
					outputCursor.nextRequiredStepBoundary ?? end,
					end
				)
			} else {
				stepLimit = insertionTime
			}
			precondition(
				stepLimit > solver.t,
				"The QSD correlation step boundary must advance time"
			)

			let step = solver.step(y: &state, upTo: stepLimit)
			try Self.prepareCorrelationState(
				&state,
				equationType: configuration.equationType,
				at: step.endTime
			)
			if case .nonLinearNormalized = configuration.equationType {
				solver.stateDidChange()
			}

			if !insertionApplied && step.endTime == insertionTime {
				try _applyCorrelationInsertion(
					request.insertion,
					at: insertionTime,
					dimension: dimension,
					guide: state.guide,
					companion: &state.companion,
					operatorStorage: &operatorStorage
				)
				insertionApplied = true
				solver.stateDidChange()
				while let outputTime = outputCursor.nextTime(
					through: insertionTime
				) {
					observer(
						outputTime,
						try Self.correlationSample(
							request: request,
							at: outputTime,
							dimension: dimension,
							state: state,
							equationType: configuration.equationType,
							operatorStorage: &operatorStorage,
							actionStorage: &actionStorage
						)
					)
				}
				continue
			}

			while let outputTime = outputCursor.nextTime(through: step.endTime) {
				precondition(
					outputTime == step.endTime,
					"A stochastic correlation output must coincide with a step boundary"
				)
				observer(
					outputTime,
					try Self.correlationSample(
						request: request,
						at: outputTime,
						dimension: dimension,
						state: state,
						equationType: configuration.equationType,
						operatorStorage: &operatorStorage,
						actionStorage: &actionStorage
					)
				)
			}
		}

		return PropagationRunSummary(
			finalTime: end,
			endReason: .reachedEndTime
		)
	}


    @inlinable
    @inline(always)
	internal static func prepareCorrelationState(
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
			state.companion.divideBLAS(by: norm)
		}
	}

    @inlinable
    @inline(always)
	internal static func correlationSample(
		request: TwoTimeCorrelationRequest,
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
		return try _correlationSample(
			request: request,
			at: time,
			dimension: dimension,
			guide: state.guide,
			companion: state.companion,
			normalization: normalization,
			operatorStorage: &operatorStorage,
			actionStorage: &actionStorage
		)
	}
}

extension QSD {
    @inlinable
    @inline(always)
	@discardableResult
	public static func solveTwoTimeCorrelation<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: QSD.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
		let engine = CPUQSDEngine()
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
