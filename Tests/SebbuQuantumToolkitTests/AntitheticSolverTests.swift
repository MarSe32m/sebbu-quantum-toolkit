// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("Antithetic solver integration", .serialized)
struct AntitheticSolverTests {
	private var configuration: HOPS.Configuration {
		hopsConfiguration(
			5, model: noiseTestModel(0), operators: [hopsMatrix([1, 0, 0, -1])],
			depth: 2)
	}
	private var problem: PureStateProblem {
		hopsProblem(markovian: [.init(rate: 2, collapseOperator: hopsMatrix([0, 1, 0, 0]))])
	}
	private var propagation: PropagationOptions<IntegrationOptions> {
		hopsPropagation(end: 0.08, maximumStep: 0.004)
	}

	private func finalState(method: Int, id: UInt64, sampling: EnsembleSampling?) throws
		-> [Complex<Double>]
	{
		var state: [Complex<Double>] = []
		let observer:
			(Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl = {
				_, value in
				state = (0..<value.count).map { value[$0] }
				return .proceed
			}
		switch (method, sampling) {
		case (0, .some(let mode)):
			try HOPS.solveTrajectory(
				problem: problem, configuration: configuration,
				propagation: propagation,
				seed: 88, trajectoryID: id, ensembleSampling: mode,
				observing: observer)
		case (0, .none):
			try HOPS.CPUEngine().solveTrajectory(
				problem: problem, configuration: configuration,
				propagation: propagation,
				seed: 88, trajectoryID: id, observing: observer)
		case (1, .some(let mode)):
			try QSD.solveTrajectory(
				problem: problem, configuration: .init(), propagation: propagation,
				seed: 88, trajectoryID: id, ensembleSampling: mode,
				observing: observer)
		case (1, .none):
			try QSD.CPUEngine().solveTrajectory(
				problem: problem, configuration: .init(), propagation: propagation,
				seed: 88, trajectoryID: id, observing: observer)
		case (_, .some(let mode)):
			try MCWF.solveTrajectory(
				problem: problem, configuration: .init(), propagation: propagation,
				seed: 88, trajectoryID: id, ensembleSampling: mode,
				observing: observer)
		case (_, .none):
			try MCWF.CPUEngine().solveTrajectory(
				problem: problem, configuration: .init(), propagation: propagation,
				seed: 88, trajectoryID: id, observing: observer)
		}
		return state
	}

	private func ensemble(method: Int, execution: TrajectoryExecution) throws -> [Complex<
		Double
	>] {
		var density: [Complex<Double>] = []
		let observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void = {
			_, value in
			density = (0..<2).flatMap { row in (0..<2).map { value[row, $0] } }
		}
		let summary: TrajectoryRunSummary
		switch method {
		case 0:
			let result = try HOPS.solveEnsemble(
				problem: problem, configuration: configuration,
				propagation: propagation,
				execution: execution, observer)
			#expect(result.bathNoise.ensembleSampling == execution.ensembleSampling)
			summary = result.summary
		case 1:
			summary = try QSD.solveEnsemble(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, observer)
		default:
			summary = try MCWF.solveEnsemble(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, observer)
		}
		#expect(summary.ensembleSampling == execution.ensembleSampling)
		#expect(
			summary.masterSeed == 88 && summary.trajectoryIDs == execution.trajectoryIDs
		)
		return density
	}

	@Test(
		"Odd ensembles match explicit trajectories, across scheduling and replay",
		arguments: 0..<3, [EnsembleSampling.independent, .antithetic])
	func ensembles(method: Int, sampling: EnsembleSampling) throws {
		var execution = TrajectoryExecution(
			trajectories: 5, seed: 88, parallelism: .serial, ensembleSampling: sampling)
		var states: [UInt64: [Complex<Double>]] = [:]
		var expected = Array(repeating: Complex<Double>.zero, count: 4)
		for id in execution.trajectoryIDs {
			let state = try finalState(method: method, id: id, sampling: sampling)
			states[id] = state
			let projector = hopsProjector(state)
			for i in expected.indices { expected[i] += projector[i] / Double(5) }
			if sampling == .independent {
				#expect(
					state
						== (try finalState(
							method: method, id: id, sampling: nil)))
			} else if id & 1 == 0 {
				#expect(
					state
						== (try finalState(
							method: method, id: id / 2,
							sampling: .independent)))
			}
		}
		let serial = try ensemble(method: method, execution: execution)
		expectHOPSClose(serial, expected, tolerance: 3e-13)
		#expect(serial == (try ensemble(method: method, execution: execution)))
		execution.parallelism = .maximumConcurrentTasks(3)
		execution.batchSize = 3
		expectHOPSClose(
			try ensemble(method: method, execution: execution), serial, tolerance: 3e-13
		)

		let observed = Mutex<[UInt64: [Complex<Double>]]>([:])
		let observer:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void =
				{ id, _, value in
					let copy = (0..<value.count).map { value[$0] }
					observed.withLock { $0[id] = copy }
				}
		let summary: TrajectoryRunSummary
		switch method {
		case 0:
			let result = try HOPS.solveTrajectories(
				problem: problem, configuration: configuration,
				propagation: propagation,
				execution: execution, observer)
			#expect(result.bathNoise.ensembleSampling == sampling)
			summary = result.summary
		case 1:
			summary = try QSD.solveTrajectories(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, observer)
		default:
			summary = try MCWF.solveTrajectories(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, observer)
		}
		#expect(
			summary.ensembleSampling == sampling
				&& summary.trajectoryIDs == execution.trajectoryIDs)
        observed.withLock { _states in
            #expect(_states.count == states.count)
            for key in _states.keys {
                guard let a = _states[key], let b = states[key] else {
                    #expect(Bool(false))
                    return
                }
                #expect(a.count == b.count)
                for (_a, _b) in zip(a, b) {
                    #expect(_a.isApproximatelyEqual(to: _b))
                }
            }
        }
	}

	@Test("White-noise draws negate in HOPS, QSD and the correlation RHS")
	func whiteNoise() throws {
		let p = problem
		let preparation = try HOPS.CPUEngine.Preparation(
			problem: p, configuration: configuration, propagation: propagation)
		var ha = HOPS.CPUEngine.RightHandSide(
			hamiltonian: p.system.hamiltonian, preparation: preparation,
			seed: 91, trajectoryID: 6, ensembleSampling: .antithetic)
		var hb = HOPS.CPUEngine.RightHandSide(
			hamiltonian: p.system.hamiltonian, preparation: preparation,
			seed: 91, trajectoryID: 7, ensembleSampling: .antithetic)
		var qa = QSD.CPUEngine.QSDRightHandSide(
			p, equationType: .nonLinearNormalized,
			seed: 91, trajectoryID: 6, ensembleSampling: .antithetic)
		var qb = QSD.CPUEngine.QSDRightHandSide(
			p, equationType: .nonLinearNormalized,
			seed: 91, trajectoryID: 7, ensembleSampling: .antithetic)
		var ca = QSD.CPUEngine.CorrelationRightHandSide(
			p, equationType: .nonLinearNormalized,
			seed: 91, trajectoryID: 6, ensembleSampling: .antithetic)
		var cb = QSD.CPUEngine.CorrelationRightHandSide(
			p, equationType: .nonLinearNormalized,
			seed: 91, trajectoryID: 7, ensembleSampling: .antithetic)
		var a = UniqueVector<Complex<Double>>.zero(1)
		var b = UniqueVector<Complex<Double>>.zero(1)
		for _ in 0..<128 {
			ha.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &a.mutableSpan)
			hb.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &b.mutableSpan)
			let reference = a[0]
			#expect(a[0] == -b[0])
			qa.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &a.mutableSpan)
			qb.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &b.mutableSpan)
			#expect(a[0] == reference && a[0] == -b[0])
			ca.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &a.mutableSpan)
			cb.sampleNormalizedNoises(t: 0, stepSize: 0.004, into: &b.mutableSpan)
			#expect(a[0] == reference && a[0] == -b[0])
		}
	}

	@Test(
		"Complementary uniforms produce one decay per pair at the half-life",
		arguments: [false, true])
	func mcwfComplementaryDecay(waitingTime: Bool) throws {
		let end = Double.log(2)
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, 0]), initial: [0, 1],
			markovian: [.init(rate: 1, collapseOperator: hopsMatrix([0, 1, 0, 0]))])
		let algorithm: MCWF.JumpAlgorithm =
			waitingTime
			? .waitingTime(eventTolerance: 1e-12, maximumEventIterations: 64)
			: .discreteTime
		var excited = 0.0
		try MCWF.solveEnsemble(
			problem: problem, configuration: .init(jumpAlgorithm: algorithm),
			propagation: hopsPropagation(end: end, maximumStep: end),
			execution: .init(
				trajectories: 32, seed: 11, parallelism: .serial,
				ensembleSampling: .antithetic)
		) { _, rho in
			excited = rho[1, 1].real
		}
		#expect(abs(excited - 0.5) < 1e-12)
	}

	private func correlation(method: Int, execution: TrajectoryExecution, twoTime: Bool) throws
		-> Complex<Double>
	{
		let identity = TimeDependentOperator.constant(hopsMatrix([1, 0, 0, 1]))
		let observable = TimeDependentOperator.constant(hopsMatrix([0, 1, 1, 0]))
		let two = TwoTimeCorrelationRequest(
			insertionTime: 0.02, insertion: .left(identity), observable: observable)
		let multi = MultiTimeOrderedCorrelationRequest(
			insertions: [.init(time: 0.02, insertion: .left(identity))],
			observable: observable)
		var value = Complex<Double>.zero
		let observer: (Double, Complex<Double>) -> PropagationControl = { _, result in
			value = result
			return .proceed
		}
		let summary: TrajectoryRunSummary
		switch method {
		case 0:
			let result =
				try twoTime
				? HOPS.solveTwoTimeCorrelation(
					problem: problem, configuration: configuration,
					request: two,
					propagation: propagation, execution: execution,
					observing: observer)
				: HOPS.solveMultiTimeOrderedCorrelation(
					problem: problem, configuration: configuration,
					request: multi,
					propagation: propagation, execution: execution,
					observing: observer)
			#expect(result.bathNoise.ensembleSampling == .antithetic)
			#expect(
				result.bathNoise.path(for: execution.trajectoryIDs.lowerBound)
					.ensembleSampling == .antithetic)
			summary = result.summary
		case 1:
			summary =
				try twoTime
				? QSD.solveTwoTimeCorrelation(
					problem: problem, configuration: .init(), request: two,
					propagation: propagation, execution: execution,
					observing: observer)
				: QSD.solveMultiTimeOrderedCorrelation(
					problem: problem, configuration: .init(), request: multi,
					propagation: propagation, execution: execution,
					observing: observer)
		default:
			summary =
				try twoTime
				? MCWF.solveTwoTimeCorrelation(
					problem: problem, configuration: .init(), request: two,
					propagation: propagation, execution: execution,
					observing: observer)
				: MCWF.solveMultiTimeOrderedCorrelation(
					problem: problem, configuration: .init(), request: multi,
					propagation: propagation, execution: execution,
					observing: observer)
		}
		#expect(
			summary.ensembleSampling == .antithetic
				&& summary.trajectoryIDs == execution.trajectoryIDs)
		return value
	}

	@Test(
		"Correlation entry points replay global IDs, including a partial pair",
		arguments: 0..<3)
	func correlations(method: Int) throws {
		let execution = TrajectoryExecution(
			trajectories: 5, startingAt: 5, seed: 88, parallelism: .serial,
			ensembleSampling: .antithetic)
		let result = try correlation(method: method, execution: execution, twoTime: false)
		#expect(
			result
				== (try correlation(
					method: method, execution: execution, twoTime: false)))
		#expect(
			result
				== (try correlation(
					method: method, execution: execution, twoTime: true)))
		var expected = Complex<Double>.zero
		for id in execution.trajectoryIDs {
			let single = TrajectoryExecution(
				trajectories: 1, startingAt: id, seed: 88, parallelism: .serial,
				ensembleSampling: .antithetic)
			expected +=
				try correlation(method: method, execution: single, twoTime: false)
				/ Double(5)
		}
		#expect((result - expected).length < 3e-13)
	}
}
