// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("Solver progress integration", .serialized)
struct ProgressIntegrationTests {
	private var problem: PureStateProblem {
		hopsProblem(markovian: [
			.init(rate: 0.4, collapseOperator: hopsMatrix([0, 1, 0, 0]))
		])
	}
	private var configuration: HOPS.Configuration {
		hopsConfiguration(
			5, model: noiseTestModel(0), operators: [hopsMatrix([1, 0, 0, -1])],
			depth: 2)
	}
	private func options(_ progress: ProgressReporting = .none) -> PropagationOptions<
		IntegrationOptions
	> {
		var result = hopsPropagation(
			end: 0.04, maximumStep: 0.001, output: .times([0.02, 0.04]))
		result.progress = progress
		return result
	}

	@Test("GKSL reports accepted time steps independently of its output schedule")
	func deterministicSteps() throws {
		func run(_ progress: ProgressReporting) throws -> [Complex<Double>] {
			var propagation = options(progress)
			propagation.output = .final
			var result: [Complex<Double>] = []
			var callbacks = 0
			try GKSL.solve(problem: problem, propagation: propagation) { _, rho in
				callbacks += 1
				result = (0..<2).flatMap { row in (0..<2).map { rho[row, $0] } }
				return .proceed
			}
			#expect(callbacks == 1)
			return result
		}
		let capture = ProgressCapture()
		let value = try run(capture.reporting(style: .bar, eta: true))
		#expect(value == (try run(.none)))
		#expect(capture.percentages.first == 0 && capture.percentages.last == 100)
		#expect(capture.percentages.count > 10 && capture.percentages.count <= 101)
		#expect(capture.newlineCount == 1)
	}

	@Test("GKSL early stops use the dense-output stop time and close the line")
	func deterministicStop() throws {
		let capture = ProgressCapture()
		var propagation = hopsPropagation(
			end: 1, maximumStep: 1, output: .times([0.25, 1]))
		propagation.progress = capture.reporting()
		let result = try GKSL.solve(
			problem: hopsProblem(hopsMatrix([0, 0, 0, 0])), propagation: propagation
		) { _, _ in .stop }
		#expect(result.finalTime == 0.25 && result.endReason == .stoppedByObserver)
		#expect(capture.percentages == [0, 25])
		#expect(capture.newlineCount == 1)
	}

	@Test("Zero-duration solves and reused options each have their own progress lifecycle")
	func zeroDurationAndReuse() throws {
		let capture = ProgressCapture()
		var propagation = hopsPropagation(end: -1, start: -1, output: .times([-1]))
		propagation.progress = capture.reporting(eta: true)
		for _ in 0..<2 {
			try GKSL.solve(problem: problem, propagation: propagation) { _, _ in
				.proceed
			}
		}
		#expect(capture.percentages == [0, 100, 0, 100])
		#expect(capture.newlineCount == 2)
	}

	@Test(
		"Deterministic correlations include preparation and every insertion segment",
		arguments: [false, true])
	func deterministicCorrelations(multiTime: Bool) throws {
		let capture = ProgressCapture()
		let identity = TimeDependentOperator.constant(hopsMatrix([1, 0, 0, 1]))
		let observable = TimeDependentOperator.constant(hopsMatrix([0, 1, 1, 0]))
		func run(_ progress: ProgressReporting) throws -> [Complex<Double>] {
			var propagation = options(progress)
			propagation.timeSpan.start = -0.02
			var result: [Complex<Double>] = []
			let observer: (Double, Complex<Double>) -> PropagationControl = {
				_, value in
				// Preparation has progressed even though there were no outputs then.
				if progress.display != nil {
					#expect((capture.percentages.last ?? 0) > 0)
				}
				result.append(value)
				return .proceed
			}
			if multiTime {
				try GKSL.solveMultiTimeOrderedCorrelation(
					problem: problem,
					request: .init(
						insertions: [
							.init(
								time: -0.01,
								insertion: .left(identity)),
							.init(
								time: 0.01,
								insertion: .right(identity)),
						], observable: observable),
					propagation: propagation, observing: observer)
			} else {
				try GKSL.solveTwoTimeCorrelation(
					problem: problem,
					request: .init(
						insertionTime: 0.01, insertion: .left(identity),
						observable: observable),
					propagation: propagation, observing: observer)
			}
			return result
		}
		#expect(try run(capture.reporting()) == run(.none))
		#expect(capture.percentages.first == 0 && capture.percentages.last == 100)
		#expect(capture.percentages.count > 10 && capture.newlineCount == 1)
	}

	private func trajectories(
		method: Int, kind: Int, propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution, invalidState: Bool = false
	) throws -> [Complex<Double>] {
		let problem = invalidState ? hopsProblem(initial: [0, 0]) : self.problem
		let samples = Mutex<[(UInt64, Double, [Complex<Double>])]>([])
		let densityObserver: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void = {
			time, value in
			let copy = (0..<2).flatMap { row in (0..<2).map { value[row, $0] } }
			samples.withLock { $0.append((0, time, copy)) }
		}
		let stateObserver:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void =
				{ id, time, value in
					let copy = (0..<value.count).map { value[$0] }
					samples.withLock { $0.append((id, time, copy)) }
				}
		let correlationObserver: (Double, Complex<Double>) -> PropagationControl = {
			time, value in
			samples.withLock { $0.append((0, time, [value])) }
			return .proceed
		}
		let identity = TimeDependentOperator.constant(hopsMatrix([1, 0, 0, 1]))
		let observable = TimeDependentOperator.constant(hopsMatrix([0, 1, 1, 0]))
		let two = TwoTimeCorrelationRequest(
			insertionTime: 0.01, insertion: .left(identity), observable: observable)
		let multi = MultiTimeOrderedCorrelationRequest(
			insertions: [.init(time: 0.01, insertion: .left(identity))],
			observable: observable)
		let summary: TrajectoryRunSummary
		switch (method, kind) {
		case (0, 0):
			summary = try HOPS.solveEnsemble(
				problem: problem, configuration: configuration,
				propagation: propagation,
				execution: execution, densityObserver
			).summary
		case (1, 0):
			summary = try QSD.solveEnsemble(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, densityObserver)
		case (2, 0):
			summary = try MCWF.solveEnsemble(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, densityObserver)
		case (0, 1):
			summary = try HOPS.solveTrajectories(
				problem: problem, configuration: configuration,
				propagation: propagation,
				execution: execution, stateObserver
			).summary
		case (1, 1):
			summary = try QSD.solveTrajectories(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, stateObserver)
		case (2, 1):
			summary = try MCWF.solveTrajectories(
				problem: problem, configuration: .init(), propagation: propagation,
				execution: execution, stateObserver)
		case (0, 2):
			summary = try HOPS.solveTwoTimeCorrelation(
				problem: problem, configuration: configuration, request: two,
				propagation: propagation, execution: execution,
				observing: correlationObserver
			).summary
		case (1, 2):
			summary = try QSD.solveTwoTimeCorrelation(
				problem: problem, configuration: .init(), request: two,
				propagation: propagation, execution: execution,
				observing: correlationObserver)
		case (2, 2):
			summary = try MCWF.solveTwoTimeCorrelation(
				problem: problem, configuration: .init(), request: two,
				propagation: propagation, execution: execution,
				observing: correlationObserver)
		case (0, _):
			summary = try HOPS.solveMultiTimeOrderedCorrelation(
				problem: problem, configuration: configuration, request: multi,
				propagation: propagation, execution: execution,
				observing: correlationObserver
			).summary
		case (1, _):
			summary = try QSD.solveMultiTimeOrderedCorrelation(
				problem: problem, configuration: .init(), request: multi,
				propagation: propagation, execution: execution,
				observing: correlationObserver)
		default:
			summary = try MCWF.solveMultiTimeOrderedCorrelation(
				problem: problem, configuration: .init(), request: multi,
				propagation: propagation, execution: execution,
				observing: correlationObserver)
		}
		#expect(
			summary.masterSeed == 91 && summary.trajectoryIDs == execution.trajectoryIDs
		)
		#expect(summary.ensembleSampling == execution.ensembleSampling)
		return samples.withLock { items in
			items.sorted { a, b in a.0 != b.0 ? a.0 < b.0 : a.1 < b.1 }.flatMap { $0.2 }
		}
	}

	@Test(
		"All ensemble and correlation APIs count global trajectories, independently of output samples",
		arguments: 0..<3, 0..<4)
	func trajectoryCompletions(method: Int, kind: Int) throws {
		for parallel in [false, true] {
			let execution = TrajectoryExecution(
				trajectories: 5, startingAt: 7, seed: 91,
				parallelism: parallel ? .maximumConcurrentTasks(3) : .serial,
				batchSize: 2, ensembleSampling: .antithetic)
			let capture = ProgressCapture()
			let reported = try trajectories(
				method: method, kind: kind,
				propagation: options(capture.reporting()), execution: execution)
			let silent = try trajectories(
				method: method, kind: kind, propagation: options(),
				execution: execution)
			expectHOPSClose(reported, silent, tolerance: 3e-13)
			let percentages = capture.percentages
			if !parallel { #expect(percentages == [0, 20, 40, 60, 80, 100]) }
			#expect(
				percentages.first == 0 && percentages.last == 100
					&& percentages.count <= 6)
			#expect(zip(percentages, percentages.dropFirst()).allSatisfy { $0 < $1 })
			#expect(capture.newlineCount == 1)
		}
	}

	@Test(
		"Worker errors propagate and close the line without completing failed trajectories",
		arguments: 0..<3, 0..<4)
	func trajectoryErrors(method: Int, kind: Int) throws {
		let capture = ProgressCapture()
		let execution = TrajectoryExecution(
			trajectories: 5, seed: 91, parallelism: .maximumConcurrentTasks(3))
		var failed = false
		do {
			_ = try trajectories(
				method: method, kind: kind,
				propagation: options(capture.reporting()),
				execution: execution, invalidState: true)
		} catch { failed = true }
		#expect(failed)
		#expect(capture.percentages == [0])
		#expect(capture.newlineCount == 1)
	}

	@Test(
		"Single trajectories report one completion or retain zero when stopped",
		arguments: 0..<3, [false, true])
	func singleTrajectories(method: Int, stopped: Bool) throws {
		for callerOwnedRNG in [false, true] {
			let capture = ProgressCapture()
			let propagation = options(capture.reporting())
			var rng = Philox4x64(seed: 17)
			let observer:
				(Double, borrowing UniqueVector<Complex<Double>>) ->
					PropagationControl = { _, _ in
						#expect(capture.percentages == [0])
						return stopped ? .stop : .proceed
					}
			if method == 0 {
				if callerOwnedRNG {
					try HOPS.solveTrajectory(
						problem: problem, configuration: configuration,
						propagation: propagation, rng: &rng,
						observing: observer)
				} else {
					try HOPS.solveTrajectory(
						problem: problem, configuration: configuration,
						propagation: propagation,
						seed: 17, trajectoryID: 4,
						ensembleSampling: .antithetic, observing: observer)
				}
			} else if method == 1 {
				if callerOwnedRNG {
					try QSD.solveTrajectory(
						problem: problem, configuration: .init(),
						propagation: propagation, rng: &rng,
						observing: observer)
				} else {
					try QSD.solveTrajectory(
						problem: problem, configuration: .init(),
						propagation: propagation,
						seed: 17, trajectoryID: 4,
						ensembleSampling: .antithetic, observing: observer)
				}
			} else {
				if callerOwnedRNG {
					try MCWF.solveTrajectory(
						problem: problem, propagation: propagation,
						rng: &rng, observing: observer)
				} else {
					try MCWF.solveTrajectory(
						problem: problem, propagation: propagation,
						seed: 17, trajectoryID: 4,
						ensembleSampling: .antithetic, observing: observer)
				}
			}
			#expect(capture.percentages == (stopped ? [0] : [0, 100]))
			#expect(capture.newlineCount == 1)
		}
	}

	@Test(
		"HOPS hierarchy and live-noise overloads share the single-trajectory lifecycle",
		arguments: 0..<3)
	func hierarchyAndNoise(kind: Int) throws {
		let capture = ProgressCapture()
		let propagation = options(capture.reporting())
		if kind == 0 {
			try HOPS.solveWithHierarchy(
				problem: problem, configuration: configuration,
				propagation: propagation,
				seed: 17, trajectoryID: 4
			) { _, _ in #expect(capture.percentages == [0]) }
		} else if kind == 1 {
			try HOPS.solveWithHierarchy(
				problem: problem, configuration: configuration,
				propagation: propagation,
				seed: 17, trajectoryID: 4,
				observingWithNoise: { _, _, _ in
					#expect(capture.percentages == [0])
					return .proceed
				})
		} else {
			try HOPS.solveTrajectory(
				problem: problem, configuration: configuration,
				propagation: propagation,
				seed: 17, trajectoryID: 4,
				observingWithNoise: { _, _, _ in
					#expect(capture.percentages == [0])
					return .proceed
				})
		}
		#expect(capture.percentages == [0, 100] && capture.newlineCount == 1)
	}
}
