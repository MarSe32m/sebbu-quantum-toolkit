// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS correlations", .serialized)
struct HOPSCorrelationTests {
	@Test("Two-time phases, complex insertion weights and both sides", arguments: 0..<6)
	func unitaryPhase(variant: Int) throws {
		let omega = 1.7
		let s = 0.237
		let times = [s, 0.51, 0.8]
		let problem = hopsProblem(hopsMatrix([0, 0, 0, Complex(omega)]), initial: [0, 1])
		let propagation = hopsPropagation(
			end: 0.8, start: -0.3, maximumStep: 0.031,
			tolerance: 1e-11, output: .times([-0.3, 0, s, 0.51, 0.8]))
		let c = Complex(1.4, -0.8)
		for right in [false, true] {
			let request = TwoTimeCorrelationRequest(
				insertionTime: s,
				insertion: right
					? .right(.constant(c * hcR)) : .left(.constant(c * hcL)),
				observable: .constant(right ? hcL : hcR))
			var observed: [Double] = []
			var values: [Complex<Double>] = []
			let summary = try HOPS.solveTwoTimeCorrelation(
				problem: problem, configuration: hcConfiguration(variant),
				request: request, propagation: propagation,
				execution: .init(trajectories: 1, seed: 17, parallelism: .serial)
			) { t, value in
				observed.append(t)
				values.append(value)
				return .proceed
			}
			#expect(observed == times)
			#expect(
				summary.masterSeed == 17
					&& summary.propagation.endReason == .reachedEndTime)
			expectHOPSClose(
				values,
				times.map {
					c
						* Complex(
							length: 1,
							phase: (right ? -1 : 1) * omega * ($0 - s))
				}, tolerance: 2e-10)
		}
	}

	@Test("Every three-event left/right ordering agrees with unitary dyads", arguments: 0..<6)
	func mixedOrdering(variant: Int) throws {
		let problem = hopsProblem()
		let propagation = hopsPropagation(
			end: 0.5, maximumStep: 0.04, tolerance: 1e-11,
			output: .times([0, 0.11, 0.23, 0.37, 0.5]))
		let operators = [hcL, hopsMatrix([1, .i, 0, Complex(0.4)]), hcR]
		for mask in 0..<8 {
			let events = zip([0.11, 0.23, 0.37], operators).enumerated().map {
				i, item in
				TimedCorrelationInsertion(
					time: item.0,
					insertion: mask & (1 << i) == 0
						? .left(.constant(item.1))
						: .right(.constant(item.1)))
			}
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: events,
				observable: .constant(
					hopsMatrix([Complex(0.3), .i, 1, Complex(-0.6)])))
			var expected: [Complex<Double>] = []
			try GKSL.solveMultiTimeOrderedCorrelation(
				problem: problem, request: request,
				propagation: propagation
			) { _, value in
				expected.append(value)
				return .proceed
			}
			expectHOPSClose(
				try hcValues(
					problem: problem, configuration: hcConfiguration(variant),
					request: request, propagation: propagation), expected,
				tolerance: 3e-10)
		}
	}

	@Test("Zero bath reproduces QSD pathwise with dynamic channels", arguments: 0..<6)
	func pathwiseQSD(variant: Int) throws {
		let dynamic = TimeDependentOperator.generatedDense(
			.init { t, output in
				output.copyElements(
					from: hopsMatrix([
						Complex(0.2 * t), Complex(1, 0.3 * t), 0,
						Complex(-0.1),
					]))
			})
		let problem = hopsProblem(markovian: [
			.init(rate: .generated { 0.5 + $0 }, collapseOperator: dynamic),
			.init(rate: 0.13, collapseOperator: hcZ),
		])
		let propagation = hopsPropagation(
			end: 0.4, maximumStep: 0.003, output: .times([0, 0.113, 0.271, 0.4]))
		let execution = TrajectoryExecution(trajectories: 5, seed: 19, parallelism: .serial)
		for mask in 0..<4 {
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: [
					.init(
						time: 0.113,
						insertion: mask & 1 == 0
							? .left(dynamic) : .right(dynamic)),
					.init(
						time: 0.271,
						insertion: mask & 2 == 0
							? .left(.constant(hcX))
							: .right(.constant(hcX))),
				], observable: .constant(hopsMatrix([1, .i, 0, -1])))
			let values = try hcValues(
				problem: problem, configuration: hcConfiguration(variant),
				request: request,
				propagation: propagation, execution: execution)
			var expected: [Complex<Double>] = []
			let equation: QSD.EquationType = [
				.linear, .nonLinear, .nonLinearNormalized,
			][variant / 2]
			try QSD.solveMultiTimeOrderedCorrelation(
				problem: problem, configuration: .init(equationType: equation),
				request: request, propagation: propagation, execution: execution
			) { _, value in
				expected.append(value)
				return .proceed
			}
			expectHOPSClose(values, expected, tolerance: 3e-12)
		}
	}

	@Test(
		"Scalar insertions preserve guide, amplitude and memory", arguments: 0..<6,
		[false, true])
	func scalarInsertions(variant: Int, hybrid: Bool) throws {
		let config = hcConfiguration(variant, colored: true)
		let problem = hopsProblem(
			markovian: hybrid ? [.init(rate: 0.4, collapseOperator: hcL)] : [])
		let c = Complex(0.3, 1.2)
		let b = Complex(-0.4, 0.7)
		let propagation = hopsPropagation(
			end: 0.4, maximumStep: hybrid ? 0.002 : 0.04,
			tolerance: 1e-10, output: .times([0.1, 0.25, 0.4]))
		let prep = try HCT.Preparation(
			problem: problem, configuration: config, propagation: propagation)
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.1, insertion: .left(.constant(c * hcI))),
				.init(time: 0.25, insertion: .right(.constant(b * hcI))),
			], observable: .constant(hcZ))
		let values = try hcPath(
			problem: problem, preparation: prep, request: request,
			propagation: propagation)
		var expected: [Complex<Double>] = []
		try HOPS.solveTrajectory(
			problem: problem, configuration: config, propagation: propagation, seed: 71,
			trajectoryID: 3
		) { t, state in
			if t >= 0.25 {
				let norm = variant < 2 ? 1 : state.normSquared
				expected.append(
					c * b * (state[0].lengthSquared - state[1].lengthSquared)
						/ norm)
			}
			return .proceed
		}
		expectHOPSClose(values, expected, tolerance: hybrid ? 2e-11 : 5e-8)
	}

	@Test("Zero insertions remain valid", arguments: 0..<6)
	func zeroCompanion(variant: Int) throws {
		let problem = hopsProblem(markovian: [.init(rate: 0.5, collapseOperator: hcL)])
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.073, insertion: .left(.constant(hcZero))),
				.init(time: 0.14, insertion: .right(.constant(hcX))),
			], observable: .constant(hcI))
		let values = try hcValues(
			problem: problem, configuration: hcConfiguration(variant, colored: true),
			request: request,
			propagation: hopsPropagation(
				end: 0.2, maximumStep: 0.005, output: .times([0.14, 0.2])))
		#expect(values == [.zero, .zero])
	}

	@Test("Dynamic operators use absolute event times")
	func dynamicOperators() throws {
		let b = TimeDependentOperator.generatedDense(
			.init { t, output in output.copyElements(from: Complex(t, 1 - t) * hcL) })
		let a = TimeDependentOperator.generatedDense(
			.init { t, output in
				output.copyElements(from: Complex(1 + t, 0.2 * t) * hcR)
			})
		let times = [-0.13, 0, 0.21]
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [.init(time: -0.13, insertion: .left(b))], observable: a)
		expectHOPSClose(
			try hcValues(
				problem: hopsProblem(hcZero, initial: [0, 1]),
				configuration: hcConfiguration(5),
				request: request,
				propagation: hopsPropagation(
					end: 0.21, start: -0.4, output: .times(times))),
			times.map { Complex(-0.13, 1.13) * Complex(1 + $0, 0.2 * $0) })
	}

	@Test("Insertion at start, end and zero-duration span")
	func endpoints() throws {
		for (start, end, s) in [(0.0, 0.3, 0.0), (0.0, 0.3, 0.3), (-0.2, -0.2, -0.2)] {
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: [.init(time: s, insertion: .left(.constant(hcL)))],
				observable: .constant(hcR))
			#expect(
				try hcValues(
					problem: hopsProblem(hcZero, initial: [0, 1]),
					configuration: hcConfiguration(5), request: request,
					propagation: hopsPropagation(end: end, start: start)) == [
						.one
					])
		}
	}

	@Test("Schedules, stopping and execution metadata are preserved")
	func outputAndSummary() throws {
		let config = hcConfiguration(5, colored: true)
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.13, insertion: .left(.constant(hcX))),
				.init(time: 0.27, insertion: .right(.constant(hcZ))),
			], observable: .constant(hcR))
		let execution = TrajectoryExecution(
			trajectoryIDs: 9..<12, randomness: .seeded(0xCAFE), parallelism: .serial)
		for stop in [false, true] {
			var times: [Double] = []
			let summary = try HOPS.solveMultiTimeOrderedCorrelation(
				problem: hopsProblem(), configuration: config, request: request,
				propagation: hopsPropagation(end: 0.5, output: .uniform(step: 0.1)),
				execution: execution
			) { t, _ in
				times.append(t)
				return stop ? .stop : .proceed
			}
			#expect(times.count == (stop ? 1 : 3) && abs(times[0] - 0.3) < 1e-15)
			#expect(summary.trajectoryIDs == 9..<12 && summary.masterSeed == 0xCAFE)
			#expect(
				summary.propagation.endReason
					== (stop ? .stoppedByObserver : .reachedEndTime))
			#expect(summary.propagation.finalTime == times.last!)
		}
		#expect(
			try hcValues(
				problem: hopsProblem(), configuration: config, request: request,
				propagation: hopsPropagation(end: 0.5, output: .times([0, 0.1])),
				execution: execution
			).isEmpty)
	}

	@Test("Parallel and partitioned ensembles retain the same streams")
	func executionReproducibility() throws {
		let problem = hopsProblem(markovian: [.init(rate: 0.3, collapseOperator: hcL)])
		let config = hcConfiguration(5, colored: true)
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.053, insertion: .left(.constant(hcX))),
				.init(time: 0.127, insertion: .right(.constant(hcL))),
			], observable: .constant(hcZ))
		let propagation = hopsPropagation(
			end: 0.2, maximumStep: 0.003, output: .times([0.127, 0.2]))
		func run(_ ids: Range<UInt64>, _ parallelism: TrajectoryParallelism) throws
			-> [Complex<Double>]
		{
			try hcValues(
				problem: problem, configuration: config, request: request,
				propagation: propagation,
				execution: .init(
					trajectoryIDs: ids, randomness: .seeded(71),
					parallelism: parallelism, batchSize: 3))
		}
		let serial = try run(7..<23, .serial)
		expectHOPSClose(
			try run(7..<23, .maximumConcurrentTasks(4)), serial, tolerance: 2e-14)
		#expect(try run(7..<23, .serial) == serial)
		let a = try run(7..<15, .serial)
		let b = try run(15..<23, .serial)
		expectHOPSClose(a.indices.map { (a[$0] + b[$0]) / 2.0 }, serial, tolerance: 2e-14)
	}

	@Test("Equal-time correlations use the prepared colored guide", arguments: 0..<6)
	func coloredEqualTime(variant: Int) throws {
		let config = hcConfiguration(variant, colored: true)
		let problem = hopsProblem(markovian: [.init(rate: 0.3, collapseOperator: hcL)])
		let propagation = hopsPropagation(end: 0.173, maximumStep: 0.003)
		let guide = try hopsFinalState(
			problem: problem, configuration: config, propagation: propagation)
		let a = hopsMatrix([1, .i, Complex(0.4), -1])
		let b = hopsMatrix([Complex(0.3), 1, .i, Complex(-0.2)])
		for right in [false, true] {
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: [
					.init(
						time: 0.173,
						insertion: right
							? .right(.constant(b)) : .left(.constant(b))
					)
				], observable: .constant(a))
			let values = try hcValues(
				problem: problem, configuration: config, request: request,
				propagation: propagation,
				execution: .init(
					trajectories: 1, startingAt: 3, seed: 71,
					parallelism: .serial))
			let action = (right ? b.dot(a) : a.dot(b)).dot(Vector(guide))
			let norm = variant < 2 ? 1.0 : guide.reduce(0.0) { $0 + $1.lengthSquared }
			expectHOPSClose(
				values,
				[
					(guide[0].conjugate * action[0] + guide[1].conjugate
						* action[1]) / norm
				], tolerance: 2e-12)
		}
	}

	@Test("Negative preparation times retain the same colored path", arguments: [0, 5])
	func translatedPreparation(variant: Int) throws {
		var config = hcConfiguration(variant, colored: true)
		config.noiseStepSize = 0.015625
		func run(_ offset: Double) throws -> [Complex<Double>] {
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: [
					.init(
						time: offset + 0.0625,
						insertion: .left(.constant(hcX))),
					.init(
						time: offset + 0.25,
						insertion: .right(.constant(hcL))),
				], observable: .constant(hcZ))
			return try hcValues(
				problem: hopsProblem(), configuration: config, request: request,
				propagation: hopsPropagation(
					end: offset + 0.375, start: offset, maximumStep: 0.03125,
					tolerance: 1e-10,
					output: .times([offset + 0.25, offset + 0.375])))
		}
		expectHOPSClose(try run(0), try run(-0.5), tolerance: 2e-10)
	}
}
