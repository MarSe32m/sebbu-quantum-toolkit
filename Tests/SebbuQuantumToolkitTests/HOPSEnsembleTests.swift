// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS ensembles")
struct HOPSEnsembleTests {
	@Test("Serial and parallel ensembles equal explicit trajectory averages", arguments: 0..<6)
	func explicitAverage(variant: Int) throws {
		let config = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2)
		let problem = hopsProblem(markovian: [
			.init(rate: 0.2, collapseOperator: hopsMatrix([0, 1, 0, 0]))
		])
		let propagation = hopsPropagation(end: 0.08, maximumStep: 0.004)
		let ids: Range<UInt64> = 10..<26
		var expected = [Complex<Double>](repeating: .zero, count: 4)
		for id in ids {
			let state = try hopsFinalState(
				problem: problem, configuration: config, propagation: propagation,
				seed: 88, id: id)
			let rho = hopsProjector(state, normalized: variant / 2 != 0)
			for i in expected.indices { expected[i] += rho[i] / Double(ids.count) }
		}
		for parallelism: TrajectoryParallelism in [.serial, .maximumConcurrentTasks(3)] {
			var actual: [Complex<Double>] = []
			let summary = try HOPS.solveEnsemble(
				problem: problem, configuration: config, propagation: propagation,
				execution: .init(
					trajectoryIDs: ids, randomness: .seeded(88),
					parallelism: parallelism, batchSize: 3)
			) { t, rho in
				#expect(t == 0.08)
				actual = (0..<4).map { rho.elements[$0] }
			}
			#expect(summary.masterSeed == 88 && summary.trajectoryIDs == ids)
			expectHOPSClose(actual, expected, tolerance: 3e-13)
		}
	}

	@Test("Parallel trajectory callbacks retain the seed and ID mapping")
	func parallelTrajectories() throws {
		let config = hopsConfiguration(
			5, model: noiseTestModel(0), operators: [hopsMatrix([1, 0, 0, -1])])
		let problem = hopsProblem()
		let propagation = hopsPropagation(end: 0.1)
		let values = Mutex<[UInt64: [Complex<Double>]]>([:])
		let ids = (UInt64.max - 8)..<UInt64.max
		try HOPS.solveTrajectories(
			problem: problem, configuration: config, propagation: propagation,
			execution: .init(
				trajectoryIDs: ids, randomness: .seeded(24),
				parallelism: .maximumConcurrentTasks(3))
		) { id, _, state in
			let copy = [state[0], state[1]]
			values.withLock { $0[id] = copy }
		}
		let results = values.withLock { $0 }
		#expect(results.count == 8)
		for id in ids {
			let expected = try hopsFinalState(
				problem: problem, configuration: config, propagation: propagation,
				seed: 24, id: id)
			expectHOPSClose(results[id]!, expected, tolerance: 0)
		}
	}

	@Test("Ensemble worker failures are thrown without emitting partial results")
	func failureHandling() throws {
		let config = hopsConfiguration(
			2, model: noiseTestModel(0), operators: [hopsMatrix([1, 0, 0, -1])])
		var emitted = false
		#expect(throws: HOPS.CPUEngine.SolverError.invalidStateNorm(time: 0)) {
			try HOPS.solveEnsemble(
				problem: hopsProblem(initial: [0, 0]), configuration: config,
				propagation: hopsPropagation(),
				execution: .init(
					trajectories: 8, seed: 1,
					parallelism: .maximumConcurrentTasks(2))
			) { _, _ in emitted = true }
		}
		#expect(!emitted)
		#expect(throws: TrajectoryEnsembleError.everyAcceptedStepOutputIsNotSupported) {
			try HOPS.solveEnsemble(
				problem: hopsProblem(), configuration: config,
				propagation: hopsPropagation(output: .everyAcceptedStep),
				execution: .init(trajectories: 2, seed: 1)
			) { _, _ in }
		}
	}

	@Test("Empty output schedules still propagate and initial nonlinear dyads are normalized")
	func outputSemantics() throws {
		for variant in 0..<6 {
			let config = hopsConfiguration(
				variant, model: .zero(channelCount: 1),
				operators: [hopsMatrix([1, 0, 0, -1])])
			var outputs = 0
			let summary = try HOPS.solveEnsemble(
				problem: hopsProblem(), configuration: config,
				propagation: hopsPropagation(output: .times([])),
				execution: .init(trajectories: 2, seed: 1, parallelism: .serial)
			) { _, _ in outputs += 1 }
			#expect(outputs == 0 && summary.propagation.finalTime == 0.3)
			try HOPS.solveEnsemble(
				problem: hopsProblem(initial: [2, 0]), configuration: config,
				propagation: hopsPropagation(end: 0),
				execution: .init(trajectories: 2, seed: 1, parallelism: .serial)
			) { _, rho in
				let value = rho[0, 0]
				#expect(value == Complex(variant < 2 ? 4.0 : 1.0))
			}
		}
	}
}
