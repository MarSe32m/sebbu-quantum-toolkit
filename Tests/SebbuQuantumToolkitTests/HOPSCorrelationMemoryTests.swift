// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS correlation memory", .serialized)
struct HOPSCorrelationMemoryTests {
	@Test(
		"The guide alone controls drift, diffusion, shifts and common normalization",
		arguments: 0..<6)
	func guideControlledGenerator(variant: Int) throws {
		let config = hopsConfiguration(
			variant, model: hopsFixtureModel(), operators: hopsFixtureOperators,
			depth: 2)
		let problem = hopsProblem(markovian: [
			.init(rate: 0.6, collapseOperator: hopsFixtureOperators[0])
		])
		let prep = try HCT.Preparation(
			problem: problem, configuration: config, propagation: hopsPropagation())
		let count = config.hierarchy.count
		let size = 2 * count
		var guide = HCT.State(
			dimension: 2, hierarchyCount: count, shiftCount: prep.shiftCount)
		var batch = HCT.State(
			dimension: 2, hierarchyCount: 3 * count, shiftCount: prep.shiftCount)
		var dg = HCT.State(dimension: 2, hierarchyCount: count, shiftCount: prep.shiftCount)
		var db = HCT.State(
			dimension: 2, hierarchyCount: 3 * count, shiftCount: prep.shiftCount)
		let c = Complex(-0.7, 1.2)
		for i in 0..<size {
			guide.amplitudes.elements[i] = Complex(
				0.1 + 0.03 * Double(i), 0.4 - 0.02 * Double(i))
			batch.amplitudes.elements[i] = guide.amplitudes.elements[i]
			batch.amplitudes.elements[size + i] = Complex(
				0.4 - 0.05 * Double(i), -0.3 + 0.02 * Double(i))
			batch.amplitudes.elements[2 * size + i] =
				c * batch.amplitudes.elements[size + i]
		}
		for i in 0..<prep.shiftCount {
			guide.shifts[i] = Complex(0.2 * Double(i + 1), -0.3)
			batch.shifts[i] = guide.shifts[i]
		}
		var singleRHS = HCT.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: prep, seed: 1,
			trajectoryID: 3)
		var batchRHS = HCT.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: prep, seed: 1,
			trajectoryID: 3)
		for i in 0..<singleRHS.physicalNoise.count {
			singleRHS.physicalNoise[i] = Complex(0.3, Double(i) - 0.4)
			batchRHS.physicalNoise[i] = singleRHS.physicalNoise[i]
		}
		singleRHS.evaluateWithCurrentNoise(t: 0.13, y: guide, into: &dg)
		batchRHS.evaluateWithCurrentNoise(t: 0.13, y: batch, into: &db)
		for i in 0..<size {
			#expect(
				(dg.amplitudes.elements[i] - db.amplitudes.elements[i]).length
					< 2e-13)
			#expect(
				(db.amplitudes.elements[2 * size + i] - c
					* db.amplitudes.elements[size + i]).length < 3e-13)
		}
		for i in 0..<prep.shiftCount {
			#expect((dg.shifts[i] - db.shifts[i]).length < 2e-14)
		}
		singleRHS.diffusion(t: 0.13, y: guide, channel: 0, into: &dg)
		batchRHS.diffusion(t: 0.13, y: batch, channel: 0, into: &db)
		for i in 0..<size {
			#expect(
				(dg.amplitudes.elements[i] - db.amplitudes.elements[i]).length
					< 2e-13)
			#expect(
				(db.amplitudes.elements[2 * size + i] - c
					* db.amplitudes.elements[size + i]).length < 3e-13)
		}
		for i in 0..<prep.shiftCount { #expect(db.shifts[i] == .zero) }
		let before = (0..<(3 * size)).map { batch.amplitudes.elements[$0] }
		let norm = guide.rootNormSquared.squareRoot()
		HCT.normalize(&batch)
		for i in before.indices {
			#expect((batch.amplitudes.elements[i] - before[i] / norm).length < 2e-14)
		}
		for i in 0..<prep.shiftCount { #expect(batch.shifts[i] == guide.shifts[i]) }
	}

	@Test("Insertions transform every tier and preserve the guide and shifts")
	func allTierInsertion() throws {
		let b = hopsMatrix([
			Complex(0.2, 0.3), Complex(-0.7, 0.1), Complex(0.6, -0.8),
			Complex(0.4, 0.5),
		])
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.1, insertion: .left(.constant(b))),
				.init(time: 0.2, insertion: .right(.constant(b))),
			], observable: .constant(hcI))
		var workspace = HCT.CorrelationWorkspace(
			request: request, hierarchyCount: 4, dimension: 2)
		#expect(workspace.branchCount == 3)
		var state = HCT.State(dimension: 2, hierarchyCount: 12, shiftCount: 2)
		for i in 0..<8 { state.amplitudes.elements[i] = Complex(0.1 * Double(i + 1), -0.2) }
		state.shifts[0] = Complex(0.3, 0.6)
		state.shifts[1] = Complex(-0.1, 0.2)
		let guide = (0..<8).map { state.amplitudes.elements[$0] }
		workspace.initializeDyad(fromGuide: &state)
		try workspace.insert(request.insertions[0], index: 0, into: &state)
		try workspace.insert(request.insertions[1], index: 1, into: &state)
		for h in 0..<4 {
			let vector = Vector([guide[2 * h], guide[2 * h + 1]])
			let ket = b.dot(vector)
			let bra = b.conjugateTranspose.dot(vector)
			for i in 0..<2 {
				#expect(state.amplitudes.elements[2 * h + i] == guide[2 * h + i])
				#expect(
					(state.amplitudes.elements[8 + 2 * h + i] - ket[i]).length
						< 1e-14)
				#expect(
					(state.amplitudes.elements[16 + 2 * h + i] - bra[i]).length
						< 1e-14)
			}
		}
		#expect(
			state.shifts[0] == Complex(0.3, 0.6)
				&& state.shifts[1] == Complex(-0.1, 0.2))
	}

	@Test(
		"A zero companion root can be repopulated from its retained auxiliaries",
		arguments: 0..<6)
	func zeroRootRetainsMemory(variant: Int) throws {
		let config = hcConfiguration(variant, colored: true, depth: 2)
		let problem = hopsProblem(hcZero, initial: [0, 1])
		let prep = try HCT.Preparation(
			problem: problem, configuration: config, propagation: hopsPropagation())
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(
					time: 0.1,
					insertion: .left(.constant(hopsMatrix([1, 0, 0, 0]))))
			], observable: .constant(hcI))
		let count = config.hierarchy.count
		let offset = 2 * count
		var state = HCT.State(
			dimension: 2, hierarchyCount: 2 * count, shiftCount: prep.shiftCount)
		var dy = HCT.State(
			dimension: 2, hierarchyCount: 2 * count, shiftCount: prep.shiftCount)
		state.amplitudes.elements[1] = .one
		state.amplitudes.elements[2] = Complex(0.3, -0.2)
		var workspace = HCT.CorrelationWorkspace(
			request: request, hierarchyCount: count, dimension: 2)
		workspace.initializeDyad(fromGuide: &state)
		try workspace.insert(request.insertions[0], index: 0, into: &state)
		#expect(
			state.amplitudes.elements[offset] == .zero
				&& state.amplitudes.elements[offset + 1] == .zero)
		#expect(state.amplitudes.elements[offset + 2] == Complex(0.3, -0.2))
		try HCT.validate(state, at: 0.1)
		var rhs = HCT.RightHandSide(
			hamiltonian: problem.system.hamiltonian, preparation: prep, seed: 1,
			trajectoryID: 3)
		rhs.physicalNoise[0] = .zero
		rhs.evaluateWithCurrentNoise(t: 0.1, y: state, into: &dy)
		#expect(dy.amplitudes.elements[offset + 1].length > 0.1)
	}

	@Test("Adaptive retries and extra samples replay the same colored path", arguments: [0, 5])
	func adaptiveReplay(variant: Int) throws {
		var config = hcConfiguration(variant, colored: true)
		config.noiseStepSize = 0.03
		let problem = hopsProblem(hopsMatrix([0, 12, 12, 0]))
		let request = MultiTimeOrderedCorrelationRequest(
			insertions: [
				.init(time: 0.077, insertion: .left(.constant(hcL))),
				.init(time: 0.143, insertion: .right(.constant(hcX))),
			], observable: .constant(hcR))
		let fine = try hcValues(
			problem: problem, configuration: config, request: request,
			propagation: hopsPropagation(
				end: 0.25, maximumStep: 0.001, tolerance: 1e-11))
		let coarse = try hcValues(
			problem: problem, configuration: config, request: request,
			propagation: hopsPropagation(end: 0.25, maximumStep: 0.25, tolerance: 1e-10)
		)
		let sampled = try hcValues(
			problem: problem, configuration: config, request: request,
			propagation: hopsPropagation(
				end: 0.25, maximumStep: 0.25, tolerance: 1e-10,
				output: .times([0.15, 0.199, 0.25])))
		expectHOPSClose(coarse, fine, tolerance: 2e-7)
		expectHOPSClose([sampled.last!], fine, tolerance: 2e-7)
	}
}
