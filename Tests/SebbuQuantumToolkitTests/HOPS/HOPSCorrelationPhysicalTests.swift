// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HOPS correlation physical validation", .serialized)
struct HOPSCorrelationPhysicalTests {
	@Test(
		"Colored correlations agree with an explicit damped environment", arguments: 0..<6,
		[false, true])
	func explicitEnvironment(variant: Int, hybrid: Bool) throws {
		let w = Complex(0.7, 0.9)
		let eta = [Complex(0.55, 0.15), Complex(-0.2, 0.25)]
		let operators = [hcL, hcZ]
		let model = CorrelatedBathModel(
			channelCount: 2,
			latentBaths: [
				noiseTestBath([w], eta.map { [(2 * w.real).squareRoot() * $0] })
			])
		let config = hopsConfiguration(
			variant, model: model, operators: operators, depth: 6, noiseStep: 0.005)
		let problem = hopsProblem(
			markovian: hybrid ? [.init(rate: 0.3, collapseOperator: hcL)] : [])
		let propagation = hopsPropagation(
			end: 1, maximumStep: hybrid ? 0.0025 : 0.02,
			tolerance: 2e-7, output: .times([0.53, 0.76, 1]))
		let prep = try HCT.Preparation(
			problem: problem, configuration: config, propagation: propagation)
		let observable = hopsMatrix([Complex(0.3), 1, Complex(0.2, 0.4), Complex(-0.5)])
		let sequences: [[TimedCorrelationInsertion]] = [
			[.init(time: 0.31, insertion: .left(.constant(hcX)))],
			[.init(time: 0.31, insertion: .right(.constant(hcR)))],
			[
				.init(time: 0.21, insertion: .left(.constant(hcX))),
				.init(time: 0.53, insertion: .right(.constant(hcL))),
			],
		]
		for events in sequences {
			let request = MultiTimeOrderedCorrelationRequest(
				insertions: events, observable: .constant(observable))
			let reference = try explicitEnvironmentReference(
				levels: 6, w: w, eta: eta, operators: operators,
				hybrid: hybrid, request: request, propagation: propagation)
			let converged = try explicitEnvironmentReference(
				levels: 8, w: w, eta: eta, operators: operators,
				hybrid: hybrid, request: request, propagation: propagation)
			expectHOPSClose(reference, converged, tolerance: 2e-6)
			let trajectories = 768
            
			var sums = [Complex<Double>](repeating: .zero, count: reference.count)
			var squares = [Double](repeating: 0, count: reference.count)
			for id in 0..<trajectories {
				let samples = try hcPath(
					problem: problem, preparation: prep, request: request,
					propagation: propagation, id: UInt64(id))
				for i in samples.indices {
					sums[i] += samples[i]
					squares[i] += samples[i].lengthSquared
				}
			}
			for i in reference.indices {
				let mean = sums[i] / Double(trajectories)
				let variance = max(
					0,
					(squares[i] - Double(trajectories) * mean.lengthSquared)
						/ Double(trajectories - 1))
				let bound =
					6 * (variance / Double(trajectories)).squareRoot() + 0.0015
				#expect(
					bound < 0.15,
					"The stochastic reference must remain informative")
				#expect(
					(mean - reference[i]).length < bound,
					"variant \(variant), hybrid \(hybrid), sample \(i): \(mean) vs \(reference[i]), bound \(bound)"
				)
			}
		}
	}

	@Test("Markovian radiative damping has the analytic phase and decay", arguments: 0..<6)
	func analyticDamping(variant: Int) throws {
		let gamma = 0.8
		let omega = 0.7
		let s = 0.31
		let times = [s, 0.57, 0.9]
		let problem = hopsProblem(
			hopsMatrix([0, 0, 0, Complex(omega)]), initial: [0, 1],
			markovian: [.init(rate: gamma, collapseOperator: hcL)])
		var samples: [Complex<Double>] = []
		try HOPS.solveTwoTimeCorrelation(
			problem: problem, configuration: hcConfiguration(variant),
			request: .init(
				insertionTime: s, insertion: .right(.constant(hcR)),
				observable: .constant(hcL)),
			propagation: hopsPropagation(
				end: 0.9, maximumStep: 0.002, output: .times(times)),
            execution: .init(trajectories: 2048, seed: 0xDAA9, parallelism: .automatic)
		) { _, value in
			samples.append(value)
			return .proceed
		}
		let expected = times.map { t in
			Complex<Double>(
				length: Double.exp(-gamma * s - 0.5 * gamma * (t - s)),
				phase: -omega * (t - s))
		}
		expectHOPSClose(samples, expected, tolerance: 0.045)
	}
}

// This reference evolves the system AND a damped oscillator. Applying regression
// only to the reduced system would erase the bath correlations being tested.
private func explicitEnvironmentReference(
	levels: Int, w: Complex<Double>, eta: [Complex<Double>],
	operators: [Matrix<Complex<Double>>], hybrid: Bool,
	request: MultiTimeOrderedCorrelationRequest,
	propagation: PropagationOptions<IntegrationOptions>
) throws -> [Complex<Double>] {
	let d = 2 * levels
	let systemH = hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)])
	var h = liftCorrelationSystem(systemH, levels: levels)
	var m = Matrix<Complex<Double>>.zeros(rows: 2, columns: 2)
	for p in eta.indices {
		for i in 0..<2 {
			for j in 0..<2 { m[i, j] += eta[p].conjugate * operators[p][i, j] }
		}
	}
	var annihilation = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
	for s in 0..<2 {
		for n in 0..<levels {
			h[s * levels + n, s * levels + n] += Complex(w.imaginary * Double(n))
		}
		for n in 0..<(levels - 1) {
			annihilation[s * levels + n, s * levels + n + 1] = Complex(
				Double(n + 1).squareRoot())
		}
		for r in 0..<2 {
			for n in 0..<(levels - 1) {
				let value = m[s, r] * Double(n + 1).squareRoot()
				h[s * levels + n + 1, r * levels + n] += value
				h[r * levels + n, s * levels + n + 1] += value.conjugate
			}
		}
	}
	var initial = [Complex<Double>](repeating: .zero, count: d)
	initial[0] = Complex(0.6)
	initial[levels] = Complex(0, 0.8)
	var channels = [MarkovianChannel(rate: 2 * w.real, collapseOperator: annihilation)]
	if hybrid {
		channels.append(
			.init(
				rate: 0.3,
				collapseOperator: liftCorrelationSystem(hcL, levels: levels)))
	}
	let problem = PureStateProblem(
		initialState: Vector(initial), system: QuantumSystem(h), markovianChannels: channels
	)
	func lift(_ op: TimeDependentOperator) -> TimeDependentOperator {
		var matrix = UniqueMatrix<Complex<Double>>.zeros(rows: 2, columns: 2)
		op.insert(t: 0, into: &matrix)
		let copy = hopsMatrix((0..<4).map { matrix.elements[$0] })
		return .constant(liftCorrelationSystem(copy, levels: levels))
	}
	let events = request.insertions.map { event in
		let insertion: CorrelationInsertion
		switch event.insertion {
		case .left(let op): insertion = .left(lift(op))
		case .right(let op): insertion = .right(lift(op))
		}
		return TimedCorrelationInsertion(time: event.time, insertion: insertion)
	}
	let referencePropagation = hopsPropagation(
		end: propagation.timeSpan.end, start: propagation.timeSpan.start,
		maximumStep: 0.03, tolerance: 1e-11, output: propagation.output)
	var result: [Complex<Double>] = []
	try GKSL.solveMultiTimeOrderedCorrelation(
		problem: problem,
		request: .init(insertions: events, observable: lift(request.observable)),
		propagation: referencePropagation
	) { _, value in
		result.append(value)
		return .proceed
	}
	return result
}

private func liftCorrelationSystem(_ matrix: Matrix<Complex<Double>>, levels: Int) -> Matrix<
	Complex<Double>
> {
	var result = Matrix<Complex<Double>>.zeros(rows: 2 * levels, columns: 2 * levels)
	for i in 0..<2 {
		for j in 0..<2 {
			for n in 0..<levels {
				result[i * levels + n, j * levels + n] = matrix[i, j]
			}
		}
	}
	return result
}
