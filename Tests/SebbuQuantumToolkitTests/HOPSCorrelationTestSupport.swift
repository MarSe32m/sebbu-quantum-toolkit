// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

typealias HCT = HOPS.CPUEngine
let hcI = hopsMatrix([1, 0, 0, 1])
let hcL = hopsMatrix([0, 1, 0, 0])
let hcR = hopsMatrix([0, 0, 1, 0])
let hcX = hopsMatrix([0, 1, 1, 0])
let hcZ = hopsMatrix([1, 0, 0, -1])
let hcZero = hopsMatrix([0, 0, 0, 0])
func hcConfiguration(_ variant: Int, colored: Bool = false, depth: Int = 5) -> HOPS.Configuration {
	let w = Complex(0.7, 0.9)
	let model: CorrelatedBathModel =
		colored
		? .init(
			channelCount: 1,
			latentBaths: [
				noiseTestBath([w], [[Complex((2 * w.real * 0.4).squareRoot())]])
			]) : .zero(channelCount: 1)
	return hopsConfiguration(
		variant, model: model, operators: [hcL], depth: depth, noiseStep: 0.005)
}
func hcValues<Hamiltonian>(
	problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
	request: MultiTimeOrderedCorrelationRequest,
	propagation: PropagationOptions<IntegrationOptions>,
	execution: TrajectoryExecution = .init(trajectories: 1, seed: 71, parallelism: .serial)
) throws -> [Complex<Double>] {
	var result: [Complex<Double>] = []
	try HOPS.solveMultiTimeOrderedCorrelation(
		problem: problem, configuration: configuration, request: request,
		propagation: propagation, execution: execution
	) { _, value in
		result.append(value)
		return .proceed
	}
	return result
}
func hcPath<Hamiltonian>(
	problem: PureStateProblem<Hamiltonian>, preparation: HCT.Preparation,
	request: MultiTimeOrderedCorrelationRequest,
	propagation: PropagationOptions<IntegrationOptions>,
	id: UInt64 = 3
) throws -> [Complex<Double>] {
	var result: [Complex<Double>] = []
	_ = try HCT().solveMultiTimeCorrelationTrajectory(
		problem: problem, preparation: preparation,
		request: request, propagation: propagation, seed: 71, trajectoryID: id
	) { _, value in result.append(value) }
	return result
}
