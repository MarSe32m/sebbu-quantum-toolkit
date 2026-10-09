// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

@Suite("HOPS potential validation")
struct HOPSPotentialValidationTests {
    @Test("Explicit paths require matching operators and channels")
    func explicitValidation() throws {
        let problem = hopsProblem()
        let base = hcConfiguration(0)
        var config = base
        config.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: .zero(channelCount: 1))
        #expect(throws: HCT.SolverError.missingPotentialNoise) {
            try HOPS.solveTrajectory(problem: problem, configuration: config, propagation: hopsPropagation(),
                bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: ZeroDiffusiveNoise(channelCount: 0)) { _, _ in .proceed }
        }
        #expect(throws: HCT.SolverError.missingPotentialNoise) {
            try HOPS.solveWithHierarchy(problem: problem, configuration: config, propagation: hopsPropagation(),
                bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: ZeroDiffusiveNoise(channelCount: 0)) { _, _ in .proceed }
        }
        #expect(throws: HCT.SolverError.potentialNoiseChannelCountMismatch(expected: 1, actual: 2)) {
            try HOPS.solveTrajectory(problem: problem, configuration: config, propagation: hopsPropagation(),
                bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: ZeroDiffusiveNoise(channelCount: 0),
                potentialNoise: ZeroMultiChannelNoise(channelCount: 2)) { _, _ in .proceed }
        }
        #expect(throws: HCT.SolverError.potentialNoiseChannelCountMismatch(expected: 0, actual: 1)) {
            try HOPS.solveTrajectory(problem: problem, configuration: base, propagation: hopsPropagation(),
                bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: ZeroDiffusiveNoise(channelCount: 0),
                potentialNoise: ZeroMultiChannelNoise(channelCount: 1)) { _, _ in .proceed }
        }
        config.stochasticPotential = .init(couplingOperator: .constant(Matrix<Complex<Double>>.identity(rows: 3)), covarianceModel: .zero(channelCount: 1))
        #expect(throws: HCT.SolverError.operatorDimensionMismatch) {
            try HOPS.solveTrajectory(problem: problem, configuration: config, propagation: hopsPropagation(), seed: 1, trajectoryID: 0) { _, _ in .proceed }
        }
    }

    @Test("Invalid exponential inputs fail clearly")
    func exponentialInputs() async {
        await #expect(processExitsWith: .failure) {
            _ = CorrelatedBathModel.independentExponentials(amplitudes: [1], poles: [])
        }
        await #expect(processExitsWith: .failure) {
            _ = CorrelatedBathModel.independentExponentials(amplitudes: [-1], poles: [Complex(1)])
        }
        await #expect(processExitsWith: .failure) {
            _ = CorrelatedBathModel.independentExponentials(amplitudes: [.nan], poles: [Complex(1)])
        }
        await #expect(processExitsWith: .failure) {
            _ = CorrelatedBathModel.independentExponentials(amplitudes: [0], poles: [Complex(0, 1)])
        }
        await #expect(processExitsWith: .failure) {
            _ = CorrelatedBathModel.independentExponentials(amplitudes: [1], poles: [Complex(1, .infinity)])
        }
    }

    @Test("Potential channel, grid and PSD validation")
    func sourceInputs() async {
        await #expect(processExitsWith: .failure) {
            _ = StochasticPotential(couplingOperator: .constant(hcL), covarianceModel: .zero(channelCount: 2))
        }
        await #expect(processExitsWith: .failure) {
            _ = StochasticPotential(couplingOperator: .constant(hcL), covarianceModel: .zero(channelCount: 1), noiseStepSize: 0)
        }
        await #expect(processExitsWith: .failure) {
            _ = StochasticPotential.FFTOptions(dtMax: 0.1, deltaOmegaMax: .nan, omegaMax: 2)
        }
        await #expect(processExitsWith: .failure) {
            _ = HCT.NoiseGridCursor((origin: 2, step: 0.1), start: 1)
        }
        await #expect(processExitsWith: .failure) {
            _ = HCT.NoiseGridCursor((origin: 1e20, step: 1e-10), start: 1e20)
        }
        await #expect(processExitsWith: .failure) {
            _ = potentialTestDefinition(.init(couplingOperators: [.constant(hcL), .constant(hcR)],
                fft: .init(dtMax: 0.1, deltaOmegaMax: 1, omegaMax: 2), powerDensity: { _ in hopsMatrix([1, 2, 2, 1]) }))
        }
        await #expect(processExitsWith: .failure) {
            _ = potentialTestDefinition(.init(couplingOperators: [.constant(hcL)],
                fft: .init(dtMax: 0.1, deltaOmegaMax: 1, omegaMax: 2), powerDensity: { _ in hcI }))
        }
    }
}
