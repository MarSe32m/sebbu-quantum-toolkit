// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import SebbuQuantumToolkit

/// Raw numerical scales throughout: no temperature unit or physical constant
/// is interpreted by HOPS. Run with the Examples --stochastic-potential flag.
public func exampleHOPSStochasticPotentials() throws {
    let l = Matrix<Complex<Double>>(elements: [0, 1, 0, 0], rows: 2, columns: 2)
    let h = Matrix<Complex<Double>>(elements: [0, 0, 0, Complex(1.2)], rows: 2, columns: 2)
    let problem = PureStateProblem(initialState: Vector.init([0, 1]), system: .init(h))
    let vacuum = CorrelatedBathModel.independentExponentials(amplitudes: [0.09], poles: [Complex(0.3, 1.2)])
    let thermal = CorrelatedBathModel.independentExponentials(amplitudes: [0.018], poles: [Complex(0.3, 1.2)])
    let hierarchy = HOPS.Hierarchy(environment: .init(couplingOperators: [.constant(l)], bath: vacuum),
                                   truncation: .maximumTier(5))
    var configuration = HOPS.Configuration(hierarchy: hierarchy, equationType: .nonLinearNormalized,
        noiseStepSize: 0.01, stochasticPotential: .init(couplingOperator: .constant(l), covarianceModel: thermal))
    let propagation = PropagationOptions(timeSpan: SimulationTimeSpan(start: 2, end: 3),
        output: OutputSchedule.uniform(step: 0.1), integration: IntegrationOptions(
            minimumStepSize: 0, maximumStepSize: 0.02, absoluteTolerance: 1e-8, relativeTolerance: 1e-8))

    // One coefficient path is shared by every hierarchy tier. Bath noise z and
    // potential coefficients y remain separately named and separately seeded.
    let run = try HOPS.solveTrajectory(problem: problem, configuration: configuration, propagation: propagation,
        seed: 735, trajectoryID: 2, ensembleSampling: .antithetic,
        observingWithNoises: { time, state, noise in
            print(time, "excited:", state[1].lengthSquared, "z:", noise.bath[0], "y:", noise.potential[0])
            return .proceed
        })
    run.potentialNoise!.forEachSample(at: [2.0, 2.25, 2.5, 2.75, 3]) { time, y in
        print("Replayed y", time, y[0])
    }
    let ensemble = try HOPS.solveEnsemble(problem: problem, configuration: configuration, propagation: propagation,
        execution: .init(trajectories: 256, seed: 735, ensembleSampling: .antithetic)) { time, rho in
            print("Ensemble", time, rho[1, 1].real)
        }
    var replay = ensemble.potentialNoise!.path(for: 2).makeSampler(windowDuration: 0.1)
    var y = UniqueVector<Complex<Double>>.zero(1)
    replay.sample(2.25, into: &y.mutableSpan)

    // A user-supplied covariance is fitted explicitly, once, before propagation.
    let fitTimes = (0...128).map { Double($0) * 0.05 }
    let fitOptions = CorrelatedBathFitter.Options(maximumPencilPoleCount: 3, targetRelativeRMSError: 1e-6)
    let fit = try CorrelatedBathFitter.fitBathCorrelation(times: fitTimes, options: fitOptions) { t in
        Complex<Double>(0.018) * .exp(-Complex(0.3, 1.2) * t)
    }
    print("Covariance fit diagnostics:", fit.diagnostics)
    configuration.stochasticPotential = .init(couplingOperator: .constant(l), covarianceModel: fit.model)
    _ = try HOPS.solveTrajectory(problem: problem, configuration: configuration, propagation: propagation,
        seed: 735, trajectoryID: 2) { time, state in
            print("Fitted covariance", time, state[1].lengthSquared); return .proceed
        }

    // FFT accepts P(omega) = n(omega) J(omega), already thermally weighted.
    // This illustrative smooth positive-frequency P is a different covariance
    // from the full-line rational OU model above. It is not model.spectralDensity.
    configuration.stochasticPotential = .init(couplingOperator: .constant(l),
        fft: .init(dtMax: 0.02, deltaOmegaMax: 0.03, omegaMax: 5), powerDensity: { omega in
            0.01 * omega * Double.exp(-omega)
        })
    _ = try HOPS.solveTrajectory(problem: problem, configuration: configuration, propagation: propagation,
        seed: 735, trajectoryID: 2, observingWithNoises: { time, _, noise in
            print("FFT y", time, noise.potential[0]); return .proceed
        })

    // A custom realized coefficient overrides the configured source. Supplying
    // zero bath noise still retains vacuum memory; this is a pathwise example,
    // not an ensemble approximation. Undamped rotation needs no stable OU pole.
    _ = try HOPS.solveWithHierarchy(problem: problem, configuration: configuration, propagation: propagation,
        bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: ZeroDiffusiveNoise(channelCount: 0),
        potentialNoise: RotatingPotentialCoefficient(amplitude: Complex(0.1, 0.03), frequency: 1.2),
        observingWithNoises: { time, hierarchy, noise in
            print("Custom path", time, "tiers:", hierarchy.count, "y:", noise.potential[0]); return .proceed
        })
}

private struct RotatingPotentialCoefficient: ~Copyable, MultiChannelNoiseRealization {
    let amplitude: Complex<Double>
    let frequency: Double
    let channelCount = 1
    mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>) {
        values[0] = amplitude * .exp(Complex(0, -frequency * time))
    }
}

/// Non-Hermitian L = sigma_- coupled to a thermal damped oscillator.
/// This deliberately costly validation is separate from the default test suite.
/// Repeat with smaller maximumStep, larger trajectories and deeper truncations.
/// Run with the Examples --thermal-pseudomode flag.
public func exampleHOPSThermalPseudomodeComparison(
    endTime: Double = 4, coupling: Double = 0.3, frequency: Double = 1.2,
    damping: Double = 0.3, occupation: Double = 0.25,
    hierarchyDepth: Int = 5, oscillatorDimension: Int = 7,
    trajectories: Int = 4096, maximumStep: Double = 0.02,
    checkConvergence: Bool = true
) throws {
    precondition(endTime > 0 && coupling >= 0 && damping > 0 && occupation >= 0)
    precondition(hierarchyDepth > 0 && oscillatorDimension > 1 && trajectories > 0)
    let l = Matrix<Complex<Double>>(elements: [0, 1, 0, 0], rows: 2, columns: 2)
    let h = Matrix<Complex<Double>>(elements: [0, 0, 0, Complex(frequency)], rows: 2, columns: 2)
    let pole = Complex(damping, frequency)
    let vacuum = CorrelatedBathModel.independentExponentials(amplitudes: [coupling * coupling], poles: [pole])
    let thermal = CorrelatedBathModel.independentExponentials(amplitudes: [occupation * coupling * coupling], poles: [pole])
    let propagation = PropagationOptions(timeSpan: SimulationTimeSpan(start: 0, end: endTime),
        output: OutputSchedule.uniform(step: 0.05), integration: IntegrationOptions(
            minimumStepSize: 0, maximumStepSize: maximumStep, absoluteTolerance: 1e-8, relativeTolerance: 1e-8))

    func hops(depth: Int) throws -> [Double] {
        let configuration = HOPS.Configuration(hierarchy: .init(
            environment: .init(couplingOperators: [.constant(l)], bath: vacuum), truncation: .maximumTier(depth)),
            equationType: .nonLinearNormalized, noiseStepSize: maximumStep / 2,
            stochasticPotential: .init(couplingOperator: .constant(l), covarianceModel: thermal))
        var populations: [Double] = []
        try HOPS.solveEnsemble(problem: .init(initialState: Vector.init([0, 1]), system: .init(h)),
            configuration: configuration, propagation: propagation,
            execution: .init(trajectories: trajectories, seed: 0x71E2, ensembleSampling: .antithetic)) {
                _, rho in populations.append(rho[1, 1].real)
            }
        return populations
    }

    func explicitMode(dimension n: Int) throws -> (populations: [Double], maximumEdgePopulation: Double) {
        let d = 2 * n
        var fullH = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
        var annihilation = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
        // Basis |g,0>...|g,n-1>, |e,0>...|e,n-1>.
        for tls in 0..<2 {
            for level in 0..<n {
                fullH[tls * n + level, tls * n + level] = Complex(Double(tls + level) * frequency)
                if level > 0 { annihilation[tls * n + level - 1, tls * n + level] = Complex(Double(level).squareRoot()) }
            }
        }
        for level in 0..<(n - 1) {
            let g = Complex(coupling * Double(level + 1).squareRoot())
            fullH[level + 1, n + level] += g
            fullH[n + level, level + 1] += g
        }
        let q = occupation / (1 + occupation)
        var weights: [Double] = []
        var weight = 1.0
        for _ in 0..<n { weights.append(weight); weight *= q }
        let sum = weights.reduce(0, +)
        var rho0 = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
        for level in 0..<n { rho0[n + level, n + level] = Complex(weights[level] / sum) }
        var populations: [Double] = []
        var edge = 0.0
        try GKSL.solve(problem: .init(initialState: rho0, system: .init(fullH), markovianChannels: [
            .init(rate: 2 * damping * (occupation + 1), collapseOperator: annihilation),
            .init(rate: 2 * damping * occupation, collapseOperator: annihilation.conjugateTranspose)
        ]), propagation: propagation) { _, rho in
            var population = 0.0
            for level in 0..<n { population += rho[n + level, n + level].real }
            populations.append(population)
            edge = max(edge, rho[n - 1, n - 1].real + rho[d - 1, d - 1].real)
            return .proceed
        }
        print("Initial omitted thermal probability for cutoff", n, "=", weight)
        return (populations, edge)
    }

    func discrepancy(_ a: [Double], _ b: [Double]) -> Double {
        precondition(a.count == b.count)
        return zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }
    let sampled = try hops(depth: hierarchyDepth)
    let reference = try explicitMode(dimension: oscillatorDimension)
    print("Maximum HOPS/GKSL excited-population difference:", discrepancy(sampled, reference.populations))
    print("Maximum occupation of highest oscillator level:", reference.maximumEdgePopulation)
    if checkConvergence {
        let refinedMode = try explicitMode(dimension: oscillatorDimension + 2)
        let refinedHOPS = try hops(depth: hierarchyDepth + 2)
        print("Oscillator cutoff change:", discrepancy(reference.populations, refinedMode.populations))
        print("Hierarchy-depth change (same reference seeds):", discrepancy(sampled, refinedHOPS))
        print("Refined HOPS/GKSL difference:", discrepancy(refinedHOPS, refinedMode.populations))
    }
}
