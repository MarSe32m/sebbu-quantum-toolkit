// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

@Suite("HOPS potential physics", .serialized)
struct HOPSPotentialPhysicsTests {
    @Test("One path survives two-time and multi-time insertions", arguments: 0..<6)
    func correlationContinuity(variant: Int) throws {
        let projector = hopsMatrix([0, 0, 0, 1])
        var config = hcConfiguration(variant)
        config.stochasticPotential = .init(couplingOperator: .constant(projector), covarianceModel: potentialTestModel(), noiseStepSize: 0.027)
        let problem = hopsProblem(hcZero, initial: [1, 0])
        let propagation = hopsPropagation(end: 0.31, maximumStep: 0.08, tolerance: 1e-11)
        let execution = TrajectoryExecution(trajectories: 1, seed: 831, parallelism: .serial)
        let insertionTime = 0.073
        var actual = Complex<Double>.zero
        let result = try HOPS.solveMultiTimeOrderedCorrelation(
            problem: problem, configuration: config,
            request: .init(insertions: [
                .init(time: insertionTime, insertion: .left(.constant(hcR))),
                .init(time: 0.181, insertion: .left(.constant(projector)))
            ], observable: .constant(hcL)), propagation: propagation, execution: execution
        ) { _, value in actual = value; return .proceed }
        let path = try #require(result.potentialNoise).path(for: 0)
        // The realized OU path is piecewise linear, so trapezoidal integration
        // split at its actual mesh nodes is an independent exact phase reference.
        var times = [insertionTime]
        let step = try #require(path.stepSize)
        for k in 0...20 {
            let t = path.meshOrigin.addingProduct(Double(k), step)
            if t > insertionTime && t < propagation.timeSpan.end { times.append(t) }
        }
        times.append(propagation.timeSpan.end)
        var previousTime = times[0]
        var previousForce = 0.0
        var phase = 0.0
        path.forEachSample(at: times) { time, value in
            let force = 2 * value[0].real
            phase += 0.5 * (time - previousTime) * (force + previousForce)
            previousTime = time; previousForce = force
        }
        let expected = Complex<Double>(length: 1, phase: -phase)
        #expect((actual - expected).length < 3e-9)
        var twoTime = Complex<Double>.zero
        _ = try HOPS.solveTwoTimeCorrelation(
            problem: problem, configuration: config,
            request: .init(insertionTime: insertionTime, insertion: .left(.constant(hcR)), observable: .constant(hcL)),
            propagation: propagation, execution: execution
        ) { _, value in twoTime = value; return .proceed }
        #expect((twoTime - expected).length < 3e-9)
    }

    @Test("Hermitian thermal force matches analytic Gaussian pure dephasing")
    func pureDephasing() throws {
        // alpha_0 = 0 isolates the thermal part of alpha_beta = alpha_0 + 2 Re K.
        // L = |1><1|, rho_01(t) = 1/2 exp[-2 Re integral_0^t (t-s) K(s) ds].
        let amplitude = 1.0
        let pole = Complex<Double>(0.7, 1.2)
        let time = 0.6
        let integral = amplitude * (Complex(time) / pole - (Complex<Double>.one - .exp(-pole * time)) / (pole * pole))
        let phaseVariance = 4 * integral.real
        let expected = 0.5 * Double.exp(-0.5 * phaseVariance)
        let count = 2048
        let pairVariance = 0.25 * (0.5 * (1 + Double.exp(-2 * phaseVariance)) - Double.exp(-phaseVariance))
        // Six analytic standard errors for independent antithetic pairs, plus
        // an interpolation/integration allowance much smaller than sampling error.
        let tolerance = 6 * (pairVariance / Double(count / 2)).squareRoot() + 0.0002
        var config = hcConfiguration(4)
        config.stochasticPotential = .init(couplingOperator: .constant(hopsMatrix([0, 0, 0, 1])),
            covarianceModel: .independentExponentials(amplitudes: [amplitude], poles: [pole]), noiseStepSize: 0.01)
        let component = Complex<Double>(1 / Double(2).squareRoot())
        let problem = hopsProblem(hcZero, initial: [component, component])
        var coherence = Complex<Double>.zero
        _ = try HOPS.solveEnsemble(problem: problem, configuration: config,
            propagation: hopsPropagation(end: time, maximumStep: 0.02, tolerance: 1e-8),
            execution: .init(trajectories: count, seed: 5731, parallelism: .serial, ensembleSampling: .antithetic)
        ) { _, rho in coherence = rho[0, 1] }
        #expect(abs(coherence.real - expected) < tolerance)
        #expect(abs(coherence.imaginary) < 1e-11)
    }
}
