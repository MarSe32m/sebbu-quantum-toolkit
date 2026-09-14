// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import SebbuQuantumToolkit

public func exampleHOPSRadiativeDamping(endTime: Double) {
    let system = QuantumSystem(
        Matrix.init(elements: [.zero, -.one, -.one, .one], rows: 2, columns: 2)
    )
    let markovianChannel = MarkovianChannel(
        rate: .constant(0.1),
        collapseOperator: .constant(
            .init(
                Matrix.init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
            )
        )
    )
    let markovianChannel2 = MarkovianChannel(
        rate: .generated({ t in 0.05 }),
        collapseOperator: .constant(
            .init(
                Matrix.init(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2)
            )
        )
    )
    let problem = PureStateProblem(
        initialState: Vector.init([Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]),
        system: system,
        markovianChannels: [markovianChannel, markovianChannel2]
    )
    let timeSpan: [Double] = .linearSpace(0.0, endTime, 0.01)
    let propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .uniform(step: 0.01),
        integration: IntegrationOptions(
            minimumStepSize: 0.01,
            maximumStepSize: 0.01,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    let L = TimeDependentOperator.constant(Matrix<Complex<Double>>.init(elements: [.zero, .zero, .zero, 1], rows: 2, columns: 2))
    let bath = CorrelatedBathModel.zero(channelCount: 1)
    let environment = HOPS.Environment(couplingOperator: L, bath: bath)
    let hierarchy = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(0))
    let configuration = HOPS.Configuration(
        hierarchy: hierarchy,
        equationType: .nonLinearNormalized,
        shiftType: .meanField)
    let trajectories = 4096
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HOPS.solveEnsemble(
                problem: problem,
                configuration: configuration,
                propagation: propagationOptions,
                execution: TrajectoryExecution(
                    trajectories: trajectories,
                    seed: 1234
                )
            ) { _, densityMatrix in
                X.append(2 * densityMatrix[0, 1].real)
                Y.append(2 * densityMatrix[0, 1].imaginary)
                Z.append((densityMatrix[0, 0] - densityMatrix[1, 1]).real)
            }
        }
        print("HOPS simulation took:", executionTime)
    } catch {
        print("Failed to solve HOPS master equation: \(error)")
    }
    plt.figure()
    plt.plot(x: timeSpan, y: X, label: "<X>")
    plt.plot(x: timeSpan, y: Y, label: "<Y>")
    plt.plot(x: timeSpan, y: Z, label: "<Z>")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<O>")
    plt.show()
    plt.close()
}

fileprivate func spectralDensity(omega: Double, A: Double, cutoff: Double) -> Double {
    A * omega * omega * omega * .exp(-(omega * omega) / (cutoff * cutoff))
}

fileprivate func spectralDensityByOmega(omega: Double, A: Double, cutoff: Double) -> Double {
    A * omega * omega * .exp(-(omega * omega) / (cutoff * cutoff))
}

fileprivate func bcf(t: Double, J: (Double) -> Double) -> Complex<Double> {
    Quad.integrate(a: 0, b: .infinity) { omega in
            Complex(length: J(omega), phase: -omega * t)
    }
}

fileprivate struct IBMBath {
    let bath: CorrelatedBathModel
    let renormalizationEnergy: Double
}

fileprivate func makeIBMBath(A: Double, cutoff: Double) throws -> IBMBath {
    if A == .zero { return IBMBath(bath: .zero(channelCount: 1), renormalizationEnergy: .zero) }
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, cutoff: cutoff)
    }
    let tau: [Double] = .linearSpace(0, 10, 100)
    let BCF = tau.map { bcf(t: $0) { omega in
        spectralDensity(omega: omega, A: A, cutoff: cutoff)
    }}
    let bath = try CorrelatedBathFitter.fitBathCorrelation(times: tau, values: BCF, options: .init(maximumPencilPoleCount: 3)).model
    return IBMBath(bath: bath, renormalizationEnergy: renormalizationEnergy)
}

public func exampleHOPSIBM(endTime: Double, trajectories: Int = 4096) {
    let A = 0.27
    let cutoff = 1.447
    
    let tau: [Double] = .linearSpace(0, 10, 1000)
    let BCF = tau.map { bcf(t: $0) { omega in
        spectralDensity(omega: omega, A: A, cutoff: cutoff)
    }}
    let ibmBath: IBMBath
    do {
        ibmBath = try makeIBMBath(A: A, cutoff: cutoff)
    } catch {
        print("Bath model construction failed with error:", error)
        return
    }
    let bath = ibmBath.bath
    let renormalizationEnergy = ibmBath.renormalizationEnergy
    plt.figure()
    plt.plot(x: tau, y: BCF.real, label: "Re BCF")
    plt.plot(x: tau, y: BCF.imaginary, label: "Im BCF")
    plt.plot(x: tau, y: tau.map { bath.bathCorrelation(at: $0)[0,0].real }, label: "Re BCF fit", linestyle: "--")
    plt.plot(x: tau, y: tau.map { bath.bathCorrelation(at: $0)[0,0].imaginary }, label: "Im BCF fit", linestyle: "--")
    plt.xlabel("t")
    plt.ylabel("bcf")
    plt.legend()
    plt.show()
    plt.close()
    print(bath.poleCount)
    let system = QuantumSystem(
        Matrix.init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    )
    let problem = PureStateProblem(
        initialState: Vector.init([Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]),
        system: system
    )
    let timeSpan: [Double] = .linearSpace(0.0, endTime, 0.01)
    let propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .uniform(step: 0.01),
        integration: IntegrationOptions(
            minimumStepSize: 0.0001,
            maximumStepSize: 0.01,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    let L = TimeDependentOperator.constant(Matrix<Complex<Double>>.init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2))
    let environment = HOPS.Environment(couplingOperator: L, bath: bath)
    let hierarchy = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(4))
    let configuration = HOPS.Configuration(
        hierarchy: hierarchy,
        equationType: .nonLinearNormalized,
        shiftType: .meanField)
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HOPS.solveEnsemble(
                problem: problem,
                configuration: configuration,
                propagation: propagationOptions,
                execution: TrajectoryExecution(
                    trajectories: trajectories,
                    seed: 1234
                )
            ) { _, densityMatrix in
                X.append(2 * densityMatrix[0, 1].real)
                Y.append(2 * densityMatrix[0, 1].imaginary)
                Z.append((densityMatrix[0, 0] - densityMatrix[1, 1]).real)
            }
        }
        print("HOPS simulation took:", executionTime)
    } catch {
        print("Failed to solve HOPS master equation: \(error)")
    }
    plt.figure()
    plt.plot(x: timeSpan, y: X, label: "<X>")
    plt.plot(x: timeSpan, y: Y, label: "<Y>")
    plt.plot(x: timeSpan, y: Z, label: "<Z>")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<O>")
    plt.show()
    plt.close()
}

/// Converge preparation time, trajectories, hierarchy depth and noise/integration steps.
public func exampleHOPSResonanceFluorescenceSpectrum(
    A: Double = .zero, cutoff: Double = 1.447,
    maximumTier: Int = 4,
    trajectories: Int = 8192, steadyTime: Double = 200, delayTime: Double = 200,
    maximumStep: Double = 0.01
) {
    let tSteady = steadyTime
    let ibmBath: IBMBath
    do {
        ibmBath = try makeIBMBath(A: A, cutoff: cutoff)
    } catch {
        print("Failed to create IBM bath: \(error)")
        return
    }
    let bath = ibmBath.bath
    let renormalizationEnergy = ibmBath.renormalizationEnergy
    print("Renormalization energy:", renormalizationEnergy)
    let environment = HOPS.Environment(couplingOperator: .constant(Matrix<Complex<Double>>(
        elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)), bath: bath)
    let system = QuantumSystem(
        Matrix.init(elements: [.zero, Complex(0.5), Complex(0.5), Complex(renormalizationEnergy)], rows: 2, columns: 2)
    )
    let sigmaMinus: ConstantOperator = .init(Matrix.init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2))
    let sigmaPlus: ConstantOperator = .init(Matrix.init(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2))
    
    let markovianChannel = MarkovianChannel(
        rate: .constant(0.075),
        collapseOperator: .constant(sigmaMinus)
    )
    let initialComponent = Complex<Double>(1 / 2.0.squareRoot())
    let problem = PureStateProblem(
        initialState: Vector([initialComponent, initialComponent]),
        system: system,
        markovianChannels: [markovianChannel]
    )
    let configuration = HOPS.Configuration(
        hierarchy: .init(environment: environment, truncation: .maximumTier(maximumTier)),
        equationType: .nonLinearNormalized, shiftType: .meanField, noiseStepSize: maximumStep)
    var propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: tSteady),
        output: .final,
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: maximumStep,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    var steadyState: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
    var sigmaMinusExpectation: Complex<Double> = .zero
    do {
        let executionTime = try ContinuousClock().measure {
            try HOPS.solveEnsemble(
                problem: problem, configuration: configuration,
                propagation: propagationOptions,
                execution: .init(trajectories: trajectories, seed: 0x57EAD7)
            ) { time, rho in
                steadyState = .init(copying: rho)
                sigmaMinusExpectation = steadyState.dot(sigmaMinus.matrix).trace
            }
        }
        print("HOPS steady-state simulation took:", executionTime)
    } catch {
        print("Failed to solve HOPS steady state: \(error)")
        return
    }
    let insertionTime = 0.0
    let request = TwoTimeCorrelationRequest(
        insertionTime: insertionTime,
        insertion: .right(.constant(sigmaPlus)),
        observable: .constant(sigmaMinus)
    )
    let times: [Double] = .linearSpace(insertionTime, insertionTime + delayTime, 10000)
    // The separate ensemble supplies only the one-time stationary expectation.
    // Its reduced density matrix cannot initialize the system-bath correlations.
    // Every correlation guide prepares its full hierarchy from -tSteady to zero,
    // retaining its auxiliaries, OU sampler and accumulated guide shift memory.
    propagationOptions = PropagationOptions(
        timeSpan: .init(start: -tSteady, end: times.last!),
        output: .times(times),
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: maximumStep,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    var correlationFunction: [Complex<Double>] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HOPS.solveTwoTimeCorrelation(
                problem: problem,
                configuration: configuration,
                request: request,
                propagation: propagationOptions,
                execution: TrajectoryExecution(
                    trajectories: trajectories,
                    seed: 0xC0FFEE
                )
            ) { t, sample in
                correlationFunction.append(sample - sigmaMinusExpectation.lengthSquared)
                return .proceed
            }
        }
        print("Two time correlation function solve took:", executionTime)
    } catch {
        print("Failed to solve two time correlation function:", error)
        return
    }
    plt.figure()
    plt.plot(x: times, y: correlationFunction.real, label: "Re C(t)")
    plt.plot(x: times, y: correlationFunction.imaginary, label: "Im C(t)")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("C(t)")
    plt.show()
    plt.close()
    
    let omegaSpace: [Double] = .linearSpace(-3, 3, 2000)
    var spectrum: [Double] = []
    let correlationFunctionSpline = CubicHermiteSpline(x: times, y: correlationFunction)
    for omega in omegaSpace {
        let s = Trapezoid.integrate(y: { t in
                .exp(.i * omega * (t - insertionTime)) * correlationFunctionSpline.sample(t)
        }, x: times)
        spectrum.append(s.real)
    }
    let max = spectrum.max()!
    spectrum = spectrum.map { $0 / max }
    
    plt.figure()
    plt.plot(x: omegaSpace, y: spectrum, label: "S(w)")
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("S(w)")
    plt.show()
    plt.close()
}
