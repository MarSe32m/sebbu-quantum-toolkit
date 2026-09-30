// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import SebbuQuantumToolkit

public func exampleHEOMRadiativeDamping(endTime: Double) {
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
            minimumStepSize: 0.0001,
            maximumStepSize: 1,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        ),
        progress: .console(
            style: .bar,
            label: "HEOM Radiative Damping"
        )
    )
    let L = TimeDependentOperator.constant(Matrix<Complex<Double>>.init(elements: [.zero, .zero, .zero, 1], rows: 2, columns: 2))
    let bath = CorrelatedBathModel.zero(channelCount: 1)
    let environment = BathEnvironment(couplingOperator: L, bath: bath)
    let hierarchy = HEOM.Hierarchy(environment: environment, truncation: .maximumTier(0))
    let configuration = HEOM.Configuration(
        hierarchy: hierarchy,
        shiftType: .meanField,
        parallelism: .automatic
    )
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HEOM.solve(
                problem: problem,
                configuration: configuration,
                propagation: propagationOptions
            ) { _, densityMatrix in
                X.append(2 * densityMatrix[0, 1].real)
                Y.append(2 * densityMatrix[0, 1].imaginary)
                Z.append((densityMatrix[0, 0] - densityMatrix[1, 1]).real)
                return .proceed
            }
        }
        print("HEOM simulation took:", executionTime)
    } catch {
        print("Failed to solve HEOM master equation: \(error)")
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

fileprivate func exactIBMSolution(t: Double, epsilon: Double, initialState: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>]) -> (X: Double, Y: Double, Z: Double) {
    var F: Complex<Double> = .zero
    for (g, w) in zip(G, W) {
        F += g * (.exp(-w * t) + t * w - 1) / (w * w)
    }
    let rho_gg = initialState[0, 0].real
    let rho_ge = initialState[0, 1] * .exp(.i * epsilon * t) * .exp(-F.conjugate)
    let rho_ee = 1 - rho_gg
    return (2 * rho_ge.real, -2 * rho_ge.imaginary, rho_gg - rho_ee)
}

public func exampleHEOMIBM(endTime: Double) {
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
    let initialState: Vector<Complex<Double>> = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]
    let initialRho = initialState.outer(initialState.conjugate)
    let problem = PureStateProblem(
        initialState: initialState,
        system: system
    )
    var timeSpan: [Double] = []
    let propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .uniform(step: 0.01),
        integration: IntegrationOptions(
            minimumStepSize: 0.0001,
            maximumStepSize: 1,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        ),
        progress: .console(
            style: .bar,
            label: "HEOM IBM"
        )
    )
    let L = TimeDependentOperator.constant(Matrix<Complex<Double>>.init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2))
    let environment = BathEnvironment(couplingOperator: L, bath: bath)
    let hierarchy = HEOM.Hierarchy(environment: environment, truncation: .maximumTier(4))
    let configuration = HEOM.Configuration(
        hierarchy: hierarchy,
        shiftType: .meanField,
        parallelism: .automatic
    )
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HEOM.solve(
                problem: problem,
                configuration: configuration,
                propagation: propagationOptions
            ) { time, densityMatrix in
                timeSpan.append(time)
                X.append(2 * densityMatrix[0, 1].real)
                Y.append(-2 * densityMatrix[0, 1].imaginary)
                Z.append((densityMatrix[0, 0] - densityMatrix[1, 1]).real)
                return .proceed
            }
        }
        print("HEOM simulation took:", executionTime)
    } catch {
        print("Failed to solve HEOM master equation: \(error)")
    }
    let G = bath.oneSidedExponentialTerms.map { $0.residue[0, 0] }
    let W = bath.oneSidedExponentialTerms.map { $0.pole }
    let exactExpectationValues = timeSpan.map { 
        exactIBMSolution(
            t: $0, 
            epsilon: renormalizationEnergy, 
            initialState: initialRho, 
            G: G, 
            W: W
        )
    }
    let exactX = exactExpectationValues.map { $0.X }
    let exactY = exactExpectationValues.map { $0.Y }
    let exactZ = exactExpectationValues.map { $0.Z }

    plt.figure()

    plt.plot(x: timeSpan, y: exactX, label: "Exact <X>")
    plt.plot(x: timeSpan, y: exactY, label: "Exact <Y>")
    plt.plot(x: timeSpan, y: exactZ, label: "Exact <Z>")

    plt.plot(x: timeSpan, y: X, label: "HOPS <X>", linestyle: "--")
    plt.plot(x: timeSpan, y: Y, label: "HOPS <Y>", linestyle: "--")
    plt.plot(x: timeSpan, y: Z, label: "HOPS <Z>", linestyle: "--")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<O>")
    plt.show()
    plt.close()
}

/// Converge preparation time, trajectories, hierarchy depth and noise/integration steps.
public func exampleHEOMResonanceFluorescenceSpectrum(
    A: Double = .zero, cutoff: Double = 1.447,
    maximumTier: Int = 4,
    steadyTime: Double = 200, delayTime: Double = 200,
    maximumStep: Double = 1,
    plotCorrelation: Bool = false
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
    let environment = BathEnvironment(couplingOperator: .constant(Matrix<Complex<Double>>(
        elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)), bath: bath)
    let system = QuantumSystem(
        Matrix.init(elements: [.zero, Complex(0.175), Complex(0.175), Complex(renormalizationEnergy)], rows: 2, columns: 2)
    )
    let sigmaMinusMatrix = Matrix<Complex<Double>>(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaMinus: ConstantOperator = .init(sigmaMinusMatrix)
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
    let hierarchy = HEOM.Hierarchy(environment: environment, truncation: .maximumTier(maximumTier))
    let configuration = HEOM.Configuration(
        hierarchy: hierarchy,
        shiftType: .meanField,
        parallelism: .automatic
    )
    var propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: tSteady),
        output: .final,
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: maximumStep,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        ),
        progress: .console(
            style: .bar,
            label: "HEOM Steady State"
        )
    )
    var steadyState: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
    var sigmaMinusExpectation: Complex<Double> = .zero
    do {
        let executionTime = try ContinuousClock().measure {
            try HEOM.solve(
                problem: problem, configuration: configuration,
                propagation: propagationOptions
            ) { time, rho in
                steadyState = .init(copying: rho)
                sigmaMinusExpectation = steadyState.dot(sigmaMinusMatrix).trace
                return .proceed
            }
        }
        print("HEOM steady-state simulation took:", executionTime)
    } catch {
        print("Failed to solve HEOM steady state: \(error)")
        return
    }
    let insertionTime = 0.0
    let request = TwoTimeCorrelationRequest(
        insertionTime: insertionTime,
        insertion: .right(.constant(sigmaPlus)),
        observable: .constant(sigmaMinus)
    )
    let times: [Double] = .linearSpace(insertionTime, insertionTime + delayTime, 10000)
    propagationOptions = PropagationOptions(
        timeSpan: .init(start: -tSteady, end: times.last!),
        output: .times(times),
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: maximumStep,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        ),
        progress: .console(
            style: .bar,
            label: "HEOM RF Spectrum"
        )
    )
    var correlationFunction: [Complex<Double>] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try HEOM.solveTwoTimeCorrelation(
                problem: problem,
                configuration: configuration,
                request: request,
                propagation: propagationOptions
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
    if plotCorrelation {
        plt.figure()
        plt.plot(x: times, y: correlationFunction.real, label: "Re C(t)")
        plt.plot(x: times, y: correlationFunction.imaginary, label: "Im C(t)")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("C(t)")
        plt.show()
        plt.close()
    }
    
    let omegaSpace: [Double] = .linearSpace(-1, 1, 2000)
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
    plt.plot(x: omegaSpace, y: spectrum, label: "HEOM S(w)")
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("S(w)")
//    plt.show()
//    plt.close()
}
