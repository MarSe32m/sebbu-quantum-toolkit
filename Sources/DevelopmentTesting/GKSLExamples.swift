// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import SebbuQuantumToolkit

public func exampleGKSLRadiativeDamping(endTime: Double) {
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
    let problem = DensityMatrixProblem(
        initialState: Matrix.init(elements: (0..<4).map { _ in Complex<Double>(0.5) }, rows: 2, columns: 2),
        system: system,
        markovianChannels: [markovianChannel, markovianChannel2]
    )
    let timeSpan: [Double] = .linearSpace(0.0, endTime, 0.1)
    let propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .times(timeSpan),
        integration: IntegrationOptions(
            minimumStepSize: 0.001,
            maximumStepSize: 0.5,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try GKSL.solve(
                problem: problem,
                propagation: propagationOptions
            ) { time, rho in
                X.append(2 * rho[0, 1].real)
                Y.append(2 * rho[0, 1].imaginary)
                Z.append(rho[0, 0].real - rho[1, 1].real)
                return .proceed
            }
        }
        print("GKSL simulation took:", executionTime)
    } catch {
        print("Failed to solve GKSL master equation: \(error)")
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

public func exampleGKSLResonanceFluorescenceSpectrum() {
    let system = QuantumSystem(
        Matrix.init(elements: [.zero, Complex(0.5), Complex(0.5), .zero], rows: 2, columns: 2)
    )
    let sigmaMinus: ConstantOperator = .init(Matrix.init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2))
    let sigmaPlus: ConstantOperator = .init(Matrix.init(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2))
    
    let markovianChannel = MarkovianChannel(
        rate: .constant(0.075),
        collapseOperator: .constant(sigmaMinus)
    )
    let problem = DensityMatrixProblem(
        initialState: Matrix.init(elements: (0..<4).map { _ in Complex<Double>(0.5) }, rows: 2, columns: 2),
        system: system,
        markovianChannels: [markovianChannel]
    )
    let tSteady = 200.0
    var propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: tSteady),
        output: .final,
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: 0.5,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    var steadyState: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
    var sigmaMinusExpectation: Complex<Double> = .zero
    do {
        let executionTime = try ContinuousClock().measure {
            try GKSL.solve(
                problem: problem,
                propagation: propagationOptions
            ) { time, rho in
                steadyState = .init(copying: rho)
                sigmaMinusExpectation = steadyState.dot(sigmaMinus.matrix).trace
                return .proceed
            }
        }
        print("GKSL simulation took:", executionTime)
    } catch {
        print("Failed to solve GKSL master equation: \(error)")
    }
    let insertionTime = 0.0
    let request = TwoTimeCorrelationRequest(
        insertionTime: insertionTime,
        insertion: .right(.constant(sigmaPlus)),
        observable: .constant(sigmaMinus)
    )
    let times: [Double] = .linearSpace(insertionTime, insertionTime + tSteady, 10000)
    // We could also construct a new DensityMatrixProblem where the steady state is the initial
    // state but this is also valid.
    propagationOptions = PropagationOptions(
        timeSpan: .init(start: -1000, end: times.last!),
        output: .times(times),
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: 0.5,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )
    var correlationFunction: [Complex<Double>] = []
    do {
        let executionTime = try ContinuousClock().measure {
            try GKSL.solveTwoTimeCorrelation(
                problem: problem,
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
                .exp(-.i * omega * (t - insertionTime)) * correlationFunctionSpline.sample(t)
        }, x: times)
        spectrum.append(s.real)
    }
    
    
    plt.figure()
    plt.plot(x: omegaSpace, y: spectrum, label: "S(w)")
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("S(w)")
    plt.show()
    plt.close()
}
