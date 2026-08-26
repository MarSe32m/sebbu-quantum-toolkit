// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import SebbuQuantumToolkit

public func testGKLSRadiativeDamping(endTime: Double) {
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
    let timeSpan: [Double] = .linearSpace(0.0, endTime, 1000)
    let propagationOptions = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .times(timeSpan),
        integration: IntegrationOptions(
            minimumStepSize: 0.001,
            maximumStepSize: 0.1,
            absoluteTolerance: 1e-8,
            relativeTolerance: 1e-8
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
                    Y.append(-2 * rho[0, 1].imaginary)
                    Z.append(rho[0, 0].real - rho[1, 1].real)
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
