// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import SebbuQuantumToolkit

/// Compares the reduced dynamics of a TLS coupled to one damped vibrational
/// mode using
///
///  1. HOPS with a single-exponential zero-temperature bath correlation, and
///  2. GKSL with the vibration represented explicitly as a truncated oscillator.
///
/// The explicit-mode model is
///
///     H = H_TLS + omegaVib b^dagger b
///         + g |e><e| (b + b^dagger),
///
/// with a GKSL damping channel `kappa D[b]`.  For an oscillator initially in
/// vacuum this gives the bath correlation
///
///     alpha(t) = g^2 exp[-(kappa / 2 + i omegaVib) t].
///
/// `CorrelatedBathModel` stores the latent OU factor residue rather than the
/// exponential BCF coefficient directly.  For a single pole
///
///     W = kappa / 2 + i omegaVib
///
/// the residue `r = g sqrt(kappa)` gives
///
///     |r|^2 / (W + W*) = g^2,
///
/// so the HOPS and explicit-mode GKSL models describe the same environment.
///
/// Increase `trajectories` to reduce the remaining Monte-Carlo noise in the
/// HOPS curves.  `vibrationalDimension` is used as the oscillator Fock cutoff,
/// while the corresponding single-direction HOPS hierarchy is truncated at
/// tier `vibrationalDimension - 1`.
public func exampleTLSVibrationalModeHOPSGKSLComparison(
    endTime: Double = 15.0,
    tlsEnergy: Double = 1.0,
    tlsDrive: Double = 0.35,
    vibrationalFrequency: Double = 1.2,
    coupling: Double = 0.30,
    modeDamping: Double = 0.40,
    vibrationalDimension: Int = 10,
    trajectories: Int = 16_384,
    outputStep: Double = 0.02,
    maximumStep: Double = 0.01
) {
    precondition(endTime > 0)
    precondition(vibrationalFrequency > 0)
    precondition(coupling >= 0)
    precondition(modeDamping > 0, "The latent HOPS pole must have a positive real part.")
    precondition(vibrationalDimension >= 2)
    precondition(trajectories > 0)
    precondition(outputStep > 0)
    precondition(maximumStep > 0)

    let halfDrive = 0.5 * tlsDrive
    let tlsHamiltonian = Matrix<Complex<Double>>(
        elements: [
            .zero, Complex(halfDrive),
            Complex(halfDrive), Complex(tlsEnergy)
        ],
        rows: 2,
        columns: 2
    )

    let excitedStateProjector = Matrix<Complex<Double>>(
        elements: [
            .zero, .zero,
            .zero, .one
        ],
        rows: 2,
        columns: 2
    )

    let initialTLSComponent = Complex<Double>(1 / 2.0.squareRoot())
    let outputTimes: [Double] = .linearSpace(0.0, endTime, outputStep)
    let propagation = PropagationOptions(
        timeSpan: .init(start: 0.0, end: endTime),
        output: .times(outputTimes),
        integration: IntegrationOptions(
            minimumStepSize: 1e-8,
            maximumStepSize: maximumStep,
            absoluteTolerance: 1e-9,
            relativeTolerance: 1e-9
        )
    )

    // MARK: HOPS model

    // For kappa D[b], <b(t) b^dagger(0)> = exp[-(kappa/2 + i omegaVib)t].
    let pole = Complex<Double>(0.5 * modeDamping, vibrationalFrequency)

    // CorrelatedBathModel with one pole produces
    // alpha(t) = |r|^2 / (W + W*) * exp(-Wt).
    // Choosing r = g sqrt(kappa) therefore gives coefficient g^2.
    let latentResidue = Complex<Double>(coupling * modeDamping.squareRoot())
    let bath = CorrelatedBathModel(
        channelCount: 1,
        latentBaths: [
            .init(
                poles: [pole],
                residues: Matrix(
                    elements: [latentResidue],
                    rows: 1,
                    columns: 1
                )
            )
        ]
    )

    let hopsEnvironment = HOPS.Environment(
        couplingOperator: .constant(excitedStateProjector),
        bath: bath
    )
    let hopsConfiguration = HOPS.Configuration(
        hierarchy: .init(
            environment: hopsEnvironment,
            truncation: .maximumTier(vibrationalDimension - 1)
        ),
        equationType: .nonLinearNormalized,
        shiftType: .meanField,
        noiseStepSize: maximumStep
    )
    let hopsProblem = PureStateProblem(
        initialState: Vector([initialTLSComponent, initialTLSComponent]),
        system: QuantumSystem(tlsHamiltonian)
    )

    var hopsX: [Double] = []
    var hopsY: [Double] = []
    var hopsZ: [Double] = []
    hopsX.reserveCapacity(outputTimes.count)
    hopsY.reserveCapacity(outputTimes.count)
    hopsZ.reserveCapacity(outputTimes.count)

    do {
        let executionTime = try ContinuousClock().measure {
            try HOPS.solveEnsemble(
                problem: hopsProblem,
                configuration: hopsConfiguration,
                propagation: propagation,
                execution: TrajectoryExecution(
                    trajectories: trajectories,
                    seed: 0x51A61E
                )
            ) { _, rho in
                appendPauliExpectations(
                    coherence: rho[0, 1],
                    groundPopulation: rho[0, 0].real,
                    excitedPopulation: rho[1, 1].real,
                    x: &hopsX,
                    y: &hopsY,
                    z: &hopsZ
                )
            }
        }
        print("TLS + vibration HOPS simulation took:", executionTime)
    } catch {
        print("Failed to solve the HOPS TLS + vibration example:", error)
        return
    }

    // MARK: Explicit-mode GKSL model

    let oscillatorDimension = vibrationalDimension
    let fullDimension = 2 * oscillatorDimension

    // Product basis ordering: |g,0>, ..., |g,N-1>, |e,0>, ..., |e,N-1>.
    func productIndex(_ tls: Int, _ vibrationalLevel: Int) -> Int {
        tls * oscillatorDimension + vibrationalLevel
    }

    var fullHamiltonian = Matrix<Complex<Double>>.zeros(
        rows: fullDimension,
        columns: fullDimension
    )

    // H_TLS tensor I_vib.
    for tlsRow in 0..<2 {
        for tlsColumn in 0..<2 {
            let element = tlsHamiltonian[tlsRow, tlsColumn]
            for n in 0..<oscillatorDimension {
                fullHamiltonian[
                    productIndex(tlsRow, n),
                    productIndex(tlsColumn, n)
                ] += element
            }
        }
    }

    // I_TLS tensor omegaVib b^dagger b.
    for tls in 0..<2 {
        for n in 0..<oscillatorDimension {
            fullHamiltonian[productIndex(tls, n), productIndex(tls, n)] +=
                Complex(Double(n) * vibrationalFrequency)
        }
    }

    // g |e><e| tensor (b + b^dagger).
    // The projector means that only the excited TLS block is modified.
    if oscillatorDimension > 1 && coupling != .zero {
        for n in 0..<(oscillatorDimension - 1) {
            let matrixElement = Complex(coupling * Double(n + 1).squareRoot())
            let lower = productIndex(1, n)
            let upper = productIndex(1, n + 1)
            fullHamiltonian[lower, upper] += matrixElement
            fullHamiltonian[upper, lower] += matrixElement
        }
    }

    // I_TLS tensor b.
    var annihilation = Matrix<Complex<Double>>.zeros(
        rows: fullDimension,
        columns: fullDimension
    )
    for tls in 0..<2 {
        for n in 1..<oscillatorDimension {
            annihilation[
                productIndex(tls, n - 1),
                productIndex(tls, n)
            ] = Complex(Double(n).squareRoot())
        }
    }

    let vibrationalDamping = MarkovianChannel(
        rate: modeDamping,
        collapseOperator: annihilation
    )

    // |+x><+x| tensor |0><0|.
    var initialDensityMatrix = Matrix<Complex<Double>>.zeros(
        rows: fullDimension,
        columns: fullDimension
    )
    let groundVacuum = productIndex(0, 0)
    let excitedVacuum = productIndex(1, 0)
    initialDensityMatrix[groundVacuum, groundVacuum] = Complex(0.5)
    initialDensityMatrix[groundVacuum, excitedVacuum] = Complex(0.5)
    initialDensityMatrix[excitedVacuum, groundVacuum] = Complex(0.5)
    initialDensityMatrix[excitedVacuum, excitedVacuum] = Complex(0.5)

    let gkslProblem = DensityMatrixProblem(
        initialState: initialDensityMatrix,
        system: QuantumSystem(fullHamiltonian),
        markovianChannels: [vibrationalDamping]
    )

    var gkslX: [Double] = []
    var gkslY: [Double] = []
    var gkslZ: [Double] = []

    do {
        let executionTime = try ContinuousClock().measure {
            try GKSL.solve(
                problem: gkslProblem,
                propagation: propagation
            ) { _, rho in
                // Trace out the explicit vibrational mode.
                var groundPopulation = 0.0
                var excitedPopulation = 0.0
                var coherence = Complex<Double>.zero

                for n in 0..<oscillatorDimension {
                    groundPopulation += rho[
                        productIndex(0, n), productIndex(0, n)
                    ].real
                    excitedPopulation += rho[
                        productIndex(1, n), productIndex(1, n)
                    ].real
                    coherence += rho[
                        productIndex(0, n), productIndex(1, n)
                    ]
                }

                appendPauliExpectations(
                    coherence: coherence,
                    groundPopulation: groundPopulation,
                    excitedPopulation: excitedPopulation,
                    x: &gkslX,
                    y: &gkslY,
                    z: &gkslZ
                )
                return .proceed
            }
        }
        print("TLS + explicit vibration GKSL simulation took:", executionTime)
    } catch {
        print("Failed to solve the explicit-mode GKSL example:", error)
        return
    }

    // MARK: Numerical comparison

    print(
        "max |Delta <sigma_x>| =",
        maximumAbsoluteDifference(hopsX, gkslX)
    )
    print(
        "max |Delta <sigma_y>| =",
        maximumAbsoluteDifference(hopsY, gkslY)
    )
    print(
        "max |Delta <sigma_z>| =",
        maximumAbsoluteDifference(hopsZ, gkslZ)
    )

    // MARK: Plot

    plt.figure()
    plt.plot(x: outputTimes, y: hopsX, label: "HOPS <sigma_x>")
    plt.plot(
        x: outputTimes,
        y: gkslX,
        label: "GKSL explicit mode <sigma_x>",
        linestyle: "--"
    )
    plt.plot(x: outputTimes, y: hopsY, label: "HOPS <sigma_y>")
    plt.plot(
        x: outputTimes,
        y: gkslY,
        label: "GKSL explicit mode <sigma_y>",
        linestyle: "--"
    )
    plt.plot(x: outputTimes, y: hopsZ, label: "HOPS <sigma_z>")
    plt.plot(
        x: outputTimes,
        y: gkslZ,
        label: "GKSL explicit mode <sigma_z>",
        linestyle: "--"
    )
    plt.xlabel("t")
    plt.ylabel("<sigma_i>")
    plt.title("TLS coupled to one damped vibrational mode")
    plt.legend()
    plt.show()
    plt.close()
}

/// Uses the standard Pauli convention
/// sigma_y = [[0, -i], [i, 0]], hence <sigma_y> = -2 Im rho_ge.
private func appendPauliExpectations(
    coherence: Complex<Double>,
    groundPopulation: Double,
    excitedPopulation: Double,
    x: inout [Double],
    y: inout [Double],
    z: inout [Double]
) {
    x.append(2 * coherence.real)
    y.append(-2 * coherence.imaginary)
    z.append(groundPopulation - excitedPopulation)
}

private func maximumAbsoluteDifference(
    _ lhs: [Double],
    _ rhs: [Double]
) -> Double {
    precondition(lhs.count == rhs.count)
    var maximum = 0.0
    for index in lhs.indices {
        maximum = Swift.max(maximum, abs(lhs[index] - rhs[index]))
    }
    return maximum
}
