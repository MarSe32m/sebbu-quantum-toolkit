// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

let heomShifts: [HEOM.ShiftType] = [.none, .meanField]

func heomSingleBath(strength: Double = 0.16, pole: Complex<Double> = Complex(0.6, 1.2)) -> CorrelatedBathModel
{
    .init(
        channelCount: 1,
        latentBaths: [
            .init(
                poles: [pole],
                residues: .init(
                    elements: [Complex((2 * pole.real * strength).squareRoot())], rows: 1, columns: 1))
        ])
}

func heomConfiguration(
    shift: HEOM.ShiftType = .none, depth: Int = 8,
    model: CorrelatedBathModel = heomSingleBath(),
    operators: [Matrix<Complex<Double>>] = [hopsMatrix([0, 0, 0, 1])]
) -> HEOM.Configuration {
    .init(
        hierarchy: .init(
            environment: .init(couplingOperators: operators.map { .constant($0) }, bath: model),
            truncation: .maximumTier(depth)), shiftType: shift)
}

func heomIndices(_ hierarchy: HEOM.Hierarchy) -> [[Int]] {
    (0..<hierarchy.count).map { index in
        var row: [Int] = []
        hierarchy.multiIndex(of: index) { s in for i in 0..<s.count { row.append(s[i]) } }
        return row
    }
}

func heomFinal<Hamiltonian: HamiltonianFunction>(
    problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
    propagation: PropagationOptions<IntegrationOptions> = hopsPropagation(end: 1)
) throws -> [Complex<Double>] {
    var result: [Complex<Double>] = []
    try HEOM.solve(problem: problem, configuration: configuration, propagation: propagation) { _, rho in
        result = (0..<(rho.rows * rho.columns)).map { rho.elements[$0] }
        return .proceed
    }
    return result
}

func heomTensor(_ a: Matrix<Complex<Double>>, _ b: Matrix<Complex<Double>>) -> Matrix<Complex<Double>> {
    Matrix(rows: a.rows * b.rows, columns: a.columns * b.columns) { elements in
        for i in 0..<a.rows {
            for j in 0..<a.columns {
                for k in 0..<b.rows {
                    for l in 0..<b.columns {
                        elements[(i * b.rows + k) * a.columns * b.columns + j * b.columns + l] =
                            a[i, j] * b[k, l]
                    }
                }
            }
        }
    }
}

func heomIdentity(_ d: Int) -> Matrix<Complex<Double>> {
    var result = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
    for i in 0..<d { result[i, i] = .one }
    return result
}

/// Independent enlarged-system reference: a damped oscillator initially in
/// vacuum. Its BCF is g^2 exp[-(kappa/2 + i omega)t].
func heomPseudomodeProblem(
    h: Matrix<Complex<Double>>, coupling: Matrix<Complex<Double>>,
    rho: Matrix<Complex<Double>>, oscillatorDimension n: Int,
    strength: Double = 0.16, pole: Complex<Double> = Complex(0.6, 1.2), rate: Double = 0
) -> DensityMatrixProblem<ConstantHamiltonian> {
    var b = Matrix<Complex<Double>>.zeros(rows: n, columns: n)
    var number = b
    var vacuum = b
    vacuum[0, 0] = .one
    for j in 1..<n {
        b[j - 1, j] = Complex(Double(j).squareRoot())
        number[j, j] = Complex(Double(j))
    }
    let id = heomIdentity(2)
    let oscillatorIdentity = heomIdentity(n)
    var total = heomTensor(h, oscillatorIdentity)
    let energy = heomTensor(id, number)
    let interaction = heomTensor(coupling, b.conjugateTranspose)
    let adjoint = interaction.conjugateTranspose
    for i in 0..<total.elements.count {
        total.elements[i] +=
            pole.imaginary * energy.elements[i] + strength.squareRoot()
            * (interaction.elements[i] + adjoint.elements[i])
    }
    var channels = [
        MarkovianChannel(rate: .constant(2 * pole.real), collapseOperator: .constant(heomTensor(id, b)))
    ]
    if rate > 0 {
        channels.append(
            .init(
                rate: .constant(rate),
                collapseOperator: .constant(heomTensor(hopsMatrix([0, 1, 0, 0]), oscillatorIdentity))))
    }
    return .init(initialState: heomTensor(rho, vacuum), system: .init(total), markovianChannels: channels)
}
