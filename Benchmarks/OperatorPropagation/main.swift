// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuQuantumToolkit
import SebbuScience

// End-to-end timings include preparation. Identical explicit CSR and dense
// Hamiltonians isolate representation choice; bath storage stays fixed.
let arguments = Array(CommandLine.arguments.dropFirst())
func option(_ name: String, _ fallback: String) -> String {
    guard let i = arguments.firstIndex(of: name) else { return fallback }
    precondition(i + 1 < arguments.count, "Missing value for \(name)")
    return arguments[i + 1]
}
let dimensions = option("--dimensions", "2,32,128").split(separator: ",").map { Int($0)! }
let methods = option("--methods", "hops,heom").split(separator: ",").map(String.init)
let storage = option("--storage", "dense,sparse").split(separator: ",").map(String.init)
let tier = Int(option("--tier", "3"))!
let repeats = Int(option("--repeats", "3"))!
let workers = Int(option("--workers", "1"))!
let end = Double(option("--end", "0.02"))!
let step = Double(option("--step", "0.002"))!
let structure = option("--structure", "banded")
let bathStorage = option("--bath-storage", "dense")
precondition(
    dimensions.allSatisfy { $0 >= 2 } && tier >= 0 && repeats > 0 && workers > 0 && end > 0 && step > 0)
precondition(methods.allSatisfy { ["hops", "heom"].contains($0) })
precondition(storage.allSatisfy { ["dense", "sparse"].contains($0) })
precondition(["diagonal", "banded", "dense"].contains(structure))
precondition(["dense", "sparse"].contains(bathStorage))

let bath = CorrelatedBathModel(
    channelCount: 1,
    latentBaths: [
        .init(
            poles: [Complex(0.7, 0.9), Complex(1.2, 1.7), Complex(0.9, -0.4)],
            residues: Matrix(elements: [Complex(0.1), Complex(0.08), Complex(0.06)], rows: 1, columns: 3))
    ])
let propagation = PropagationOptions<IntegrationOptions>(
    timeSpan: .init(start: 0, end: end), output: .final,
    integration: .init(
        minimumStepSize: 0, maximumStepSize: step, absoluteTolerance: 1e-9, relativeTolerance: 1e-9))
print(
    "method,dimension,structure,nnz,hierarchy,workers,bath_storage,storage,repeat,seconds,checksum_real,checksum_imaginary"
)
for d in dimensions {
    var dense = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
    var values: [Complex<Double>] = []
    var columns: [Int] = []
    var rows = [0]
    for i in 0..<d {
        for j in 0..<d where i == j || (structure == "banded" && abs(i - j) == 1) || structure == "dense" {
            let value = i == j ? Complex(0.2 * Double(i) / Double(d)) : Complex(0.15 / Double(1 + abs(i - j)))
            dense[i, j] = value
            values.append(value)
            columns.append(j)
        }
        rows.append(values.count)
    }
    let csr = CSRMatrix(rows: d, columns: d, values: values, rowIndices: rows, columnIndices: columns)
    let coupling = CSRMatrix<Complex<Double>>(
        rows: d, columns: d,
        values: (0..<d).map { Complex(Double($0) / Double(d)) },
        rowIndices: Array(0...d), columnIndices: Array(0..<d))
    var denseCoupling = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
    for i in 0..<d { denseCoupling[i, i] = Complex(Double(i) / Double(d)) }
    let bathOperator = bathStorage == "sparse" ? ConstantOperator(coupling) : ConstantOperator(denseCoupling)
    let environment = BathEnvironment(couplingOperator: .constant(bathOperator), bath: bath)
    let hc = HOPS.Configuration(
        hierarchy: .init(environment: environment, truncation: .maximumTier(tier)),
        equationType: .linear, noiseStepSize: step)
    var ec = HEOM.Configuration(hierarchy: .init(environment: environment, truncation: .maximumTier(tier)))
    ec.parallelism = .maximumWorkers(workers)
    let initial = Vector<Complex<Double>>((0..<d).map { _ in Complex(1 / Double(d).squareRoot()) })
    for method in methods {
        for representation in storage {
            let h: TimeDependentOperator =
                representation == "sparse" ? .constant(ConstantOperator(csr)) : .constant(dense)
            let problem = PureStateProblem(initialState: initial, system: .init(dimension: d, hamiltonian: h))
            // The first run warms caches and is excluded from the CSV.
            for repeatIndex in -1..<repeats {
                var checksum = Complex<Double>.zero
                let duration = try ContinuousClock().measure {
                    if method == "hops" {
                        try HOPS.solveTrajectory(
                            problem: problem, configuration: hc, propagation: propagation, seed: 713,
                            trajectoryID: 0
                        ) { _, psi in
                            for i in 0..<d { checksum += Double(i + 1) * psi[i] }
                            return .proceed
                        }
                    } else {
                        try HEOM.solve(problem: problem, configuration: ec, propagation: propagation) {
                            _, rho in
                            for i in 0..<d {
                                for j in 0..<d { checksum += Double(1 + i * d + j) * rho[i, j] }
                            }
                            return .proceed
                        }
                    }
                }
                if repeatIndex >= 0 {
                    let seconds =
                        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
                    print(
                        "\(method),\(d),\(structure),\(values.count),\(method == "heom" ? ec.hierarchy.count : hc.hierarchy.count),\(method == "heom" ? workers : 1),\(bathStorage),\(representation),\(repeatIndex),\(seconds),\(checksum.real),\(checksum.imaginary)"
                    )
                }
            }
        }
    }
}
