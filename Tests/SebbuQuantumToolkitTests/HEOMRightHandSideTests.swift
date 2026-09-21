// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Test("HEOM drift matches a literal multi-channel, multi-pole reference", arguments: heomShifts, [1, 3])
func heomDriftReference(shift: HEOM.ShiftType, workers: Int) throws {
    let model = hopsFixtureModel()
    let h = hopsMatrix([Complex(0.1), Complex(0.2, 0.1), Complex(0.2, -0.1), Complex(-0.3)])
    let l = hopsMatrix([0, 1, 0, 0])
    var configuration = heomConfiguration(
        shift: shift, depth: 2, model: model, operators: hopsFixtureOperators)
    configuration.parallelism = .maximumWorkers(workers)
    let problem = DensityMatrixProblem(
        initialState: hopsMatrix([Complex(0.3), Complex(0.1), Complex(0.1), Complex(0.7)]),
        system: QuantumSystem(h),
        markovianChannels: [.init(rate: .generated { t in 0.2 + t }, collapseOperator: .constant(l))])
    let count = configuration.hierarchy.count
    let pCount = model.poleCount
    let shiftCount = shift == .none ? 0 : pCount
    var state = HEOM.CPUEngine.State(dimension: 2, hierarchyCount: count, shiftCount: shiftCount, copies: 2)
    var derivative = HEOM.CPUEngine.State(
        dimension: 2, hierarchyCount: count, shiftCount: shiftCount, copies: 2)
    for i in 0..<(state.ados.rows * 4) {
        state.ados.elements[i] = Complex(Double.sin(Double(i + 1)), Double.cos(Double(i + 2))) * 0.1
    }
    for i in 0..<4 { state.ados.elements[i] = problem.initialState.elements[i] }
    for i in 0..<shiftCount { state.shifts[i] = Complex(0.03 * Double(i + 1), -0.02 * Double(i + 2)) }
    let failure = HEOM.CPUEngine.Failure()
    var rhs = try HEOM.CPUEngine.RightHandSide(
        problem: problem, configuration: configuration, failure: failure, copies: 2)
    rhs.evaluate(t: 0.17, y: state, dy: &derivative)
    try failure.check()

    // Assemble Lambda and M from the original model, independently of the
    // engine's shared contractions, flattened adjacency tables and kernels.
    var lambda = Array(repeating: hopsMatrix([0, 0, 0, 0]), count: pCount)
    var memory = lambda
    var offset = 0
    for bath in model.latentBaths {
        for p in 0..<bath.poleCount {
            for i in hopsFixtureOperators.indices {
                for j in 0..<4 {
                    lambda[offset + p].elements[j] +=
                        bath.residues[i, p].conjugate * hopsFixtureOperators[i].elements[j]
                }
            }
        }
        for p in 0..<bath.poleCount {
            for q in 0..<bath.poleCount {
                let covariance = Complex<Double>.one / (bath.poles[p] + bath.poles[q].conjugate)
                for j in 0..<4 {
                    memory[offset + p].elements[j] += covariance * lambda[offset + q].elements[j]
                }
            }
        }
        offset += bath.poleCount
    }
    let amplitudes = (0..<(state.ados.rows * 4)).map { state.ados.elements[$0] }
    func matrix(_ index: Int) -> Matrix<Complex<Double>> {
        hopsMatrix((0..<4).map { amplitudes[index * 4 + $0] })
    }
    let indices = heomIndices(configuration.hierarchy)
    let ids = Dictionary(uniqueKeysWithValues: indices.enumerated().map { ($0.element, $0.offset) })
    let poles = model.latentBaths.flatMap(\.poles)
    var expected = Array(repeating: Complex<Double>.zero, count: count * 8)
    var shiftedH = h
    var means = Array(repeating: Complex<Double>.zero, count: pCount)
    if shift == .meanField {
        let root = matrix(0)
        for p in 0..<pCount {
            let product = memory[p].dot(root)
            means[p] = (product[0, 0] + product[1, 1]) / (root[0, 0] + root[1, 1])
            #expect((derivative.shifts[p] - (-poles[p] * state.shifts[p] + means[p])).length < 1e-13)
            let adjoint = lambda[p].conjugateTranspose
            for j in 0..<4 {
                shiftedH.elements[j] +=
                    .i
                    * (state.shifts[p].conjugate * lambda[p].elements[j] - state.shifts[p]
                        * adjoint.elements[j])
            }
        }
    }
    let loss = l.conjugateTranspose.dot(l)
    for block in 0..<2 {
        for ado in 0..<count {
            let rho = matrix(block * count + ado)
            var value = hopsMatrix([0, 0, 0, 0])
            func add(_ m: Matrix<Complex<Double>>, _ factor: Complex<Double>) {
                for j in 0..<4 { value.elements[j] += factor * m.elements[j] }
            }
            add(shiftedH.dot(rho), -.i)
            add(rho.dot(shiftedH), .i)
            add(l.dot(rho).dot(l.conjugateTranspose), Complex(0.37))
            add(loss.dot(rho), Complex(-0.185))
            add(rho.dot(loss), Complex(-0.185))
            let ns = indices[ado]
            for p in 0..<pCount {
                add(rho, -Double(ns[p]) * poles[p] - Double(ns[p + pCount]) * poles[p].conjugate)
                var ketChild = ns
                ketChild[p] += 1
                if let id = ids[ketChild] {
                    let child = matrix(block * count + id)
                    let adjoint = lambda[p].conjugateTranspose
                    let weight = Complex<Double>(Double(ns[p] + 1).squareRoot())
                    add(adjoint.dot(child), -weight)
                    add(child.dot(adjoint), weight)
                }
                var braChild = ns
                braChild[p + pCount] += 1
                if let id = ids[braChild] {
                    let child = matrix(block * count + id)
                    let weight = Complex<Double>(Double(ns[p + pCount] + 1).squareRoot())
                    add(lambda[p].dot(child), weight)
                    add(child.dot(lambda[p]), -weight)
                }
                var ketParent = ns
                ketParent[p] -= 1
                if let id = ids[ketParent] {
                    let parent = matrix(block * count + id)
                    let weight = Complex<Double>(Double(ns[p]).squareRoot())
                    add(memory[p].dot(parent), weight)
                    if shift == .meanField { add(parent, -weight * means[p]) }
                }
                var braParent = ns
                braParent[p + pCount] -= 1
                if let id = ids[braParent] {
                    let parent = matrix(block * count + id)
                    let weight = Complex<Double>(Double(ns[p + pCount]).squareRoot())
                    add(parent.dot(memory[p].conjugateTranspose), weight)
                    if shift == .meanField { add(parent, -weight * means[p].conjugate) }
                }
            }
            for j in 0..<4 { expected[(block * count + ado) * 4 + j] = value.elements[j] }
        }
    }
    expectHOPSClose((0..<expected.count).map { derivative.ados.elements[$0] }, expected, tolerance: 3e-13)
}

@Test("HEOM tiny and BLAS matrix kernels respect complex adjoints and accumulation", arguments: [2, 5])
func heomMatrixKernels(d: Int) {
    let a = UniqueMatrix<Complex<Double>>(
        elements: (0..<(d * d)).map { Complex(Double($0) * 0.07, 0.1) }, rows: d, columns: d)
    let b = UniqueMatrix<Complex<Double>>(
        elements: (0..<(d * d)).map { Complex(-0.2, Double($0) * 0.09) }, rows: d, columns: d)
    let c = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
    let scale = Complex<Double>(0.4, -0.2)
    for adjointA in [false, true] {
        for adjointB in [false, true] {
            for adding in [false, true] {
                for j in 0..<(d * d) { c.elements[j] = Complex(0.3, 0.1) }
                HEOM.CPUEngine.MatrixAction.product(
                    a.elements, b.elements, dimension: d, adjointA: adjointA, adjointB: adjointB,
                    scale: scale, adding: adding, into: c.elements)
                for i in 0..<d {
                    for j in 0..<d {
                        var expected = adding ? Complex<Double>(0.3, 0.1) : .zero
                        for k in 0..<d {
                            let left = adjointA ? a[k, i].conjugate : a[i, k]
                            let right = adjointB ? b[j, k].conjugate : b[k, j]
                            expected += scale * left * right
                        }
                        #expect((c[i, j] - expected).length < 1e-12)
                    }
                }
            }
        }
    }
}

@Test("HEOM error control protects the root and shift from dilution by many ADOs")
func heomErrorControl() {
    var high = HEOM.CPUEngine.State(dimension: 2, hierarchyCount: 100, shiftCount: 1)
    var low = HEOM.CPUEngine.State(dimension: 2, hierarchyCount: 100, shiftCount: 1)
    let start = HEOM.CPUEngine.State(dimension: 2, hierarchyCount: 100, shiftCount: 1)
    high.ados.elements[0] = Complex(1e-4)
    #expect(
        high.normalizedError(
            comparedTo: low, relativeTo: start, absoluteTolerance: 1e-5, relativeTolerance: 0) > 9.9)
    low.assign(high)
    high.shifts[0] = Complex(2e-4)
    #expect(
        high.normalizedError(
            comparedTo: low, relativeTo: start, absoluteTolerance: 1e-5, relativeTolerance: 0) > 19.9)
    high.shifts[0] = Complex(.nan, 0)
    #expect(
        high.normalizedError(
            comparedTo: low, relativeTo: start, absoluteTolerance: 1e-5, relativeTolerance: 0
        ).isInfinite)
}
