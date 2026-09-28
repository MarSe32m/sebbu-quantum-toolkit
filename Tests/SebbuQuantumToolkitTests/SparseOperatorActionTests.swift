// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

// Build CSR explicitly, independently of the dense reference. The final row
// is empty and entries are complex/non-Hermitian to distinguish A, Aᵀ and A†.
func explicitSparseFixture(_ d: Int) -> (CSRMatrix<Complex<Double>>, Matrix<Complex<Double>>) {
    var values: [Complex<Double>] = []
    var columns: [Int] = []
    var rows = [0]
    var dense = Matrix<Complex<Double>>.zeros(rows: d, columns: d)
    for i in 0..<d {
        if i + 1 < d {
            values += [Complex(0.1 * Double(i + 1), -0.07), Complex(0.23, 0.11 * Double(i + 1))]
            columns += [i, i + 1]
            dense[i, i] = Complex(0.1 * Double(i + 1), -0.07)
            dense[i, i + 1] = Complex(0.23, 0.11 * Double(i + 1))
        }
        rows.append(values.count)
    }
    return (.init(rows: d, columns: d, values: values, rowIndices: rows, columnIndices: columns), dense)
}

func sparseFixtureState(_ d: Int) -> Vector<Complex<Double>> {
    Vector((0..<d).map { Complex(0.1 + 0.03 * Double($0), -0.07 + 0.02 * Double($0)) })
}

func sparseFixtureOperator(_ d: Int, sparse: Bool, expansion: Bool = false) -> TimeDependentOperator {
    let (csr, dense) = explicitSparseFixture(d)
    let op = sparse ? ConstantOperator(csr) : ConstantOperator(dense)
    return expansion
        ? .linearCombination(
            .init(
                coefficients: [.generated { Complex(1 + $0, -0.3 * $0) }, .constant(Complex(0.2))],
                operators: [op, .init(dense.conjugateTranspose)])) : .constant(op)
}

func sparseFixtureProblem(_ d: Int, sparseH: Bool, sparseC: Bool, expansion: Bool = false) -> PureStateProblem
{
    .init(
        initialState: sparseFixtureState(d),
        system: .init(
            dimension: d, hamiltonian: sparseFixtureOperator(d, sparse: sparseH, expansion: expansion)),
        markovianChannels: [
            .init(
                rate: .generated { 0.2 + $0 },
                collapseOperator: sparseFixtureOperator(d, sparse: sparseC, expansion: expansion))
        ])
}

@Suite("Prepared sparse operator actions")
struct SparseOperatorActionTests {
    @Test(arguments: [2, 5, 13])
    func actions(d: Int) {
        let (csr, dense) = explicitSparseFixture(d)
        let sparse = _PreparedOperator(.constant(csr), dimension: d)
        let reference = _PreparedOperator(.constant(dense), dimension: d)
        #expect(sparse.buffer.rows == 0)
        #expect(sparse.action.terms[0].matrix.isSparse)
        let x = UniqueVector(copying: sparseFixtureState(d))
        let a = UniqueVector<Complex<Double>>.zero(d)
        let b = UniqueVector<Complex<Double>>.zero(d)
        let scale = Complex(0.37, -0.61)
        for adjoint in [false, true] {
            for adding in [false, true] {
                sparse.action.vector(
                    x.components, into: a.components, scale: scale, adding: adding, adjoint: adjoint)
                reference.action.vector(
                    x.components, into: b.components, scale: scale, adding: adding, adjoint: adjoint,
                    blas: true)
                expectHOPSClose((0..<d).map { a[$0] }, (0..<d).map { b[$0] })
            }
        }
        let rho = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        let sa = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        let da = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        for i in 0..<d * d { rho.elements[i] = Complex(0.03 * Double(i), -0.01 * Double(i + 1)) }
        for right in [false, true] {
            for adjoint in [false, true] {
                for adding in [false, true] {
                    sparse.action.matrix(
                        rho.elements, into: sa.elements, right: right, adjoint: adjoint, scale: scale,
                        adding: adding)
                    reference.action.matrix(
                        rho.elements, into: da.elements, right: right, adjoint: adjoint, scale: scale,
                        adding: adding)
                    expectHOPSClose((0..<d * d).map { sa.elements[$0] }, (0..<d * d).map { da.elements[$0] })
                }
            }
        }
        sparse.action.commutator(rho.elements, into: sa.elements)
        reference.action.commutator(rho.elements, into: da.elements)
        expectHOPSClose((0..<d * d).map { sa.elements[$0] }, (0..<d * d).map { da.elements[$0] })
        sparse.action.commutator(rho.elements, into: sa.elements, scale: scale, adding: true, adjoint: true)
        reference.action.commutator(rho.elements, into: da.elements, scale: scale, adding: true, adjoint: true)
        expectHOPSClose((0..<d * d).map { sa.elements[$0] }, (0..<d * d).map { da.elements[$0] })
        var trace = Complex<Double>.zero
        for i in 0..<d {
            for j in 0..<d { trace += dense[i, j] * rho[j, i] }
        }
        #expect((sparse.action.traceProduct(rho.elements) - trace).length < 1e-11)
        #expect((reference.action.traceProduct(rho.elements) - trace).length < 1e-11)
        let mean = sparse.action.expectation(x.components, scratch: a.components)
        let expectedMean = reference.action.expectation(x.components, scratch: b.components)
        #expect((mean - expectedMean).length < 1e-11)
    }

    @Test func materializationAndEmptyCSR() {
        let (csr, dense) = explicitSparseFixture(5)
        var result = UniqueMatrix<Complex<Double>>.zeros(rows: 5, columns: 5)
        for i in 0..<25 { result.elements[i] = Complex(42, 17) }
        TimeDependentOperator.constant(csr).insert(t: 0, into: &result)
        expectHOPSClose((0..<25).map { result.elements[$0] }, dense.elements)
        let empty = UniqueCSRMatrix(
            copying: CSRMatrix<Complex<Double>>(
                rows: 5, columns: 5, values: [], rowIndices: [0, 0, 0, 0, 0, 0], columnIndices: []))
        let source = TimeDependentOperator.constant(empty)
        source.insert(t: 0, into: &result)
        #expect((0..<25).allSatisfy { result.elements[$0] == .zero })
        let prepared = _PreparedOperator(source, dimension: 5)
        let x = UniqueVector(copying: sparseFixtureState(5))
        let y = UniqueVector<Complex<Double>>.zero(5)
        for adjoint in [false, true] {
            prepared.action.vector(x.components, into: y.components, adjoint: adjoint)
            #expect((0..<5).allSatisfy { y[$0] == .zero })
        }
    }

    @Test(arguments: [false, true], [0, 1, 2])
    func expansions(aggregate: Bool, storage: Int) {
        let d = 5
        let (csr, dense) = explicitSparseFixture(d)
        let calls = Mutex(0)
        let source = TimeDependentOperator.linearCombination(
            .init(
                coefficients: [
                    .generated { t in
                        calls.withLock { $0 += 1 }
                        return Complex(t, -0.3)
                    }, .constant(Complex(0.2, 0.1)),
                ],
                operators: [
                    storage == 0 ? .init(dense) : .init(csr), storage == 2 ? .init(csr) : .init(dense),
                ]))
        var prepared = _PreparedOperator(source, dimension: d, aggregateDense: aggregate)
        #expect(calls.withLock { $0 } == 0)
        let x = UniqueVector(copying: sparseFixtureState(d))
        let y = UniqueVector<Complex<Double>>.zero(d)
        let expected = UniqueVector<Complex<Double>>.zero(d)
        var matrix = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        for time in [0.0, 0.31, 0.31] {
            calls.withLock { $0 = 0 }
            prepared.updateChecked(at: time)
            for _ in 0..<17 { prepared.action.vector(x.components, into: y.components, adjoint: true) }
            #expect(calls.withLock { $0 } == 1)
            source.insert(t: time, into: &matrix)
            _MatrixActionView(matrix).vector(x.components, into: expected.components, adjoint: true)
            expectHOPSClose((0..<d).map { y[$0] }, (0..<d).map { expected[$0] })
        }
    }

    @Test(arguments: [false, true], [0, 1, 2])
    func constantExpansions(aggregate: Bool, storage: Int) {
        let (csr, dense) = explicitSparseFixture(5)
        let first = Complex(0.3, -0.2), second = Complex(-0.1, 0.7)
        let source = TimeDependentOperator.linearCombination(.init(
            coefficients: [.constant(first), .constant(second)],
            operators: [storage == 0 ? .init(dense) : .init(csr), storage == 2 ? .init(csr) : .init(dense)]))
        var prepared = _PreparedOperator(source, dimension: 5, aggregateDense: aggregate)
        let reference = _PreparedOperator(.constant(dense), dimension: 5)
        let x = UniqueVector(copying: sparseFixtureState(5))
        let y = UniqueVector<Complex<Double>>.zero(5)
        let expected = UniqueVector<Complex<Double>>.zero(5)
        for t in [0.0, 0.4] {
            prepared.updateChecked(at: t)
            prepared.action.vector(x.components, into: y.components)
            reference.action.vector(x.components, into: expected.components, scale: first + second)
            expectHOPSClose((0..<5).map { y[$0] }, (0..<5).map { expected[$0] })
        }
    }

    @Test func factorizedLossAndChangingCoefficient() {
        let (csr, dense) = explicitSparseFixture(5)
        let sparse = _PreparedMarkovianChannel(
            .init(rate: .constant(1), collapseOperator: .constant(csr)), dimension: 5)
        #expect(sparse.loss.rows == 0)
        let x = UniqueVector(copying: sparseFixtureState(5))
        let a = UniqueVector<Complex<Double>>.zero(5)
        let b = UniqueVector<Complex<Double>>.zero(5)
        let scratch = UniqueVector<Complex<Double>>.zero(5)
        var scaled = _PreparedMarkovianChannel(
            .init(
                rate: .constant(1),
                collapseOperator: .linearCombination(
                    .init(
                        coefficients: [.generated { Complex($0) }], operators: [.init(dense)]))), dimension: 5
        )
        for t in [0.0, 1.0, 0.2] {
            scaled.updateChecked(at: t)
            scaled.action.lossVector(x.components, into: a.components, scratch: scratch.components)
            sparse.action.lossVector(
                x.components, into: b.components, scratch: scratch.components, scale: Complex(t * t))
            expectHOPSClose((0..<5).map { a[$0] }, (0..<5).map { b[$0] })
        }
    }

    @Test func generatedDenseCanReplaceStorage() {
        var prepared = _PreparedOperator(
            .generatedDense(
                .init { t, output in
                    output = .zeros(rows: 3, columns: 3)
                    for i in 0..<3 { output[i, i] = Complex(t + Double(i), -0.1) }
                }), dimension: 3)
        let x = UniqueVector(copying: sparseFixtureState(3))
        let y = UniqueVector<Complex<Double>>.zero(3)
        for t in [0.0, 0.4, 0.7] {
            prepared.updateChecked(at: t)
            prepared.action.vector(x.components, into: y.components)
            expectHOPSClose((0..<3).map { y[$0] }, (0..<3).map { Complex(t + Double($0), -0.1) * x[$0] })
        }
    }
}
