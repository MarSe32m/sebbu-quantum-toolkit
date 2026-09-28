// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

/// Borrowed row-major storage. The immutable owner must outlive each action.
/// No COW containers or closure contexts enter the hierarchy kernels.
@usableFromInline
internal struct _MatrixActionView {
    @usableFromInline let dimension: Int
    @usableFromInline let values: UnsafePointer<Complex<Double>>?
    @usableFromInline let rows: UnsafePointer<Int>?
    @usableFromInline let columns: UnsafePointer<Int>?

    @inlinable init(dimension: Int, values: UnsafePointer<Complex<Double>>) {
        self.dimension = dimension
        self.values = values
        rows = nil
        columns = nil
    }

    @inlinable init(_ matrix: borrowing UniqueMatrix<Complex<Double>>) {
        self.init(dimension: matrix.rows, values: UnsafePointer(matrix.elements))
    }

    @inlinable init(_ matrix: borrowing UniqueCSRMatrix<Complex<Double>>) {
        dimension = matrix.rows
        // Unique storage stays at these addresses until its owner is mutated.
        // Empty CSR values/columns may be nil and are never dereferenced.
        values = matrix.withValues { $0.span.withUnsafeBufferPointer { $0.baseAddress } }
        rows = matrix.withRowIndices { $0.span.withUnsafeBufferPointer { $0.baseAddress! } }
        columns = matrix.withColumnIndices { $0.span.withUnsafeBufferPointer { $0.baseAddress } }
    }

    @inlinable var isSparse: Bool { rows != nil }

    @inlinable @inline(always)
    func vector(
        _ x: UnsafePointer<Complex<Double>>, into y: UnsafeMutablePointer<Complex<Double>>,
        scale: Complex<Double> = .one, adding: Bool = false, adjoint: Bool = false,
        blas: Bool = false
    ) {
        if scale == .zero && adding { return }
        //TODO: Return early when scale == .zero && !adding
        let d = dimension
        if let rows {
            if adjoint {
                if !adding { for i in 0..<d { y[i] = .zero } }
                for i in 0..<d {
                    for k in rows[i]..<rows[i + 1] { y[columns![k]] += scale * values![k].conjugate * x[i] }
                }
            } else {
                for i in 0..<d {
                    var sum = Complex<Double>.zero
                    for k in rows[i]..<rows[i + 1] { sum += values![k] * x[columns![k]] }
                    y[i] = (adding ? y[i] : .zero) + scale * sum
                }
            }
        } else if d == 2 {
            // Keep TLS actions scalar and unrolled, including adjoints.
            let a = values!
            let v0 =
                (adjoint ? a[0].conjugate : a[0]) * x[0]
                + (adjoint ? a[2].conjugate : a[1]) * x[1]
            let v1 =
                (adjoint ? a[1].conjugate : a[2]) * x[0]
                + (adjoint ? a[3].conjugate : a[3]) * x[1]
            y[0] = (adding ? y[0] : .zero) + scale * v0
            y[1] = (adding ? y[1] : .zero) + scale * v1
        } else if blas && d > 4 {
            BLAS.zgemv(
                layout: .rowMajor, transpose: adjoint ? .conjugateTranspose : .noTranspose,
                m: d, n: d, alpha: scale, a: values!, lda: d, x: x, incX: 1,
                beta: adding ? .one : .zero, y: y, incY: 1)
        } else {
            for i in 0..<d {
                var sum = Complex<Double>.zero
                for j in 0..<d {
                    sum += (adjoint ? values![j * d + i].conjugate : values![i * d + j]) * x[j]
                }
                y[i] = (adding ? y[i] : .zero) + scale * sum
            }
        }
    }

    /// rho*A scatters CSR row k into output column j. This uses A, never
    /// A†, unless explicitly requested, including for non-Hermitian H.
    @inlinable @inline(always)
    func matrix(
        _ rho: UnsafePointer<Complex<Double>>, into output: UnsafeMutablePointer<Complex<Double>>,
        right: Bool = false, adjoint: Bool = false,
        scale: Complex<Double> = .one, adding: Bool = true
    ) {
        if scale == .zero && adding { return }
        //TODO: Return early when scale == .zero && !adding
        let d = dimension
        guard let rows else {
            _DenseMatrixAction.product(
                right ? rho : values!, right ? values! : rho,
                dimension: d, adjointA: !right && adjoint, adjointB: right && adjoint,
                scale: scale, adding: adding, into: output)
            return
        }
        if !adding { for i in 0..<d * d { output[i] = .zero } }
        for row in 0..<d {
            for k in rows[row]..<rows[row + 1] {
                let column = columns![k]
                let i = adjoint ? column : row
                let j = adjoint ? row : column
                let value = scale * (adjoint ? values![k].conjugate : values![k])
                if right {
                    for r in 0..<d { output[r &* d &+ j] += rho[r &* d &+ i] * value }
                } else {
                    for c in 0..<d { output[i &* d &+ c] += value * rho[j &* d &+ c] }
                }
            }
        }
    }

    @inlinable
    func add(into output: UnsafeMutablePointer<Complex<Double>>, scale: Complex<Double> = .one) {
        if scale == .zero { return }
        if let rows {
            for i in 0..<dimension {
                for k in rows[i]..<rows[i + 1] { output[i &* dimension &+ columns![k]] += scale * values![k] }
            }
        } else {
            for i in 0..<dimension * dimension { output[i] += scale * values![i] }
        }
    }

    /// Tr(A*rho), without forming a matrix product. In particular, this is a
    /// transpose contraction, not a Hilbert-Schmidt (adjoint) inner product.
    @inlinable
    func traceProduct(_ rho: UnsafePointer<Complex<Double>>) -> Complex<Double> {
        var result = Complex<Double>.zero
        let d = dimension
        for i in 0..<d {
            if let rows {
                for k in rows[i]..<rows[i + 1] {
                    result += values![k] * rho[columns![k] &* d &+ i]
                }
            } else {
                for j in 0..<d { result += values![i &* d &+ j] * rho[j &* d &+ i] }
            }
        }
        return result
    }
}

@usableFromInline
internal struct _OperatorTermView {
    @usableFromInline let matrix: _MatrixActionView
    @usableFromInline var coefficient: Complex<Double>
    @inlinable init(matrix: _MatrixActionView, coefficient: Complex<Double>) {
        self.matrix = matrix
        self.coefficient = coefficient
    }
}

/// Stage data borrowed from _PreparedOperator, valid until its next update or
/// destruction. HEOM publishes this POD view, then joins all workers before
/// mutating any backing storage. Coefficients are already evaluated.
@usableFromInline
internal struct _OperatorAction {
    @usableFromInline let terms: UnsafePointer<_OperatorTermView>
    @usableFromInline let count: Int
    @inlinable init(terms: UnsafePointer<_OperatorTermView>, count: Int) {
        self.terms = terms
        self.count = count
    }
    @inlinable var dimension: Int { terms[0].matrix.dimension }
    @inlinable var dense: UnsafePointer<Complex<Double>>? {
        count == 1 && !terms[0].matrix.isSparse && terms[0].coefficient == .one ? terms[0].matrix.values : nil
    }

    @inlinable @inline(always)
    func vector(
        _ x: UnsafePointer<Complex<Double>>, into y: UnsafeMutablePointer<Complex<Double>>,
        scale: Complex<Double> = .one, adding: Bool = false, adjoint: Bool = false,
        blas: Bool = false
    ) {
        for i in 0..<count {
            let c = adjoint ? terms[i].coefficient.conjugate : terms[i].coefficient
            terms[i].matrix.vector(
                x, into: y, scale: scale * c, adding: adding || i > 0,
                adjoint: adjoint, blas: blas)
        }
    }

    @inlinable @inline(always)
    func matrix(
        _ rho: UnsafePointer<Complex<Double>>, into output: UnsafeMutablePointer<Complex<Double>>,
        right: Bool = false, adjoint: Bool = false,
        scale: Complex<Double> = .one, adding: Bool = true
    ) {
        for i in 0..<count {
            let c = adjoint ? terms[i].coefficient.conjugate : terms[i].coefficient
            terms[i].matrix.matrix(
                rho, into: output, right: right, adjoint: adjoint,
                scale: scale * c, adding: adding || i > 0)
        }
    }

    @inlinable
    func commutator(
        _ rho: UnsafePointer<Complex<Double>>, into output: UnsafeMutablePointer<Complex<Double>>,
        scale: Complex<Double> = -.i, adding: Bool = false, adjoint: Bool = false
    ) {
        matrix(rho, into: output, adjoint: adjoint, scale: scale, adding: adding)
        matrix(rho, into: output, right: true, adjoint: adjoint, scale: -scale)
    }

    @inlinable
    @inline(always)
    func add(into output: UnsafeMutablePointer<Complex<Double>>, scale: Complex<Double> = .one) {
        for i in 0..<count { terms[i].matrix.add(into: output, scale: scale * terms[i].coefficient) }
    }

    @inlinable
    @inline(always)
    func traceProduct(_ rho: UnsafePointer<Complex<Double>>) -> Complex<Double> {
        var result = Complex<Double>.zero
        for i in 0..<count {
            result += terms[i].coefficient * terms[i].matrix.traceProduct(rho)
        }
        return result
    }

    @inlinable
    @inline(always)
    func expectation(_ x: UnsafePointer<Complex<Double>>, scratch: UnsafeMutablePointer<Complex<Double>>)
        -> Complex<Double>
    {
        vector(x, into: scratch)
        var result = Complex<Double>.zero
        for i in 0..<dimension { result += x[i].conjugate * scratch[i] }
        return result
    }
}

@usableFromInline
internal enum _ConstantOperatorStorage: ~Copyable, Sendable {
    case dense(UniqueMatrix<Complex<Double>>)
    case sparse(UniqueCSRMatrix<Complex<Double>>)

    @inlinable
    internal init(_ source: ConstantOperator) {
        switch source.storage {
        case .dense(let matrix): self = .dense(.init(copying: matrix))
        case .sparse(let matrix): self = .sparse(.init(copying: matrix))
        }
    }

    @inlinable
    @inline(always)
    var view: _MatrixActionView {
        switch self {
        case .dense(let matrix): _MatrixActionView(matrix)
        case .sparse(let matrix): _MatrixActionView(matrix)
        }
    }
}

/// Constants are copied only during preparation. Mixed expansions retain CSR
/// terms; density methods aggregate dense terms into one reusable GEMM buffer.
@usableFromInline
internal struct _PreparedOperator: ~Copyable {
    @usableFromInline let dimension: Int
    @usableFromInline let isConstant: Bool
    @usableFromInline let storage: UniqueArray<_ConstantOperatorStorage>
    @usableFromInline let coefficients: UniqueArray<_PreparedTimeFunction<Complex<Double>>>
    @usableFromInline let generator: DynamicDenseOperator.GeneratorFunction?
    @usableFromInline let aggregateDense: Bool
    @usableFromInline var buffer: UniqueMatrix<Complex<Double>>
    @usableFromInline var views: UniqueVector<_OperatorTermView>

    @inlinable
    internal init(_ source: TimeDependentOperator, dimension: Int, aggregateDense: Bool = false) {
        precondition(
            source.firstDimensionMismatch(expected: dimension) == nil,
            "Operator dimensions do not match the system")
        self.dimension = dimension
        self.isConstant = source.isConstant
        var storage = UniqueArray<_ConstantOperatorStorage>()
        var coefficients = UniqueArray<_PreparedTimeFunction<Complex<Double>>>()
        var generator: DynamicDenseOperator.GeneratorFunction?
        switch source {
        case .constant(let op):
            storage.append(_ConstantOperatorStorage(op))
            coefficients.append(.constant(.one))
        case .linearCombination(let expansion):
            for i in expansion.operators.indices {
                storage.append(_ConstantOperatorStorage(expansion.operators[i]))
                coefficients.append(.init(expansion.coefficients[i]))
            }
        case .generatedDense(let op): generator = op.generator
        }
        var denseCount = 0
        for i in 0..<storage.count { if !storage[i].view.isSparse { denseCount += 1 } }
        // A one-term dense expansion also benefits from a staged dense buffer:
        // collapse preparation can then retain its precomputed loss path even
        // when the scalar coefficient changes. Constants need no stage copy.
        let singleDenseExpansion: Bool
        if case .linearCombination = source {
            singleDenseExpansion = storage.count == 1 && denseCount == 1
        } else {
            singleDenseExpansion = false
        }
        let aggregate = aggregateDense && (denseCount > 1 || singleDenseExpansion)
        self.aggregateDense = aggregate
        let buffer = UniqueMatrix<Complex<Double>>.zeros(
            rows: generator != nil || aggregate ? dimension : 0,
            columns: generator != nil || aggregate ? dimension : 0)
        var initialViews: [_OperatorTermView] = []
        if generator != nil || aggregate {
            initialViews.append(.init(matrix: .init(buffer), coefficient: .one))
        }
        for i in 0..<storage.count {
            let view = storage[i].view
            // Generated coefficients are deliberately not called during setup.
            let value: Complex<Double>
            switch coefficients[i] {
            case .constant(let c): value = c
            case .generated: value = .zero
            }
            if aggregate && !view.isSparse {
                view.add(into: buffer.elements, scale: value)
            } else {
                initialViews.append(.init(matrix: view, coefficient: value))
            }
        }
        self.storage = storage
        self.coefficients = coefficients
        self.generator = generator
        self.buffer = buffer
        self.views = .init(initialViews)
    }

    @inlinable
    @inline(always)
    var action: _OperatorAction {
        .init(terms: UnsafePointer(views.components), count: views.count)
    }

    /// HEOM reports malformed generated dimensions via its failure slot.
    @inlinable
    mutating func update(at time: Double) -> Bool {
        if isConstant { return true }
        if let generator {
            generator(time, &buffer)
            guard buffer.rows == dimension && buffer.columns == dimension else { return false }
            views[0] = .init(matrix: .init(buffer), coefficient: .one)
            return true
        }
        if aggregateDense { buffer.zeroElements() }
        var index = aggregateDense ? 1 : 0
        for i in 0..<storage.count {
            let value = coefficients[i](time)
            let view = storage[i].view
            if aggregateDense && !view.isSparse {
                view.add(into: buffer.elements, scale: value)
            } else {
                views[index].coefficient = value
                index += 1
            }
        }
        return true
    }

    @inlinable
    @inline(always)
    mutating func updateChecked(at time: Double) {
        let valid = update(at: time)
        precondition(valid, "Generated operator dimensions do not match the system")
    }
}
