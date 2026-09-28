// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Sparse/mixed collapse operators retain C†(C x), avoiding loss-matrix fill-in.
@usableFromInline
internal struct _PreparedMarkovianChannel: ~Copyable {
    @usableFromInline
    let rate: _PreparedTimeFunction<Double>
    @usableFromInline
    var op: _PreparedOperator
    @usableFromInline
    var loss: UniqueMatrix<Complex<Double>>
    
    @inlinable
    @inline(always)
    var action: _CollapseAction {
        .init(op: op.action, loss: loss.rows == 0 ? nil : UnsafePointer(loss.elements))
    }

    @inlinable
    init(_ source: MarkovianChannel, dimension: Int) {
        rate = .init(source.rate)
        let op = _PreparedOperator(source.collapseOperator, dimension: dimension, aggregateDense: true)
        let dense = op.action.dense
        let loss = UniqueMatrix<Complex<Double>>.zeros(
            rows: dense == nil ? 0 : dimension,
            columns: dense == nil ? 0 : dimension)
        if op.isConstant, let dense {
            _DenseMatrixAction.product(
                dense, dense, dimension: dimension, adjointA: true,
                adding: false, into: loss.elements)
        }
        self.op = op
        self.loss = loss
    }

    @inlinable
    mutating func update(at time: Double) -> Bool {
        guard op.update(at: time) else { return false }
        if !op.isConstant && loss.rows > 0, let dense = op.action.dense {
            _DenseMatrixAction.product(
                dense, dense, dimension: op.dimension, adjointA: true,
                adding: false, into: loss.elements)
        }
        return true
    }

    @inlinable
    mutating func updateChecked(at time: Double) {
        let valid = update(at: time)
        precondition(valid, "Generated collapse dimensions do not match the system")
    }
}

@usableFromInline
internal struct _CollapseAction {
    @usableFromInline
    let op: _OperatorAction
    @usableFromInline
    let loss: UnsafePointer<Complex<Double>>?
    
    @inlinable
    init(op: _OperatorAction, loss: UnsafePointer<Complex<Double>>?) {
        self.op = op
        self.loss = loss
    }

    @inlinable
    @inline(always)
    func lossVector(
        _ state: UnsafePointer<Complex<Double>>,
        into output: UnsafeMutablePointer<Complex<Double>>,
        scratch: UnsafeMutablePointer<Complex<Double>>,
        scale: Complex<Double> = .one,
        adding: Bool = false,
        blas: Bool = false
    ) {
        if let loss {
            _MatrixActionView(dimension: op.dimension, values: loss).vector(
                state, into: output, scale: scale, adding: adding, blas: blas)
        } else {
            op.vector(state, into: scratch, blas: blas)
            op.vector(scratch, into: output, scale: scale, adding: adding, adjoint: true, blas: blas)
        }
    }

    @inlinable
    func dissipator(
        _ rho: UnsafePointer<Complex<Double>>,
        into output: UnsafeMutablePointer<Complex<Double>>,
        scratch: UnsafeMutablePointer<Complex<Double>>,
        rate: Double
    ) {
        op.matrix(rho, into: scratch, adding: false)
        op.matrix(scratch, into: output, right: true, adjoint: true, scale: Complex(rate))
        let half = Complex<Double>(-0.5 * rate)
        if let loss {
            _DenseMatrixAction.product(loss, rho, dimension: op.dimension, scale: half, into: output)
            _DenseMatrixAction.product(rho, loss, dimension: op.dimension, scale: half, into: output)
        } else {
            // C†(C rho), followed by (rho C†)C; never form C†C.
            op.matrix(scratch, into: output, adjoint: true, scale: half)
            op.matrix(rho, into: scratch, right: true, adjoint: true, adding: false)
            op.matrix(scratch, into: output, right: true, scale: half)
        }
    }
}

@inlinable
internal func _prepareMarkovianChannels(_ sources: [MarkovianChannel], dimension: Int) -> UniqueArray<
    _PreparedMarkovianChannel
> {
    var result = UniqueArray<_PreparedMarkovianChannel>(minimumCapacity: sources.count)
    for source in sources { result.append(.init(source, dimension: dimension)) }
    return result
}

/// Guide-conditioned coefficients reused unchanged by every companion/tier.
@usableFromInline
internal struct _DiffusiveAction {
    @usableFromInline let collapse: _CollapseAction
    @usableFromInline let rate: Double
    @usableFromInline let mean: Complex<Double>
    @usableFromInline let scalar: Double
    @usableFromInline let normalized: Bool

    @inlinable init(
        collapse: _CollapseAction,
        rate: Double,
        nonlinear: Bool,
        normalized: Bool,
        guide: UnsafePointer<Complex<Double>>,
        norm: Double,
        scratch: UnsafeMutablePointer<Complex<Double>>
    ) {
        self.collapse = collapse
        self.rate = rate
        self.normalized = normalized
        if nonlinear {
            let mean = collapse.op.expectation(guide, scratch: scratch) / norm
            self.mean = mean
            if normalized {
                var lossMean = 0.0
                for i in 0..<collapse.op.dimension { lossMean += scratch[i].lengthSquared }
                scalar = rate * (0.5 * lossMean / norm - mean.lengthSquared)
            } else {
                scalar = 0
            }
        } else {
            mean = .zero
            scalar = 0
        }
    }

    /// `applied` may borrow C*state computed while obtaining the guide mean.
    @inlinable
    @inline(always)
    func drift(
        _ state: UnsafePointer<Complex<Double>>,
        into output: UnsafeMutablePointer<Complex<Double>>,
        scratch: UnsafeMutablePointer<Complex<Double>>,
        applied: UnsafePointer<Complex<Double>>? = nil
    ) {
        if let applied {
            if let loss = collapse.loss {
                _MatrixActionView(dimension: collapse.op.dimension, values: loss).vector(
                    state, into: output, scale: Complex(-0.5 * rate), adding: true)
            } else {
                collapse.op.vector(applied, into: output, scale: Complex(-0.5 * rate), adding: true, adjoint: true)
            }
            let coefficient = rate * mean.conjugate
            if coefficient != .zero {
                for i in 0..<collapse.op.dimension { output[i] += coefficient * applied[i] }
            }
        } else {
            collapse.lossVector(state, into: output, scratch: scratch, scale: Complex(-0.5 * rate), adding: true)
            if mean != .zero {
                collapse.op.vector(state, into: output, scale: rate * mean.conjugate, adding: true)
            }
        }
        if scalar != 0 {
            for i in 0..<collapse.op.dimension { output[i] += scalar * state[i] }
        }
    }

    @inlinable
    func diffusion(
        _ state: UnsafePointer<Complex<Double>>,
        into output: UnsafeMutablePointer<Complex<Double>>,
        applied: UnsafePointer<Complex<Double>>? = nil
    ) {
        let coefficient = (0.5 * rate).squareRoot()
        if let applied {
            for i in 0..<collapse.op.dimension { output[i] = coefficient * applied[i] }
        } else {
            collapse.op.vector(state, into: output, scale: Complex(coefficient))
        }
        if normalized {
            for i in 0..<collapse.op.dimension { output[i] -= coefficient * mean * state[i] }
        }
    }

    /// Dense HOPS channels keep the single common-generator action. Sparse
    /// channels use factorized actions on every hierarchy row instead.
    @inlinable
    func addDenseGenerator(into output: UnsafeMutablePointer<Complex<Double>>) -> Bool {
        guard let matrix = collapse.op.dense, let loss = collapse.loss else { return false }
        let d = collapse.op.dimension
        for i in 0..<d * d { output[i] += (-0.5 * rate) * loss[i] + (rate * mean.conjugate) * matrix[i] }
        if scalar != 0 { for i in 0..<d { output[i * d + i] += Complex(scalar) } }
        return true
    }
}
