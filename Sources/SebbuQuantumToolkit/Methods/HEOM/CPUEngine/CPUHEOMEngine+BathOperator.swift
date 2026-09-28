// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    /// One latent combination (Lambda_p or M_p). Dense contributions share a
    /// single GEMM buffer; sparse contributions borrow the prepared source CSR
    /// storage. No dense buffer is allocated for an entirely sparse combination.
    /// The source owners must outlive this value and all published actions.
    @usableFromInline
    internal struct BathOperator: ~Copyable {
        @usableFromInline
        internal struct Contribution {
            @usableFromInline let source: Int
            @usableFromInline let term: Int
            @usableFromInline let weight: Complex<Double>
            /// A negative destination denotes the aggregated dense buffer.
            @usableFromInline let destination: Int

            @inlinable
            init(source: Int, term: Int, weight: Complex<Double>, destination: Int) {
                self.source = source
                self.term = term
                self.weight = weight
                self.destination = destination
            }
        }

        @usableFromInline let contributions: UniqueVector<Contribution>
        @usableFromInline let isConstant: Bool
        @usableFromInline var dense: UniqueMatrix<Complex<Double>>
        @usableFromInline var terms: UniqueVector<_OperatorTermView>

        @inlinable
        init(sources: borrowing UniqueArray<_PreparedOperator>, weights: [Complex<Double>], dimension: Int) {
            precondition(sources.count == weights.count)
            var hasDense = false
            var constant = true
            for i in 0..<sources.count where weights[i] != .zero {
                constant = constant && sources[i].isConstant
                let action = sources[i].action
                for j in 0..<action.count where !action.terms[j].matrix.isSparse { hasDense = true }
            }
            let buffer = UniqueMatrix<Complex<Double>>.zeros(
                rows: hasDense ? dimension : 0, columns: hasDense ? dimension : 0)
            var views: [_OperatorTermView] = []
            if hasDense { views.append(.init(matrix: .init(buffer), coefficient: .one)) }
            var bindings: [Contribution] = []
            for i in 0..<sources.count where weights[i] != .zero {
                let action = sources[i].action
                for j in 0..<action.count {
                    let term = action.terms[j]
                    let destination = term.matrix.isSparse ? views.count : -1
                    bindings.append(.init(source: i, term: j, weight: weights[i], destination: destination))
                    if destination >= 0 {
                        views.append(.init(matrix: term.matrix, coefficient: weights[i] * term.coefficient))
                    } else {
                        term.matrix.add(into: buffer.elements, scale: weights[i] * term.coefficient)
                    }
                }
            }
            contributions = .init(bindings)
            isConstant = constant
            dense = buffer
            terms = .init(views)
        }

        @inlinable
        var action: _OperatorAction {
            .init(terms: UnsafePointer(terms.components), count: terms.count)
        }

        /// The centered sparse correction is applied directly to each ADO;
        /// the dense part is folded into the shared effective Hamiltonians.
        @inlinable
        var sparseAction: _OperatorAction {
            let offset = dense.rows == 0 ? 0 : 1
            return .init(terms: UnsafePointer(terms.components) + offset, count: terms.count - offset)
        }

        /// Sources have already evaluated their coefficients/generators once
        /// for this stage. Refresh values only, with no allocation or CSR copy.
        @inlinable
        mutating func update(sources: borrowing UniqueArray<_PreparedOperator>) {
            if isConstant { return }
            dense.zeroElements()
            for i in 0..<contributions.count {
                let binding = contributions[i]
                let term = sources[binding.source].action.terms[binding.term]
                let coefficient = binding.weight * term.coefficient
                if binding.destination >= 0 {
                    terms[binding.destination] = .init(matrix: term.matrix, coefficient: coefficient)
                } else {
                    term.matrix.add(into: dense.elements, scale: coefficient)
                }
            }
        }
    }

    @usableFromInline
    internal struct BathView {
        @usableFromInline let lambda: _OperatorAction
        @usableFromInline let memory: _OperatorAction
        @usableFromInline let sparseLambda: _OperatorAction

        @inlinable
        init(lambda: borrowing BathOperator, memory: borrowing BathOperator) {
            self.lambda = lambda.action
            self.memory = memory.action
            self.sparseLambda = lambda.sparseAction
        }
    }
}
