// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    /// Immutable operator storage shared by ensemble workers. Unlike bathChannels,
    /// this list is never filtered by active hierarchy directions.
    @usableFromInline
    internal struct PotentialOperatorDefinition: ~Copyable, Sendable {
        @usableFromInline let storage: UniqueArray<_ConstantOperatorStorage>
        @usableFromInline let coefficients: UniqueArray<_PreparedTimeFunction<Complex<Double>>>
        @usableFromInline let generator: DynamicDenseOperator.GeneratorFunction?

        @inlinable
        init(_ source: TimeDependentOperator) {
            var storage = UniqueArray<_ConstantOperatorStorage>()
            var coefficients = UniqueArray<_PreparedTimeFunction<Complex<Double>>>()
            var generator: DynamicDenseOperator.GeneratorFunction?
            switch source {
            case .constant(let op):
                storage.append(.init(op))
                coefficients.append(.constant(.one))
            case .linearCombination(let expansion):
                for i in expansion.operators.indices {
                    storage.append(.init(expansion.operators[i]))
                    coefficients.append(.init(expansion.coefficients[i]))
                }
            case .generatedDense(let op): generator = op.generator
            }
            self.storage = storage
            self.coefficients = coefficients
            self.generator = generator
        }
    }

    /// Per-trajectory coefficient views and generated-dense buffer. Constant
    /// dense/CSR matrices stay in Preparation. Uses the same factorized action
    /// as _PreparedOperator, including conjugation of expansion coefficients.
    @usableFromInline
    internal struct PotentialOperatorState: ~Copyable {
        @usableFromInline var buffer: UniqueMatrix<Complex<Double>>
        @usableFromInline var views: UniqueVector<_OperatorTermView>

        @inlinable
        init(_ definition: borrowing PotentialOperatorDefinition, dimension: Int) {
            let generated = definition.generator != nil
            let buffer = UniqueMatrix<Complex<Double>>.zeros(
                rows: generated ? dimension : 0, columns: generated ? dimension : 0)
            var views: [_OperatorTermView] = []
            if generated {
                views.append(.init(matrix: .init(buffer), coefficient: .one))
            } else {
                for i in 0..<definition.storage.count {
                    views.append(.init(matrix: definition.storage[i].view,
                                       coefficient: definition.coefficients[i].constantValue(or: .zero)))
                }
            }
            self.buffer = buffer
            self.views = .init(views)
        }

        @inlinable
        mutating func update(_ definition: borrowing PotentialOperatorDefinition,
                             at time: Double, dimension: Int) {
            if let generator = definition.generator {
                generator(time, &buffer)
                precondition(buffer.rows == dimension && buffer.columns == dimension,
                             "Generated potential operator dimensions do not match the system.")
                views[0] = .init(matrix: .init(buffer), coefficient: .one)
            } else {
                for i in 0..<views.count { views[i].coefficient = definition.coefficients[i](time) }
            }
        }

        @inlinable
        var action: _OperatorAction { .init(terms: UnsafePointer(views.components), count: views.count) }
    }
}
