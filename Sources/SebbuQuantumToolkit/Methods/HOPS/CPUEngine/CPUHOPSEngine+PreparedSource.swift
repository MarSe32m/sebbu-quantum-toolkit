// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    @usableFromInline
    internal typealias PreparedTimeFunction<Value: Sendable> = _PreparedTimeFunction<Value>

	@usableFromInline
	internal enum PreparedSource: ~Copyable, Sendable {
		// Constant bath operators use the immutable preparation storage.
		case precomputed
		case generated(DynamicDenseOperator.GeneratorFunction)
		case expansion(UniqueArray<Term>)

		@usableFromInline
		internal struct Term: ~Copyable, Sendable {
			@usableFromInline let coefficient: PreparedTimeFunction<Complex<Double>>
			@usableFromInline let matrix: _ConstantOperatorStorage

			@inlinable
			init(coefficient: ComplexTimeFunction, matrix: ConstantOperator) {
				self.coefficient = PreparedTimeFunction(coefficient)
				self.matrix = _ConstantOperatorStorage(matrix)
			}

			@inlinable
			borrowing func insert(
				t: Double, first: Bool,
				into output: inout UniqueMatrix<Complex<Double>>
			) {
                precondition(output.rows == matrix.view.dimension && output.columns == matrix.view.dimension)
                if first { output.zeroElements() }
                matrix.view.add(into: output.elements, scale: coefficient(t))
			}
		}

		@inlinable
		init(_ source: TimeDependentOperator) {
			if source.isConstant {
				self = .precomputed
				return
			}
			switch source {
			case .constant:
				preconditionFailure("A constant operator must be precomputed")
			case .generatedDense(let dynamic):
				self = .generated(dynamic.generator)
			case .linearCombination(let expansion):
				var terms = UniqueArray<Term>(
					minimumCapacity: expansion.operators.count)
				for i in expansion.operators.indices {
                    terms.append(Term(coefficient: expansion.coefficients[i], matrix: expansion.operators[i]))
				}
				self = .expansion(terms)
			}
		}

		@inlinable
		borrowing func insert(t: Double, into output: inout UniqueMatrix<Complex<Double>>) {
			switch self {
			case .precomputed:
				preconditionFailure("Use the precomputed operator matrices")
			case .generated(let function):
				function(t, &output)
			case .expansion(let terms):
				for i in 0..<terms.count {
					terms[i].insert(t: t, first: i == 0, into: &output)
				}
			}
		}
	}
}
