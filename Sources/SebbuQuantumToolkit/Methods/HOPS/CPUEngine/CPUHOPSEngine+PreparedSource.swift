// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// Span subscripts of copyable enums can copy their closure contexts.
	/// A noncopyable representation forces borrowing, including when the
	/// coefficient is a generated function rather than a constant.
	@usableFromInline
	internal enum PreparedTimeFunction<Value: Sendable>: ~Copyable, Sendable {
		case constant(Value)
		case generated(@Sendable (Double) -> Value)

		@inlinable
		init(_ source: TimeFunction<Value>) {
			switch source {
			case .constant(let value): self = .constant(value)
			case .generated(let function): self = .generated(function)
			}
		}

		@inlinable
		borrowing func callAsFunction(_ t: Double) -> Value {
			switch self {
			case .constant(let value): value
			case .generated(let function): function(t)
			}
		}
	}

	@usableFromInline
	internal enum PreparedSource: ~Copyable, Sendable {
		// Constant operators use the precomputed OperatorMatrices instead.
		case precomputed
		case generated(DynamicDenseOperator.GeneratorFunction)
		case expansion(UniqueArray<Term>)

		@usableFromInline
		internal struct Term: ~Copyable, Sendable {
			@usableFromInline let coefficient: PreparedTimeFunction<Complex<Double>>
			@usableFromInline let matrix: UniqueMatrix<Complex<Double>>

			@inlinable
			init(coefficient: ComplexTimeFunction, matrix: Matrix<Complex<Double>>) {
				self.coefficient = PreparedTimeFunction(coefficient)
				self.matrix = UniqueMatrix(copying: matrix)
			}

			@inlinable
			borrowing func insert(
				t: Double, first: Bool,
				into output: inout UniqueMatrix<Complex<Double>>
			) {
				precondition(
					output.rows == matrix.rows
						&& output.columns == matrix.columns,
					"Operator dimensions do not match the output buffer")
				let value = coefficient(t)
				if first {
					output.copyElements(from: matrix, multiplied: value)
				} else {
					output.add(matrix, multiplied: value)
				}
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
					terms.append(
						Term(
							coefficient: expansion.coefficients[i],
							matrix: expansion.operators[i].matrix))
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
