// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum TimeFunction<Value: Sendable>: Sendable {
	case constant(Value)
	case generated(@Sendable (Double) -> Value)
}

public typealias ScalarTimeFunction = TimeFunction<Double>
public typealias ComplexTimeFunction =
	TimeFunction<Complex<Double>>

public enum TimeDependentOperator: Sendable {
	case constant(ConstantOperator)
	case linearCombination(OperatorExpansion)
	case generatedDense(DynamicDenseOperator)
}

public struct ConstantOperator: Sendable {
	@usableFromInline
	package let matrix: Matrix<Complex<Double>>

	@inlinable
	public init(_ matrix: Matrix<Complex<Double>>) {
		self.matrix = matrix
	}

	@inlinable
	public init(_ matrix: borrowing UniqueMatrix<Complex<Double>>) {
		self.matrix = .init(copying: matrix)
	}
}

public struct OperatorExpansion: Sendable {
	@usableFromInline
	package let coefficients: [ComplexTimeFunction]
	@usableFromInline
	package let operators: [ConstantOperator]

	@inlinable
	public init(coefficients: [ComplexTimeFunction], operators: [ConstantOperator]) {
		self.coefficients = coefficients
		self.operators = operators
	}
}

public struct DynamicDenseOperator: Sendable {
	public typealias GeneratorFunction =
		@Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void

	@usableFromInline
	package let generator: GeneratorFunction

	@inlinable
	public init(_ generator: @escaping GeneratorFunction) {
		self.generator = generator
	}

	@inlinable
	public init(_ generator: @escaping @Sendable (Double) -> Matrix<Complex<Double>>) {
		self.generator = { t, output in
			let matrix = generator(t)
			output.copyElements(from: matrix)
		}
	}
}
