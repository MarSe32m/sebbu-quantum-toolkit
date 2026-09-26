// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum TimeFunction<Value: Sendable>: Sendable {
	case constant(Value)
	case generated(@Sendable (Double) -> Value)
    
	@inlinable
	@inline(always)
	public func callAsFunction(_ t: Double) -> Value {
		switch self {
		case .constant(let value):
			value
		case .generated(let function):
			function(t)
		}
	}
}

public typealias ScalarTimeFunction = TimeFunction<Double>
public typealias ComplexTimeFunction =
	TimeFunction<Complex<Double>>

public extension ScalarTimeFunction {
    @inlinable
    @inline(always)
    static func sin(frequency: Double) -> Self {
        .generated( { .sin(frequency * $0) })
    }
    
    @inlinable
    @inline(always)
    static func cos(frequency: Double) -> Self {
        .generated( { .cos(frequency * $0) })
    }
}

public extension ComplexTimeFunction {
    /// e^{i * `frequency` * t}
    @inlinable
    @inline(always)
    static func exp(frequency: Double) -> Self {
        .generated( { .init(length: 1, phase: $0 * frequency) })
    }
}

public enum TimeDependentOperator: Sendable {
	case constant(ConstantOperator)
	case linearCombination(OperatorExpansion)
	case generatedDense(DynamicDenseOperator)

	@inlinable
	package var isConstant: Bool {
		switch self {
		case .constant(_):
			return true
		case .linearCombination(let operatorExpansion):
			return operatorExpansion.isConstant
		case .generatedDense:
			return false
		}
	}
}

public struct ConstantOperator: Sendable {
    public let storage: Storage
    
    public enum Storage: Sendable {
        case dense(Matrix<Complex<Double>>)
        case sparse(CSRMatrix<Complex<Double>>)
    }
    
    @inlinable
    public init(_ matrix: Matrix<Complex<Double>>) {
        self.storage = .dense(matrix)
    }
    
    @inlinable
    public init(_ matrix: borrowing UniqueMatrix<Complex<Double>>) {
        self.storage = .dense(.init(copying: matrix))
    }
    
    @inlinable
    public init(_ sparse: CSRMatrix<Complex<Double>>) {
        self.storage = .sparse(sparse)
    }
    
    @inlinable
    public init(_ sparse: borrowing UniqueCSRMatrix<Complex<Double>>) {
        self.storage = .sparse(.init(copying: sparse))
    }
}

public extension TimeDependentOperator {
    @inlinable
    static func constant(_ matrix: Matrix<Complex<Double>>) -> Self {
        .constant(.init(matrix))
    }
    
    @inlinable
    static func constant(_ matrix: borrowing UniqueMatrix<Complex<Double>>) -> Self {
        .constant(.init(matrix))
    }
}

public struct OperatorExpansion: Sendable {
	@usableFromInline
	package let coefficients: [ComplexTimeFunction]
	@usableFromInline
	package let operators: [ConstantOperator]

	@usableFromInline
	package var isConstant: Bool {
		for coefficient in coefficients {
			if case .generated(_) = coefficient {
				return false
			}
		}
		return true
	}

	@inlinable
	public init(coefficients: [ComplexTimeFunction], operators: [ConstantOperator]) {
		precondition(
			coefficients.count == operators.count,
			"`coefficients` and `operators` must have the same count")
		precondition(!coefficients.isEmpty, "`coefficients` must not be empty")
		self.coefficients = coefficients
		self.operators = operators
	}
}

public struct DynamicDenseOperator: Sendable {
	public typealias GeneratorFunction =
		@Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void

	public let generator: GeneratorFunction

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

public extension TimeDependentOperator {
    @inlinable
    init(_ generator: @escaping DynamicDenseOperator.GeneratorFunction) {
        self = .generatedDense(.init(generator))
    }
}
