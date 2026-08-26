// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
#if swift(<6.5)
import BasicContainers
#else
#warning("Remove swift-collections dependency")
#endif

@usableFromInline
package enum _TimeDependentOperator: ~Copyable, Sendable {
	case constant(_ConstantOperator)
	case linearCombination(_OperatorExpansion)
	case generatedDense(_DynamicDenseOperator)
    
    @inlinable
    public init(_ timeDependentOperator: TimeDependentOperator) {
        switch timeDependentOperator {
        case .constant(let constantOperator):
            self = .constant(.init(constantOperator))
        case .linearCombination(let operatorExpansion):
            self = .linearCombination(.init(operatorExpansion))
        case .generatedDense(let dynamicDenseOperator):
            self = .generatedDense(.init(dynamicDenseOperator))
        }
    }
    
    @inlinable
    public func insert(t: Double, into: inout UniqueMatrix<Complex<Double>>) {
        switch self {
        case .constant(let op):
            into.copyElements(from: op.matrix)
        case .linearCombination(let op):
            for i in 0..<op.operators.count {
                if i == 0 {
                    into.copyElements(from: op.operators[i].matrix, multiplied: op.coefficients[i](t))
                } else {
                    into.add(op.operators[i].matrix, multiplied: op.coefficients[i](t))
                }
            }
        case .generatedDense(let op):
            op.generator(t, &into)
        }
    }
}

@usableFromInline
package struct _ConstantOperator: ~Copyable, Sendable {
	@usableFromInline
	package let matrix: UniqueMatrix<Complex<Double>>

    @inlinable
    public init(_ constantOperator: ConstantOperator) {
        self.matrix = .init(copying: constantOperator.matrix)
    }
    
	@inlinable
	public init(_ matrix: Matrix<Complex<Double>>) {
        self.matrix = .init(copying: matrix)
	}

	@inlinable
	public init(_ matrix: borrowing UniqueMatrix<Complex<Double>>) {
		self.matrix = .init(copying: matrix)
	}
}

@usableFromInline
package struct _OperatorExpansion: ~Copyable, Sendable {
	@usableFromInline
	package let coefficients: UniqueArray<ComplexTimeFunction>
	@usableFromInline
	package let operators: UniqueArray<_ConstantOperator>

    @usableFromInline
    package var isConstant: Bool {
        for i in coefficients.indices {
            let coefficient = coefficients[i]
            if case .generated(_) = coefficient {
                return false
            }
        }
        return true
    }
    
    @inlinable
    public init(_ operatorExpansion: OperatorExpansion) {
        self.init(coefficients: operatorExpansion.coefficients, operators: operatorExpansion.operators)
    }
    
	@inlinable
	public init(coefficients: [ComplexTimeFunction], operators: [ConstantOperator]) {
        precondition(coefficients.count == operators.count, "`coefficients` and `operators` must have the same count")
        precondition(!coefficients.isEmpty, "`coefficients` must not be empty")
        self.coefficients = .init(copying: coefficients)
        self.operators = .init(capacity: operators.count) { output in
            for op in operators {
                output.append(_ConstantOperator(op.matrix))
            }
        }
	}
}

@usableFromInline
package struct _DynamicDenseOperator: Sendable {
	public typealias GeneratorFunction =
		@Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void

	@usableFromInline
	package let generator: GeneratorFunction

    @inlinable
    public init(_ dynamicDenseOperator: DynamicDenseOperator) {
        self.generator = dynamicDenseOperator.generator
    }
    
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
