// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum ScalarTimeFunction: Sendable {
    case constant(Double)
    case generated(@Sendable (Double) -> Double)
}

public enum TimeDependentOperator: Sendable {
    case constant(QuantumOperator)
    case linearCombination(OperatorExpansion)
    case generatedDense(DynamicDenseOperator)
}

public struct QuantumOperator: Sendable {
    @usableFromInline
    internal let matrix: Matrix<Complex<Double>>
    
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
    internal let coefficients: [ScalarTimeFunction]
    @usableFromInline
    internal let operators: [QuantumOperator]
    
    @inlinable
    public init(coefficients: [ScalarTimeFunction], operators: [QuantumOperator]) {
        self.coefficients = coefficients
        self.operators = operators
    }
}

public struct DynamicDenseOperator: Sendable {
    public typealias GeneratorFunction = @Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void
    
    @usableFromInline
    internal let generator: GeneratorFunction
 
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
