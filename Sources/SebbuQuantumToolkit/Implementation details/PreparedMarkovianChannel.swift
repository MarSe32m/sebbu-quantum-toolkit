// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

@usableFromInline
internal struct _PreparedConstantMarkovianChannelOperators: ~Copyable, Sendable {
    @usableFromInline
    let rate: ScalarTimeFunction
    
    @usableFromInline
    let C: UniqueMatrix<Complex<Double>>
    @usableFromInline
    let Cdagger: UniqueMatrix<Complex<Double>>
    @usableFromInline
    let CdaggerC: UniqueMatrix<Complex<Double>>
    
    @inlinable
    public init(_ C: borrowing UniqueMatrix<Complex<Double>>, rate: ScalarTimeFunction) {
        self.rate = rate
        let Cdagger = C.conjugateTranspose
        self.C = .init(copying: C)
        self.CdaggerC = Cdagger.dot(C)
        self.Cdagger = Cdagger
    }
    
    @inlinable
    public init?(_ channel: MarkovianChannel) {
        switch channel.collapseOperator {
        case .constant(let constantOperator):
            let C = UniqueMatrix(copying: constantOperator.matrix)
            self.init(C, rate: channel.rate)
        case .linearCombination(let operatorExpansion):
            if !operatorExpansion.isConstant { return nil }
            var C: UniqueMatrix<Complex<Double>>? = nil
            for (coefficient, op) in zip(operatorExpansion.coefficients, operatorExpansion.operators) {
                guard case .constant(let coefficient) = coefficient else {
                    preconditionFailure("Operator expansion contains non-constant coefficients")
                }
                if C == nil {
                    C = .init(copying: coefficient * op.matrix)
                } else {
                    C!.add(op.matrix, multiplied: coefficient)
                }
            }
            guard let C else { preconditionFailure("Operator expansion is empty") }
            self.init(C, rate: channel.rate)
        case .generatedDense(_):
            return nil
        }
    }
}

@usableFromInline
internal struct _PreparedDynamicMarkovianChannel: ~Copyable, Sendable {
    @usableFromInline
    let rate: ScalarTimeFunction
    
    @usableFromInline
    let _operator: _TimeDependentOperator
    
    @inlinable
    public init(_ channel: MarkovianChannel) {
        self.rate = channel.rate
        self._operator = .init(channel.collapseOperator)
    }
    
    @inlinable
    @inline(always)
    public func insert(t: Double, into: inout UniqueMatrix<Complex<Double>>) {
        _operator.insert(t: t, into: &into)
    }
}
