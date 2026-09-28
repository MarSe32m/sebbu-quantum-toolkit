// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension GKSL.CPUEngine {
    @usableFromInline
    internal struct GKSLRightHandSide: ~Copyable, ODERHSFunction {
        @usableFromInline var hamiltonian: _PreparedOperator
        @usableFromInline var channels: UniqueArray<_PreparedMarkovianChannel>
        @usableFromInline var temporary: UniqueMatrix<Complex<Double>>

        @inlinable init(_ problem: DensityMatrixProblem) {
            let d = problem.system.dimension
            hamiltonian = .init(problem.system.hamiltonian, dimension: d, aggregateDense: true)
            channels = _prepareMarkovianChannels(problem.markovianChannels, dimension: d)
            temporary = .zeros(rows: d, columns: d)
        }

        @inlinable mutating func evaluate(
            t: Double, y: borrowing DensityMatrixState, dy: inout DensityMatrixState
        ) {
            hamiltonian.updateChecked(at: t)
            hamiltonian.action.commutator(y.densityMatrix.elements, into: dy.densityMatrix.elements)
            for i in 0..<channels.count {
                let rate = channels[i].rate(t)
                Self.validate(rate: rate)
                if rate == 0 { continue }
                channels[i].updateChecked(at: t)
                channels[i].action.dissipator(
                    y.densityMatrix.elements, into: dy.densityMatrix.elements,
                    scratch: temporary.elements, rate: rate)
            }
        }

        @inlinable @inline(always)
        static func validate(rate: Double) {
            precondition(rate.isFinite && rate >= 0, "A GKSL channel rate must be finite and nonnegative")
        }
    }
}
