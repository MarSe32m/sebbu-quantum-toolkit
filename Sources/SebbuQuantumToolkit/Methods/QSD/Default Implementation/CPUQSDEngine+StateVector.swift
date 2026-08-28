// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
    @usableFromInline
    internal struct StateVector: ~Copyable, FixedStepSDESolverState {
        @usableFromInline
        internal var state: UniqueVector<Complex<Double>>
        
        @inlinable
        internal init(state: borrowing UniqueVector<Complex<Double>>) {
            self.state = .init(copying: state)
        }

        @inlinable
        internal init(state: Vector<Complex<Double>>) {
            self.state = .init(copying: state)
        }
        
        @inlinable
        internal init(dimension: Int) {
            self.state = .zero(dimension)
        }
        
        @inlinable
        @inline(always)
        mutating func zero() {
            state.zeroComponents()
        }
        
        @inlinable
        @inline(always)
        mutating func assign(_ other: borrowing StateVector) {
            state.copyComponents(from: other.state)
        }
        
        @inlinable
        @inline(always)
        mutating func add(_ other: borrowing StateVector, multiplied: Double) {
            state.add(other.state, multiplied: multiplied)
        }
        
        @inlinable
        @inline(always)
        mutating func assign(_ base: borrowing StateVector, adding direction: borrowing StateVector, multipliedBy coefficient: Double) {
            state.copyComponents(from: base.state, adding: direction.state, multiplied: coefficient)
        }
        
        @inlinable
        @inline(always)
        mutating func add(_ other: borrowing StateVector, scaledBy noise: borrowing Complex<Double>) {
            state.add(other.state, multiplied: noise)
        }
    }
}
