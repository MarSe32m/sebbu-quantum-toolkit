// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public protocol HamiltonianFunction: ~Copyable, Sendable {
    func hamiltonian(t: Double, into: inout UniqueMatrix<Complex<Double>>)
}

public struct ClosureHamiltonian: ~Copyable, Sendable, HamiltonianFunction {
    @usableFromInline
    internal let closure: @Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void
    
    @inlinable
    public init(_ closure: @Sendable @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void) {
        self.closure = closure
    }
    
    @inlinable
    public func hamiltonian(t: Double, into: inout UniqueMatrix<Complex<Double>>) {
        closure(t, &into)
    }
}

public extension QuantumSystem where Hamiltonian == ClosureHamiltonian {
    @inlinable
    init(dimension: Int, hamiltonian: @Sendable @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void) {
        self.dimension = dimension
        self.hamiltonian = ClosureHamiltonian(hamiltonian)
    }
}

public struct TimeIndependentHamiltonian: ~Copyable, Sendable, HamiltonianFunction {
    @usableFromInline
    internal let _hamiltonian: UniqueMatrix<Complex<Double>>
    
    @inlinable
    public init(_ hamiltonian: borrowing UniqueMatrix<Complex<Double>>) {
        self._hamiltonian = .init(copying: hamiltonian)
    }
    
    @inlinable
    public init(_ hamiltonian: Matrix<Complex<Double>>) {
        self._hamiltonian = .init(copying: hamiltonian)
    }
    
    @inlinable
    @inline(always)
    public func hamiltonian(t: Double, into: inout UniqueMatrix<Complex<Double>>) {
        into.copyElements(from: _hamiltonian)
    }
}

public extension QuantumSystem where Hamiltonian == TimeIndependentHamiltonian {
    @inlinable
    init(_ hamiltonian: borrowing UniqueMatrix<Complex<Double>>) {
        precondition(hamiltonian.isSquare, "Hamiltonian operator must be a square matrix")
        self.dimension = hamiltonian.rows
        self.hamiltonian = TimeIndependentHamiltonian(hamiltonian)
    }
    
    @inlinable
    init(_ hamiltonian: Matrix<Complex<Double>>) {
        precondition(hamiltonian.isSquare, "Hamiltonian operator must be a square matrix")
        self.dimension = hamiltonian.rows
        self.hamiltonian = TimeIndependentHamiltonian(hamiltonian)
    }
}
