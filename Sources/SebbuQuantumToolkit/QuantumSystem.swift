// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct QuantumSystem: Sendable {
	public let dimension: Int
	public let hamiltonian: TimeDependentOperator

	@inlinable
	public init(dimension: Int, hamiltonian: TimeDependentOperator) {
		precondition(dimension > 0, "Quantum-system dimension must be positive")
		self.dimension = dimension
		self.hamiltonian = hamiltonian
	}
    
    @inlinable
    public init(_ hamiltonian: Matrix<Complex<Double>>) {
        precondition(hamiltonian.rows > 0 && hamiltonian.columns > 0, "Quantum-system dimension must be positive")
        precondition(hamiltonian.isSquare, "Quantum-system hamiltonian must be square")
        self.dimension = hamiltonian.rows
        self.hamiltonian = .constant(hamiltonian)
    }
    
    @inlinable
    public init(_ hamiltonian: borrowing UniqueMatrix<Complex<Double>>) {
        self.init(Matrix<Complex<Double>>(copying: hamiltonian))
    }
    
    @inlinable
    public init(dimension: Int, _ generator: @Sendable @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void) {
        self.dimension = dimension
        self.hamiltonian = .generatedDense(.init(generator))
    }
}
