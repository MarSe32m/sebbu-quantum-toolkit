// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct PureStateProblem<Hamiltonian: HamiltonianFunction>: Sendable {
	public let initialState: Vector<Complex<Double>>
	public let system: QuantumSystem<Hamiltonian>
	public let markovianChannels: [MarkovianChannel]

	@inlinable
	public init(
		initialState: Vector<Complex<Double>>,
		system: QuantumSystem<Hamiltonian>,
		markovianChannels: [MarkovianChannel] = []
	) {
		precondition(
			initialState.count == system.dimension,
			"Initial state dimension does not match the quantum system"
		)
		self.initialState = initialState
		self.system = system
		self.markovianChannels = markovianChannels
	}
    
    @inlinable
    public init(
        initialState: borrowing UniqueVector<Complex<Double>>,
        system: QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel] = []
    ) {
        precondition(
            initialState.count == system.dimension,
            "Initial state dimension does not match the quantum system"
        )
        self.initialState = .init(copying: initialState)
        self.system = system
        self.markovianChannels = markovianChannels
    }
}

public struct DensityMatrixProblem<Hamiltonian: HamiltonianFunction>: Sendable {
	public let initialState: Matrix<Complex<Double>>
	public let system: QuantumSystem<Hamiltonian>
	public let markovianChannels: [MarkovianChannel]

	@inlinable
	public init(
		initialState: Matrix<Complex<Double>>,
		system: QuantumSystem<Hamiltonian>,
		markovianChannels: [MarkovianChannel] = []
	) {
		precondition(
			initialState.rows == system.dimension
				&& initialState.columns == system.dimension,
			"Initial density-matrix dimension does not match the quantum system"
		)

		self.initialState = initialState
		self.system = system
		self.markovianChannels = markovianChannels
	}
    
    @inlinable
    public init(
        initialState: borrowing UniqueMatrix<Complex<Double>>,
        system: QuantumSystem<Hamiltonian>,
        markovianChannels: [MarkovianChannel] = []
    ) {
        precondition(
            initialState.rows == system.dimension
                && initialState.columns == system.dimension,
            "Initial density-matrix dimension does not match the quantum system"
        )

        self.initialState = .init(copying: initialState)
        self.system = system
        self.markovianChannels = markovianChannels
    }
    
    @inlinable
    public init(_ pureStateProblem: borrowing PureStateProblem<Hamiltonian>) {
        let initialState = pureStateProblem.initialState.outer(pureStateProblem.initialState.conjugate)
        self.init(initialState: initialState, system: pureStateProblem.system, markovianChannels: pureStateProblem.markovianChannels)
    }
}
