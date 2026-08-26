// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct PureStateProblem<
	Hamiltonian: HamiltonianFunction
>: ~Copyable, Sendable {
	public let initialState: UniqueVector<Complex<Double>>
	public let system: QuantumSystem<Hamiltonian>
	public let markovianChannels: [MarkovianChannel]

	@inlinable
	public init(
		initialState: consuming UniqueVector<Complex<Double>>,
		system: consuming QuantumSystem<Hamiltonian>,
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
}

public struct DensityMatrixProblem<
	Hamiltonian: HamiltonianFunction
>: ~Copyable, Sendable {
	public let initialState: UniqueMatrix<Complex<Double>>
	public let system: QuantumSystem<Hamiltonian>
	public let markovianChannels: [MarkovianChannel]

	@inlinable
	public init(
		initialState: consuming UniqueMatrix<Complex<Double>>,
		system: consuming QuantumSystem<Hamiltonian>,
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
}
