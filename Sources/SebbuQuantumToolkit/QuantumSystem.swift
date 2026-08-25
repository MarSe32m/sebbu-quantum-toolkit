// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct QuantumSystem<Hamiltonian: ~Copyable & HamiltonianFunction>: ~Copyable, Sendable {
	public let dimension: Int
	public let hamiltonian: Hamiltonian

	public init(dimension: Int, hamiltonian: consuming Hamiltonian) {
		self.dimension = dimension
		self.hamiltonian = hamiltonian
	}
}
