// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuScience
import Numerics


public struct PureStateProblem<
    Hamiltonian: HamiltonianFunction & ~Copyable
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
