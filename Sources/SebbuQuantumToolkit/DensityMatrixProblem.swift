// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuScience
import Numerics

public struct DensityMatrixProblem<
    Hamiltonian: HamiltonianFunction & ~Copyable
>: ~Copyable, Sendable {
    public let initialState: UniqueMatrix<Complex<Double>>
    public let system: QuantumSystem<Hamiltonian>
    public let markovianChannels: [MarkovianChannel]

    @inlinable
    public init(
        initialState: consuming UniqueMatrix<Complex<Double>>,
        system: consuming QuantumSystem<Hamiltonian>,
        markovianChannel: [MarkovianChannel] = []
    ) {
        precondition(
            initialState.rows == system.dimension &&
            initialState.columns == system.dimension,
            "Initial density-matrix dimension does not match the quantum system"
        )

        self.initialState = initialState
        self.system = system
        self.markovianChannels = markovianChannel
    }
}
