// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum GKSL {}

public extension GKSL {
    struct Configuration: Sendable {
        public init() {}
    }
}

public extension GKSL {
    protocol Implementation {
        static func solve<Hamiltonian>(
            problem: borrowing DensityMatrixProblem<Hamiltonian>,
            configuration: GKSL.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            _ forEach: (
                Double,
                borrowing UniqueMatrix<Complex<Double>>
            ) -> Void
        )

        static func solve<Hamiltonian>(
            problem: borrowing PureStateProblem<Hamiltonian>,
            configuration: GKSL.Configuration,
            propagation: PropagationOptions<IntegrationOptions>,
            _ forEach: (
                Double,
                borrowing UniqueMatrix<Complex<Double>>
            ) -> Void
        )
    }
}

public extension GKSL.Implementation {
    @inlinable
    static func solve<Hamiltonian>(
        problem: borrowing PureStateProblem<Hamiltonian>,
        configuration: GKSL.Configuration = .init(),
        propagation: PropagationOptions<IntegrationOptions>,
        _ forEach: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> Void
    )
    where Hamiltonian: HamiltonianFunction & ~Copyable {
        fatalError("TODO: Implement")
    }
}
