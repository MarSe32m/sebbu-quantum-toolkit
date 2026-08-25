// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum GKSL {}

extension GKSL {
	public struct Configuration: Sendable {
		public init() {}
	}
}

extension GKSL {
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

extension GKSL.Implementation {
	@inlinable
	public static func solve<Hamiltonian>(
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
