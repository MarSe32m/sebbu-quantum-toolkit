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

public extension GKSL {
	protocol Implementation {
        static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		)

		static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
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
	static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) {
		fatalError("TODO: Implement")
	}
}

public extension GKSL {
	protocol TwoTimeCorrelationImplementation: Implementation {
		static func solveTwoTimeCorrelation<
			Hamiltonian: HamiltonianFunction & ~Copyable
		>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			request: TwoTimeCorrelationRequest,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				Complex<Double>
			) -> Void
		)
	}
}
