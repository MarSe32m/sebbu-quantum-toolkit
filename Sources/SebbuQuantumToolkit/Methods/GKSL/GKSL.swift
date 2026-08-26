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
        static func solve<Hamiltonian: HamiltonianFunction>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws

		static func solve<Hamiltonian: HamiltonianFunction>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) throws -> Void
		)
	}
}

public extension GKSL.Implementation {
	@inlinable
	static func solve<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) throws -> Void
	) {
		fatalError("TODO: Implement")
	}
}

public extension GKSL {
	protocol TwoTimeCorrelationImplementation: Implementation {
		static func solveTwoTimeCorrelation<
			Hamiltonian: HamiltonianFunction
		>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			request: TwoTimeCorrelationRequest,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				Complex<Double>
			) -> Void
		) throws
	}
}
