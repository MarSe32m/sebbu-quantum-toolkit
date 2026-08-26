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
    protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions
        
        func solve<Hamiltonian: HamiltonianFunction>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws

		func solve<Hamiltonian: HamiltonianFunction>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws
	}
}

public extension GKSL.Implementation {
	@inlinable
	func solve<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegratorConfiguration>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) throws {
		fatalError("TODO: Implement")
	}
}

public extension GKSL {
	protocol TwoTimeCorrelationImplementation: Implementation {
		func solveTwoTimeCorrelation<
			Hamiltonian: HamiltonianFunction
		>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			request: TwoTimeCorrelationRequest,
			propagation: PropagationOptions<IntegratorConfiguration>,
			_ forEach: (
				Double,
				Complex<Double>
			) -> Void
		) throws
	}
}
