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
	public protocol Implementation: ~Copyable {
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

extension GKSL.Implementation {
	@inlinable
	public func solve<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegratorConfiguration>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) throws {
		let densityMatrixProblem = DensityMatrixProblem(problem)
		try solve(
			problem: densityMatrixProblem, configuration: configuration,
			propagation: propagation, forEach)
	}
}

extension GKSL {
	public protocol TwoTimeCorrelationImplementation: Implementation {
		/// Computes a two-time correlation using the quantum regression
		/// theorem and the same GKSL propagator as the density matrix.
		///
		/// Scheduled output times before `request.insertionTime` are skipped.
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

extension GKSL.TwoTimeCorrelationImplementation {
	@inlinable
	public func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: TwoTimeCorrelationRequest,
		propagation: PropagationOptions<IntegratorConfiguration>,
		_ forEach: (Double, Complex<Double>) -> Void
	) throws {
		let densityMatrixProblem = DensityMatrixProblem(problem)
		try solveTwoTimeCorrelation(
			problem: densityMatrixProblem,
			configuration: configuration,
			request: request,
			propagation: propagation,
			forEach
		)
	}
}
