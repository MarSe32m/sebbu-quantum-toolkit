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
        
        @discardableResult
        func solve<Hamiltonian: HamiltonianFunction>(
            problem: borrowing DensityMatrixProblem<Hamiltonian>,
            configuration: GKSL.Configuration,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (
                Double,
                borrowing UniqueMatrix<Complex<Double>>
            ) -> PropagationControl
        ) throws -> PropagationRunSummary
	}
}

extension GKSL.Implementation {
    @inlinable
    @discardableResult
    public func solve<Hamiltonian: HamiltonianFunction>(
        problem: borrowing PureStateProblem<Hamiltonian>,
        configuration: GKSL.Configuration = .init(),
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let densityMatrixProblem = DensityMatrixProblem(problem)
        return try solve(
            problem: densityMatrixProblem,
            configuration: configuration,
            propagation: propagation,
            observing: observer
        )
    }
}

extension GKSL {
	public protocol TwoTimeCorrelationImplementation: Implementation {
        /// Computes a two-time correlation using the quantum regression
        /// theorem and the same GKSL propagator as the density matrix.
        ///
        /// Scheduled output times before `request.insertionTime` are skipped.
        @discardableResult
        func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
            problem: borrowing DensityMatrixProblem<Hamiltonian>,
            configuration: GKSL.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> PropagationRunSummary
	}
}

extension GKSL.TwoTimeCorrelationImplementation {
    @inlinable
    @discardableResult
    public func solveTwoTimeCorrelation<
        Hamiltonian: HamiltonianFunction
    >(
        problem: borrowing PureStateProblem<Hamiltonian>,
        configuration: GKSL.Configuration = .init(),
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (
            Double,
            Complex<Double>
        ) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let densityMatrixProblem = DensityMatrixProblem(problem)
        return try solveTwoTimeCorrelation(
            problem: densityMatrixProblem,
            configuration: configuration,
            request: request,
            propagation: propagation,
            observing: observer
        )
	}
}

extension GKSL {
	public protocol MultiTimeOrderedCorrelationImplementation: Implementation {
		@discardableResult
		func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: GKSL.Configuration,
			request: MultiTimeOrderedCorrelationRequest,
			propagation: PropagationOptions<IntegratorConfiguration>,
			observing observer: (Double, Complex<Double>) -> PropagationControl
		) throws -> PropagationRunSummary
	}
}

extension GKSL.MultiTimeOrderedCorrelationImplementation {
	@discardableResult
	public func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		request: MultiTimeOrderedCorrelationRequest,
		propagation: PropagationOptions<IntegratorConfiguration>,
		observing observer: (Double, Complex<Double>) -> PropagationControl
	) throws -> PropagationRunSummary {
		try solveMultiTimeOrderedCorrelation(
			problem: DensityMatrixProblem(problem),
			configuration: configuration,
			request: request,
			propagation: propagation,
			observing: observer
		)
	}
}
