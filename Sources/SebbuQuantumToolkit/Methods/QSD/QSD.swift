// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum QSD: Sendable {}

extension QSD {
	public enum EquationType: Sendable {
		case linear
        case nonLinear
		case nonLinearNormalized
	}

	public struct Configuration: Sendable {
		public var equationType: EquationType

		public init(
			equationType: EquationType = .nonLinearNormalized
		) {
			self.equationType = equationType
		}
	}
}

public extension QSD {
	protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions
        
		func solve<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> PropagationControl
		) throws -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction

		/// Solves and averages the requested trajectories into density matrices.
		///
		/// The callback is invoked serially in output-time order after the parallel
		/// reduction. A fixed output schedule is required because independent
		/// trajectories need a common set of sampling times.
		@discardableResult
		func solveEnsemble<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction

		/// Solves independent trajectories. The callback can run concurrently.
		@discardableResult
		func solveTrajectories<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach:
				@Sendable (  // Will potentially be called from multiple threads for each trajectory
					UInt64,
					Double,
					borrowing UniqueVector<Complex<Double>>
				) -> Void
		) throws -> TrajectoryRunSummary
        where Hamiltonian: HamiltonianFunction
	}
}

public extension QSD {
    protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation<Hamiltonian>(
            problem: PureStateProblem<Hamiltonian>,
            configuration: QSD.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            execution: TrajectoryExecution,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> TrajectoryRunSummary
        where Hamiltonian: HamiltonianFunction
    }
}
