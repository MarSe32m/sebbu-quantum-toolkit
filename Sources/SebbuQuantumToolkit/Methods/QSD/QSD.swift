// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum QSD: Sendable {}

extension QSD {
	public enum EquationType: Sendable {
		case linear
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

	protocol RandomNumberGeneratorDrivenImplementation: Implementation {
		func solve<Hamiltonian, RNG>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			rng: inout RNG,
			observing observer: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> PropagationControl
		) throws -> TrajectoryRunSummary
		where
			Hamiltonian: HamiltonianFunction,
			RNG: RandomNumberGenerator
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
