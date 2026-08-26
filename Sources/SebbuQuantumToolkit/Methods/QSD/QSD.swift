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
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			_ forEach: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
		) throws
		where Hamiltonian: HamiltonianFunction

		@discardableResult
		func solveEnsemble<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
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
			problem: borrowing PureStateProblem<Hamiltonian>,
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
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			rng: inout RNG,
			_ forEach: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
		) throws
		where
			Hamiltonian: HamiltonianFunction,
			RNG: RandomNumberGenerator
	}
}
