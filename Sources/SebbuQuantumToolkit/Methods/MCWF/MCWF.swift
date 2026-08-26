// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum MCWF: Sendable {}

public extension MCWF {
	enum JumpAlgorithm: Sendable {
		/// Integrate the cumulative hazard and locate its crossing.
		case waitingTime(
			eventTolerance: Double,
			maximumEventIterations: Int
		)

		/// Test for jumps over discrete integration intervals.
		case discreteTime
	}

	struct Configuration: Sendable {
		public var jumpAlgorithm: JumpAlgorithm

		public init(
			jumpAlgorithm: JumpAlgorithm = .waitingTime(
				eventTolerance: 1e-10,
				maximumEventIterations: 64
			)
		) {
			self.jumpAlgorithm = jumpAlgorithm
		}
	}
}

public extension MCWF {
	protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions
        
		func solve<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
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
			problem: PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
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
			configuration: MCWF.Configuration,
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
			configuration: MCWF.Configuration,
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
