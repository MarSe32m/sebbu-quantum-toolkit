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
	protocol Implementation {
		static func solve<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			seed: UInt64,
			trajectoryID: UInt64,
			_ forEach: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
		)
		where Hamiltonian: HamiltonianFunction & ~Copyable

		@discardableResult
		static func solveEnsemble<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction & ~Copyable

		static func solveTrajectories<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			execution: TrajectoryExecution,
			_ forEach:
				@Sendable (  // Will potentially be called from multiple threads for each trajectory
					UInt64,
					Double,
					borrowing UniqueVector<Complex<Double>>
				) -> Void
		)
		where Hamiltonian: HamiltonianFunction & ~Copyable
	}

	protocol RandomNumberGeneratorDrivenImplementation: Implementation {
		static func solve<Hamiltonian, RNG>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			rng: inout RNG,
			_ forEach: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> Void
		)
		where
			Hamiltonian: HamiltonianFunction & ~Copyable,
			RNG: RandomNumberGenerator
	}
}
