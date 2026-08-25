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

extension QSD {
	protocol Implementation {
		static func solve<Hamiltonian, RNG>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
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

		static func solve<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: QSD.Configuration,
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
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction & ~Copyable
	}
}
