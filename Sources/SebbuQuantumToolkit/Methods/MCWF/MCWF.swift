// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public enum MCWF: Sendable {}

public extension MCWF {
	enum JumpAlgorithm: Sendable {
		/// Integrate the cumulative hazard and locate the sampled waiting-time
		/// threshold inside an accepted ODE step.
		///
		/// `eventTolerance` is an absolute tolerance in simulation-time units.
		case waitingTime(
			eventTolerance: Double,
			maximumEventIterations: Int
		)

		/// Test for a jump at the end of every accepted integration interval.
		///
		/// The interval jump probability is obtained from the integrated hazard,
		/// `1 - exp(-Delta Lambda)`, and at most one jump is applied per interval.
		/// `IntegrationOptions.maximumStepSize` therefore controls the temporal
		/// resolution of the discrete-time approximation.
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
			if case .waitingTime(let eventTolerance, let maximumEventIterations) =
				jumpAlgorithm
			{
				precondition(
					eventTolerance.isFinite && eventTolerance > .zero,
					"The MCWF event tolerance must be positive and finite"
				)
				precondition(
					maximumEventIterations > 0,
					"The maximum MCWF event-iteration count must be positive"
				)
			}
			self.jumpAlgorithm = jumpAlgorithm
		}
	}
}

public extension MCWF {
	protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions
        
        @discardableResult
		func solve<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: MCWF.Configuration,
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
			configuration: MCWF.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction

        @discardableResult
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
}

public extension MCWF {
    protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation<Hamiltonian>(
            problem: PureStateProblem<Hamiltonian>,
            configuration: MCWF.Configuration,
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
