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
        
		func solveTrajectory(
			problem: PureStateProblem,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> PropagationControl
		) throws -> TrajectoryRunSummary

		/// Solves and averages the requested trajectories into density matrices.
		///
		/// The callback is invoked serially in output-time order after the parallel
		/// reduction. A fixed output schedule is required because independent
		/// trajectories need a common set of sampling times.
		@discardableResult
		func solveEnsemble(
			problem: PureStateProblem,
			configuration: QSD.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws -> TrajectoryRunSummary

		/// Solves the requested trajectories. The callback can run concurrently.
		@discardableResult
		func solveTrajectories(
			problem: PureStateProblem,
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
	}
}

public extension QSD {
    protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation(
            problem: PureStateProblem,
            configuration: QSD.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            execution: TrajectoryExecution,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> TrajectoryRunSummary
    }
}

public extension QSD {
	protocol MultiTimeOrderedCorrelationImplementation: Implementation {
		@discardableResult
		func solveMultiTimeOrderedCorrelation(
			problem: PureStateProblem,
			configuration: QSD.Configuration,
			request: MultiTimeOrderedCorrelationRequest,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			observing observer: (Double, Complex<Double>) -> PropagationControl
		) throws -> TrajectoryRunSummary
	}
}
