// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
public enum HOPS {}

extension HOPS {
	public enum EquationType: Sendable {
		case linear
		case nonLinear
		case nonLinearNormalized
	}

	public enum ShiftType: Sendable {
		case none
		case meanField
	}

	public struct Configuration: Sendable {
		public let hierarchy: Hierarchy
		public var equationType: EquationType
		public var shiftType: ShiftType
		public var unravelling: MarkovianUnravelling

		/// Uniform OU mesh spacing. Nil uses the integrator's maximum step.
		/// Converge this independently of the ODE tolerances. The CPU engine
		/// retains enough history for every trial step and solver retry.
		public var noiseStepSize: Double?

		public init(
			hierarchy: Hierarchy,
			equationType: EquationType,
			shiftType: ShiftType = .none,
			unravelling: MarkovianUnravelling = .diffusive,
			noiseStepSize: Double? = nil
		) {
			self.hierarchy = hierarchy
			self.equationType = equationType
			self.shiftType = shiftType
			self.unravelling = unravelling
			if let noiseStepSize {
				precondition(noiseStepSize.isFinite && noiseStepSize > 0,
					"The OU mesh step must be finite and positive.")
			}
			self.noiseStepSize = noiseStepSize
		}
	}
}

// MARK: Implementation
public extension HOPS {
	protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions
        
        @discardableResult
        func solveTrajectory<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> PropagationControl
		) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction

		@discardableResult
        func solveEnsemble<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws -> HOPS.EnsembleRunResult
		where Hamiltonian: HamiltonianFunction

        @discardableResult
		func solveTrajectories<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach:
				@Sendable (  // Will potentially be called from multiple threads for each trajectory
					UInt64,
					Double,
					borrowing UniqueVector<Complex<Double>>
				) -> Void
		) throws -> HOPS.EnsembleRunResult
        where Hamiltonian: HamiltonianFunction
	}
}

//MARK: HierarchyProvidingImplementation, HierarchyProvidingRandomNumberGeneratorDrivenImplementation
public extension HOPS {
	protocol HierarchyProvidingImplementation: Implementation {
        @discardableResult
        func solveWithHierarchy<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing HOPS.HierarchyStateView
			) -> Void
		) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction
	}
}

//MARK: TwoTimeCorrelationImplementation
public extension HOPS {
    protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation<Hamiltonian>(
            problem: PureStateProblem<Hamiltonian>,
            configuration: HOPS.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            execution: TrajectoryExecution,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> HOPS.EnsembleRunResult
        where Hamiltonian: HamiltonianFunction
    }
}

public extension HOPS {
	protocol MultiTimeOrderedCorrelationImplementation: Implementation {
		@discardableResult
		func solveMultiTimeOrderedCorrelation<Hamiltonian>(
			problem: PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			request: MultiTimeOrderedCorrelationRequest,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			observing observer: (Double, Complex<Double>) -> PropagationControl
		) throws -> HOPS.EnsembleRunResult where Hamiltonian: HamiltonianFunction
	}
}
