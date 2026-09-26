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

	/// Package-only control used by tests and the HOPS benchmark. Normal user
	/// code keeps `.automatic`; this is deliberately not part of the public API.
	package enum BathOperatorStoragePolicy: Sendable {
		case automatic
		case dense
		case sparse
	}

	public struct Configuration: Sendable {
		public let hierarchy: Hierarchy
		public var equationType: EquationType
		public var shiftType: ShiftType
		public var unravelling: MarkovianUnravelling
		@usableFromInline
		internal var _bathOperatorStoragePolicyCode: UInt8 = 0

		package var bathOperatorStoragePolicy: BathOperatorStoragePolicy {
			get {
				switch _bathOperatorStoragePolicyCode {
				case 1: .dense
				case 2: .sparse
				default: .automatic
				}
			}
			set {
				switch newValue {
				case .automatic: _bathOperatorStoragePolicyCode = 0
				case .dense: _bathOperatorStoragePolicyCode = 1
				case .sparse: _bathOperatorStoragePolicyCode = 2
				}
			}
		}

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
        func solveTrajectory(
			problem: PureStateProblem,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing UniqueVector<Complex<Double>>
			) -> PropagationControl
		) throws -> HOPS.TrajectoryRunResult

		@discardableResult
        func solveEnsemble(
			problem: PureStateProblem,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		) throws -> HOPS.EnsembleRunResult

        @discardableResult
		func solveTrajectories(
			problem: PureStateProblem,
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
	}
}

//MARK: HierarchyProvidingImplementation, HierarchyProvidingRandomNumberGeneratorDrivenImplementation
public extension HOPS {
	protocol HierarchyProvidingImplementation: Implementation {
        @discardableResult
        func solveWithHierarchy(
			problem: PureStateProblem,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegratorConfiguration>,
			seed: UInt64,
			trajectoryID: UInt64,
			observing observer: (
				Double,
				borrowing HOPS.HierarchyStateView
			) -> Void
		) throws -> HOPS.TrajectoryRunResult
	}
}

//MARK: TwoTimeCorrelationImplementation
public extension HOPS {
    protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation(
            problem: PureStateProblem,
            configuration: HOPS.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            execution: TrajectoryExecution,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> HOPS.EnsembleRunResult
    }
}

public extension HOPS {
	protocol MultiTimeOrderedCorrelationImplementation: Implementation {
		@discardableResult
		func solveMultiTimeOrderedCorrelation(
			problem: PureStateProblem,
			configuration: HOPS.Configuration,
			request: MultiTimeOrderedCorrelationRequest,
			propagation: PropagationOptions<IntegratorConfiguration>,
			execution: TrajectoryExecution,
			observing observer: (Double, Complex<Double>) -> PropagationControl
		) throws -> HOPS.EnsembleRunResult
	}
}
