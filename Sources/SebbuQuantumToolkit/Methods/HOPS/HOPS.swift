// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum HOPS {}

extension HOPS {
	public struct Hierarchy: Sendable {
		public init() {}
		public struct Index: Sendable {}

		@inlinable
		public func multiIndex(at: Int) -> Index { fatalError("TODO: Implement") }

		@inlinable
		public func tier(at: Index) -> Int { fatalError("TODO: Implement") }

		@inlinable
		public func parentIndices(of: Index, indices: (borrowing Span<Index>) -> Void) {
			fatalError("TODO: Implement")
		}

		@inlinable
		public func childIndices(of: Index, indices: (borrowing Span<Index>) -> Void) {
			fatalError("TODO: Implement")
		}
	}

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

		public init(
			hierarchy: Hierarchy,
			equationType: EquationType,
			shiftType: ShiftType = .none,
			unravelling: MarkovianUnravelling = .diffusive
		) {
			self.hierarchy = hierarchy
			self.equationType = equationType
			self.shiftType = shiftType
			self.unravelling = unravelling
		}
	}
}

extension HOPS {
	public struct HierarchyStateView: ~Copyable {
		@usableFromInline
		package let dimension: Int
		// Total state
		@usableFromInline
		package let states: UniqueVector<Complex<Double>>

		@inlinable
		public var count: Int {
			states.count / dimension
		}

		@inlinable
		@inline(always)
		public func withPhysicalState<Result>(
			_ body: (
				borrowing UniqueVector<Complex<Double>>
			) -> Result
		) -> Result {
			let state = UniqueVector(
				_unsafeComponents: states.components, count: dimension)
			let result = body(state)
			let _ = state.consumeComponents()
			return result
		}

		@inlinable
		public func withState<Result>(
			at index: Int,
			_ body: (
				borrowing UniqueVector<Complex<Double>>
			) -> Result
		) -> Result {
			if index < 0 || index >= states.count {
				return body(.zero(dimension))
			}
			let state = UniqueVector(
				_unsafeComponents: states.components, count: dimension)
			let result = body(state)
			let _ = state.consumeComponents()
			return result
		}

		@inlinable
		public func withFullState<Result>(
			_ body: (
				borrowing UniqueVector<Complex<Double>>
			) -> Result
		) -> Result {
			body(states)
		}

		@inlinable
		deinit {
			let _ = states.consumeComponents()
		}
	}
}

public extension HOPS {
	protocol Implementation {
        static func solve<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
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
			configuration: HOPS.Configuration,
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
			configuration: HOPS.Configuration,
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
}

public extension HOPS {
	protocol RandomNumberGeneratorDrivenImplementation: Implementation {
		static func solve<Hamiltonian, RNG>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
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

public extension HOPS {
	protocol HierarchyProvidingImplementation: Implementation {
        static func solveWithHierarchy<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			seed: UInt64,
			trajectoryID: UInt64,
			_ forEach: (
				Double,
				borrowing HOPS.HierarchyStateView
			) -> Void
		)
		where Hamiltonian: HamiltonianFunction & ~Copyable
	}

	protocol HierarchyProvidingRandomNumberGeneratorDrivenImplementation:
		HierarchyProvidingImplementation
	{
		static func solveWithHierarchy<Hamiltonian, RNG>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			rng: inout RNG,
			_ forEach: (
				Double,
				borrowing HOPS.HierarchyStateView
			) -> Void
		)
		where
			Hamiltonian: HamiltonianFunction & ~Copyable,
			RNG: RandomNumberGenerator
	}

	protocol TwoTimeCorrelationImplementation: Implementation {
		@discardableResult
		static func solveTwoTimeCorrelation<Hamiltonian>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HOPS.Configuration,
			request: TwoTimeCorrelationRequest,
			propagation: PropagationOptions<IntegrationOptions>,
			execution: TrajectoryExecution,
			_ forEach: (
				Double,
				Complex<Double>
			) -> Void
		) -> TrajectoryRunSummary
		where Hamiltonian: HamiltonianFunction & ~Copyable
	}
}
