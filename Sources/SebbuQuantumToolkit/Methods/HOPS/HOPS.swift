// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
#if swift(<6.5)
import BasicContainers
#else
#warning("Remove swift-collections dependency")
#endif

public enum HOPS {}

extension HOPS {
	public final class Hierarchy: Sendable {
		public typealias Index = Int
        
        // Specifies +1 auxiliary states
        @usableFromInline
        package let childIndices: UniqueArray<Index>
        
        // Specifies -1 auxiliary states
        @usableFromInline
        package let parentIndices: UniqueArray<Index>
        
        // All of the -k \cdot W precomputed for all states
        @usableFromInline
        package let kWArray: UniqueArray<Complex<Double>>
        
        // The multi indices for each auxiliary state
        // For example, 0 -> (0, 0, ..., 0, 0), 1 -> (0, 0, ..., 0, 1), 2 -> (0, 0, ..., 1, 0) etc.
        @usableFromInline
        package let multiIndices: UniqueArray<Index>
        
        // How many indices the multi index tuple has
        @usableFromInline
        package let multiIndexCount: Int
        
        public let environment: Environment
        
        public init(environment: Environment, truncation: Truncation) {
            self.environment = environment
            let hierarchyDirections = environment.bath.poleCount
            fatalError("TODO: Implement")
        }

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
    public struct Environment: Sendable {
        public let couplingOperators: [TimeDependentOperator]
        public let bath: CorrelatedBathModel
        
        @inlinable
        public init(couplingOperators: [TimeDependentOperator], bath: CorrelatedBathModel) {
            precondition(couplingOperators.count == bath.poleCount, "There must be equal number of coupling operators and bath correlation matrix diagonal elements.")
            self.couplingOperators = couplingOperators
            self.bath = bath
        }
        
        @inlinable
        public init(couplingOperator: TimeDependentOperator, bath: CorrelatedBathModel) {
            self.init(couplingOperators: [couplingOperator], bath: bath)
        }
    }
    
    public enum Truncation: Sendable {
        case maximumTier(Int)
        case custom(@Sendable (borrowing Span<Int>) -> Bool)
    }
}

extension HOPS {
    public struct HierarchyStateView: ~Copyable, ~Escapable {
		@usableFromInline
		package let systemDimension: Int
		// Total state
		@usableFromInline
		package let states: Span<Complex<Double>>

		@inlinable
		public var count: Int {
			states.count / systemDimension
		}

        @_lifetime(copy states)
        @inlinable
        package init(systemDimension: Int, states: Span<Complex<Double>>) {
            self.systemDimension = systemDimension
            self.states = states
        }
        
        @inlinable
        @inline(always)
        public func withPhysicalState<Result>(
            _ body: (
                borrowing StateVectorView
            ) -> Result
        ) -> Result {
            withState(at: 0, body)
        }
        
        @inlinable
        @inline(always)
        public func withState<Result>(
            at index: HOPS.Hierarchy.Index,
            _ body: (borrowing StateVectorView) -> Result
        ) -> Result {
            precondition(index >= 0 && index < count)
            let span = states.extracting(index &* systemDimension ..< (index &+ 1) &* systemDimension)
            let stateVectorView = StateVectorView(elements: span)
            return body(stateVectorView)
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
		) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction

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
		) throws -> TrajectoryRunSummary
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
		) throws -> TrajectoryRunSummary
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
		) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction
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
        ) throws -> TrajectoryRunSummary
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
		) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction
	}
}
