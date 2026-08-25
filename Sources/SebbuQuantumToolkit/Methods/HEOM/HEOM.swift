// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum HEOM: Sendable {}

extension HEOM {
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

	public enum ShiftType: Sendable {
		case none
		case meanField
	}

	public struct Configuration: Sendable {
		public let hierarchy: Hierarchy
		public var shiftType: ShiftType

		public init(
			hierarchy: Hierarchy,
			shiftType: ShiftType = .none
		) {
			self.hierarchy = hierarchy
			self.shiftType = shiftType
		}
	}
}

extension HEOM {
	public struct HierarchyStateView: ~Copyable {
		//TODO: This will probably be stored as a huge vector where the ADOs are vectorized / flattened
		@usableFromInline
		package let states: UniqueVector<UniqueMatrix<Complex<Double>>>

		@inlinable
		public var count: Int {
			states.count
		}

		@inlinable
		@inline(always)
		public func withPhysicalState<Result>(
			_ body: (
				borrowing UniqueMatrix<Complex<Double>>
			) -> Result
		) -> Result {
			body(states[unchecked: 0])
		}

		@inlinable
		@inline(always)
		public func withState<Result>(
			at index: Int,
			_ body: (
				borrowing UniqueMatrix<Complex<Double>>
			) -> Result
		) -> Result {
			body(states[index])
		}

		@inlinable
		deinit {
			let _ = states.consumeComponents()
		}
	}
}

public extension HEOM {
	protocol Implementation {
		static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		)

		static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
			problem: borrowing PureStateProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		)
	}
}

public extension HEOM.Implementation {
	@inlinable
	static func solve<Hamiltonian: HamiltonianFunction & ~Copyable>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: HEOM.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) {
		fatalError("TODO: Implement")
	}
}

public extension HEOM {
	protocol HierarchyProvidingImplementation: Implementation {
		static func solveWithHierarchy<Hamiltonian: HamiltonianFunction & ~Copyable>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing HEOM.HierarchyStateView
			) -> Void
		)
	}

	protocol TwoTimeCorrelationImplementation: Implementation {
		static func solveTwoTimeCorrelation<
			Hamiltonian: HamiltonianFunction & ~Copyable
		>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			request: TwoTimeCorrelationRequest,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				Complex<Double>
			) -> Void
		)
	}
}
