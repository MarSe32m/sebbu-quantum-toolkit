// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum HEOM: Sendable {}

extension HEOM {
	public struct Hierarchy: Sendable {

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
	protocol Implementation {
		static func solve<Hamiltonian>(
			problem: borrowing DensityMatrixProblem<Hamiltonian>,
			configuration: HEOM.Configuration,
			propagation: PropagationOptions<IntegrationOptions>,
			_ forEach: (
				Double,
				borrowing UniqueMatrix<Complex<Double>>
			) -> Void
		)

		static func solve<Hamiltonian>(
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

extension HEOM.Implementation {
	@inlinable
	public static func solve<Hamiltonian>(
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
