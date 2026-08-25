// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum DefaultHEOMImplementation: Sendable {}

extension DefaultHEOMImplementation: HEOM.Implementation {
	@inlinable
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: HEOM.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) where Hamiltonian: HamiltonianFunction {
		fatalError("TODO: Implementation")
	}
}

extension HEOM: HEOM.Implementation {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) where Hamiltonian: HamiltonianFunction {
		DefaultHEOMImplementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}
}
