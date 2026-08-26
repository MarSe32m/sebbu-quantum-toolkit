// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum DefaultGKSLImplementation: Sendable {}

extension DefaultGKSLImplementation: GKSL.Implementation {
	@inlinable
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension GKSL: GKSL.Implementation {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		try DefaultGKSLImplementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}
}
