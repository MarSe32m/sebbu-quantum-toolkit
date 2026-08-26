// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct DefaultGKSLImplementation: Sendable {
    @inlinable
    public init() {}
}

extension DefaultGKSLImplementation: GKSL.Implementation {
	@inlinable
	public func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension GKSL {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        let implementation = DefaultGKSLImplementation()
		try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}
}
