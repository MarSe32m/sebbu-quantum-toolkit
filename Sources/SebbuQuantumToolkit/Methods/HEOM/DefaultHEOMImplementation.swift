// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct DefaultHEOMImplementation: Sendable {
    @inlinable
    public init() {}
}

extension DefaultHEOMImplementation: HEOM.Implementation {
	@inlinable
	public func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: HEOM.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension HEOM {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        let implementation = DefaultHEOMImplementation()
        try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}
}
