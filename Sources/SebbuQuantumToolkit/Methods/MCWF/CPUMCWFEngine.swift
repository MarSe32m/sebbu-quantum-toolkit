// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct CPUMCWFEngine: Sendable {
    @inlinable
    public init() {}
}

extension CPUMCWFEngine: MCWF.Implementation {
    @inlinable
    @discardableResult
    public func solve<Hamiltonian>(problem: PureStateProblem<Hamiltonian>, configuration: MCWF.Configuration, propagation: PropagationOptions<IntegrationOptions>, seed: UInt64, trajectoryID: UInt64, observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction {
        throw ImplementationError.notImplemented
    }
    
    @inlinable
    @discardableResult
    public func solve<Hamiltonian, RNG>(problem: PureStateProblem<Hamiltonian>, configuration: MCWF.Configuration, propagation: PropagationOptions<IntegrationOptions>, rng: inout RNG, observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl) throws -> TrajectoryRunSummary where Hamiltonian : HamiltonianFunction, RNG : RandomNumberGenerator {
        throw ImplementationError.notImplemented
    }
    
	@inlinable
	public func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		seed: UInt64,
		trajectoryID: UInt64,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}

	@inlinable
	public func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<IntegrationOptions>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        throw ImplementationError.notImplemented
	}
}

extension MCWF {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		rng: inout RNG,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
        let implementation = CPUMCWFEngine()
        try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, forEach)
	}

	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		seed: UInt64,
		trajectoryID: UInt64,
		_ forEach: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed,
			trajectoryID: trajectoryID, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        return try implementation.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

	@inlinable
	@inline(always)
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>,
		configuration: MCWF.Configuration,
		propagation: PropagationOptions<CPUMCWFEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (UInt64, Double, borrowing UniqueVector<Complex<Double>>) -> Void
	) throws -> TrajectoryRunSummary
    where Hamiltonian: HamiltonianFunction {
        let implementation = CPUMCWFEngine()
        return try implementation.solveTrajectories(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

}
