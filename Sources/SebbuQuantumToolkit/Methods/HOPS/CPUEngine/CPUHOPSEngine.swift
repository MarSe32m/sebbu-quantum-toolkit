// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS {
	public struct CPUEngine: Sendable {
		@inlinable
		public init() {}

		public enum SolverError: Error, Equatable, Sendable {
			case unsupportedUnravelling
			case operatorDimensionMismatch
			case invalidStateNorm(time: Double)
			case nonFiniteState(time: Double)
		}
	}
}

extension HOPS.CPUEngine: HOPS.Implementation, HOPS.HierarchyProvidingImplementation {}

extension HOPS {
	@inlinable
	@inline(always)
	@discardableResult
	public static func solveWithHierarchy<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, rng: inout RNG,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) ->
			PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let engine = CPUEngine()
		return try engine.solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, observing: observer)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solveWithHierarchy<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observing observer: (Double, borrowing HOPS.HierarchyStateView) -> Void
	) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, observing: observer)
	}

	@discardableResult
	public static func solveWithHierarchy<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, rng: inout RNG,
		observingWithNoise observer: (
			Double, borrowing HOPS.HierarchyStateView, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let engine = CPUEngine()
		return try engine.solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, observingWithNoise: observer)
	}

	@discardableResult
	public static func solveWithHierarchy<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observingWithNoise observer: (
			Double, borrowing HOPS.HierarchyStateView, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveWithHierarchy(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, observingWithNoise: observer)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solveTrajectory<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, rng: inout RNG,
		observing observer: (
			Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let engine = CPUEngine()
		return try engine.solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, observing: observer)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solveTrajectory<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observing observer: (
			Double, borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, observing: observer)
	}

	@discardableResult
	public static func solveTrajectory<Hamiltonian, RNG>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, rng: inout RNG,
		observingWithNoise observer: (
			Double, borrowing UniqueVector<Complex<Double>>, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult
	where Hamiltonian: HamiltonianFunction, RNG: RandomNumberGenerator {
		let engine = CPUEngine()
		return try engine.solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			rng: &rng, observingWithNoise: observer)
	}

	@discardableResult
	public static func solveTrajectory<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>, seed: UInt64,
		trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent,
		observingWithNoise observer: (
			Double, borrowing UniqueVector<Complex<Double>>, borrowing Span<Complex<Double>>
		) -> PropagationControl
	) throws -> HOPS.TrajectoryRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveTrajectory(
			problem: problem, configuration: configuration, propagation: propagation,
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, observingWithNoise: observer)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solveEnsemble<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach: (
			Double, borrowing SebbuScience.UniqueMatrix<ComplexModule.Complex<Double>>
		) -> Void
	) throws -> HOPS.EnsembleRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveEnsemble(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}

	@inlinable
	@inline(always)
	@discardableResult
	public static func solveTrajectories<Hamiltonian>(
		problem: PureStateProblem<Hamiltonian>, configuration: HOPS.Configuration,
		propagation: PropagationOptions<CPUEngine.IntegratorConfiguration>,
		execution: TrajectoryExecution,
		_ forEach:
			@Sendable (
				UInt64, Double,
				borrowing SebbuScience.UniqueVector<ComplexModule.Complex<Double>>
			) -> Void
	) throws -> HOPS.EnsembleRunResult where Hamiltonian: HamiltonianFunction {
		let engine = CPUEngine()
		return try engine.solveTrajectories(
			problem: problem, configuration: configuration, propagation: propagation,
			execution: execution, forEach)
	}
}
