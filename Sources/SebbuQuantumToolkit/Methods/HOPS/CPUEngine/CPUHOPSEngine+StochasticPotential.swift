// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @discardableResult
    public func solveTrajectory<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        var root = UniqueVector<Complex<Double>>.zero(problem.system.dimension)
        return try solveExplicitNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise
        ) { time, state, _ in
            for j in 0..<root.count { root[j] = state.amplitudes.elements[j] }
            return observer(time, root)
        }
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @_disfavoredOverload
    @discardableResult
    public func solveTrajectory<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        var root = UniqueVector<Complex<Double>>.zero(problem.system.dimension)
        return try solveExplicitNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise
        ) { time, state, noise in
            for j in 0..<root.count { root[j] = state.amplitudes.elements[j] }
            return observer(time, root, noise)
        }
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public func solveTrajectory(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64, trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        var root = UniqueVector<Complex<Double>>.zero(problem.system.dimension)
        return try solveObservingNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling
        ) { time, state, noise in
            for j in 0..<root.count { root[j] = state.amplitudes.elements[j] }
            return observer(time, root, noise)
        }
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public func solveTrajectory<RNG: RandomNumberGenerator>(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        let seed = rng.next()
        let trajectoryID = rng.next() % UInt64.max
        return try solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, observingWithNoises: observer)
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @discardableResult
    public func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        return try solveExplicitNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise
        ) { time, state, _ in
            return Self.withHierarchyView(state) { observer(time, $0) }
        }
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @_disfavoredOverload
    @discardableResult
    public func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        return try solveExplicitNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise
        ) { time, state, noise in
            return Self.withHierarchyView(state) { observer(time, $0, noise) }
        }
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public func solveWithHierarchy(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64, trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        return try solveObservingNoises(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling
        ) { time, state, noise in
            return Self.withHierarchyView(state) { observer(time, $0, noise) }
        }
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public func solveWithHierarchy<RNG: RandomNumberGenerator>(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        let seed = rng.next()
        let trajectoryID = rng.next() % UInt64.max
        return try solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, observingWithNoises: observer)
    }

    @discardableResult
    public func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        guard configuration.stochasticPotential == nil else { throw SolverError.missingPotentialNoise }
        return try solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise,
            potentialNoise: ZeroMultiChannelNoise(channelCount: 0), observing: observer)
    }
}

extension HOPS {
    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @discardableResult
    public static func solveTrajectory<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observing observer: (Double, borrowing UniqueVector<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise,
            observing: observer)
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @_disfavoredOverload
    @discardableResult
    public static func solveTrajectory<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise,
            observingWithNoises: observer)
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public static func solveTrajectory(
        problem: PureStateProblem,
        configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64,
        trajectoryID: UInt64,
        ensembleSampling: EnsembleSampling = .independent,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        try CPUEngine().solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling,
            observingWithNoises: observer)
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public static func solveTrajectory<RNG: RandomNumberGenerator>(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
        observingWithNoises observer: (Double, borrowing UniqueVector<Complex<Double>>, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        try CPUEngine().solveTrajectory(
            problem: problem, configuration: configuration, propagation: propagation,
            rng: &rng,
            observingWithNoises: observer)
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @discardableResult
    public static func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise,
            observing: observer)
    }

    /// Consumes explicit coefficient paths; returns no invented seed provenance.
    @_disfavoredOverload
    @discardableResult
    public static func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise,
            observingWithNoises: observer)
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public static func solveWithHierarchy(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        seed: UInt64, trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling,
            observingWithNoises: observer)
    }

    /// Borrows the bath and potential coefficients used by this trajectory.
    @_disfavoredOverload
    @discardableResult
    public static func solveWithHierarchy<RNG: RandomNumberGenerator>(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        rng: inout RNG,
        observingWithNoises observer: (Double, borrowing HOPS.HierarchyStateView, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            rng: &rng,
            observingWithNoises: observer)
    }

    @discardableResult
    public static func solveWithHierarchy<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        observing observer: (Double, borrowing HOPS.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration, propagation: propagation,
            bathNoise: bathNoise, markovianNoise: markovianNoise, observing: observer)
    }
}
