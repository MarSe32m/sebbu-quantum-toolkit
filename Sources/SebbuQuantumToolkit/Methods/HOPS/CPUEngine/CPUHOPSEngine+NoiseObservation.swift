// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
    /// Non-owning access to the two stable, RHS-owned output vectors. Used only
    /// inside propagate while the owning RHS/solver is explicitly kept alive.
    @usableFromInline
    internal struct NoiseObservation {
        @usableFromInline let bath: UnsafePointer<Complex<Double>>
        @usableFromInline let bathCount: Int
        @usableFromInline let potential: UnsafePointer<Complex<Double>>
        @usableFromInline let potentialCount: Int

        @inlinable
        init(bath: UnsafePointer<Complex<Double>>, bathCount: Int,
             potential: UnsafePointer<Complex<Double>>, potentialCount: Int) {
            self.bath = bath
            self.bathCount = bathCount
            self.potential = potential
            self.potentialCount = potentialCount
        }

        @inlinable
        func observe(_ time: Double, _ state: borrowing State,
                     _ observer: (Double, borrowing State, borrowing HOPS.NoiseView) -> PropagationControl)
            -> PropagationControl {
            // An absent potential has no allocated coefficient storage. Span's
            // pointer initializer requires alignment even when count is zero.
            let view = HOPS.NoiseView(
                bath: Span(_unsafeStart: bath, count: bathCount),
                potential: potentialCount == 0
                    ? Span<Complex<Double>>()
                    : Span(_unsafeStart: potential, count: potentialCount))
            return observer(time, state, view)
        }
    }

    internal func solveObservingNoises(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>, seed: UInt64, trajectoryID: UInt64,
        ensembleSampling: EnsembleSampling,
        observing observer: (Double, borrowing State, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> HOPS.TrajectoryRunResult {
        precondition(trajectoryID < UInt64.max, "The trajectory ID must fit in a half-open range.")
        let preparation = try Preparation(problem: problem, configuration: configuration, propagation: propagation)
        let summary = try propagate(
            problem: problem, preparation: preparation, propagation: propagation,
            rhs: Self.makeSeededRightHandSide(
                hamiltonian: problem.system.hamiltonian, preparation: preparation,
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling),
            observing: observer)
        return .init(
            summary: .init(trajectoryIDs: trajectoryID..<(trajectoryID + 1), masterSeed: seed,
                           ensembleSampling: ensembleSampling, propagation: summary),
            bathNoise: .init(definition: preparation.noiseDefinition, masterSeed: seed,
                             trajectoryID: trajectoryID, ensembleSampling: ensembleSampling),
            potentialNoise: preparation.potentialPath(
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling))
    }

    internal func solveExplicitNoises<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable,
        PotentialNoise: MultiChannelNoiseRealization & ~Copyable
    >(
        problem: PureStateProblem, configuration: HOPS.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        bathNoise: consuming BathNoise, markovianNoise: consuming MarkovianNoise,
        potentialNoise: consuming PotentialNoise,
        observing observer: (Double, borrowing State, borrowing HOPS.NoiseView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let bathCount = configuration.hierarchy.environment.bath.channelCount
        let potentialCount = configuration.stochasticPotential?.couplingOperators.count ?? 0
        guard bathNoise.channelCount == bathCount else {
            throw SolverError.bathNoiseChannelCountMismatch(expected: bathCount, actual: bathNoise.channelCount)
        }
        guard markovianNoise.channelCount == problem.markovianChannels.count else {
            throw SolverError.markovianNoiseChannelCountMismatch(
                expected: problem.markovianChannels.count, actual: markovianNoise.channelCount)
        }
        guard potentialNoise.channelCount == potentialCount else {
            throw SolverError.potentialNoiseChannelCountMismatch(expected: potentialCount, actual: potentialNoise.channelCount)
        }
        // Explicit paths must not prepare or invoke the configured OU/FFT source.
        let preparation = try Preparation(
            problem: problem, configuration: configuration, propagation: propagation, preparingStandardNoise: false)
        return try propagate(
            problem: problem, preparation: preparation, propagation: propagation,
            rhs: RightHandSide(
                hamiltonian: problem.system.hamiltonian, preparation: preparation,
                bathNoise: bathNoise, markovianNoise: markovianNoise, potentialNoise: potentialNoise),
            observing: observer)
    }
}
