// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public extension HOPS {
    /// The versioned algorithm that defines a replayable colored HOPS bath-noise path.
    ///
    /// A version identifies all pathwise choices that affect `z(t)`, including
    /// stationary initialization, exact OU mesh transitions, Gaussian stream
    /// derivation, physical/latent mixing, and interpolation between mesh nodes.
    enum BathNoiseGenerationAlgorithm: UInt8, Sendable {
        case correlatedOUV1 = 1
    }
}

extension HOPS {
    /// Shared immutable definition of a colored-noise realization family.
    ///
    /// Ensemble provenance stores this object once and individual paths only add
    /// their trajectory ID. Keeping the model here also prevents public HOPS APIs
    /// from exposing the lower-level sliding-window implementation type.
    @usableFromInline
    final class BathNoiseDefinition: Sendable {
        @usableFromInline let model: CorrelatedBathModel
        @usableFromInline internal let timeSpan: SimulationTimeSpan
        @usableFromInline let stepSize: Double
        @usableFromInline let generationAlgorithm: BathNoiseGenerationAlgorithm

        @usableFromInline
        init(
            model: CorrelatedBathModel,
            timeSpan: SimulationTimeSpan,
            stepSize: Double,
            generationAlgorithm: BathNoiseGenerationAlgorithm = .correlatedOUV1
        ) {
            precondition(stepSize.isFinite && stepSize > 0,
                         "The bath-noise mesh step must be finite and positive.")
            self.model = model
            self.timeSpan = timeSpan
            self.stepSize = stepSize
            self.generationAlgorithm = generationAlgorithm
        }
    }
}

public extension HOPS {
    /// A lightweight description of one complete multichannel colored bath-noise realization.
    ///
    /// `BathNoisePath` identifies the stochastic path used by one HOPS trajectory;
    /// it is not the probability distribution/process itself. Physical channels can
    /// be correlated, so one path always represents the full physical noise vector
    /// `z(t)` together with its latent coordinates.
    ///
    /// The path stores no generated samples. Calling ``makeSampler(windowDuration:)``
    /// creates a bounded-memory cursor that lazily regenerates the realization.
    /// A fresh sampler replays it from the mesh origin. For long sequential
    /// post-processing, prefer ``forEachSample(at:_:)``.
    ///
    /// The realization is defined by its reproducible OU mesh and interpolation
    /// rule, not by a log of adaptive-integrator RHS query times. Replaying the
    /// path therefore reproduces `z(t)` at arbitrary valid times without recording
    /// Runge–Kutta stage evaluations.
    ///
    /// Exact replay is defined by ``generationAlgorithm``. This descriptor covers
    /// the colored HOPS bath noise only; Markovian white-noise unravelling is not
    /// part of this path. ``ensembleSampling`` is part of the replay identity.
    /// Antithetic paths negate the reference noise, before state-dependent shifts.
    struct BathNoisePath: Sendable {
        @usableFromInline internal let definition: BathNoiseDefinition

        public let masterSeed: UInt64
        public let trajectoryID: UInt64
        public let ensembleSampling: EnsembleSampling

        public var channelCount: Int { definition.model.channelCount }
        public var latentCount: Int { definition.model.poleCount }
        public var stepSize: Double { definition.stepSize }
        public var timeSpan: SimulationTimeSpan { definition.timeSpan }
        public var meshOrigin: Double { definition.timeSpan.start }
        public var generationAlgorithm: BathNoiseGenerationAlgorithm {
            definition.generationAlgorithm
        }

        @usableFromInline
        internal init(
            definition: BathNoiseDefinition,
            masterSeed: UInt64,
            trajectoryID: UInt64,
            ensembleSampling: EnsembleSampling = .independent
        ) {
            self.definition = definition
            self.masterSeed = masterSeed
            self.trajectoryID = trajectoryID
            self.ensembleSampling = ensembleSampling
        }

        /// Creates a bounded-lookback cursor over this realization.
        ///
        /// Sampling allocates its working storage only during construction. Queries
        /// may be repeated or made out of order while the required OU mesh nodes
        /// remain in the retained window. Requests older than
        /// ``BathNoiseSampler/earliestAvailableTime`` follow the lower-level OU
        /// sampler contract and fail a precondition. Construct a fresh sampler to
        /// replay the path from the beginning.
        func makeSampler(windowDuration: Double) -> BathNoiseSampler {
            BathNoiseSampler(path: self, windowDuration: windowDuration)
        }

        /// Sequentially replays physical bath noise at nondecreasing times.
        ///
        /// The callback receives the complete correlated physical-noise vector. The
        /// implementation retains only a bounded OU window and one reusable output
        /// vector; it never collects the path into an array.
        func forEachSample<Times: Sequence>(
            at times: Times,
            _ body: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
        ) where Times.Element == Double {
            var sampler = makeSampler(windowDuration: 0)
            var values = UniqueVector<Complex<Double>>.zero(channelCount)
            var previous: Double?
            for time in times {
                precondition(time.isFinite, "Bath-noise sample times must be finite.")
                if let previous {
                    precondition(time >= previous,
                                 "Bath-noise replay times must be nondecreasing.")
                }
                sampler.sample(time, into: &values.mutableSpan)
                body(time, values)
                previous = time
            }
        }
    }

    /// A mutable bounded-memory cursor over a ``BathNoisePath``.
    ///
    /// The sampler lazily regenerates the same versioned OU mesh used by HOPS and
    /// linearly interpolates that mesh at arbitrary query times. It keeps only a
    /// sliding lookback window, not the complete realization.
    struct BathNoiseSampler: ~Copyable, Sendable {
        @usableFromInline internal var rng: TrajectoryRandomNumberGenerator
        @usableFromInline internal var process: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess
        @usableFromInline internal let timeSpan: SimulationTimeSpan

        public var channelCount: Int { process.channelCount }
        public var latentCount: Int { process.latentCount }
        public var stepSize: Double { process.step }
        public var latentBathRanges: [Range<Int>] { process.latentBathRanges }
        public var earliestAvailableTime: Double { process.earliestAvailableTime }
        public var latestGeneratedTime: Double { process.latestGeneratedTime }

        @usableFromInline
        internal init(path: BathNoisePath, windowDuration: Double) {
            precondition(windowDuration.isFinite && windowDuration >= 0,
                         "The bath-noise window duration must be finite and nonnegative.")
            self.timeSpan = path.timeSpan
            var rng = TrajectoryRandomNumberGenerator(
                seed: path.masterSeed,
                trajectoryID: path.trajectoryID,
                ensembleSampling: path.ensembleSampling,
                purpose: .coloredNoiseGeneration
            )
            switch path.generationAlgorithm {
            case .correlatedOUV1:
                let generator = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
                    model: path.definition.model,
                    windowDuration: windowDuration,
                    start: path.meshOrigin,
                    step: path.stepSize
                )
                self.process = generator.generate(generator: &rng.gaussian)
            }
            self.rng = rng
        }

        @usableFromInline
        internal init(
            path: BathNoisePath,
            preparedGenerator: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator
        ) {
            self.timeSpan = path.timeSpan
            var rng = TrajectoryRandomNumberGenerator(
                seed: path.masterSeed,
                trajectoryID: path.trajectoryID,
                ensembleSampling: path.ensembleSampling,
                purpose: .coloredNoiseGeneration
            )
            switch path.generationAlgorithm {
            case .correlatedOUV1:
                self.process = preparedGenerator.generate(generator: &rng.gaussian)
            }
            self.rng = rng
        }

        /// Writes all physical correlated bath channels `z_i(t)`.
        @inlinable
        mutating func sample(
            _ time: Double,
            into values: inout MutableSpan<Complex<Double>>
        ) {
            validate(time)
            process.sample(time, into: &values, generator: &rng.gaussian)
        }

        /// Writes all latent OU coordinates `x_p(t)` in latent-bath/pole order.
        @inlinable
        mutating func sampleLatent(
            _ time: Double,
            into values: inout MutableSpan<Complex<Double>>
        ) {
            validate(time)
            process.sampleLatent(time, into: &values, generator: &rng.gaussian)
        }

        /// Writes physical and latent coordinates of exactly the same realization.
        @inlinable
        mutating func sample(
            _ time: Double,
            physical: inout MutableSpan<Complex<Double>>,
            latent: inout MutableSpan<Complex<Double>>
        ) {
            validate(time)
            process.sample(
                time,
                physical: &physical,
                latent: &latent,
                generator: &rng.gaussian
            )
        }

        @inlinable
        @inline(always)
        internal func validate(_ time: Double) {
            precondition(
                time.isFinite && time >= timeSpan.start && time <= timeSpan.end,
                "The requested bath-noise time lies outside this trajectory's time span."
            )
        }
    }

    /// Cheap replay provenance for every trajectory in a HOPS ensemble.
    ///
    /// The correlated bath model, mesh, time span, master seed and algorithm
    /// version are stored once. ``path(for:)`` adds only the requested trajectory
    /// identifier and shares the immutable definition.
    struct EnsembleNoisePaths: Sendable {
        @usableFromInline internal let definition: BathNoiseDefinition

        public let masterSeed: UInt64
        public let trajectoryIDs: Range<UInt64>
        public let ensembleSampling: EnsembleSampling

        public var channelCount: Int { definition.model.channelCount }
        public var latentCount: Int { definition.model.poleCount }
        public var stepSize: Double { definition.stepSize }
        public var timeSpan: SimulationTimeSpan { definition.timeSpan }
        public var generationAlgorithm: BathNoiseGenerationAlgorithm {
            definition.generationAlgorithm
        }

        @usableFromInline
        internal init(
            definition: BathNoiseDefinition,
            masterSeed: UInt64,
            trajectoryIDs: Range<UInt64>,
            ensembleSampling: EnsembleSampling = .independent
        ) {
            self.definition = definition
            self.masterSeed = masterSeed
            self.trajectoryIDs = trajectoryIDs
            self.ensembleSampling = ensembleSampling
        }

        /// Returns the replayable colored-noise path for one ensemble trajectory.
        @inlinable
        func path(for trajectoryID: UInt64) -> BathNoisePath {
            precondition(
                trajectoryIDs.contains(trajectoryID),
                "The requested trajectory ID does not belong to this ensemble."
            )
            return BathNoisePath(
                definition: definition,
                masterSeed: masterSeed,
                trajectoryID: trajectoryID,
                ensembleSampling: ensembleSampling
            )
        }
    }

    /// HOPS-specific result for a single trajectory.
    struct TrajectoryRunResult: Sendable {
        public let summary: TrajectoryRunSummary
        public let bathNoise: BathNoisePath

        public var trajectoryIDs: Range<UInt64> { summary.trajectoryIDs }
        public var masterSeed: UInt64 { summary.masterSeed }
        public var ensembleSampling: EnsembleSampling { summary.ensembleSampling }
        public var propagation: PropagationRunSummary { summary.propagation }

        public init(summary: TrajectoryRunSummary, bathNoise: BathNoisePath) {
            self.summary = summary
            self.bathNoise = bathNoise
        }
    }

    /// HOPS-specific result for an ensemble or explicit trajectory batch.
    struct EnsembleRunResult: Sendable {
        public let summary: TrajectoryRunSummary
        public let bathNoise: EnsembleNoisePaths

        public var trajectoryIDs: Range<UInt64> { summary.trajectoryIDs }
        public var masterSeed: UInt64 { summary.masterSeed }
        public var ensembleSampling: EnsembleSampling { summary.ensembleSampling }
        public var propagation: PropagationRunSummary { summary.propagation }

        public init(summary: TrajectoryRunSummary, bathNoise: EnsembleNoisePaths) {
            self.summary = summary
            self.bathNoise = bathNoise
        }
    }
}
