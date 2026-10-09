// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS {
    @usableFromInline
    internal final class PotentialNoiseDefinition: Sendable {
        internal enum Source: Sendable {
            case ou(UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator)
            case fft(GaussianFFTMultiNoiseProcessGenerator, StochasticPotential.FFTOptions)
        }

        internal let source: Source
        internal let timeSpan: SimulationTimeSpan
        internal let windowDuration: Double
        internal let channelCount: Int
        internal let isZero: Bool

        @usableFromInline
        internal init(
            _ potential: StochasticPotential,
            propagation: PropagationOptions<IntegrationOptions>,
            fallbackStep: Double
        ) {
            let timeSpan = propagation.timeSpan
            let windowDuration = propagation.integration.maximumStepSize
            let channelCount = potential.couplingOperators.count
            self.timeSpan = timeSpan
            self.windowDuration = windowDuration
            self.channelCount = channelCount
            switch potential.source {
            case .covarianceModel(let model):
                let generator = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
                    model: model, windowDuration: windowDuration, start: timeSpan.start,
                    step: potential.noiseStepSize ?? fallbackStep)
                source = .ou(generator)
                isZero = model.latentBaths.allSatisfy { $0.residues.elements.allSatisfy { $0 == .zero } }
            case .fft(let options, let powerDensity):
                // A zero-duration solve still has a stationary coefficient at its origin.
                let generator = GaussianFFTMultiNoiseProcessGenerator(
                    tMax: max(timeSpan.end - timeSpan.start, options.dtMax),
                    dtMax: options.dtMax, deltaOmegaMax: options.deltaOmegaMax,
                    omegaMax: options.omegaMax, spectralDensity: powerDensity)
                precondition(generator.channelCount == channelCount,
                             "Expected one potential operator per FFT channel.")
                source = .fft(generator, options)
                isZero = generator.factors.allSatisfy { $0.elements.allSatisfy { $0 == .zero } }
            }
        }
    }
}

public extension HOPS {
    /// Identifies all sampling, purpose-domain, interpolation and origin choices.
    enum PotentialNoiseGenerationAlgorithm: UInt8, Sendable {
        case correlatedOUV1 = 1
        case positiveFrequencyFFTV1 = 2
    }

    /// Immutable replay identity of the unconjugated physical coefficients y_i.
    /// No quantum noise or state-dependent shift is included.
    ///
    /// OU replay uses bounded memory. Queries can repeat or move backwards only
    /// within the sampler's retained window; start a new sampler for older times.
    /// FFT replay freezes spectral weights and never reevaluates a spectral closure.
    struct PotentialNoisePath: Sendable {
        @usableFromInline internal let definition: PotentialNoiseDefinition
        public let masterSeed: UInt64
        public let trajectoryID: UInt64
        public let ensembleSampling: EnsembleSampling

        public var channelCount: Int { definition.channelCount }
        public var timeSpan: SimulationTimeSpan { definition.timeSpan }
        /// FFT paths are queried internally at t - meshOrigin.
        public var meshOrigin: Double { timeSpan.start }
        public var covarianceModel: CorrelatedBathModel? {
            if case .ou(let generator) = definition.source { return generator.model }
            return nil
        }
        public var stepSize: Double? {
            if case .ou(let generator) = definition.source { return generator.step }
            return nil
        }
        public var fftOptions: StochasticPotential.FFTOptions? {
            if case .fft(_, let options) = definition.source { return options }
            return nil
        }
        public var generationAlgorithm: PotentialNoiseGenerationAlgorithm {
            switch definition.source {
            case .ou: .correlatedOUV1
            case .fft: .positiveFrequencyFFTV1
            }
        }

        @usableFromInline
        internal init(definition: PotentialNoiseDefinition, masterSeed: UInt64,
                      trajectoryID: UInt64, ensembleSampling: EnsembleSampling = .independent) {
            self.definition = definition
            self.masterSeed = masterSeed
            self.trajectoryID = trajectoryID
            self.ensembleSampling = ensembleSampling
        }

        /// Nil retains the propagation's maximum-step lookback for OU paths.
        public func makeSampler(windowDuration: Double? = nil) -> PotentialNoiseSampler {
            .init(definition: definition, seed: masterSeed, trajectoryID: trajectoryID,
                  ensembleSampling: ensembleSampling, windowDuration: windowDuration)
        }

        public func forEachSample<Times: Sequence>(
            at times: Times, _ body: (Double, borrowing UniqueVector<Complex<Double>>) -> Void
        ) where Times.Element == Double {
            var sampler = makeSampler(windowDuration: 0)
            var values = UniqueVector<Complex<Double>>.zero(channelCount)
            var previous: Double?
            for time in times {
                precondition(time.isFinite && (previous == nil || time >= previous!),
                             "Potential replay times must be finite and nondecreasing.")
                sampler.sample(time, into: &values.mutableSpan)
                body(time, values)
                previous = time
            }
        }
    }

    /// Mutable trajectory-owned state. The absent and zero-source cases draw no
    /// Gaussian values and own no OU or FFT workspace.
    struct PotentialNoiseSampler: ~Copyable, MultiChannelNoiseRealization, Sendable {
        internal var rng: TrajectoryRandomNumberGenerator?
        internal var ou: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess?
        internal var fft: GaussianFFTMultiNoiseProcess?
        internal let timeSpan: SimulationTimeSpan?
        public let channelCount: Int
        public private(set) var integrationGrid: (origin: Double, step: Double)?

        @usableFromInline
        internal init(definition: PotentialNoiseDefinition?, seed: UInt64, trajectoryID: UInt64,
                      ensembleSampling: EnsembleSampling, windowDuration: Double? = nil) {
            if let windowDuration {
                precondition(windowDuration.isFinite && windowDuration >= 0,
                             "The potential lookback must be finite and nonnegative.")
            }
            self.channelCount = definition?.channelCount ?? 0
            self.timeSpan = definition?.timeSpan
            self.ou = nil
            self.fft = nil
            self.rng = nil
            self.integrationGrid = nil
            guard let definition, !definition.isZero else { return }
            var rng = TrajectoryRandomNumberGenerator(
                seed: seed, trajectoryID: trajectoryID, ensembleSampling: ensembleSampling,
                purpose: .stochasticPotentialGeneration)
            switch definition.source {
            case .ou(let prepared):
                let generator: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator
                if let windowDuration, windowDuration != prepared.windowDuration {
                    generator = .init(model: prepared.model, windowDuration: windowDuration,
                                      start: prepared.start, step: prepared.step)
                } else { generator = prepared }
                self.ou = generator.generate(generator: &rng.gaussian)
                self.integrationGrid = (origin: generator.start, step: generator.step)
            case .fft(let generator, _):
                self.fft = generator.generateProcess(generator: &rng.gaussian)
            }
            self.rng = rng
        }

        public mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>) {
            precondition(values.count == channelCount, "Expected one value per potential channel.")
            if let timeSpan {
                precondition(time.isFinite && time >= timeSpan.start && time <= timeSpan.end,
                             "Potential sample time lies outside this trajectory's time span.")
            }
            if ou != nil {
                ou!.sample(time, into: &values, generator: &rng!.gaussian)
            } else if let fft {
                fft.sample(time - timeSpan!.start, into: &values)
            } else {
                for i in values.indices { values[unchecked: i] = .zero }
            }
        }
    }

    /// Shares one frozen source definition across all trajectory replay identities.
    struct EnsemblePotentialNoisePaths: Sendable {
        @usableFromInline internal let definition: PotentialNoiseDefinition
        public let masterSeed: UInt64
        public let trajectoryIDs: Range<UInt64>
        public let ensembleSampling: EnsembleSampling
        public var channelCount: Int { definition.channelCount }
        public var timeSpan: SimulationTimeSpan { definition.timeSpan }

        @usableFromInline
        internal init(definition: PotentialNoiseDefinition, masterSeed: UInt64,
                      trajectoryIDs: Range<UInt64>, ensembleSampling: EnsembleSampling) {
            self.definition = definition
            self.masterSeed = masterSeed
            self.trajectoryIDs = trajectoryIDs
            self.ensembleSampling = ensembleSampling
        }

        public func path(for trajectoryID: UInt64) -> PotentialNoisePath {
            precondition(trajectoryIDs.contains(trajectoryID),
                         "The requested trajectory does not belong to this ensemble.")
            return .init(definition: definition, masterSeed: masterSeed,
                         trajectoryID: trajectoryID, ensembleSampling: ensembleSampling)
        }
    }

    /// Borrowed reference coefficients at an observation time. Neither span
    /// includes nonlinear shifts; no potential is represented by an empty span.
    struct NoiseView: ~Copyable, ~Escapable {
        public let bath: Span<Complex<Double>>
        public let potential: Span<Complex<Double>>

        @_lifetime(copy bath, copy potential)
        @inlinable
        internal init(bath: Span<Complex<Double>>, potential: Span<Complex<Double>>) {
            self.bath = bath
            self.potential = potential
        }
    }
}
