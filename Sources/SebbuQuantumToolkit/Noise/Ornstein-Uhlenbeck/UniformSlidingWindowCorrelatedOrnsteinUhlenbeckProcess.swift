// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Immutable preparation for stationary, proper complex OU noise from a fitted bath.
///
/// Pass `CorrelatedBathFitter.Result.model` directly. Within each latent bath,
/// `K[p,q] = 1 / (W[p] + conj(W[q]))` and
/// `Q[p,q] = step * phi((W[p] + conj(W[q])) * step)`, where
/// `phi(v) = (1 - exp(-v)) / v`. The exact mesh update is `x' = F x + LQ u`,
/// initialized with `x = LK u`, for independent proper unit-variance Gaussians.
/// Physical noise is `z = R x`, so `E[z(t) z(s)^dagger] = model.bathCorrelation(t-s)`.
/// The noises returned here are **unconjugated**.
///
/// Covariance factorization occurs only in this initializer. Reuse this value
/// across trajectories, with an independent RNG and process for each trajectory.
public struct UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator: Sendable {
    public let model: CorrelatedBathModel
    public let start: Double
    public let step: Double
    public let windowDuration: Double

    public var channelCount: Int { model.channelCount }
    public let latentCount: Int

    /// Flattening preserves bath order, then pole order, in the supplied model.
    /// No latent coordinates are removed or rescaled during preparation.
    public let latentBathRanges: [Range<Int>]

    internal struct Block: Sendable {
        let range: Range<Int>
        let decay: [Complex<Double>]
        let stationaryFactor: Matrix<Complex<Double>>
        let innovationFactor: Matrix<Complex<Double>>
    }

    internal let blocks: [Block]
    internal let windowSize: Int
    internal let maximumBlockSize: Int

    /// - Parameters:
    ///   - windowDuration: Nonnegative lookback retained for solver retries.
    ///     Two additional mesh nodes accommodate interpolation at either end.
    ///   - start: Origin of the uniform noise mesh; may be negative.
    ///   - step: Positive mesh spacing. Covariances are exact at mesh nodes;
    ///     linear interpolation between nodes must be converged separately.
    public init(
        model: CorrelatedBathModel, windowDuration: Double,
        start: Double = 0, step: Double
    ) {
        precondition(start.isFinite, "The mesh origin must be finite.")
        precondition(step.isFinite && step > 0, "The mesh step must be finite and positive.")
        precondition((start + step).isFinite && start + step > start,
                     "The mesh step must be representable at the origin.")
        precondition(windowDuration.isFinite && windowDuration >= 0,
                     "The window duration must be finite and nonnegative.")
        let intervals = (windowDuration / step).rounded(.up)
        precondition(intervals.isFinite && intervals < Double(Int.max / 4),
                     "The requested noise window is too large.")
        let windowSize = Int(intervals) + 2
        precondition(model.poleCount == 0 || windowSize <= Int.max / model.poleCount,
                     "The requested noise buffer is too large.")

        var blocks: [Block] = []
        var ranges: [Range<Int>] = []
        var offset = 0
        var maximumBlockSize = 0
        for bath in model.latentBaths {
            let count = bath.poleCount
            let range = offset..<(offset + count)
            let decay = bath.poles.map { Complex<Double>.exp(-$0 * step) }
            precondition(decay.allSatisfy { $0.real.isFinite && $0.imaginary.isFinite },
                         "The OU transition is not representable.")
            let covariance = bath.stationaryCovariance
            var innovation = Matrix<Complex<Double>>.zeros(rows: count, columns: count)
            for p in 0..<count {
                for q in 0..<count {
                    // Avoid cancellation in K - F K F^dagger for a fine mesh.
                    let v = (bath.poles[p] + bath.poles[q].conjugate) * step
                    innovation[p, q] = step * .phiOneMinusExpMinus(v)
                }
            }
            blocks.append(Block(
                range: range, decay: decay,
                stationaryFactor: _noiseCovarianceFactor(covariance),
                innovationFactor: _noiseCovarianceFactor(innovation)
            ))
            ranges.append(range)
            offset += count
            maximumBlockSize = max(maximumBlockSize, count)
        }
        self.model = model
        self.latentCount = offset
        self.start = start
        self.step = step
        self.windowDuration = windowDuration
        self.windowSize = windowSize
        self.blocks = blocks
        self.latentBathRanges = ranges
        self.maximumBlockSize = maximumBlockSize
    }

    /// Allocates trajectory storage and draws its stationary initial state.
    /// Subsequent `sample`, `sampleLatent`, and `reset` calls reuse that storage.
    public func generate<Generator: RandomNumberGenerator>(
        generator: inout Generator
    ) -> UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess {
        UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess(
            preparation: self, generator: &generator
        )
    }
}

/// A stationary OU path with bounded storage and allocation-free span sampling.
///
/// All mutable buffers use `SebbuScience.UniqueVector`, so moving a process
/// cannot introduce copy-on-write allocations. The caller owns the RNG. Keep
/// using the same RNG stream for this trajectory; cached queries never use it.
/// As usual, a caller-supplied RNG must itself avoid allocations for the complete
/// sampling call to be allocation-free.
///
/// Sampling lazily generates a uniform mesh. Repeated or out-of-order queries
/// reuse the same path while their bracketing nodes remain in the window.
/// Queries before `earliestAvailableTime` fail; choose a window long enough for
/// the ODE solver's rejected steps. Interpolation does not sample OU bridges.
public struct UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess: ~Copyable, Sendable {
    internal let preparation: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator
    internal var samples: UniqueVector<Complex<Double>>
    internal var gaussians: UniqueVector<Complex<Double>>
    internal var interpolated: UniqueVector<Complex<Double>>
    internal var newestIndex: Int = 0

    public var channelCount: Int { preparation.channelCount }
    public var latentCount: Int { preparation.latentCount }
    public var step: Double { preparation.step }
    public var latentBathRanges: [Range<Int>] { preparation.latentBathRanges }

    public var earliestAvailableTime: Double {
        time(at: max(0, newestIndex - preparation.windowSize + 1))
    }

    public var latestGeneratedTime: Double { time(at: newestIndex) }

    public init<Generator: RandomNumberGenerator>(
        model: CorrelatedBathModel, windowDuration: Double,
        start: Double = 0, step: Double, generator: inout Generator
    ) {
        self.init(
            preparation: .init(model: model, windowDuration: windowDuration, start: start, step: step),
            generator: &generator
        )
    }

    internal init<Generator: RandomNumberGenerator>(
        preparation: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator,
        generator: inout Generator
    ) {
        self.preparation = preparation
        // UniqueVector's empty initializer owns no allocation.
        if preparation.latentCount == 0 {
            self.samples = .init()
            self.gaussians = .init()
            self.interpolated = .init()
        } else {
            self.samples = .zero(preparation.windowSize * preparation.latentCount)
            self.gaussians = .zero(preparation.maximumBlockSize)
            self.interpolated = .zero(preparation.latentCount)
        }
        reset(generator: &generator)
    }

    /// Starts a new stationary realization at the original mesh origin,
    /// without allocating or recomputing factors. Previously cached nodes expire.
    public mutating func reset<Generator: RandomNumberGenerator>(generator: inout Generator) {
        newestIndex = 0
        for a in 0..<preparation.blocks.count {
            let block = preparation.blocks[a]
            let count = block.range.count
            for q in 0..<count {
                gaussians[q] = generator.nextNormal(stdev: Double(0.5).squareRoot())
            }
            for p in 0..<count {
                var value = Complex<Double>.zero
                for q in 0..<count {
                    value += block.stationaryFactor[p, q] * gaussians[q]
                }
                samples[block.range.lowerBound + p] = value
            }
        }
    }

    /// Writes all physical noises `z_i(t)`. The span must have `channelCount` elements.
    public mutating func sample<Generator: RandomNumberGenerator>(
        _ t: Double, into physical: inout MutableSpan<Complex<Double>>,
        generator: inout Generator
    ) {
        precondition(physical.count == channelCount, "Expected one output per physical channel.")
        interpolate(at: t, generator: &generator)
        mix(into: &physical)
    }

    /// Writes all latent noises `x_p(t)`, flattened in `latentBathRanges` order.
    /// The span must have `latentCount` elements (zero for a zero-bath model).
    public mutating func sampleLatent<Generator: RandomNumberGenerator>(
        _ t: Double, into latent: inout MutableSpan<Complex<Double>>,
        generator: inout Generator
    ) {
        precondition(latent.count == latentCount, "Expected one output per latent pole.")
        interpolate(at: t, generator: &generator)
        for p in 0..<latentCount { latent[p] = interpolated[p] }
    }

    /// Writes both representations of exactly the same realization, with `z = R x`.
    /// Both spans must have their respective channel/latent counts.
    public mutating func sample<Generator: RandomNumberGenerator>(
        _ t: Double, physical: inout MutableSpan<Complex<Double>>,
        latent: inout MutableSpan<Complex<Double>>, generator: inout Generator
    ) {
        precondition(physical.count == channelCount, "Expected one output per physical channel.")
        precondition(latent.count == latentCount, "Expected one output per latent pole.")
        interpolate(at: t, generator: &generator)
        for p in 0..<latentCount { latent[p] = interpolated[p] }
        mix(into: &physical)
    }

    private func time(at index: Int) -> Double {
        // Integer mesh indices prevent cumulative floating-point drift.
        preparation.start.addingProduct(Double(index), step)
    }

    private mutating func interpolate<Generator: RandomNumberGenerator>(
        at t: Double, generator: inout Generator
    ) {
        precondition(t.isFinite && t >= earliestAvailableTime,
                     "The requested time precedes the retained noise window or is non-finite.")
        let coordinate = (t - preparation.start) / step
        precondition(coordinate.isFinite && coordinate >= 0 && coordinate < 0x1p52,
                     "The requested mesh index is not representable.")
        var lower = Int(coordinate.rounded(.down))
        // Resolve rounding at mesh nodes by comparing physical times. Do not
        // snap arbitrary nearby solver times to a different point on the path.
        if lower > 0 && t < time(at: lower) { lower -= 1 }
        if t >= time(at: lower + 1) { lower += 1 }
        let lowerTime = time(at: lower)
        let upper = t == lowerTime ? lower : lower + 1
        let upperTime = time(at: upper)
        precondition(lowerTime <= t && t <= upperTime && upperTime.isFinite,
                     "The requested time cannot be bracketed on this mesh.")
        advance(through: upper, generator: &generator)
        precondition(lower >= max(0, newestIndex - preparation.windowSize + 1),
                     "The interpolation nodes are outside the retained noise window.")
        let fraction = lower == upper ? 0 : (t - lowerTime) / (upperTime - lowerTime)
        let left = (lower % preparation.windowSize) * latentCount
        let right = (upper % preparation.windowSize) * latentCount
        for p in 0..<latentCount {
            let a = samples[left + p]
            interpolated[p] = a + fraction * (samples[right + p] - a)
        }
    }

    private mutating func advance<Generator: RandomNumberGenerator>(
        through index: Int, generator: inout Generator
    ) {
        if latentCount == 0 {
            newestIndex = max(newestIndex, index)
            return
        }
        while newestIndex < index {
            let previous = (newestIndex % preparation.windowSize) * latentCount
            let next = ((newestIndex + 1) % preparation.windowSize) * latentCount
            for a in 0..<preparation.blocks.count {
                let block = preparation.blocks[a]
                let count = block.range.count
                for q in 0..<count {
                    gaussians[q] = generator.nextNormal(stdev: Double(0.5).squareRoot())
                }
                for p in 0..<count {
                    var innovation = Complex<Double>.zero
                    for q in 0..<count {
                        innovation += block.innovationFactor[p, q] * gaussians[q]
                    }
                    let pole = block.range.lowerBound + p
                    samples[next + pole] = block.decay[p] * samples[previous + pole] + innovation
                }
            }
            newestIndex += 1
        }
    }

    private func mix(into physical: inout MutableSpan<Complex<Double>>) {
        for i in 0..<channelCount {
            var value = Complex<Double>.zero
            for a in 0..<preparation.blocks.count {
                let range = preparation.blocks[a].range
                let residues = preparation.model.latentBaths[a].residues
                for p in 0..<range.count {
                    value += residues[i, p] * interpolated[range.lowerBound + p]
                }
            }
            physical[i] = value
        }
    }
}
