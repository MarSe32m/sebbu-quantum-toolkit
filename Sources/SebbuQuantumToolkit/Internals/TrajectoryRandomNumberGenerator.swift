// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Philox4x64 stream used to derive a trajectory from a master seed, channel number,
/// trajectory identifier and a purpose.
@usableFromInline
internal struct TrajectoryRandomNumberGenerator: RandomNumberGenerator, Sendable {
	@usableFromInline
	internal enum Purpose: UInt64 {
		case unspecified = 0
		case gaussianWhiteNoise
		case coloredNoiseGeneration
		case mcwfJumps
	}

	@usableFromInline
	internal var gaussian: GaussianStream

	@inlinable
	internal var isAntithetic: Bool { gaussian.isAntithetic }

	@inlinable
	internal init(
		seed: UInt64, channel: UInt64 = 0, trajectoryID: UInt64 = 0,
		ensembleSampling: EnsembleSampling = .independent, purpose: Purpose = .unspecified
	) {
		var splitMix = SplitMix64(seed: seed)
		let streamID = ensembleSampling == .antithetic ? trajectoryID / 2 : trajectoryID
		let generator = Philox4x64(
			key: .init(splitMix.next(), splitMix.next()),
			counter: .init(
				0 /* block index */,
				channel /* channel */,
				streamID /* trajectory or pair id */,
				purpose.rawValue /* purpose */
			)
		)
		self.gaussian = GaussianStream(
			generator: generator,
			isAntithetic: ensembleSampling == .antithetic && trajectoryID & 1 != 0
		)
	}

	/// Integer reflection reaches generic uniform users, including the midpoint
	/// inverse CDF in UniquePDPSolver. Generic Gaussian code must instead receive
	/// `&rng.gaussian` (see GaussianStream).
	@inlinable
	internal mutating func next() -> UInt64 {
		let value = gaussian.generator.next()
		return isAntithetic ? ~value : value
	}

	/// The upstream 53-bit midpoint rounds its largest value to 1. Keep the open
	/// interval without extra draws; all other conversions remain unchanged.
	@inlinable
	internal mutating func nextUnitDoubleOpen() -> Double {
		Self.unitDoubleOpen(from: next())
	}

	@inlinable
	internal static func unitDoubleOpen(from word: UInt64) -> Double {
		min((Double(word >> 11) + 0.5) * 0x1.0p-53, Double(1).nextDown)
	}

	/// Raw-word adapter for SebbuScience's Marsaglia polar Gaussian sampler.
	/// Its helpers are statically dispatched protocol extensions, so overriding
	/// nextNormal alone cannot intercept generic OU/FFT sampling.
	///
	/// For m = raw >> 11 the signed uniform is m * 2^-52 - 1. Mapping m to
	/// (-m) modulo 2^53 negates every interior value exactly. The unmatched -1
	/// endpoint stays -1 and is always rejected. Thus polar radii, rejection
	/// decisions and raw draw counts are identical for both partners. Retaining
	/// the low bits makes the word mapping a bijection too.
	/// This view is only for Gaussian helpers, not uniform or PDP sampling.
	@usableFromInline
	internal struct GaussianStream: RandomNumberGenerator, Sendable {
		@usableFromInline internal var generator: Philox4x64
		@usableFromInline internal let isAntithetic: Bool

		@inlinable
		internal init(generator: Philox4x64, isAntithetic: Bool) {
			self.generator = generator
			self.isAntithetic = isAntithetic
		}

		@inlinable
		internal static func reflectedWord(_ value: UInt64) -> UInt64 {
			let lowMask: UInt64 = (1 << 11) - 1
			return (0 &- (value & ~lowMask)) | (value & lowMask)
		}

		@inlinable
		internal mutating func next() -> UInt64 {
			let value = generator.next()
			return isAntithetic ? Self.reflectedWord(value) : value
		}
	}

	@inlinable
	internal mutating func nextGaussian() -> (Double, Double) {
		gaussian.nextGaussian()
	}

	@inlinable
	internal mutating func nextGaussian(count: Int) -> [Double] {
		precondition(count >= 0)
		// SebbuScience 0.4.12 overfills even-sized Gaussian arrays. Keep the
		// same pair consumption, checking capacity after either component.
		return .init(capacity: count) { span in
			while !span.isFull {
				let pair = gaussian.nextGaussian()
				span.append(pair.0)
				if !span.isFull { span.append(pair.1) }
			}
		}
	}

	@inlinable
	internal mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(mean: Float = 0.0, stdev: Float = 1.0) -> Float {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Complex<
		Double
	> {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(mean: Complex<Double>, stdev: Double = 1.0) -> Complex<
		Double
	> {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(mean: Float = 0.0, stdev: Float = 1.0) -> Complex<Float> {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(mean: Complex<Float>, stdev: Float = 1.0) -> Complex<
		Float
	> {
		gaussian.nextNormal(mean: mean, stdev: stdev)
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Double = 0.0, stdev: Double = 1.0)
		-> [Double]
	{
		nextGaussian(count: count).map { $0 * stdev + mean }
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Float = 0.0, stdev: Float = 1.0)
		-> [Float]
	{
		(nextNormal(count: count, mean: Double(mean), stdev: Double(stdev)) as [Double]).map
		{ Float($0) }
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Double = 0.0, stdev: Double = 1.0)
		-> [Complex<Double>]
	{
		let reals: [Double] = nextNormal(count: 2 * count, mean: mean, stdev: stdev)
		return (0..<count).map { Complex(reals[2 * $0], reals[2 * $0 + 1]) }
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Complex<Double>, stdev: Double = 1.0)
		-> [Complex<Double>]
	{
		let reals: [Double] = nextNormal(count: 2 * count, stdev: stdev)
		return (0..<count).map { Complex(reals[2 * $0], reals[2 * $0 + 1]) + mean }
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Float = 0.0, stdev: Float = 1.0)
		-> [Complex<Float>]
	{
		let reals: [Float] = nextNormal(count: 2 * count, mean: mean, stdev: stdev)
		return (0..<count).map { Complex(reals[2 * $0], reals[2 * $0 + 1]) }
	}

	@inlinable
	internal mutating func nextNormal(count: Int, mean: Complex<Float>, stdev: Float = 1.0)
		-> [Complex<Float>]
	{
		let reals: [Float] = nextNormal(count: 2 * count, stdev: stdev)
		return (0..<count).map { Complex(reals[2 * $0], reals[2 * $0 + 1]) + mean }
	}
}
