// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// One realized multichannel colored path, rather than a process distribution.
///
/// `sample` must fill every physical channel without allocating. Adaptive
/// integrators may repeat times or go backwards within an attempted step;
/// sampling must return the same path values on those queries. A streaming
/// realization must retain at least `integration.maximumStepSize` of lookback.
/// HOPS consumes `z_i(t)` and applies the conjugation in its equations itself.
///
/// Custom paths do not automatically define a valid statistical unraveling.
/// Their covariance must match the bath for ordinary ensemble simulations.
public protocol MultiChannelNoiseRealization: ~Copyable {
	var channelCount: Int { get }

	/// Optional uniform interpolation mesh at whose nodes adaptive ODE steps
	/// should end. The origin must be no later than the propagation start and
	/// the step must be finite, positive, and resolvable at that origin.
	/// Smooth paths normally use the default `nil`.
	var integrationGrid: (origin: Double, step: Double)? { get }

	mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>)
}

extension MultiChannelNoiseRealization where Self: ~Copyable {
	@inlinable
	public var integrationGrid: (origin: Double, step: Double)? { nil }
}

/// A prescribed zero colored path. Deterministic bath memory remains active.
public struct ZeroMultiChannelNoise: MultiChannelNoiseRealization, Sendable {
	public let channelCount: Int

	@inlinable
	public init(channelCount: Int) {
		precondition(channelCount >= 0, "The noise channel count must be nonnegative.")
		self.channelCount = channelCount
	}

	@inlinable
	public mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>)
	{
		precondition(values.count == channelCount, "Expected one value per noise channel.")
		for i in values.indices { values[unchecked: i] = .zero }
	}
}

/// The normalized complex values driving one diffusive trajectory.
///
/// The stochastic integrator samples once per step at its start time, passes
/// the actual (possibly output-shortened) step size, multiplies these values
/// by `sqrt(stepSize)`, and reuses them at both Heun stages. Do not multiply
/// by `sqrt(stepSize)` in the realization. Fill the entire span without allocating.
/// The standard SQT convention has independent real and imaginary components
/// of variance one; `GaussianDiffusiveNoise` preserves that convention exactly.
/// Custom statistics need not represent an ensemble unraveling of the problem.
public protocol DiffusiveNoiseRealization: ~Copyable {
	/// A type-wide promise that every realization supplies only zeros. HOPS
	/// may use an adaptive ODE solver and omit noise sampling entirely.
	static var isIdenticallyZero: Bool { get }
	var channelCount: Int { get }
	mutating func sampleNormalized(
		at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>
	)
}

extension DiffusiveNoiseRealization where Self: ~Copyable {
	@inlinable
	public static var isIdenticallyZero: Bool { false }
}

/// SQT's purpose-separated, reproducible Gaussian white-noise realization.
public struct GaussianDiffusiveNoise: DiffusiveNoiseRealization, Sendable {
	public let channelCount: Int
	@usableFromInline
    internal var rng: TrajectoryRandomNumberGenerator

	@inlinable
	public init(
		channelCount: Int, seed: UInt64, trajectoryID: UInt64,
		ensembleSampling: EnsembleSampling = .independent
	) {
		precondition(channelCount >= 0, "The noise channel count must be nonnegative.")
		self.channelCount = channelCount
		self.rng = .init(
			seed: seed, trajectoryID: trajectoryID,
			ensembleSampling: ensembleSampling, purpose: .gaussianWhiteNoise)
	}

	@inlinable
	public mutating func sampleNormalized(
		at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>
	) {
		precondition(values.count == channelCount, "Expected one value per noise channel.")
		for i in values.indices { values[unchecked: i] = rng.nextNormal() }
	}
}

/// A prescribed zero Wiener realization. Dissipative drift is still applied;
/// this is not the ensemble-averaged open-system evolution.
public struct ZeroDiffusiveNoise: DiffusiveNoiseRealization, Sendable {
	@inlinable
    @inline(always)
    public static var isIdenticallyZero: Bool { true }
	
    public let channelCount: Int

	@inlinable
	public init(channelCount: Int) {
		precondition(channelCount >= 0, "The noise channel count must be nonnegative.")
		self.channelCount = channelCount
	}

	@inlinable
	public mutating func sampleNormalized(
		at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>
	) {
		precondition(values.count == channelCount, "Expected one value per noise channel.")
		for i in values.indices { values[unchecked: i] = .zero }
	}
}
