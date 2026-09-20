// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// How global trajectory IDs select stochastic realizations.
public enum EnsembleSampling: Sendable, Equatable {
	/// One Philox stream per trajectory ID (the default).
	case independent

	/// Pair global IDs `(0, 1), (2, 3), ...` using stream `trajectoryID / 2`.
	///
	/// Even IDs use the ordinary realization and odd IDs use its antithetic partner.
	/// Centered Gaussian draws become `-x`, including both complex components
	/// and stationary OU initial conditions. Uniform draws are reflected toward
	/// `1 - u`, preserving the sampler's endpoint convention: a 53-bit `[0, 1)`
	/// grid uses `(1 - 2^-53) - u`. PDP midpoint uniforms are complemented exactly
	/// before the inverse CDF is evaluated.
	///
	/// Pairing is independent of request order, batching and workers. Partial
	/// pairs and odd ensemble sizes are supported without adding trajectories.
	/// Only reference noise is reflected. State-dependent Girsanov and nuHOPS
	/// shifts are still calculated independently for each trajectory.
	case antithetic
}
