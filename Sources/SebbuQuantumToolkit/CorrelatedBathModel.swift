// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// A physical rational model of a matrix-valued bath correlation function.
///
/// Each latent bath is driven by one independent white-noise input. All pole
/// states inside a latent bath share that input. With poles `W[a, mu]` and
/// residue matrices `R[a]`, the model is
///
/// ```text
/// H_a(omega) = sum_mu R[a][:, mu] / (W[a, mu] - i omega)
/// J(omega)   = H(omega) H(omega)^dagger
/// alpha(tau) = sum_a R[a] exp(-W[a] tau) K[a] R[a]^dagger,
/// K[a][mu, nu] = 1 / (W[a, mu] + W[a, nu]^*)
/// ```
///
/// Consequently, `spectralDensity(at:)` is Hermitian positive semidefinite
/// for every real frequency by construction.
public struct CorrelatedBathModel: Sendable {
	/// One independent latent bath and all OU pole states driven by it.
	public struct LatentBath: Sendable {
		/// Stable OU poles. Every real part is strictly positive.
		public let poles: [Complex<Double>]

		/// The physical-channel mixing vectors, stored as an `N x K` matrix.
		/// Column `mu` belongs to `poles[mu]`.
		public let residues: Matrix<Complex<Double>>

		public var poleCount: Int { poles.count }
		public var channelCount: Int { residues.rows }

		public init(
			poles: [Complex<Double>],
			residues: Matrix<Complex<Double>>
		) {
			precondition(!poles.isEmpty, "A latent bath must contain a pole")
			precondition(
				residues.columns == poles.count,
				"The residue matrix must have one column per pole"
			)
			precondition(residues.rows > 0, "At least one physical channel is required")
			precondition(
				poles.allSatisfy {
					$0.real.isFinite && $0.imaginary.isFinite && $0.real > 0
				},
				"Every pole must be finite and have a positive real part"
			)
			precondition(
				residues.elements.allSatisfy {
					$0.real.isFinite && $0.imaginary.isFinite
				},
				"Latent residues must be finite"
			)
			self.poles = poles
			self.residues = residues
		}

		/// The stationary covariance of the latent OU pole states.
		public var stationaryCovariance: Matrix<Complex<Double>> {
			Matrix(rows: poleCount, columns: poleCount) { elements in
				for mu in 0..<poleCount {
					for nu in 0..<poleCount {
						elements[mu * poleCount + nu] =
							Complex<Double>.one
							/ (poles[mu] + poles[nu].conjugate)
					}
				}
			}
		}
	}

	/// A one-sided exponential matrix residue, useful for conventional HOPS
	/// and HEOM interfaces.
	public struct ExponentialTerm: Sendable {
		public let pole: Complex<Double>
		public let residue: Matrix<Complex<Double>>

		public init(
			pole: Complex<Double>,
			residue: Matrix<Complex<Double>>
		) {
			self.pole = pole
			self.residue = residue
		}
	}

	public let channelCount: Int
	public let latentBaths: [LatentBath]

	/// Total number of latent OU states and, therefore, hierarchy directions.
	public var poleCount: Int {
		latentBaths.reduce(0) { $0 + $1.poleCount }
	}

	public init(channelCount: Int, latentBaths: [LatentBath]) {
		precondition(channelCount > 0, "At least one physical channel is required")
		precondition(
			latentBaths.allSatisfy { $0.channelCount == channelCount },
			"Every latent bath must have the same physical-channel count"
		)
		self.channelCount = channelCount
		self.latentBaths = latentBaths
	}

	public static func zero(channelCount: Int) -> Self {
		Self(channelCount: channelCount, latentBaths: [])
	}

	/// Evaluates the stable spectral factor `H(omega)`.
	///
	/// Its columns correspond to the independent latent baths.
	public func spectralFactor(at frequency: Double) -> Matrix<Complex<Double>> {
		precondition(frequency.isFinite, "The frequency must be finite")
		let imaginaryFrequency = Complex<Double>(0, frequency)
		return Matrix(rows: channelCount, columns: latentBaths.count) { elements in
			elements.initialize(repeating: .zero)
			for (bathIndex, bath) in latentBaths.enumerated() {
				for channel in 0..<channelCount {
					var value = Complex<Double>.zero
					for pole in bath.poles.indices {
						value +=
							bath.residues[channel, pole]
							/ (bath.poles[pole] - imaginaryFrequency)
					}
					elements[channel * latentBaths.count + bathIndex] = value
				}
			}
		}
	}

	/// Evaluates `J(omega) = H(omega) H(omega)^dagger`.
	public func spectralDensity(at frequency: Double) -> Matrix<Complex<Double>> {
		if latentBaths.isEmpty {
			return .zeros(rows: channelCount, columns: channelCount)
		}
		let factor = spectralFactor(at: frequency)
		return factor.dot(factor.conjugateTranspose)
	}

	/// Evaluates the stationary matrix-valued BCF.
	///
	/// Positive lags use the latent OU representation directly. Negative lags
	/// are obtained from stationarity, `alpha(-tau) = alpha(tau)^dagger`.
	public func bathCorrelation(at lag: Double) -> Matrix<Complex<Double>> {
		precondition(lag.isFinite, "The lag must be finite")
		if lag < 0 {
			return bathCorrelation(at: -lag).conjugateTranspose
		}

		var result = Matrix<Complex<Double>>.zeros(
			rows: channelCount,
			columns: channelCount
		)
		for bath in latentBaths {
			for mu in bath.poles.indices {
				let decay = Complex<Double>.exp(-bath.poles[mu] * lag)
				for nu in bath.poles.indices {
					let covariance =
						decay
						/ (bath.poles[mu] + bath.poles[nu].conjugate)
					for i in 0..<channelCount {
						let left = bath.residues[i, mu] * covariance
						for j in 0..<channelCount {
							result[i, j] +=
								left
								* bath.residues[j, nu].conjugate
						}
					}
				}
			}
		}
		return result
	}

	/// Converts the latent representation into one-sided exponential terms.
	///
	/// The returned residue matrices need not individually be Hermitian or
	/// positive. Joint physicality is carried by the latent factor.
	public var oneSidedExponentialTerms: [ExponentialTerm] {
		var terms: [ExponentialTerm] = []
		terms.reserveCapacity(poleCount)
		for bath in latentBaths {
			for mu in bath.poles.indices {
				var residue = Matrix<Complex<Double>>.zeros(
					rows: channelCount,
					columns: channelCount
				)
				for i in 0..<channelCount {
					for j in 0..<channelCount {
						var right = Complex<Double>.zero
						for nu in bath.poles.indices {
							right +=
								bath.residues[j, nu].conjugate
								/ (bath.poles[mu]
									+ bath.poles[nu].conjugate)
						}
						residue[i, j] = bath.residues[i, mu] * right
					}
				}
				terms.append(
					ExponentialTerm(pole: bath.poles[mu], residue: residue))
			}
		}
		return terms
	}
}
