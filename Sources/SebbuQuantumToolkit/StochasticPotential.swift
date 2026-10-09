// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// A trajectory-local Hermitian potential V(t) = sum_i [y_i L_i† + conj(y_i) L_i].
///
/// Built-in sources produce stationary proper complex Gaussian coefficients:
/// E[y_i(t) conj(y_j(s))] = K_ij(t-s), E[y_i(t)y_j(s)] = 0.
/// For Hermitian L, V = 2 Re(y) L, with real-force covariance 2 Re(K).
/// All quantities are raw numbers in caller-chosen consistent units.
///
/// For thermal HOPS, put only the vacuum BCF in the hierarchy and use this
/// source for K_beta. No potential poles, Girsanov shifts or importance weights
/// are added to the hierarchy. Custom realized coefficients can override the
/// source through the explicit-noise solve overloads.
public struct StochasticPotential: Sendable {
    public struct FFTOptions: Sendable {
        public let dtMax: Double
        public let deltaOmegaMax: Double
        public let omegaMax: Double

        public init(dtMax: Double, deltaOmegaMax: Double, omegaMax: Double) {
            precondition(dtMax.isFinite && dtMax > 0,
                         "FFT time spacing must be finite and positive.")
            precondition(deltaOmegaMax.isFinite && deltaOmegaMax > 0,
                         "FFT frequency spacing must be finite and positive.")
            precondition(omegaMax.isFinite && omegaMax > 0,
                         "FFT frequency cutoff must be finite and positive.")
            self.dtMax = dtMax
            self.deltaOmegaMax = deltaOmegaMax
            self.omegaMax = omegaMax
        }
    }

    public enum Source: Sendable {
        case covarianceModel(CorrelatedBathModel)
        /// Already weighted positive-frequency power P(omega), with no implicit
        /// pi factor: K(tau) = integral_0^omegaMax P(omega) exp(-i omega tau) d omega.
        /// Each matrix must be Hermitian PSD. Evaluated once per solve preparation.
        case fft(options: FFTOptions,
                 powerDensity: @Sendable (Double) -> Matrix<Complex<Double>>)
    }

    public let couplingOperators: [TimeDependentOperator]
    public let source: Source
    /// OU spacing override; nil uses HOPS.Configuration.noiseStepSize, then
    /// the integrator maximum step. Does not affect FFT discretization.
    public let noiseStepSize: Double?

    public init(
        couplingOperators: [TimeDependentOperator], source: Source,
        noiseStepSize: Double? = nil
    ) {
        precondition(!couplingOperators.isEmpty, "A potential needs at least one operator.")
        if case .covarianceModel(let model) = source {
            precondition(model.channelCount == couplingOperators.count,
                         "Expected one potential operator per covariance channel.")
        }
        if let noiseStepSize {
            precondition(noiseStepSize.isFinite && noiseStepSize > 0,
                         "The potential OU mesh step must be finite and positive.")
        }
        self.couplingOperators = couplingOperators
        self.source = source
        self.noiseStepSize = noiseStepSize
    }

    public init(couplingOperator: TimeDependentOperator, source: Source,
                noiseStepSize: Double? = nil) {
        self.init(couplingOperators: [couplingOperator], source: source,
                  noiseStepSize: noiseStepSize)
    }

    public init(couplingOperators: [TimeDependentOperator], covarianceModel: CorrelatedBathModel,
                noiseStepSize: Double? = nil) {
        self.init(couplingOperators: couplingOperators, source: .covarianceModel(covarianceModel),
                  noiseStepSize: noiseStepSize)
    }

    public init(couplingOperator: TimeDependentOperator, covarianceModel: CorrelatedBathModel,
                noiseStepSize: Double? = nil) {
        self.init(couplingOperators: [couplingOperator], covarianceModel: covarianceModel,
                  noiseStepSize: noiseStepSize)
    }

    /// Scalar positive-frequency FFT convenience, using the same matrix sampler.
    public init(
        couplingOperator: TimeDependentOperator,
        fft options: FFTOptions,
        powerDensity: @Sendable @escaping (Double) -> Double
    ) {
        self.init(couplingOperator: couplingOperator, source: .fft(options: options) { omega in
            let value = powerDensity(omega)
            precondition(value.isFinite && value >= 0,
                         "Power density must be finite and nonnegative.")
            return .init(elements: [Complex(value)], rows: 1, columns: 1)
        })
    }

    public init(
        couplingOperators: [TimeDependentOperator],
        fft options: FFTOptions,
        powerDensity: @Sendable @escaping (Double) -> Matrix<Complex<Double>>
    ) {
        self.init(
            couplingOperators: couplingOperators,
            source: .fft(options: options, powerDensity: powerDensity)
        )
    }
}
