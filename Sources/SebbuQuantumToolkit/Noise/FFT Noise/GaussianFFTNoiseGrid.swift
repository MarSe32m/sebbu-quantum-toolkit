// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// Common midpoint quadrature and padded time mesh for scalar and matrix spectra.
@usableFromInline
internal struct _GaussianFFTNoiseGrid: Sendable {
    @usableFromInline internal let fftCount: Int
    @usableFromInline internal let frequencyStep: Double
    @usableFromInline internal let frequencies: [Double]
    @usableFromInline internal let times: [Double]
    @usableFromInline internal let midpointPhases: [Complex<Double>]

    @inlinable
    internal init(tMax: Double, dtMax: Double, deltaOmegaMax: Double, omegaMax: Double?) {
        precondition(tMax.isFinite && tMax > 0, "tMax must be finite and positive.")
        precondition(dtMax.isFinite && dtMax > 0, "dtMax must be finite and positive.")
        precondition(deltaOmegaMax.isFinite && deltaOmegaMax > 0,
                     "deltaOmegaMax must be finite and positive.")
        let cutoff = omegaMax ?? (.pi / dtMax)
        precondition(cutoff.isFinite && cutoff > 0,
                     "The frequency cutoff must be finite and positive.")

        // Keep the requested interval within half the FFT period, and reduce
        // the spacing slightly to integrate the exact cutoff.
        let maximumFrequencyStep = min(deltaOmegaMax, .pi / tMax)
        let requestedBinCount = (cutoff / maximumFrequencyStep).rounded(.up)
        let largestFFTCount = 1 << (Int.bitWidth - 2)
        precondition(requestedBinCount.isFinite
                     && requestedBinCount <= Double(largestFFTCount / 4),
                     "The requested frequency grid is too large.")
        let binCount = max(1, Int(requestedBinCount))
        let frequencyStep = cutoff / Double(binCount)
        precondition(frequencyStep.isFinite && frequencyStep > 0,
                     "The frequency spacing is not representable.")
        let period = (2 * Double.pi) / frequencyStep
        precondition(period.isFinite, "The FFT period is not representable.")

        // Pad with zeros rather than evaluating J at additional frequencies.
        // At least four FFT samples per shortest period also keep the occupied
        // band away from the time-grid Nyquist boundary. dtMax can refine further.
        var fftCount = 8
        while fftCount < 4 * binCount || period / Double(fftCount) > dtMax {
            precondition(fftCount < largestFFTCount,
                         "The requested time grid is too large.")
            fftCount *= 2
        }
        let timeStep = period / Double(fftCount)
        precondition(timeStep.isFinite && timeStep > 0,
                     "The time spacing is not representable.")

        var lastIndex = max(1, Int((tMax / timeStep).rounded(.up)))
        // Floating-point division and multiplication can round oppositely.
        if Double(lastIndex) * timeStep < tMax { lastIndex += 1 }
        precondition(lastIndex < fftCount, "The FFT grid must cover tMax.")
        let times = (0...lastIndex).map { Double($0) * timeStep }
        let frequencies = (0..<binCount).map { (Double($0) + 0.5) * frequencyStep }
        self.fftCount = fftCount
        self.frequencyStep = frequencyStep
        self.frequencies = frequencies
        self.times = times
        self.midpointPhases = times.map {
            Complex<Double>(length: 1, phase: -0.5 * frequencyStep * $0)
        }
    }
}
