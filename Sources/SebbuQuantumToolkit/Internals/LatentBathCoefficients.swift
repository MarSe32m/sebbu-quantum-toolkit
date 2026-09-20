// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

/// The common latent contractions used by HOPS and HEOM:
/// Lambda_p = sum_i conj(R_ip) L_i; M_p = sum_i downward_ip L_i.
/// Retains pole order and full within-bath covariance, including complex phases.
@usableFromInline
internal struct _LatentBathCoefficients {
    @usableFromInline let poles: [Complex<Double>]
    @usableFromInline let upward: Matrix<Complex<Double>>
    @usableFromInline let downward: Matrix<Complex<Double>>

    @inlinable
    init(_ model: CorrelatedBathModel) {
        poles = model.latentBaths.flatMap(\.poles)
        var up = Matrix<Complex<Double>>.zeros(rows: model.channelCount, columns: poles.count)
        var down = Matrix<Complex<Double>>.zeros(rows: model.channelCount, columns: poles.count)
        var offset = 0
        for bath in model.latentBaths {
            let covariance = bath.stationaryCovariance
            for i in 0..<model.channelCount {
                for p in 0..<bath.poleCount {
                    up[i, offset + p] = bath.residues[i, p]
                    var value = Complex<Double>.zero
                    for q in 0..<bath.poleCount {
                        value += covariance[p, q] * bath.residues[i, q].conjugate
                    }
                    down[i, offset + p] = value
                }
            }
            offset += bath.poleCount
        }
        upward = up
        downward = down
    }
}
