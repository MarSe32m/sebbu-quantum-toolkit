// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

@usableFromInline
internal func _noiseCovarianceFactor(
    _ covariance: Matrix<Complex<Double>>
) -> Matrix<Complex<Double>> {
    precondition(covariance.rows > 0 && covariance.isSquare,
                 "A noise covariance must be a nonempty square matrix.")
    var scale = 0.0
    for value in covariance.elements {
        precondition(value.real.isFinite && value.imaginary.isFinite,
                     "Noise covariances must be finite.")
        scale = max(scale, max(abs(value.real), abs(value.imaginary)))
    }
    let tolerance = 100 * Double.ulpOfOne * Double(covariance.rows)
        * max(scale, Double.leastNormalMagnitude)
    for i in 0..<covariance.rows {
        for j in i..<covariance.columns {
            precondition((covariance[i, j] - covariance[j, i].conjugate).length <= tolerance,
                         "A noise covariance must be Hermitian.")
        }
    }
    return MatrixOperations.positiveSemidefiniteSquareRoot(covariance)
}
