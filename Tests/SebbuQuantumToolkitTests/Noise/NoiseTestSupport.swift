// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

func noiseTestBath(
    _ poles: [Complex<Double>], _ rows: [[Complex<Double>]]
) -> CorrelatedBathModel.LatentBath {
    .init(poles: poles, residues: Matrix(elements: rows.flatMap { $0 }, rows: rows.count, columns: poles.count))
}

func noiseTestModel(_ kind: Int) -> CorrelatedBathModel {
    switch kind {
    case 0: // Single channel, with an independently known G exp(-W t) covariance.
        return .init(channelCount: 1, latentBaths: [
            noiseTestBath([Complex(0.7, 1.2)], [[Complex((2 * 0.7 * 1.3).squareRoot())]])
        ])
    case 1: // Identical poles in different baths must still be independent.
        return .init(channelCount: 3, latentBaths: [
            noiseTestBath([Complex(0.7, 1.2)], [[1], [0], [0]]),
            noiseTestBath([Complex(0.7, 1.2)], [[0], [Complex(0.8, 0.3)], [0]])
        ])
    case 2: // Two distinct poles driven by the SAME bath; complex cross covariance.
        return .init(channelCount: 2, latentBaths: [
            noiseTestBath([Complex(0.4, 1.1), Complex(1.2, -0.8)], [
                [Complex(1.0, 0.3), Complex(-0.4, 0.2)],
                [Complex(0.2, -0.7), Complex(0.8, 0.1)]
            ])
        ])
    default: // Partial sharing, multiple inputs and a rectangular mixing matrix.
        return .init(channelCount: 3, latentBaths: [
            noiseTestBath([Complex(0.4, 1.1), Complex(1.2, -0.8)], [
                [Complex(1.0, 0.3), Complex(-0.4, 0.2)],
                [Complex(0.2, -0.7), Complex(0.8, 0.1)], [0, 0]
            ]),
            noiseTestBath([Complex(0.9, 0.6), Complex(1.7, -0.3)], [
                [0, 0], [Complex(0.3, 0.2), Complex(0.1, -0.2)],
                [Complex(0.5, -0.4), Complex(0.9, 0.2)]
            ])
        ])
    }
}

func expectNoiseMatricesClose(
    _ actual: Matrix<Complex<Double>>, _ expected: Matrix<Complex<Double>>,
    tolerance: Double, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(actual.rows == expected.rows && actual.columns == expected.columns, sourceLocation: sourceLocation)
    for i in actual.elements.indices {
        #expect((actual.elements[i] - expected.elements[i]).length <= tolerance,
                sourceLocation: sourceLocation)
    }
}

/// Ensemble moments and six-standard-error bounds for proper Gaussian noise.
/// Expected covariances come from independent analytic models, not sample fits.
struct NoiseTestMoments {
    let dimension: Int
    var count = 0
    var mean0: [Complex<Double>]
    var mean1: [Complex<Double>]
    var equal0: Matrix<Complex<Double>>
    var equal1: Matrix<Complex<Double>>
    var lagged: Matrix<Complex<Double>>
    var pseudo: Matrix<Complex<Double>>
    var pseudo0: Matrix<Complex<Double>>
    var pseudo1: Matrix<Complex<Double>>

    init(dimension: Int) {
        self.dimension = dimension
        mean0 = .init(repeating: .zero, count: dimension)
        mean1 = mean0
        equal0 = .zeros(rows: dimension, columns: dimension)
        equal1 = equal0
        lagged = equal0
        pseudo = equal0
        pseudo0 = equal0
        pseudo1 = equal0
    }

    mutating func record(_ first: [Complex<Double>], _ second: [Complex<Double>]) {
        count += 1
        for i in 0..<dimension {
            mean0[i] += first[i]
            mean1[i] += second[i]
            for j in 0..<dimension {
                equal0[i, j] += first[i] * first[j].conjugate
                equal1[i, j] += second[i] * second[j].conjugate
                lagged[i, j] += second[i] * first[j].conjugate
                pseudo[i, j] += second[i] * first[j]
                pseudo0[i, j] += first[i] * first[j]
                pseudo1[i, j] += second[i] * second[j]
            }
        }
    }

    func check(
        equal: Matrix<Complex<Double>>, lag: Matrix<Complex<Double>>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let n = Double(count)
        for i in 0..<dimension {
            let meanTolerance = 6 * (max(0, equal[i, i].real) / n).squareRoot() + 1e-12
            #expect((mean0[i] / n).length <= meanTolerance, sourceLocation: sourceLocation)
            #expect((mean1[i] / n).length <= meanTolerance, sourceLocation: sourceLocation)
            for j in 0..<dimension {
                let error = (max(0, equal[i, i].real * equal[j, j].real) / n).squareRoot()
                let tolerance = 6 * error + 1e-12
                #expect((equal0[i, j] / n - equal[i, j]).length <= tolerance, sourceLocation: sourceLocation)
                #expect((equal1[i, j] / n - equal[i, j]).length <= tolerance, sourceLocation: sourceLocation)
                #expect((lagged[i, j] / n - lag[i, j]).length <= tolerance, sourceLocation: sourceLocation)
                // Properness has up to twice the covariance-estimator variance.
                #expect((pseudo[i, j] / n).length <= 1.5 * tolerance, sourceLocation: sourceLocation)
                #expect((pseudo0[i, j] / n).length <= 1.5 * tolerance, sourceLocation: sourceLocation)
                #expect((pseudo1[i, j] / n).length <= 1.5 * tolerance, sourceLocation: sourceLocation)
            }
        }
    }
}

struct NoiseCountingRNG: RandomNumberGenerator {
    var base: SplitMix64
    var count = 0
    init(seed: UInt64) { base = SplitMix64(seed: seed) }
    mutating func next() -> UInt64 {
        count += 1
        return base.next()
    }
}
