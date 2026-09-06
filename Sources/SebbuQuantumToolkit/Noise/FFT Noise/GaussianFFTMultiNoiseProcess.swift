// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct GaussianFFTMultiNoiseProcessGenerator: Sendable {
    @usableFromInline
    internal let tMax: Double
    @usableFromInline
    internal let dtMax: Double
    @usableFromInline
    internal let deltaOmegaMax: Double
    @usableFromInline
    internal let omegaMax: Double?
    @usableFromInline
    internal let spectralDensity: @Sendable (Double) -> Matrix<Complex<Double>>
    
    @inlinable
    public init(tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, spectralDensity: @Sendable @escaping (_ omega: Double) -> Matrix<Complex<Double>>) {
        precondition(tMax > 0)
        precondition(dtMax > 0)
        precondition(deltaOmegaMax > 0)
        self.tMax = tMax
        self.spectralDensity = spectralDensity
        self.dtMax = dtMax
        self.deltaOmegaMax = deltaOmegaMax
        self.omegaMax = omegaMax
    }
    
    @inlinable
    @inline(always)
    public func generate<Generator: RandomNumberGenerator>(generator: inout Generator) -> [GaussianFFTNoiseProcess] {
        // Set frequency resoluation based on tMax
        let deltaOmega = min(deltaOmegaMax, .pi / tMax)
        
        // Compute minimum N so that dt <= dtMax and optionally omegaMax is covered
        var N = omegaMax != nil ? Int(omegaMax! / deltaOmega).nextPowerOf2 : 1024
        N = max(1024, N)
        var dt = 2.0 * .pi / (Double(N) * deltaOmega)
        while dt > dtMax {
            N <<= 1
            dt = 2.0 * .pi / (Double(N) * deltaOmega)
        }
        
        let omegaMax = Double(N - 1) * deltaOmega
        let omegaSpace = [Double].linearSpace(0, omegaMax, N)
        
        // Generate correlated Gaussian coefficients
        var signals: [[Complex<Double>]] = []
        var A: Matrix<Complex<Double>> = .zeros(rows: 1, columns: 1)
        for (index, omega) in omegaSpace.enumerated() {
            let J = spectralDensity(omega)
            if signals.isEmpty {
                for _ in 0..<J.rows {
                    signals.append(.init(repeating: .zero, count: omegaSpace.count))
                }
                A = .zeros(rows: J.rows, columns: J.columns)
            }
            let (eigenValues, eigenVectors) = try! MatrixOperations.diagonalizeHermitian(J)
            let sqrtD: Matrix<Complex<Double>> = .diagonal(from: eigenValues.map { Complex(($0 * deltaOmega).squareRoot()) })
            let U: Matrix<Complex<Double>> = .from(columns: eigenVectors.map { $0.components })
            U.dot(sqrtD, into: &A)
            let gaussians: [Complex<Double>] = signals.indices.map { _ in
                generator.nextNormal(stdev: Double(0.5).squareRoot())
            }
            let x = Vector(gaussians)
            let xi = A.dot(x)
            for i in 0..<xi.count {
                signals[i][index] = xi[i]
            }
        }
        // FFT
        let noises = signals.map { FFT.fft($0).spectrum }
        
        var lastIndex = max(1, Int((tMax / dt).rounded(.up)))
        if Double(lastIndex) * dt < tMax { lastIndex += 1 }
        precondition(lastIndex < N, "The FFT grid must cover tMax.")
        let trimmedTime = (0...lastIndex).map { Double($0) * dt }
        let trimmedNoises = noises.map { Array($0[0...lastIndex]) }
        
        // Interpolate
        let splines = trimmedNoises.map { CubicHermiteSpline(x: trimmedTime, y: $0) }
        return splines.map { GaussianFFTNoiseProcess(spline: $0) }
    }
}
