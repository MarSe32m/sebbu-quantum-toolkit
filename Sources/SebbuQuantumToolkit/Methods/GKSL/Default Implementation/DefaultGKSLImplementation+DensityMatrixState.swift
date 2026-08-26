// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension DefaultGKSLImplementation {
    @usableFromInline
    internal struct DensityMatrixState: ~Copyable, AdaptiveStepODESolverState {
        public var densityMatrix: UniqueMatrix<Complex<Double>>
        
        public var norm: Double { densityMatrix.frobeniusNorm }
        
        @inlinable
        public init(_ densityMatrix: borrowing UniqueMatrix<Complex<Double>>) {
            self.densityMatrix = .init(copying: densityMatrix)
        }
        
        @inlinable
        public init(_ densityMatrix: Matrix<Complex<Double>>) {
            self.densityMatrix = .init(copying: densityMatrix)
        }
        
        @inlinable
        public init(dimension: Int) {
            self.densityMatrix = .zeros(rows: dimension, columns: dimension)
        }
        
        @inlinable
        @inline(always)
        public func errorNorm(to other: borrowing DensityMatrixState) -> Double {
            self.densityMatrix.frobeniusDistance(to: other.densityMatrix)
        }
        
        @inlinable
        public func normalizedError(comparedTo lowerOrderEstimate: borrowing DensityMatrixState, relativeTo stepStart: borrowing DensityMatrixState, absoluteTolerance: Double, relativeTolerance: Double) -> Double {
            let dimensionSquared = Double(densityMatrix.rows * densityMatrix.columns)
            var error: Double = .zero
            for i in 0..<densityMatrix.rows {
                for j in 0..<densityMatrix.columns {
                    let difference54 = (self.densityMatrix[i, j] - lowerOrderEstimate.densityMatrix[i, j]).length
                    let tolerances = absoluteTolerance
                    + relativeTolerance * Swift.max(self.densityMatrix[i, j].length, stepStart.densityMatrix[i, j].length)
                    let ratio = difference54 / tolerances
                    error += ratio * ratio
                }
            }
            return (error / dimensionSquared).squareRoot()
        }
        
        @inlinable
        @inline(always)
        public mutating func assign(_ other: borrowing DensityMatrixState) {
            densityMatrix.copyElements(from: other.densityMatrix)
        }
        
        @inlinable
        @inline(always)
        public mutating func add(_ other: borrowing DensityMatrixState, multiplied: Double) {
            densityMatrix.add(other.densityMatrix, multiplied: multiplied)
        }
        
        @inlinable
        @inline(always)
        public mutating func assign(_ base: borrowing DensityMatrixState, adding direction: borrowing DensityMatrixState, multipliedBy coefficient: Double) {
            densityMatrix.copyElements(from: base.densityMatrix, adding: direction.densityMatrix, multiplied: Complex(coefficient))
        }
    }
}
