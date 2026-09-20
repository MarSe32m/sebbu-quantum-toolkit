// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    /// One row-major matrix per factorially scaled ADO. Centered correlations
    /// carry a guide block followed by the companion block; only the guide
    /// determines the displacement. Shifts are integrated at every RK stage.
    @usableFromInline
    internal struct State: ~Copyable, AdaptiveStepODESolverState {
        @usableFromInline
        internal var ados: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var shifts: UniqueVector<Complex<Double>>

        @inlinable
        internal init(dimension: Int, hierarchyCount: Int, shiftCount: Int, copies: Int = 1) {
            let (matrixSize, matrixOverflow) = dimension.multipliedReportingOverflow(by: dimension)
            let (rows, rowOverflow) = hierarchyCount.multipliedReportingOverflow(by: copies)
            let (elements, elementOverflow) = rows.multipliedReportingOverflow(by: matrixSize)
            precondition(
                !matrixOverflow && !rowOverflow && !elementOverflow
                    && elements <= Int.max / MemoryLayout<Complex<Double>>.stride,
                "The HEOM state storage size overflows Int.")
            ados = .zeros(rows: rows, columns: matrixSize)
            shifts = shiftCount == 0 ? .init() : .zero(shiftCount)
        }

        @inlinable
        internal var norm: Double {
            var result = 0.0
            for i in 0..<(ados.rows * ados.columns) {
                result = max(result, ados.elements[i].length)
            }
            for i in 0..<shifts.count { result = max(result, shifts[i].length) }
            return result
        }

        @inlinable
        internal func errorNorm(to other: borrowing State) -> Double {
            var result = 0.0
            for i in 0..<(ados.rows * ados.columns) {
                result = max(
                    result,
                    (ados.elements[i] - other.ados.elements[i])
                        .length)
            }
            for i in 0..<shifts.count {
                result = max(result, (shifts[i] - other.shifts[i]).length)
            }
            return result
        }

        // A maximum component error prevents a large number of small auxiliaries
        // from diluting the error in the physical root or displacement memory.
        @inlinable
        internal func normalizedError(
            comparedTo low: borrowing State, relativeTo start: borrowing State,
            absoluteTolerance: Double, relativeTolerance: Double
        ) -> Double {
            var result = 0.0
            for i in 0..<(ados.rows * ados.columns) {
                result = max(
                    result,
                    Self.scaledError(
                        ados.elements[i], low.ados.elements[i],
                        start.ados.elements[i], absoluteTolerance,
                        relativeTolerance))
            }
            for i in 0..<shifts.count {
                result = max(
                    result,
                    Self.scaledError(
                        shifts[i], low.shifts[i], start.shifts[i],
                        absoluteTolerance, relativeTolerance))
            }
            return result
        }

        @inlinable
        @inline(always)
        internal static func scaledError(
            _ high: Complex<Double>, _ low: Complex<Double>,
            _ start: Complex<Double>, _ atol: Double, _ rtol: Double
        ) -> Double {
            let error = (high - low).length
            let scale = atol + rtol * max(high.length, start.length)
            guard error.isFinite && scale.isFinite else { return .infinity }
            return scale > 0 ? error / scale : (error == 0 ? 0 : .infinity)
        }

        @inlinable
        @inline(always)
        mutating func zero() {
            ados.zeroElements()
            if shifts.count > 0 { shifts.zeroComponents() }
        }

        @inlinable
        @inline(always)
        mutating func assign(_ other: borrowing State) {
            ados.copyElements(from: other.ados)
            if shifts.count > 0 { shifts.copyComponents(from: other.shifts) }
        }

        @inlinable
        @inline(always)
        mutating func add(_ other: borrowing State, multiplied coefficient: Double) {
            ados.add(other.ados, multiplied: coefficient)
            if shifts.count > 0 { shifts.add(other.shifts, multiplied: coefficient) }
        }

        @inlinable
        @inline(always)
        mutating func assign(
            _ base: borrowing State, adding direction: borrowing State,
            multipliedBy coefficient: Double
        ) {
            ados.copyElements(
                from: base.ados, adding: direction.ados,
                multiplied: Complex(coefficient))
            if shifts.count > 0 {
                shifts.copyComponents(
                    from: base.shifts, adding: direction.shifts,
                    multiplied: coefficient)
            }
        }

    }
}
