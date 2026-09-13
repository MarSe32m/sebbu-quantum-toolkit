// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// One row per factorially scaled auxiliary. The mean-field memory is
	/// integrated with the hierarchy, but is never rescaled with its root norm.
    @usableFromInline
	internal struct State: ~Copyable, AdaptiveStepODESolverState, FixedStepSDESolverState {
		@usableFromInline
        internal var amplitudes: UniqueMatrix<Complex<Double>>
		@usableFromInline
        internal var shifts: UniqueVector<Complex<Double>>

        @inlinable
		internal init(dimension: Int, hierarchyCount: Int, shiftCount: Int) {
			amplitudes = .zeros(rows: hierarchyCount, columns: dimension)
			shifts = shiftCount == 0 ? .init() : .zero(shiftCount)
		}

        @inlinable
		internal var rootNormSquared: Double {
			var result = 0.0
			for j in 0..<amplitudes.columns {
				result += amplitudes[unchecked: 0, unchecked: j].lengthSquared
			}
			return result
		}

        @inlinable
		internal var norm: Double {
			var result = 0.0
			for i in 0..<(amplitudes.rows * amplitudes.columns) {
				result = max(result, amplitudes.elements[i].length)
			}
			for i in 0..<shifts.count { result = max(result, shifts[i].length) }
			return result
		}

        @inlinable
		internal func errorNorm(to other: borrowing State) -> Double {
			var result = 0.0
			for i in 0..<(amplitudes.rows * amplitudes.columns) {
				result = max(
					result,
					(amplitudes.elements[i] - other.amplitudes.elements[i])
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
			for i in 0..<(amplitudes.rows * amplitudes.columns) {
				result = max(
					result,
					Self.scaledError(
						amplitudes.elements[i], low.amplitudes.elements[i],
						start.amplitudes.elements[i], absoluteTolerance,
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
			amplitudes.zeroElements()
			if shifts.count > 0 { shifts.zeroComponents() }
		}

        @inlinable
        @inline(always)
		mutating func assign(_ other: borrowing State) {
			amplitudes.copyElements(from: other.amplitudes)
			if shifts.count > 0 { shifts.copyComponents(from: other.shifts) }
		}

        @inlinable
        @inline(always)
		mutating func add(_ other: borrowing State, multiplied coefficient: Double) {
			amplitudes.add(other.amplitudes, multiplied: coefficient)
			if shifts.count > 0 { shifts.add(other.shifts, multiplied: coefficient) }
		}

        @inlinable
        @inline(always)
		mutating func assign(
			_ base: borrowing State, adding direction: borrowing State,
			multipliedBy coefficient: Double
		) {
			amplitudes.copyElements(
				from: base.amplitudes, adding: direction.amplitudes,
				multiplied: Complex(coefficient))
			if shifts.count > 0 {
				shifts.copyComponents(
					from: base.shifts, adding: direction.shifts,
					multiplied: coefficient)
			}
		}

        @inlinable
        @inline(always)
		mutating func add(
			_ other: borrowing State, scaledBy noise: borrowing Complex<Double>
		) {
			amplitudes.add(other.amplitudes, multiplied: noise)
			if shifts.count > 0 { shifts.add(other.shifts, multiplied: noise) }
		}
	}
}
