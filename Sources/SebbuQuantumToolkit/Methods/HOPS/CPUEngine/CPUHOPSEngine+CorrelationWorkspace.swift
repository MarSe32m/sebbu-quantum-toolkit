// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// Complete hierarchies occupy consecutive row blocks. The first block is
	/// always the guide; one-sided requests need two blocks, mixed requests three.
	internal struct CorrelationWorkspace: ~Copyable {
		let hierarchyCount: Int
		let dimension: Int
		let branchCount: Int
		let ket: Int
		let bra: Int
		var operatorStorage: UniqueMatrix<Complex<Double>>
		var actionStorage: UniqueMatrix<Complex<Double>>

		init(
			request: MultiTimeOrderedCorrelationRequest, hierarchyCount: Int,
			dimension: Int
		) {
			var left = false
			var right = false
			for event in request.insertions {
				switch event.insertion {
				case .left: left = true
				case .right: right = true
				}
			}
			// Use locals: capturing partially initialized noncopyable self in
			// precondition's autoclosure can cause premature destruction in Swift 6.3.
			let branchCount = left && right ? 3 : 2
			precondition(
				hierarchyCount <= Int.max / branchCount / dimension
					/ MemoryLayout<Complex<Double>>.stride,
				"The correlation hierarchy buffer is too large.")
			self.hierarchyCount = hierarchyCount
			self.dimension = dimension
			self.branchCount = branchCount
			self.ket = left ? 1 : 0
			self.bra = right ? (left ? 2 : 1) : 0
			operatorStorage = .zeros(rows: dimension, columns: dimension)
			actionStorage = .zeros(rows: hierarchyCount, columns: dimension)
		}

		func initializeDyad(fromGuide state: inout State) {
			let count = hierarchyCount * dimension
			for branch in 1..<branchCount {
				for i in 0..<count {
					state.amplitudes.elements[branch * count + i] =
						state.amplitudes.elements[i]
				}
			}
			// One shift vector belongs to the guide. Neither insertions nor
			// initialization restart this memory or the colored-noise sampler.
		}

		mutating func insert(
			_ event: TimedCorrelationInsertion, index: Int, into state: inout State
		) throws {
			_correlationInsertionOperator(event.insertion).insert(
				t: event.time, into: &operatorStorage)
			guard
				operatorStorage.rows == dimension
					&& operatorStorage.columns == dimension
			else {
				throw
					MultiTimeOrderedCorrelationError
					.insertionOperatorDimensionMismatch(
						index: index, expected: dimension,
						rows: operatorStorage.rows,
						columns: operatorStorage.columns)
			}
			let branch: Int
			let adjoint: Bool
			switch event.insertion {
			case .left: (branch, adjoint) = (ket, false)
			case .right: (branch, adjoint) = (bra, true)
			}
			precondition(branch != 0, "Insertions must never modify the guide.")
			let offset = branch * hierarchyCount * dimension
			// B acts on every ket tier and B dagger on every bra tier. A zero
			// physical companion can retain nonzero memory in its auxiliaries.
			for h in 0..<hierarchyCount {
				for i in 0..<dimension {
					var value = Complex<Double>.zero
					for j in 0..<dimension {
						let coefficient =
							adjoint
							? operatorStorage[j, i].conjugate
							: operatorStorage[i, j]
						value +=
							coefficient
							* state.amplitudes.elements[
								offset + h * dimension + j]
					}
					actionStorage[h, i] = value
				}
			}
			for i in 0..<(hierarchyCount * dimension) {
				state.amplitudes.elements[offset + i] = actionStorage.elements[i]
			}
		}

		mutating func sample(
			observable: TimeDependentOperator, at time: Double,
			state: borrowing State, equationType: HOPS.EquationType
		) throws -> Complex<Double> {
			observable.insert(t: time, into: &operatorStorage)
			guard
				operatorStorage.rows == dimension
					&& operatorStorage.columns == dimension
			else {
				throw MultiTimeOrderedCorrelationError.observableDimensionMismatch(
					expected: dimension, rows: operatorStorage.rows,
					columns: operatorStorage.columns)
			}
			let ketOffset = ket * hierarchyCount * dimension
			let braOffset = bra * hierarchyCount * dimension
			var result = Complex<Double>.zero
			for i in 0..<dimension {
				var value = Complex<Double>.zero
				for j in 0..<dimension {
					value +=
						operatorStorage[i, j]
						* state.amplitudes.elements[ketOffset + j]
				}
				result += state.amplitudes.elements[braOffset + i].conjugate * value
			}
			// Only the guide supplies the importance weight. Companion norms
			// carry insertion amplitudes and must never divide the sample.
			return equationType == .linear ? result : result / state.rootNormSquared
		}

		/// Apply an insertion before a coincident observation, and report whether
		/// cached derivatives must be invalidated. All events are exact boundaries.
		mutating func process(
			at time: Double, state: inout State, insertionIndex: inout Int,
			cursor: inout OutputCursor, request: MultiTimeOrderedCorrelationRequest,
			equationType: HOPS.EquationType,
			observing observer: (Double, Complex<Double>) -> Void
		) throws -> Bool {
			var inserted = false
			if insertionIndex < request.insertions.count,
				request.insertions[insertionIndex].time == time
			{
				if insertionIndex == 0 { initializeDyad(fromGuide: &state) }
				try insert(
					request.insertions[insertionIndex], index: insertionIndex,
					into: &state)
				try HOPS.CPUEngine.validate(state, at: time)
				insertionIndex += 1
				inserted = true
			}
			if insertionIndex == request.insertions.count {
				while let outputTime = cursor.nextTime(through: time) {
					precondition(outputTime == time)
					observer(
						outputTime,
						try sample(
							observable: request.observable, at: time,
							state: state, equationType: equationType))
				}
			}
			return inserted
		}
	}
}
