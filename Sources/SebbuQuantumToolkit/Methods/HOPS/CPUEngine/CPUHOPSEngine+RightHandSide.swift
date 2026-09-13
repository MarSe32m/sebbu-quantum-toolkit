// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

#if swift(<6.5)
	import BasicContainers
#else
	#warning("Remove swift-collections dependency")
#endif

extension HOPS.CPUEngine {
	@usableFromInline
	internal struct RightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable, ~Escapable,
		ODERHSFunction,
		SDERHSFunction
	{
		@usableFromInline
		let dimension: Int
		@usableFromInline
		let poles: Span<Complex<Double>>
		@usableFromInline
		let bathChannels: Span<Preparation.BathChannel>
		@usableFromInline
		let markovianOperators: Span<PreparedOperator>
		@usableFromInline
		let rates: Span<PreparedTimeFunction<Double>>
		@usableFromInline
		let hierarchy: HierarchyTables
		@usableFromInline
		let hamiltonian: Hamiltonian
		@usableFromInline
		var coloredRNG: TrajectoryRandomNumberGenerator
		@usableFromInline
		var whiteRNG: TrajectoryRandomNumberGenerator
		@usableFromInline
		var noise: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcess
		@usableFromInline
		var physicalNoise: UniqueVector<Complex<Double>>
		@usableFromInline
		var original: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var transpose: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var adjointTranspose: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var lossTranspose: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var generator: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var gathered: UniqueMatrix<Complex<Double>>

		@usableFromInline
		let equationType: HOPS.EquationType

		@usableFromInline
		let shiftType: HOPS.ShiftType

		// The solver cannot outlive the shared immutable preparation. Its hot
		// path accesses spans and value fields, never the owning classes.
		@_lifetime(borrow preparation)
		@inlinable
		init(
			hamiltonian: Hamiltonian, preparation: borrowing Preparation, seed: UInt64,
			trajectoryID: UInt64
		) {
			self.dimension = preparation.dimension
			self.poles = Self.borrowPoles(preparation.poles, owner: preparation)
			self.bathChannels = _hopsBorrowStorage(
				preparation.bathChannels, owner: preparation)
			self.markovianOperators = _hopsBorrowStorage(
				preparation.markovianOperators, owner: preparation)
			self.rates = _hopsBorrowStorage(preparation.rates, owner: preparation)
			let hierarchy = preparation.configuration.hierarchy
			self.hierarchy = _overrideLifetime(
				HierarchyTables(hierarchy), borrowing: preparation)
			self.hamiltonian = hamiltonian
			var rng = TrajectoryRandomNumberGenerator(
				seed: seed, trajectoryID: trajectoryID,
				purpose: .coloredNoiseGeneration)
			self.noise = preparation.noise.generate(generator: &rng)
			self.coloredRNG = rng
			self.whiteRNG = .init(
				seed: seed, trajectoryID: trajectoryID, purpose: .gaussianWhiteNoise
			)
			self.physicalNoise = .zero(preparation.noise.channelCount)
			let d = preparation.dimension
			self.original = .zeros(rows: d, columns: d)
			self.transpose = .zeros(rows: d, columns: d)
			self.adjointTranspose = .zeros(rows: d, columns: d)
			self.lossTranspose = .zeros(rows: d, columns: d)
			self.generator = .zeros(rows: d, columns: d)
			self.gathered = .zeros(
				rows: preparation.configuration.hierarchy.count, columns: d)
			self.equationType = preparation.configuration.equationType
			self.shiftType = preparation.configuration.shiftType
		}

		@_lifetime(borrow owner)
		@inlinable
		internal static func borrowPoles(
			_ poles: borrowing UniqueVector<Complex<Double>>,
			owner: borrowing Preparation
		) -> Span<Complex<Double>> {
			_overrideLifetime(
				Span(_unsafeStart: poles.components, count: poles.count),
				borrowing: owner)
		}

		@inlinable
		@inline(always)
		mutating func evaluate(t: Double, y: borrowing State, dy: inout State) {
			drift(t: t, y: y, into: &dy)
		}

		@inlinable
		@inline(always)
		mutating func drift(t: Double, y: borrowing State, into dy: inout State) {
			if !poles.isEmpty {
				noise.sample(
					t, into: &physicalNoise.mutableSpan, generator: &coloredRNG)
			}
			evaluateWithCurrentNoise(t: t, y: y, into: &dy)
		}

		/// Also permits deterministic, pathwise equation tests with a prescribed
		/// physical noise vector. Only `drift` advances the OU sampler.
		@inlinable
		mutating func evaluateWithCurrentNoise(
			t: Double, y: borrowing State, into dy: inout State
		) {
			// Copy only the span descriptors so pattern matching a borrowed
			// operator does not extend an exclusive access to the whole RHS.
			let bathChannels = self.bathChannels
			let markovianOperators = self.markovianOperators
			let nonlinear = equationType != .linear
			let displaced = shiftType == .meanField
			let normalized = equationType == .nonLinearNormalized
			let norm = y.rootNormSquared
			let inverseNorm = nonlinear || displaced ? 1 / norm : 1
			dy.zero()
			hamiltonian.hamiltonian(t: t, into: &original)
			precondition(
				original.rows == dimension
					&& original.columns == dimension,
				"Hamiltonian dimensions do not match the system.")
			for i in 0..<generator.rows {
				for j in 0..<generator.columns {
					generator[unchecked: i, unchecked: j] =
						-.i * original[unchecked: j, unchecked: i]
				}
			}
			for p in 0..<y.shifts.count {
				dy.shifts[p] = -poles[p] * y.shifts[p]
			}

			for i in 0..<bathChannels.count {
				switch bathChannels[i].op.constant {
				case .some(let op):
					Self.accumulateBath(
						bathChannels[i], transpose: op.transpose,
						adjointTranspose: op.adjointTranspose,
						noise: physicalNoise[bathChannels[i].physicalIndex],
						hierarchy: hierarchy, nonlinear: nonlinear,
						displaced: displaced, inverseNorm: inverseNorm,
						y: y, dy: &dy,
						generator: &generator, gathered: &gathered)
				case .none:
					bathChannels[i].op.source.insert(t: t, into: &original)
					OperatorMatrices.transpose(
						original, into: &transpose,
						adjointInto: &adjointTranspose)
					Self.accumulateBath(
						bathChannels[i], transpose: transpose,
						adjointTranspose: adjointTranspose,
						noise: physicalNoise[bathChannels[i].physicalIndex],
						hierarchy: hierarchy, nonlinear: nonlinear,
						displaced: displaced, inverseNorm: inverseNorm,
						y: y, dy: &dy,
						generator: &generator, gathered: &gathered)
				}
			}

			for i in 0..<markovianOperators.count {
				let rate = Self.checkedRate(rates[i](t))
				if rate == 0 { continue }
				switch markovianOperators[i].constant {
				case .some(let constant):
					Self.accumulateMarkovianDrift(
						transpose: constant.transpose,
						loss: constant.lossTranspose,
						rate: rate, nonlinear: nonlinear,
						normalized: normalized, y: y,
						inverseNorm: inverseNorm, generator: &generator)
				case .none:
					markovianOperators[i].source.insert(t: t, into: &original)
					OperatorMatrices.transpose(
						original, into: &transpose,
						adjointInto: &adjointTranspose)
					transpose.dotBLAS(adjointTranspose, into: &lossTranspose)
					Self.accumulateMarkovianDrift(
						transpose: transpose, loss: lossTranspose,
						rate: rate, nonlinear: nonlinear,
						normalized: normalized, y: y,
						inverseNorm: inverseNorm, generator: &generator)
				}
			}

			// Apply the common system generator to every auxiliary.
			Self.apply(generator, to: y.amplitudes, adding: true, into: &dy.amplitudes)
			for h in 0..<hierarchy.count {
				let damping = hierarchy.damping[h]
				let offset = h * dimension
				for j in 0..<dimension {
					dy.amplitudes.elements[offset + j] +=
						damping * y.amplitudes.elements[offset + j]
				}
			}
			if normalized {
				var inner = Complex<Double>.zero
				for j in 0..<dimension {
					inner +=
						y.amplitudes.elements[j].conjugate
						* dy.amplitudes.elements[j]
				}
				// Common real gauge for every tier, NOT for the shift memory.
				// The normalized Markovian Stratonovich drift is already tangent
				// to the sphere; this also removes the colored root norm drift.
				dy.amplitudes.add(y.amplitudes, multiplied: -inner.real / norm)
			}
		}

		@inlinable
		internal static func accumulateBath(
			_ channel: borrowing Preparation.BathChannel,
			transpose: borrowing UniqueMatrix<Complex<Double>>,
			adjointTranspose: borrowing UniqueMatrix<Complex<Double>>,
			noise: Complex<Double>, hierarchy: borrowing HierarchyTables,
			nonlinear: Bool, displaced: Bool, inverseNorm: Double,
			y: borrowing State, dy: inout State,
			generator: inout UniqueMatrix<Complex<Double>>,
			gathered: inout UniqueMatrix<Complex<Double>>
		) {
			let mean =
				nonlinear || displaced
				? expectation(transpose, y: y) * inverseNorm : .zero
			var physicalShift = Complex<Double>.zero
			if y.shifts.count > 0 {
				for i in 0..<channel.directions.count {
					let direction = channel.directions[i]
					physicalShift +=
						direction.upward * y.shifts[direction.index]
					dy.shifts[direction.index] += direction.downward * mean
				}
			}
			generator.add(
				transpose,
				multiplied: noise.conjugate
					+ (nonlinear ? physicalShift.conjugate : .zero))
			if displaced {
				generator.add(adjointTranspose, multiplied: -physicalShift)
				if nonlinear {
					addDiagonal(
						physicalShift * mean.conjugate, into: &generator)
				}
			}

			// M_p = sum_i S_pi L_i and Lambda_p^dagger = sum_i R_ip L_i^dagger.
			// Gather latent neighbours first: matrix applications scale with the
			// number of physical operators rather than the number of poles.
			gather(
				channel.directions, parents: true, hierarchy: hierarchy,
				y: y.amplitudes, into: &gathered)
			apply(transpose, to: gathered, adding: true, into: &dy.amplitudes)
			if displaced { dy.amplitudes.add(gathered, multiplied: -mean) }
			gather(
				channel.directions, parents: false, hierarchy: hierarchy,
				y: y.amplitudes, into: &gathered)
			apply(
				adjointTranspose, to: gathered, multiplied: -.one,
				adding: true, into: &dy.amplitudes)
			if nonlinear { dy.amplitudes.add(gathered, multiplied: mean.conjugate) }
		}

		@inlinable
		internal static func gather(
			_ directions: borrowing UniqueArray<Preparation.Direction>, parents: Bool,
			hierarchy: borrowing HierarchyTables,
			y: borrowing UniqueMatrix<Complex<Double>>,
			into output: inout UniqueMatrix<Complex<Double>>
		) {
			output.zeroElements()
			for h in 0..<hierarchy.count {
				let offset = h * y.columns
				let edgeOffset = h * hierarchy.multiIndexCount
				for i in 0..<directions.count {
					let direction = directions[i]
					let edge = edgeOffset + direction.index
					let neighbour =
						parents
						? hierarchy.parentIndices[edge]
						: hierarchy.childIndices[edge]
					if neighbour < 0 { continue }
					let coefficient =
						parents
						? direction.downward * hierarchy.parentWeights[edge]
						: direction.upward * hierarchy.childWeights[edge]
					if coefficient == .zero { continue }
					let source = neighbour * y.columns
					for j in 0..<y.columns {
						output.elements[offset + j] +=
							coefficient * y.elements[source + j]
					}
				}
			}
		}

		@inlinable
		internal static func accumulateMarkovianDrift(
			transpose: borrowing UniqueMatrix<Complex<Double>>,
			loss: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double, nonlinear: Bool, normalized: Bool, y: borrowing State,
			inverseNorm: Double, generator: inout UniqueMatrix<Complex<Double>>
		) {
			generator.add(loss, multiplied: -0.5 * rate)
			if nonlinear {
				let mean = expectation(transpose, y: y) * inverseNorm
				generator.add(transpose, multiplied: rate * mean.conjugate)
				if normalized {
					let lossMean = expectation(loss, y: y).real * inverseNorm
					// Stratonovich drift for stochastic Heun, as in QSD.
					addDiagonal(
						Complex(
							rate * (0.5 * lossMean - mean.lengthSquared)
						), into: &generator)
				}
			}
		}

		@inlinable
		mutating func diffusion(
			t: Double, y: borrowing State, channel: Int, into dy: inout State
		) {
			let markovianOperators = self.markovianOperators
			dy.zero()
			let rate = Self.checkedRate(rates[channel](t))
			if rate == 0 { return }
			switch markovianOperators[channel].constant {
			case .some(let constant):
				Self.assignDiffusion(
					constant.transpose, rate: rate,
					normalized: equationType
						== .nonLinearNormalized, y: y, dy: &dy)
			case .none:
				markovianOperators[channel].source.insert(t: t, into: &original)
				OperatorMatrices.transpose(
					original, into: &transpose, adjointInto: &adjointTranspose)
				Self.assignDiffusion(
					transpose, rate: rate,
					normalized: equationType
						== .nonLinearNormalized, y: y, dy: &dy)
			}
		}

		@inlinable
		internal static func assignDiffusion(
			_ transpose: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double, normalized: Bool, y: borrowing State, dy: inout State
		) {
			// nextNormal() has variance one in each real component.
			let coefficient = (0.5 * rate).squareRoot()
			apply(
				transpose, to: y.amplitudes, multiplied: Complex(coefficient),
				adding: false, into: &dy.amplitudes)
			if normalized {
				let mean = expectation(transpose, y: y) / y.rootNormSquared
				dy.amplitudes.add(y.amplitudes, multiplied: -coefficient * mean)
			}
			// Shifts have finite variation and no direct Wiener increment.
		}

		/// A one-row hierarchy is a matrix-vector product. GEMV avoids GEMM's
		/// packing/workspace overhead (and shared workspace contention in
		/// some BLAS implementations) for the Markovian-only limit.
		@inlinable
		internal static func apply(
			_ transpose: borrowing UniqueMatrix<Complex<Double>>,
			to states: borrowing UniqueMatrix<Complex<Double>>,
			multiplied coefficient: Complex<Double> = .one,
			adding: Bool, into output: inout UniqueMatrix<Complex<Double>>
		) {
			precondition(
				states.columns == transpose.rows
					&& output.rows == states.rows
					&& output.columns == transpose.columns)
			if states.rows == 1 {
				// Stored operators are O^T. This is a plain transpose, never
				// a conjugate transpose: (O^T)^T psi = O psi.
				BLAS.zgemv(
					layout: .rowMajor, transpose: .transpose,
					m: transpose.rows, n: transpose.columns,
					alpha: coefficient, a: transpose.elements,
					lda: transpose.columns,
					x: states.elements, incX: 1, beta: adding ? .one : .zero,
					y: output.elements, incY: 1)
			} else if adding {
				states.dotBLAS(
					transpose, multiplied: coefficient, addingInto: &output)
			} else {
				states.dotBLAS(transpose, multiplied: coefficient, into: &output)
			}
		}

		@inlinable
		mutating func sampleNormalizedNoises(
			t: Double, stepSize: Double,
			into noises: inout MutableSpan<Complex<Double>>
		) {
			precondition(noises.count == markovianOperators.count)
			for i in 0..<noises.count { noises[i] = whiteRNG.nextNormal() }
		}

		@inlinable
		@inline(always)
		internal static func checkedRate(_ rate: Double) -> Double {
			precondition(
				rate.isFinite && rate >= 0,
				"Markovian rates must be finite and nonnegative.")
			return rate
		}

		@inlinable
		internal static func expectation(
			_ transpose: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing State
		) -> Complex<Double> {
			var result = Complex<Double>.zero
			for j in 0..<transpose.columns {
				var value = Complex<Double>.zero
				for i in 0..<transpose.rows {
					value +=
						y.amplitudes.elements[i]
						* transpose[unchecked: i, unchecked: j]
				}
				result += y.amplitudes.elements[j].conjugate * value
			}
			return result
		}

		@inlinable
		@inline(always)
		internal static func addDiagonal(
			_ value: Complex<Double>, into matrix: inout UniqueMatrix<Complex<Double>>
		) {
			for i in 0..<matrix.rows { matrix[unchecked: i, unchecked: i] += value }
		}
	}
}
