// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
#if swift(<6.5)
import BasicContainers
#else
#warning("Remove swift-collections dependency")
#endif

extension HOPS.CPUEngine {
    @usableFromInline
	internal struct RightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable, ODERHSFunction,
		SDERHSFunction
	{
		@usableFromInline
        let preparation: Preparation
        @usableFromInline
        let hierarchy: HOPS.Hierarchy
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
        
        @inlinable
		init(
			hamiltonian: Hamiltonian, preparation: Preparation, seed: UInt64,
			trajectoryID: UInt64
		) {
			self.preparation = preparation
            self.hierarchy = preparation.configuration.hierarchy
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

        @inlinable
        @inline(always)
		mutating func evaluate(t: Double, y: borrowing State, dy: inout State) {
			drift(t: t, y: y, into: &dy)
		}

		@inlinable
        @inline(always)
        mutating func drift(t: Double, y: borrowing State, into dy: inout State) {
			noise.sample(t, into: &physicalNoise.mutableSpan, generator: &coloredRNG)
			evaluateWithCurrentNoise(t: t, y: y, into: &dy)
		}

		/// Also permits deterministic, pathwise equation tests with a prescribed
		/// physical noise vector. Only `drift` advances the OU sampler.
        @inlinable
		mutating func evaluateWithCurrentNoise(
			t: Double, y: borrowing State, into dy: inout State
		) {
			//let configuration = preparation.configuration
			let nonlinear = equationType != .linear
			let displaced = shiftType == .meanField
			let normalized = equationType == .nonLinearNormalized
			let norm = y.rootNormSquared
			let inverseNorm = nonlinear || displaced ? 1 / norm : 1
			dy.zero()
			hamiltonian.hamiltonian(t: t, into: &original)
			precondition(
				original.rows == preparation.dimension
					&& original.columns == preparation.dimension,
				"Hamiltonian dimensions do not match the system.")
			for i in 0..<generator.rows {
				for j in 0..<generator.columns {
					generator[unchecked: i, unchecked: j] =
						-.i * original[unchecked: j, unchecked: i]
				}
			}
			for p in 0..<y.shifts.count {
				dy.shifts[p] = -preparation.poles[p] * y.shifts[p]
			}

			for i in 0..<preparation.bathChannels.count {
				let channel = preparation.bathChannels[i]
				if let op = channel.op.constant {
					Self.accumulateBath(
						channel, transpose: op.transpose,
						adjointTranspose: op.adjointTranspose,
						noise: physicalNoise[channel.physicalIndex],
						hierarchy: hierarchy, nonlinear: nonlinear,
						displaced: displaced, inverseNorm: inverseNorm,
						y: y, dy: &dy,
						generator: &generator, gathered: &gathered)
				} else {
					channel.op.source.insert(t: t, into: &original)
					OperatorMatrices.transpose(
						original, into: &transpose,
						adjointInto: &adjointTranspose)
					Self.accumulateBath(
						channel, transpose: transpose,
						adjointTranspose: adjointTranspose,
						noise: physicalNoise[channel.physicalIndex],
						hierarchy: hierarchy, nonlinear: nonlinear,
						displaced: displaced, inverseNorm: inverseNorm,
						y: y, dy: &dy,
						generator: &generator, gathered: &gathered)
				}
			}

			for i in 0..<preparation.markovianOperators.count {
				let rate = Self.checkedRate(preparation.rates[i](t))
				if rate == 0 { continue }
				let op = preparation.markovianOperators[i]
				if let constant = op.constant {
					Self.accumulateMarkovianDrift(
						transpose: constant.transpose,
						loss: constant.lossTranspose,
						rate: rate, nonlinear: nonlinear,
						normalized: normalized, y: y,
						inverseNorm: inverseNorm, generator: &generator)
				} else {
					op.source.insert(t: t, into: &original)
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

			// The common system generator acts on all auxiliaries in one GEMM.
			y.amplitudes.dotBLAS(generator, addingInto: &dy.amplitudes)
			for h in 0..<hierarchy.count {
				let damping = hierarchy.kWArray[h]
				let offset = h * preparation.dimension
				for j in 0..<preparation.dimension {
					dy.amplitudes.elements[offset + j] +=
						damping * y.amplitudes.elements[offset + j]
				}
			}
			if normalized {
				var inner = Complex<Double>.zero
				for j in 0..<preparation.dimension {
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
			_ channel: Preparation.BathChannel,
			transpose: borrowing UniqueMatrix<Complex<Double>>,
			adjointTranspose: borrowing UniqueMatrix<Complex<Double>>,
			noise: Complex<Double>, hierarchy: HOPS.Hierarchy,
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
			gathered.dotBLAS(transpose, addingInto: &dy.amplitudes)
			if displaced { dy.amplitudes.add(gathered, multiplied: -mean) }
			gather(
				channel.directions, parents: false, hierarchy: hierarchy,
				y: y.amplitudes, into: &gathered)
			gathered.dotBLAS(
				adjointTranspose, multiplied: -.one, addingInto: &dy.amplitudes)
			if nonlinear { dy.amplitudes.add(gathered, multiplied: mean.conjugate) }
		}

		@inlinable
        internal static func gather(
			_ directions: [Preparation.Direction], parents: Bool,
			hierarchy: HOPS.Hierarchy, y: borrowing UniqueMatrix<Complex<Double>>,
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
			dy.zero()
			let rate = Self.checkedRate(preparation.rates[channel](t))
			if rate == 0 { return }
			let op = preparation.markovianOperators[channel]
			if let constant = op.constant {
				Self.assignDiffusion(
					constant.transpose, rate: rate,
					normalized: preparation.configuration.equationType
						== .nonLinearNormalized, y: y, dy: &dy)
			} else {
				op.source.insert(t: t, into: &original)
				OperatorMatrices.transpose(
					original, into: &transpose, adjointInto: &adjointTranspose)
				Self.assignDiffusion(
					transpose, rate: rate,
					normalized: preparation.configuration.equationType
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
			y.amplitudes.dotBLAS(
				transpose, multiplied: Complex(coefficient), into: &dy.amplitudes)
			if normalized {
				let mean = expectation(transpose, y: y) / y.rootNormSquared
				dy.amplitudes.add(y.amplitudes, multiplied: -coefficient * mean)
			}
			// Shifts have finite variation and no direct Wiener increment.
		}

        @inlinable
		mutating func sampleNormalizedNoises(
			t: Double, stepSize: Double,
			into noises: inout MutableSpan<Complex<Double>>
		) {
			precondition(noises.count == preparation.markovianOperators.count)
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
