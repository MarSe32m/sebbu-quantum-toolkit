// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS
import SebbuScience

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
		var loss: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var generator: UniqueMatrix<Complex<Double>>
		@usableFromInline
		var down: UniqueVector<Complex<Double>>
		@usableFromInline
		var up: UniqueVector<Complex<Double>>

		@usableFromInline
		let equationType: HOPS.EquationType

		@usableFromInline
		let shiftType: HOPS.ShiftType

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
			self.loss = .zeros(rows: d, columns: d)
			self.generator = .zeros(rows: d, columns: d)
			self.down = .zero(d)
			self.up = .zero(d)
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
			let bathChannels = self.bathChannels
			let markovianOperators = self.markovianOperators
			let nonlinear = equationType != .linear
			let displaced = shiftType == .meanField
			let normalized = equationType == .nonLinearNormalized
			let norm = y.rootNormSquared
			let inverseNorm = nonlinear || displaced ? 1 / norm : 1
			dy.zero()
			hamiltonian.hamiltonian(t: t, into: &generator)
            generator.multiply(by: -.i)
			for p in 0..<y.shifts.count {
				dy.shifts[unchecked: p] =
					-poles[unchecked: p] * y.shifts[unchecked: p]
			}
            // This is currently super unoptimal
            // The desired thing is such that
            // 1. Create Heff <- -iH + sum z_i L_i - nuHOPS shifts etc.
            //    This operator will be applied to every tier, i.e., is diagonal on the hierarchy
            // 2. Deal with the physical tier first to obtain dpsi_0
            // 3. From 2 Re (psi dot dpsi_0), we obtain the common real gauge Gamma for normalization.
            // 4. Now we go through each tier and compute their derivative
            //  4.1. Apply Heff to dpsi_k = Heff * psi_k - (Gamma + kW) psi_k
            //  4.2. Go through parents dpsi_k += L psi_k-1
            //  4.3. Go through childred dpsi_k -= L^dagger psi_k+1
            // This way we traverse the hierarchy only once instead of doing the stuff we do now
            // where we traverse it for each bath channel and for each markovian operator, and then
            // again for the diagonal stuff, and then again for the common real gauge if normalized...
            for channel in bathChannels {
                switch channel.op.constant {
                    case .some(let op):
                        Self.accumulateBath(
                            channel, matrix: op.matrix,
                            noise: physicalNoise[
                                channel.physicalIndex],
                            hierarchy: hierarchy, nonlinear: nonlinear,
                            displaced: displaced, inverseNorm: inverseNorm,
                            y: y, dy: &dy,
                            generator: &generator, down: &down, up: &up)
                    case .none:
                        channel.op.source.insert(
                            t: t, into: &original)
                        Self.validateOperator(original, dimension: dimension)
                        Self.accumulateBath(
                            channel, matrix: original,
                            noise: physicalNoise[
                                channel.physicalIndex],
                            hierarchy: hierarchy, nonlinear: nonlinear,
                            displaced: displaced, inverseNorm: inverseNorm,
                            y: y, dy: &dy,
                            generator: &generator, down: &down, up: &up)
                }
            }

			for i in 0..<markovianOperators.count {
                let rate = Self.checkedRate(rates[unchecked: i](t))
				if rate == 0 { continue }
				switch markovianOperators[unchecked: i].constant {
				case .some(let constant):
					Self.accumulateMarkovianDrift(
						matrix: constant.matrix,
						loss: constant.loss,
						rate: rate, nonlinear: nonlinear,
						normalized: normalized, y: y,
						inverseNorm: inverseNorm, generator: &generator)
				case .none:
					markovianOperators[unchecked: i].source.insert(
						t: t, into: &original)
					Self.validateOperator(original, dimension: dimension)
					OperatorApplication.loss(original, into: &loss)
					Self.accumulateMarkovianDrift(
						matrix: original, loss: loss,
						rate: rate, nonlinear: nonlinear,
						normalized: normalized, y: y,
						inverseNorm: inverseNorm, generator: &generator)
				}
			}

			// Apply the common system generator to every auxiliary.
			OperatorApplication.apply(
				generator, to: y.amplitudes, adding: true, into: &dy.amplitudes)
			// The guide is first; means, shifts and the common gauge use its root only.
			for branch in stride(from: 0, to: y.amplitudes.rows, by: hierarchy.count) {
				for h in 0..<hierarchy.count {
					let damping = hierarchy.damping[h]
					let offset = (branch + h) * dimension
					for j in 0..<dimension {
						dy.amplitudes.elements[offset + j] +=
							damping * y.amplitudes.elements[offset + j]
					}
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
			matrix: borrowing UniqueMatrix<Complex<Double>>,
			noise: Complex<Double>, hierarchy: borrowing HierarchyTables,
			nonlinear: Bool, displaced: Bool, inverseNorm: Double,
			y: borrowing State, dy: inout State,
			generator: inout UniqueMatrix<Complex<Double>>,
			down: inout UniqueVector<Complex<Double>>,
			up: inout UniqueVector<Complex<Double>>
		) {
			let mean =
				nonlinear || displaced
				? expectation(matrix, y: y) * inverseNorm : .zero
			var physicalShift = Complex<Double>.zero
			if y.shifts.count > 0 {
				for i in 0..<channel.directions.count {
					let direction = channel.directions[i]
					physicalShift +=
						direction.upward
						* y.shifts[unchecked: direction.index]
					dy.shifts[unchecked: direction.index] +=
						direction.downward * mean
				}
			}
			generator.add(
				matrix,
				multiplied: noise.conjugate
					+ (nonlinear ? physicalShift.conjugate : .zero))
			if displaced {
				for i in 0..<matrix.rows {
					for j in 0..<matrix.columns {
						generator[unchecked: i, unchecked: j] -=
							physicalShift
							* matrix[unchecked: j, unchecked: i]
							.conjugate
					}
				}
				if nonlinear {
					addDiagonal(
						physicalShift * mean.conjugate, into: &generator)
				}
			}

			accumulateNeighbours(
				channel.connections, matrix: matrix,
				hierarchyCount: hierarchy.count,
				mean: displaced ? mean : .zero,
				adjointMean: nonlinear ? mean.conjugate : .zero,
				y: y.amplitudes, into: &dy.amplitudes, down: &down, up: &up)
		}

		@inlinable
		internal static func accumulateMarkovianDrift(
			matrix: borrowing UniqueMatrix<Complex<Double>>,
			loss: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double, nonlinear: Bool, normalized: Bool, y: borrowing State,
			inverseNorm: Double, generator: inout UniqueMatrix<Complex<Double>>
		) {
			generator.add(loss, multiplied: -0.5 * rate)
			if nonlinear {
				let mean = expectation(matrix, y: y) * inverseNorm
				generator.add(matrix, multiplied: rate * mean.conjugate)
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
			let rate = Self.checkedRate(rates[unchecked: channel](t))
			if rate == 0 { return }
			switch markovianOperators[unchecked: channel].constant {
			case .some(let constant):
				Self.assignDiffusion(
					constant.matrix, rate: rate,
					normalized: equationType
						== .nonLinearNormalized, y: y, dy: &dy)
			case .none:
				markovianOperators[unchecked: channel].source.insert(
					t: t, into: &original)
				Self.validateOperator(original, dimension: dimension)
				Self.assignDiffusion(
					original, rate: rate,
					normalized: equationType
						== .nonLinearNormalized, y: y, dy: &dy)
			}
		}

		@inlinable
		internal static func assignDiffusion(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double, normalized: Bool, y: borrowing State, dy: inout State
		) {
			// nextNormal() has variance one in each real component.
			let coefficient = (0.5 * rate).squareRoot()
			OperatorApplication.apply(
				matrix, to: y.amplitudes, multiplied: Complex(coefficient),
				adding: false, into: &dy.amplitudes)
			if normalized {
				let mean = expectation(matrix, y: y) / y.rootNormSquared
				dy.amplitudes.add(y.amplitudes, multiplied: -coefficient * mean)
			}
			// Shifts have finite variation and no direct Wiener increment.
		}

		@inlinable
        @inline(always)
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
			_ matrix: borrowing UniqueMatrix<Complex<Double>>, y: borrowing State
		) -> Complex<Double> {
			var result = Complex<Double>.zero
			for i in 0..<matrix.rows {
				var value = Complex<Double>.zero
				for j in 0..<matrix.columns {
					value +=
						matrix[unchecked: i, unchecked: j]
						* y.amplitudes.elements[j]
				}
				result += y.amplitudes.elements[i].conjugate * value
			}
			return result
		}

		@inlinable
        @inline(always)
		internal static func validateOperator(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>, dimension: Int
		) {
			precondition(
				matrix.rows == dimension && matrix.columns == dimension,
				"Generated operator dimensions do not match the system.")
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
