// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
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

		/// One instantaneous matrix per active physical bath channel. Constant
		/// matrices are copied once when the trajectory RHS is constructed;
		/// generated/expanded matrices are refreshed once per RHS evaluation.
		@usableFromInline
		var bathMatrices: UniqueArray<UniqueMatrix<Complex<Double>>>
		@usableFromInline
		var bathMeans: UniqueVector<Complex<Double>>

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
			hamiltonian: Hamiltonian,
			preparation: borrowing Preparation,
			seed: UInt64,
			trajectoryID: UInt64
		) {
			self.dimension = preparation.dimension
			self.poles = Self.borrowPoles(
				preparation.poles, owner: preparation)
			self.bathChannels = _hopsBorrowStorage(
				preparation.bathChannels, owner: preparation)
			self.markovianOperators = _hopsBorrowStorage(
				preparation.markovianOperators, owner: preparation)
			self.rates = _hopsBorrowStorage(
				preparation.rates, owner: preparation)
			self.hierarchy = HierarchyTables(preparation)
			self.hamiltonian = hamiltonian

			var rng = TrajectoryRandomNumberGenerator(
				seed: seed,
				trajectoryID: trajectoryID,
				purpose: .coloredNoiseGeneration)
			self.noise = preparation.noise.generate(generator: &rng)
			self.coloredRNG = rng
			self.whiteRNG = .init(
				seed: seed,
				trajectoryID: trajectoryID,
				purpose: .gaussianWhiteNoise)
			self.physicalNoise = .zero(preparation.noise.channelCount)

			let d = preparation.dimension
			var bathMatrices =
				UniqueArray<UniqueMatrix<Complex<Double>>>(
					minimumCapacity: preparation.bathChannels.count)
			for i in 0..<preparation.bathChannels.count {
				switch preparation.bathChannels[i].op.constant {
				case .some(let constant):
					bathMatrices.append(
						UniqueMatrix(copying: constant.matrix))
				case .none:
					bathMatrices.append(
						.zeros(rows: d, columns: d))
				}
			}
			self.bathMatrices = bathMatrices
			self.bathMeans = .zero(preparation.bathChannels.count)

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
		mutating func evaluate(
			t: Double,
			y: borrowing State,
			dy: inout State
		) {
			drift(t: t, y: y, into: &dy)
		}

		@inlinable
		@inline(always)
		mutating func drift(
			t: Double,
			y: borrowing State,
			into dy: inout State
		) {
			if !poles.isEmpty {
				noise.sample(
					t,
					into: &physicalNoise.mutableSpan,
					generator: &coloredRNG)
			}
			evaluateWithCurrentNoise(t: t, y: y, into: &dy)
		}

		/// Also permits deterministic, pathwise equation tests with a prescribed
		/// physical noise vector. Only `drift` advances the OU sampler.
		@inlinable
		mutating func evaluateWithCurrentNoise(
			t: Double,
			y: borrowing State,
			into dy: inout State
		) {
			let bathChannels = self.bathChannels
			let markovianOperators = self.markovianOperators
			let nonlinear = equationType != .linear
			let displaced = shiftType == .meanField
			let normalized = equationType == .nonLinearNormalized
			let norm = y.rootNormSquared
			let inverseNorm = nonlinear || displaced ? 1 / norm : 1

			dy.zero()

			// 1. Build the common effective system generator H_eff
			hamiltonian.hamiltonian(t: t, into: &generator)
			generator.multiply(by: -.i)

			for p in 0..<y.shifts.count {
				dy.shifts[unchecked: p] =
					-poles[unchecked: p] * y.shifts[unchecked: p]
			}

			// Materialize each dynamic physical bath operator exactly once.
			do {
				var matrices = bathMatrices.mutableSpan
				for i in 0..<bathChannels.count {
					switch bathChannels[i].op.constant {
					case .some(_):
						break
					case .none:
						bathChannels[i].op.source.insert(
							t: t, into: &matrices[i])
						Self.validateOperator(
							matrices[i], dimension: dimension)
					}
				}
			}

			// Compute guide-root means, shift equations and all colored-noise /
			// nonlinear / nuHOPS contributions to the common system generator
			for i in 0..<bathChannels.count {
                bathMeans[unchecked: i] = Self.accumulateBathGenerator(
                    bathChannels[unchecked: i],
					matrix: bathMatrices[i],
                    noise: physicalNoise[bathChannels[unchecked: i].physicalIndex],
					nonlinear: nonlinear,
					displaced: displaced,
					inverseNorm: inverseNorm,
					y: y,
					dy: &dy,
					generator: &generator)
			}

			// Markovian drift is also common to every hierarchy row, so fold it
			// into H_eff before touching the hierarchy
			for i in 0..<markovianOperators.count {
				let rate = Self.checkedRate(rates[unchecked: i](t))
				if rate == 0 { continue }

				switch markovianOperators[unchecked: i].constant {
				case .some(let constant):
					Self.accumulateMarkovianDrift(
						matrix: constant.matrix,
						loss: constant.loss,
						rate: rate,
						nonlinear: nonlinear,
						normalized: normalized,
						y: y,
						inverseNorm: inverseNorm,
						generator: &generator)
				case .none:
					markovianOperators[unchecked: i].source.insert(
						t: t, into: &original)
					Self.validateOperator(
						original, dimension: dimension)
					OperatorApplication.loss(
						original, into: &loss)
					Self.accumulateMarkovianDrift(
						matrix: original,
						loss: loss,
						rate: rate,
						nonlinear: nonlinear,
						normalized: normalized,
						y: y,
						inverseNorm: inverseNorm,
						generator: &generator)
				}
			}

			// 2. Evaluate the complete guide physical tier first
			Self.evaluateHierarchyRow(
				branch: 0,
				tier: 0,
				gauge: 0,
				hierarchy: hierarchy,
				generator: generator,
				bathMatrices: bathMatrices,
				means: bathMeans,
				nonlinear: nonlinear,
				displaced: displaced,
				y: y.amplitudes,
				into: &dy.amplitudes,
				down: &down,
				up: &up)

			// 3. Obtain the common real normalization gauge from the complete
			//    root derivative, including its child terms
			var gauge = 0.0
			if normalized {
				var inner = Complex<Double>.zero
				for j in 0..<dimension {
					inner +=
						y.amplitudes.elements[j].conjugate
						* dy.amplitudes.elements[j]
				}
				gauge = inner.real / norm

				// We already evaluated this row with gauge == 0.
				for j in 0..<dimension {
					dy.amplitudes.elements[j] -=
						gauge * y.amplitudes.elements[j]
				}
			}

			// 4. Traverse every remaining hierarchy row exactly once
			for h in 1..<hierarchy.count {
				Self.evaluateHierarchyRow(
					branch: 0,
					tier: h,
					gauge: gauge,
					hierarchy: hierarchy,
					generator: generator,
					bathMatrices: bathMatrices,
					means: bathMeans,
					nonlinear: nonlinear,
					displaced: displaced,
					y: y.amplitudes,
					into: &dy.amplitudes,
					down: &down,
					up: &up)
			}
            
            // For correlation functions
			// Correlation companions occupy complete consecutive hierarchy
			// blocks. They use the guide's means, shifts and normalization gauge
			for branch in stride(
				from: hierarchy.count,
				to: y.amplitudes.rows,
				by: hierarchy.count
			) {
				for h in 0..<hierarchy.count {
					Self.evaluateHierarchyRow(
						branch: branch,
						tier: h,
						gauge: gauge,
						hierarchy: hierarchy,
						generator: generator,
						bathMatrices: bathMatrices,
						means: bathMeans,
						nonlinear: nonlinear,
						displaced: displaced,
						y: y.amplitudes,
						into: &dy.amplitudes,
						down: &down,
						up: &up)
				}
			}
		}

		/// Add one physical bath channel to the common effective system
		/// generator and to the finite-variation shift equations.
		///
		/// The hierarchy parent/child terms are deliberately *not* evaluated
		/// here; they are handled later by the target-tier action table.
		@inlinable
		internal static func accumulateBathGenerator(
			_ channel: borrowing Preparation.BathChannel,
			matrix: borrowing UniqueMatrix<Complex<Double>>,
			noise: Complex<Double>,
			nonlinear: Bool,
			displaced: Bool,
			inverseNorm: Double,
			y: borrowing State,
			dy: inout State,
			generator: inout UniqueMatrix<Complex<Double>>
		) -> Complex<Double> {
			let mean =
				nonlinear || displaced
				? expectation(matrix, y: y) * inverseNorm
				: .zero

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
				multiplied:
					noise.conjugate
					+ (nonlinear
						? physicalShift.conjugate
						: .zero))

			if displaced {
				for i in 0..<matrix.rows {
					for j in 0..<matrix.columns {
						generator[unchecked: i, unchecked: j] -=
							physicalShift
							* matrix[
								unchecked: j,
								unchecked: i
							].conjugate
					}
				}
				if nonlinear {
					addDiagonal(
						physicalShift * mean.conjugate,
						into: &generator)
				}
			}

			return mean
		}

		@inlinable
		internal static func accumulateMarkovianDrift(
			matrix: borrowing UniqueMatrix<Complex<Double>>,
			loss: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double,
			nonlinear: Bool,
			normalized: Bool,
			y: borrowing State,
			inverseNorm: Double,
			generator: inout UniqueMatrix<Complex<Double>>
		) {
			generator.add(loss, multiplied: -0.5 * rate)
			if nonlinear {
				let mean =
					expectation(matrix, y: y) * inverseNorm
				generator.add(
					matrix,
					multiplied: rate * mean.conjugate)
				if normalized {
					let lossMean =
						expectation(loss, y: y).real * inverseNorm
					// Stratonovich drift for stochastic Heun, as in QSD.
					addDiagonal(
						Complex(
							rate
								* (0.5 * lossMean
									- mean.lengthSquared)),
						into: &generator)
				}
			}
		}

		@inlinable
		mutating func diffusion(
			t: Double,
			y: borrowing State,
			channel: Int,
			into dy: inout State
		) {
			let markovianOperators = self.markovianOperators
			dy.zero()
			let rate =
				Self.checkedRate(rates[unchecked: channel](t))
			if rate == 0 { return }

			switch markovianOperators[unchecked: channel].constant {
			case .some(let constant):
				Self.assignDiffusion(
					constant.matrix,
					rate: rate,
					normalized:
						equationType == .nonLinearNormalized,
					y: y,
					dy: &dy)
			case .none:
				markovianOperators[unchecked: channel].source.insert(
					t: t, into: &original)
				Self.validateOperator(
					original, dimension: dimension)
				Self.assignDiffusion(
					original,
					rate: rate,
					normalized:
						equationType == .nonLinearNormalized,
					y: y,
					dy: &dy)
			}
		}

		@inlinable
		internal static func assignDiffusion(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			rate: Double,
			normalized: Bool,
			y: borrowing State,
			dy: inout State
		) {
			// nextNormal() has variance one in each real component.
			let coefficient = (0.5 * rate).squareRoot()
			OperatorApplication.apply(
				matrix,
				to: y.amplitudes,
				multiplied: Complex(coefficient),
				adding: false,
				into: &dy.amplitudes)
			if normalized {
				let mean =
					expectation(matrix, y: y)
					/ y.rootNormSquared
				dy.amplitudes.add(
					y.amplitudes,
					multiplied: -coefficient * mean)
			}
		}

		@inlinable
		@inline(always)
		mutating func sampleNormalizedNoises(
			t: Double,
			stepSize: Double,
			into noises: inout MutableSpan<Complex<Double>>
		) {
			precondition(
				noises.count == markovianOperators.count)
			for i in 0..<noises.count {
				noises[i] = whiteRNG.nextNormal()
			}
		}

		@inlinable
		@inline(always)
		internal static func checkedRate(
			_ rate: Double
		) -> Double {
			precondition(
				rate.isFinite && rate >= 0,
				"Markovian rates must be finite and nonnegative.")
			return rate
		}

		@inlinable
		internal static func expectation(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			y: borrowing State
		) -> Complex<Double> {
			var result = Complex<Double>.zero
			for i in 0..<matrix.rows {
				var value = Complex<Double>.zero
				for j in 0..<matrix.columns {
					value +=
                    Relaxed.product(matrix[unchecked: i, unchecked: j], y.amplitudes.elements[j])
				}
				result +=
					y.amplitudes.elements[i].conjugate * value
			}
			return result
		}

		@inlinable
		@inline(always)
		internal static func validateOperator(
			_ matrix: borrowing UniqueMatrix<Complex<Double>>,
			dimension: Int
		) {
			precondition(
				matrix.rows == dimension
					&& matrix.columns == dimension,
				"Generated operator dimensions do not match the system.")
		}

		@inlinable
		@inline(always)
		internal static func addDiagonal(
			_ value: Complex<Double>,
			into matrix: inout UniqueMatrix<Complex<Double>>
		) {
			for i in 0..<matrix.rows {
				matrix[unchecked: i, unchecked: i] += value
			}
		}
	}
}
