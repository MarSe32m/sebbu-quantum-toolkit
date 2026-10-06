// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	@usableFromInline
	internal struct RightHandSide<
        BathNoise: MultiChannelNoiseRealization & ~Copyable,
        MarkovianNoise: DiffusiveNoiseRealization & ~Copyable
    >: ~Copyable, ~Escapable, ODERHSFunction, SDERHSFunction
	{
		@usableFromInline
		let dimension: Int
		@usableFromInline
		let poles: Span<Complex<Double>>
		@usableFromInline
		let bathChannels: Span<Preparation.BathChannel>
        @usableFromInline
        var markovianOperators: UniqueArray<_PreparedMarkovianChannel>
        @usableFromInline
        var markovianActions: UniqueArray<_DiffusiveAction>
		@usableFromInline
		let hierarchy: HierarchyTables
		@usableFromInline
		var hamiltonian: _PreparedOperator
		@usableFromInline
        var bathNoise: BathNoise
        @usableFromInline
        var markovianNoise: MarkovianNoise
		@usableFromInline
		var physicalNoise: UniqueVector<Complex<Double>>

		/// Scratch matrices only for time-dependent physical bath channels.
		/// Constant matrices remain owned by the shared Preparation and are
		/// borrowed through `bathChannels`.
		@usableFromInline
		var dynamicBathMatrices: UniqueArray<UniqueMatrix<Complex<Double>>>
		@usableFromInline
		var bathMeans: UniqueVector<Complex<Double>>

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
			hamiltonian: TimeDependentOperator,
			preparation: borrowing Preparation,
            bathNoise: consuming BathNoise,
            markovianNoise: consuming MarkovianNoise
		) {
			self.dimension = preparation.dimension
			self.poles = Self.borrowPoles(
				preparation.poles, owner: preparation)
			self.bathChannels = _hopsBorrowStorage(
				preparation.bathChannels, owner: preparation)
            self.markovianOperators = _prepareMarkovianChannels(preparation.markovianChannels, dimension: preparation.dimension)
            self.markovianActions = .init(minimumCapacity: preparation.markovianChannels.count)
			self.hierarchy = HierarchyTables(preparation)
			self.hamiltonian = .init(hamiltonian, dimension: preparation.dimension)

            self.bathNoise = bathNoise
            self.markovianNoise = markovianNoise
            self.physicalNoise = .zero(preparation.configuration.hierarchy.environment.bath.channelCount)

			let d = preparation.dimension
			var dynamicBathMatrices =
				UniqueArray<UniqueMatrix<Complex<Double>>>(
					minimumCapacity: preparation.dynamicBathMatrixCount)
			for _ in 0..<preparation.dynamicBathMatrixCount {
				dynamicBathMatrices.append(
					.zeros(rows: d, columns: d))
			}
			self.dynamicBathMatrices = dynamicBathMatrices
			self.bathMeans = .zero(preparation.bathChannels.count)

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
		internal static func withDenseBathMatrix(
			_ channel: borrowing Preparation.BathChannel,
			dynamicBathMatrices:
				borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
			_ body: (borrowing UniqueMatrix<Complex<Double>>) -> Void
		) {
			switch channel.op {
			case .dense(let matrix):
				body(matrix)
			case .dynamic(_, let matrixIndex):
				body(dynamicBathMatrices[matrixIndex])
			case .sparse:
				preconditionFailure(
					"The specialized d == 2 path must remain dense.")
			}
		}

		@inlinable
		@inline(always)
		internal static func applyBathForward(
			_ channel: borrowing Preparation.BathChannel,
			dynamicBathMatrices:
				borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
			x: UnsafePointer<Complex<Double>>,
			y: UnsafeMutablePointer<Complex<Double>>,
			coefficient: Complex<Double> = .one
		) {
			switch channel.op {
			case .dense(let matrix):
				OperatorApplication.vector(
					matrix, x: x, y: y,
					coefficient: coefficient, adding: true)
			case .sparse(let sparse):
				OperatorApplication.vector(
					sparse.matrix, x: x, y: y,
					coefficient: coefficient, adding: true)
			case .dynamic(_, let matrixIndex):
				OperatorApplication.vector(
					dynamicBathMatrices[matrixIndex],
					x: x, y: y,
					coefficient: coefficient, adding: true)
			}
		}

		@inlinable
		@inline(always)
		internal static func applyBathAdjoint(
			_ channel: borrowing Preparation.BathChannel,
			dynamicBathMatrices:
				borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
			x: UnsafePointer<Complex<Double>>,
			y: UnsafeMutablePointer<Complex<Double>>,
			coefficient: Complex<Double> = .one
		) {
			switch channel.op {
			case .dense(let matrix):
				OperatorApplication.vector(
					matrix, adjoint: true, x: x, y: y,
					coefficient: coefficient, adding: true)
			case .sparse(let sparse):
				OperatorApplication.vector(
					sparse.adjoint, x: x, y: y,
					coefficient: coefficient, adding: true)
			case .dynamic(_, let matrixIndex):
				OperatorApplication.vector(
					dynamicBathMatrices[matrixIndex],
					adjoint: true, x: x, y: y,
					coefficient: coefficient, adding: true)
			}
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
            bathNoise.sample(t, into: &physicalNoise.mutableSpan)
			evaluateWithCurrentNoise(t: t, y: y, into: &dy)
		}

		/// Also permits deterministic, pathwise equation tests with a prescribed
		/// physical noise vector. Only `drift` samples the supplied colored realization.
		@inlinable
		mutating func evaluateWithCurrentNoise(
			t: Double,
			y: borrowing State,
			into dy: inout State
		) {
			let bathChannels = self.bathChannels
			let nonlinear = equationType != .linear
			let displaced = shiftType == .meanField
			let normalized = equationType == .nonLinearNormalized
			let norm = y.rootNormSquared
			let inverseNorm = nonlinear || displaced ? 1 / norm : 1

			dy.zero()

            // Base H stays separate. Every coefficient is evaluated before
            // any guide, companion or hierarchy-row action.
            hamiltonian.updateChecked(at: t)
            generator.zeroElements()

			for p in 0..<y.shifts.count {
				dy.shifts[unchecked: p] =
					-poles[unchecked: p] * y.shifts[unchecked: p]
			}

			// Materialize each dynamic physical bath operator exactly once.
			// Constant matrices stay in Preparation and are never copied.
			do {
				var matrices = dynamicBathMatrices.mutableSpan
				for i in 0..<bathChannels.count {
					switch bathChannels[i].op {
					case .dynamic(let source, let matrixIndex):
						source.insert(
							t: t, into: &matrices[matrixIndex])
						Self.validateOperator(
							matrices[matrixIndex],
							dimension: dimension)
                    case .dense: continue
                    case  .sparse: continue
					}
				}
			}

			// Compute guide-root means, shift equations and all colored-noise /
			// nonlinear / nuHOPS contributions to the common system generator
			for i in 0..<bathChannels.count {
				bathMeans[unchecked: i] =
					Self.accumulateBathGenerator(
					bathChannels[unchecked: i],
					dynamicBathMatrices: dynamicBathMatrices,
					noise:
						physicalNoise[
							bathChannels[unchecked: i].physicalIndex],
					nonlinear: nonlinear,
					displaced: displaced,
					inverseNorm: inverseNorm,
					y: y,
					dy: &dy,
					generator: &generator)
			}

            // Dense channels are fused into the common correction generator;
            // sparse/mixed channels retain factorized actions for every row.
            var hasCorrection = !bathChannels.isEmpty
            markovianActions.removeAll() // UniqueArray preserves its reserved capacity.
            for i in 0..<markovianOperators.count {
                let rate = Self.checkedRate(markovianOperators[i].rate(t))
                if rate == 0 { continue }
                markovianOperators[i].updateChecked(at: t)
                let action = _DiffusiveAction(collapse: markovianOperators[i].action, rate: rate,
                    nonlinear: nonlinear, normalized: normalized,
                    guide: y.amplitudes.elements, norm: norm, scratch: down.components)
                if action.addDenseGenerator(into: generator.elements) { hasCorrection = true }
                else { markovianActions.append(action) }
            }

			// 2. Evaluate the complete guide physical tier first
			Self.evaluateHierarchyRow(
				branch: 0,
				tier: 0,
				gauge: 0,
				hierarchy: hierarchy,
				generator: generator,
                hamiltonian: hamiltonian.action,
                hasCorrection: hasCorrection,
                markovianActions: markovianActions,
				bathChannels: bathChannels,
				dynamicBathMatrices: dynamicBathMatrices,
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
                hamiltonian: hamiltonian.action,
                hasCorrection: hasCorrection,
                markovianActions: markovianActions,
					bathChannels: bathChannels,
					dynamicBathMatrices: dynamicBathMatrices,
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
                hamiltonian: hamiltonian.action,
                hasCorrection: hasCorrection,
                markovianActions: markovianActions,
						bathChannels: bathChannels,
						dynamicBathMatrices: dynamicBathMatrices,
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
			dynamicBathMatrices:
				borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
			noise: Complex<Double>,
			nonlinear: Bool,
			displaced: Bool,
			inverseNorm: Double,
			y: borrowing State,
			dy: inout State,
			generator: inout UniqueMatrix<Complex<Double>>
		) -> Complex<Double> {
			switch channel.op {
			case .dense(let matrix):
				return accumulateDenseBathGenerator(
					channel,
					matrix: matrix,
					noise: noise,
					nonlinear: nonlinear,
					displaced: displaced,
					inverseNorm: inverseNorm,
					y: y,
					dy: &dy,
					generator: &generator)
			case .dynamic(_, let matrixIndex):
				return accumulateDenseBathGenerator(
					channel,
					matrix: dynamicBathMatrices[matrixIndex],
					noise: noise,
					nonlinear: nonlinear,
					displaced: displaced,
					inverseNorm: inverseNorm,
					y: y,
					dy: &dy,
					generator: &generator)
			case .sparse(let sparse):
				return accumulateSparseBathGenerator(
					channel,
					matrix: sparse.matrix,
					noise: noise,
					nonlinear: nonlinear,
					displaced: displaced,
					inverseNorm: inverseNorm,
					y: y,
					dy: &dy,
					generator: &generator)
			}
		}

		@inlinable
		internal static func accumulateDenseBathGenerator(
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
		internal static func accumulateSparseBathGenerator(
			_ channel: borrowing Preparation.BathChannel,
			matrix: borrowing UniqueCSRMatrix<Complex<Double>>,
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
				? OperatorApplication.expectation(
					matrix, state: y.amplitudes.elements) * inverseNorm
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

			let forwardCoefficient =
				noise.conjugate
					+ (nonlinear
						? physicalShift.conjugate
						: .zero)
			let adjointCoefficient =
				displaced ? -physicalShift : .zero
			OperatorApplication.addSparseBathContributions(
				matrix,
				forwardCoefficient: forwardCoefficient,
				adjointCoefficient: adjointCoefficient,
				into: &generator)

			if displaced && nonlinear {
				addDiagonal(
					physicalShift * mean.conjugate,
					into: &generator)
			}
			return mean
		}

        @inlinable
        mutating func diffusion(t: Double, y: borrowing State, channel: Int, into dy: inout State) {
            dy.zero()
            let rate = Self.checkedRate(markovianOperators[channel].rate(t))
            if rate == 0 { return }
            markovianOperators[channel].updateChecked(at: t)
            let normalized = equationType == .nonLinearNormalized
            let action = _DiffusiveAction(collapse: markovianOperators[channel].action, rate: rate,
                nonlinear: normalized, normalized: normalized,
                guide: y.amplitudes.elements, norm: y.rootNormSquared, scratch: down.components)
            for row in 0..<y.amplitudes.rows {
                action.diffusion(y.amplitudes.elements + row * dimension,
                                 into: dy.amplitudes.elements + row * dimension)
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
            markovianNoise.sampleNormalized(at: t, stepSize: stepSize, into: &noises)
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
