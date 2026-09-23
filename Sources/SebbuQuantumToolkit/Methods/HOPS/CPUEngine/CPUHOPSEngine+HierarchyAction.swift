// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine.RightHandSide {
	/// Evaluate exactly one target hierarchy row.
	///
	/// This is the hot HOPS kernel. The target row is visited once: first the
	/// common effective system generator and diagonal hierarchy damping/gauge
	/// are applied, then all physical bath channel actions connected to this
	/// target are accumulated.
	@inlinable
	@inline(always)
	internal static func evaluateHierarchyRow(
		branch: Int,
		tier h: Int,
		gauge: Double,
		hierarchy: borrowing HOPS.CPUEngine.HierarchyTables,
		generator: borrowing UniqueMatrix<Complex<Double>>,
		bathChannels: borrowing Span<HOPS.CPUEngine.Preparation.BathChannel>,
		dynamicBathMatrices:
			borrowing UniqueArray<UniqueMatrix<Complex<Double>>>,
		means: borrowing UniqueVector<Complex<Double>>,
		nonlinear: Bool,
		displaced: Bool,
		y: borrowing UniqueMatrix<Complex<Double>>,
		into output: inout UniqueMatrix<Complex<Double>>,
		down: inout UniqueVector<Complex<Double>>,
		up: inout UniqueVector<Complex<Double>>
	) {
		let d = generator.rows
		let branchBase = branch &* d
		let targetOffset = branchBase &+ h &* d
		let state = y.elements + targetOffset
		let target = output.elements + targetOffset

		// H_eff psi_k
        HOPS.CPUEngine.OperatorApplication.vector(
			generator, x: state, y: target, adding: false)

		// hierarchy.damping[h] is already -sum_p k_p W_p.
		let diagonal = hierarchy.damping[h] - Complex(gauge)
		if diagonal != .zero {
			for j in 0..<d {
				target[j] += diagonal * state[j]
			}
		}

        let actionStart = hierarchy.actionStarts[unchecked: h]
        let actionEnd = hierarchy.actionStarts[unchecked: h + 1]

		if d == 2 {
			for actionIndex in actionStart..<actionEnd {
				let action = hierarchy.actions[actionIndex]
				Self.withDenseBathMatrix(
					bathChannels[unchecked: action.channel],
					dynamicBathMatrices: dynamicBathMatrices
				) { matrix in
					let mean = displaced ? means[unchecked: action.channel] : .zero
                    let adjointMean = nonlinear ? means[unchecked: action.channel].conjugate : .zero

					var d0 = Complex<Double>.zero
					var d1 = Complex<Double>.zero
					for edgeIndex in action.parentStart..<action.parentEnd {
                        let edge = hierarchy.parents[unchecked: edgeIndex]
						let source = branchBase &+ edge.source
						d0 += edge.weight * y.elements[source]
						d1 += edge.weight * y.elements[source &+ 1]
					}

					var u0 = Complex<Double>.zero
					var u1 = Complex<Double>.zero
					for edgeIndex in action.childStart..<action.childEnd {
						let edge = hierarchy.children[edgeIndex]
						let source = branchBase + edge.source
						u0 += edge.weight * y.elements[source]
						u1 += edge.weight * y.elements[source + 1]
					}

					if action.parentStart != action.parentEnd {
						target[0] +=
							(matrix[unchecked: 0, unchecked: 0] - mean) * d0
							+ matrix[unchecked: 0, unchecked: 1] * d1
						target[1] +=
							matrix[unchecked: 1, unchecked: 0] * d0
							+ (matrix[unchecked: 1, unchecked: 1] - mean) * d1
					}

					if action.childStart != action.childEnd {
						target[0] -=
							(matrix[unchecked: 0, unchecked: 0].conjugate
								- adjointMean) * u0
							+ matrix[unchecked: 1, unchecked: 0].conjugate * u1
						target[1] -=
							matrix[unchecked: 0, unchecked: 1].conjugate * u0
							+ (matrix[unchecked: 1, unchecked: 1].conjugate
								- adjointMean) * u1
					}
				}
			}
			return
		}

		for actionIndex in actionStart..<actionEnd {
            let action = hierarchy.actions[unchecked: actionIndex]

			if action.parentStart != action.parentEnd {
				down.zeroComponents()
				for edgeIndex in action.parentStart..<action.parentEnd {
					let edge = hierarchy.parents[unchecked: edgeIndex]
					let source = branchBase &+ edge.source
					down.components._unsafeAdd(
						y.elements + source,
						multiplied: edge.weight,
						count: d)
				}

				Self.applyBathForward(
                    bathChannels[unchecked: action.channel],
					dynamicBathMatrices: dynamicBathMatrices,
					x: down.components, y: target)

				if displaced {
					let mean = means[action.channel]
					if mean != .zero {
						for j in 0..<d {
							target[j] -= mean * down.components[j]
						}
					}
				}
			}

			if action.childStart != action.childEnd {
				up.zeroComponents()
				for edgeIndex in action.childStart..<action.childEnd {
					let edge = hierarchy.children[unchecked: edgeIndex]
					let source = branchBase &+ edge.source
					up.components._unsafeAdd(
						y.elements + source,
						multiplied: edge.weight,
						count: d)
				}

				Self.applyBathAdjoint(
                    bathChannels[unchecked: action.channel],
					dynamicBathMatrices: dynamicBathMatrices,
					x: up.components, y: target,
					coefficient: -.one)

				if nonlinear {
					let adjointMean = means[action.channel].conjugate
					if adjointMean != .zero {
						for j in 0..<d {
							target[j] += adjointMean * up.components[j]
						}
					}
				}
			}
		}
	}
}
