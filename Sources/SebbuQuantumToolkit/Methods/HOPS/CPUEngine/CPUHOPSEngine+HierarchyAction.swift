// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine.RightHandSide {
	/// Contract latent neighbours before applying the physical operator. Only
	/// two ket-sized scratch vectors are needed, independent of hierarchy size.
	@inlinable
	internal static func accumulateNeighbours(
		_ connections: borrowing HOPS.CPUEngine.BathConnections,
		matrix: borrowing UniqueMatrix<Complex<Double>>, hierarchyCount: Int,
		mean: Complex<Double>, adjointMean: Complex<Double>,
		y: borrowing UniqueMatrix<Complex<Double>>,
		into output: inout UniqueMatrix<Complex<Double>>,
		down: inout UniqueVector<Complex<Double>>, up: inout UniqueVector<Complex<Double>>
	) {
		let d = matrix.rows
		precondition(
			y.columns == d && y.rows % hierarchyCount == 0
				&& output.rows == y.rows && output.columns == d)
		if d == 2 {
			// The common TLS case stays in registers, including its gathered
			// neighbours. The shifted operators are the same for every branch.
			let a = matrix[unchecked: 0, unchecked: 0] - mean
			let b = matrix[unchecked: 0, unchecked: 1]
			let c = matrix[unchecked: 1, unchecked: 0]
			let e = matrix[unchecked: 1, unchecked: 1] - mean
			let aa = matrix[unchecked: 0, unchecked: 0].conjugate - adjointMean
			let bb = c.conjugate
			let cc = b.conjugate
			let ee = matrix[unchecked: 1, unchecked: 1].conjugate - adjointMean
			for branch in stride(from: 0, to: y.rows, by: hierarchyCount) {
				let base = 2 &* branch
				for h in 0..<hierarchyCount {
					var d0 = Complex<Double>.zero
					var d1 = Complex<Double>.zero
					var u0 = Complex<Double>.zero
					var u1 = Complex<Double>.zero
					for i in connections.parentStarts[
						h]..<connections.parentStarts[h + 1]
					{
						let edge = connections.parents[i]
						let source = base + edge.source
						d0 += edge.weight * y.elements[source]
						d1 += edge.weight * y.elements[source + 1]
					}
					for i in connections.childStarts[
						h]..<connections.childStarts[h + 1]
					{
						let edge = connections.children[i]
						let source = base + edge.source
						u0 += edge.weight * y.elements[source]
						u1 += edge.weight * y.elements[source + 1]
					}
					let target = base + 2 * h
					output.elements[target] +=
						(a * d0 + b * d1) - (aa * u0 + bb * u1)
					output.elements[target + 1] +=
						(c * d0 + e * d1) - (cc * u0 + ee * u1)
				}
			}
		} else {
			for branch in stride(from: 0, to: y.rows, by: hierarchyCount) {
				let base = d * branch
				for h in 0..<hierarchyCount {
					down.zeroComponents()
					up.zeroComponents()
					for i in connections.parentStarts[
						h]..<connections.parentStarts[h &+ 1]
					{
						let edge = connections.parents[i]
						let source = base &+ edge.source
						for j in 0..<d {
							down.components[j] +=
								edge.weight * y.elements[source &+ j]
						}
					}
					for i in connections.childStarts[
						h]..<connections.childStarts[h &+ 1]
					{
						let edge = connections.children[i]
						let source = base &+ edge.source
						for j in 0..<d {
							up.components[j] +=
								edge.weight * y.elements[source &+ j]
						}
					}
					let target = output.elements + (base &+ d &* h)
					HOPS.CPUEngine.OperatorApplication.vector(
						matrix, x: down.components, y: target, adding: true)
					HOPS.CPUEngine.OperatorApplication.vector(
						matrix, adjoint: true,
						x: up.components, y: target, coefficient: -.one,
						adding: true)
					if mean != .zero || adjointMean != .zero {
						for j in 0..<d {
							target[j] +=
								adjointMean * up.components[j]
								- mean * down.components[j]
						}
					}
				}
			}
		}
	}
}
