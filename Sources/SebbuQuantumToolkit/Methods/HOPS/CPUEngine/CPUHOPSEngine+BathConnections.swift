// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

#if swift(<6.5)
	import BasicContainers
#endif

extension HOPS.CPUEngine {

	/// Weighted neighbours in one hierarchy block. Offsets are measured in
	/// amplitudes; a correlation branch adds its own block offset at runtime.
	@usableFromInline
	internal struct BathConnections: ~Copyable, Sendable {
		@usableFromInline
		struct Edge: Sendable {
			@usableFromInline let source: Int
			@usableFromInline let weight: Complex<Double>
			@inlinable init(source: Int, weight: Complex<Double>) {
				self.source = source
				self.weight = weight
			}
		}
		@usableFromInline let parentStarts: UniqueArray<Int>
		@usableFromInline let childStarts: UniqueArray<Int>
		@usableFromInline let parents: UniqueArray<Edge>
		@usableFromInline let children: UniqueArray<Edge>

		@inlinable
		init(
			hierarchy: HOPS.Hierarchy, dimension: Int,
			directions: borrowing UniqueArray<Preparation.Direction>
		) {
			var parentStarts = UniqueArray<Int>(minimumCapacity: hierarchy.count + 1)
			var childStarts = UniqueArray<Int>(minimumCapacity: hierarchy.count + 1)
			var parents = UniqueArray<Edge>()
			var children = UniqueArray<Edge>()
			for h in 0..<hierarchy.count {
				parentStarts.append(parents.count)
				childStarts.append(children.count)
				for i in 0..<directions.count {
					let direction = directions[i]
					let edge = h * hierarchy.multiIndexCount + direction.index
					let parent = hierarchy.parentIndices[edge]
					let child = hierarchy.childIndices[edge]
					if parent >= 0 && direction.downward != .zero {
						parents.append(
							.init(
								source: parent * dimension,
								weight: direction.downward
									* hierarchy.parentWeights[
										edge]))
					}
					if child >= 0 && direction.upward != .zero {
						children.append(
							.init(
								source: child * dimension,
								weight: direction.upward
									* hierarchy.childWeights[
										edge]))
					}
				}
			}
			parentStarts.append(parents.count)
			childStarts.append(children.count)
			self.parentStarts = parentStarts
			self.childStarts = childStarts
			self.parents = parents
			self.children = children
		}
	}
}
