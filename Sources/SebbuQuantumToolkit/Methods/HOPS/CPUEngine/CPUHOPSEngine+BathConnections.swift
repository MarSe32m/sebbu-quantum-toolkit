// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HOPS.CPUEngine {
	/// CSR-like hierarchy action table grouped by target hierarchy state.
	///
	/// Each target state stores only the physical bath channels that actually
	/// have a parent and/or child contribution. Within one channel action the
	/// latent directions have already been contracted into weighted source
	/// edges, so the RHS can gather one parent and one child ket and apply the
	/// physical coupling operator only once.
	@usableFromInline
	internal struct BathConnections: ~Copyable, Sendable {
		@usableFromInline
		struct Edge: Sendable {
			@usableFromInline let source: Int
			@usableFromInline let weight: Complex<Double>

			@inlinable
			init(source: Int, weight: Complex<Double>) {
				self.source = source
				self.weight = weight
			}
		}

		@usableFromInline
		struct Action: Sendable {
			@usableFromInline let channel: Int
			@usableFromInline let parentStart: Int
			@usableFromInline let parentEnd: Int
			@usableFromInline let childStart: Int
			@usableFromInline let childEnd: Int

			@inlinable
			init(
				channel: Int,
				parentStart: Int, parentEnd: Int,
				childStart: Int, childEnd: Int
			) {
				self.channel = channel
				self.parentStart = parentStart
				self.parentEnd = parentEnd
				self.childStart = childStart
				self.childEnd = childEnd
			}
		}

		/// `actionStarts[h]..<actionStarts[h + 1]` are the channel actions for
		/// target hierarchy state `h`.
		@usableFromInline let actionStarts: UniqueArray<Int>
		@usableFromInline let actions: UniqueArray<Action>
		@usableFromInline let parents: UniqueArray<Edge>
		@usableFromInline let children: UniqueArray<Edge>

		@inlinable
		init(
			hierarchy: HOPS.Hierarchy,
			dimension: Int,
			channels: borrowing UniqueArray<Preparation.BathChannel>
		) {
			var actionStarts = UniqueArray<Int>(
				minimumCapacity: hierarchy.count + 1)
			var actions = UniqueArray<Action>()
			var parents = UniqueArray<Edge>()
			var children = UniqueArray<Edge>()

			for h in 0..<hierarchy.count {
				actionStarts.append(actions.count)

				for channelIndex in 0..<channels.count {
					let parentStart = parents.count
					let childStart = children.count

					for directionIndex in 0..<channels[channelIndex].directions.count {
						let direction =
							channels[channelIndex].directions[directionIndex]
						let edge =
							h * hierarchy.multiIndexCount &+ direction.index

						let parent = hierarchy.parentIndices[edge]
						if parent >= 0 && direction.downward != .zero {
							parents.append(
								.init(
									source: parent &* dimension,
									weight:
										direction.downward
										* hierarchy.parentWeights[edge]))
						}

						let child = hierarchy.childIndices[edge]
						if child >= 0 && direction.upward != .zero {
							children.append(
								.init(
									source: child &* dimension,
									weight:
										direction.upward
										* hierarchy.childWeights[edge]))
						}
					}

					if parentStart != parents.count || childStart != children.count {
						actions.append(
							.init(
								channel: channelIndex,
								parentStart: parentStart,
								parentEnd: parents.count,
								childStart: childStart,
								childEnd: children.count))
					}
				}
			}

			actionStarts.append(actions.count)
			self.actionStarts = actionStarts
			self.actions = actions
			self.parents = parents
			self.children = children
		}
	}
}
