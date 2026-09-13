// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

#if swift(<6.5)
	import BasicContainers
#endif

extension HOPS.CPUEngine {
	/// Borrow the immutable tables once per trajectory. Accessing the owning
	/// class's noncopyable fields inside a gather loop otherwise introduces
	/// ownership traffic on the hierarchy shared by all workers.
	@usableFromInline
	internal struct HierarchyTables: ~Escapable {
		@usableFromInline let count: Int
		@usableFromInline let multiIndexCount: Int
		@usableFromInline let damping: Span<Complex<Double>>
		@usableFromInline let parentIndices: Span<Int>
		@usableFromInline let childIndices: Span<Int>
		@usableFromInline let parentWeights: Span<Double>
		@usableFromInline let childWeights: Span<Double>

		@_lifetime(borrow hierarchy)
		@inlinable
		init(_ hierarchy: borrowing HOPS.Hierarchy) {
			// Swift 6.3 ends the borrow of a noncopyable class field at its
			// accessor. These immutable allocations actually live as long as
			// their owner, which the returned view is required to borrow.
			count = hierarchy.count
			multiIndexCount = hierarchy.multiIndexCount
			damping = _hopsBorrowStorage(hierarchy.kWArray, owner: hierarchy)
			parentIndices = _hopsBorrowStorage(
				hierarchy.parentIndices, owner: hierarchy)
			childIndices = _hopsBorrowStorage(hierarchy.childIndices, owner: hierarchy)
			parentWeights = _hopsBorrowStorage(
				hierarchy.parentWeights, owner: hierarchy)
			childWeights = _hopsBorrowStorage(hierarchy.childWeights, owner: hierarchy)
		}
	}
}

// Keep the field accessor's borrow alive until its span has been rebound to
// the immutable owner. All call sites pass storage owned by that exact object.
@_lifetime(borrow owner)
@inlinable
internal func _hopsBorrowStorage<Element: ~Copyable, Owner: AnyObject>(
	_ storage: borrowing UniqueArray<Element>, owner: borrowing Owner
) -> Span<Element> {
	_overrideLifetime(storage.span, borrowing: owner)
}
