// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

extension HOPS.CPUEngine {
	/// Borrow all immutable hierarchy-side tables once per trajectory.
	@usableFromInline
	internal struct HierarchyTables: ~Escapable {
		@usableFromInline let count: Int
		@usableFromInline let damping: Span<Complex<Double>>
		@usableFromInline let actionStarts: Span<Int>
		@usableFromInline let actions: Span<BathConnections.Action>
		@usableFromInline let parents: Span<BathConnections.Edge>
		@usableFromInline let children: Span<BathConnections.Edge>

		@_lifetime(borrow preparation)
		@inlinable
		init(_ preparation: borrowing Preparation) {
			let hierarchy = preparation.configuration.hierarchy
			count = hierarchy.count

			// Rebind field storage to the Preparation owner. This is needed on
			// Swift 6.3/6.4 because the borrow of a noncopyable class field
			// otherwise ends at the field accessor.
			damping = _hopsBorrowStorage(
				hierarchy.kWArray, owner: preparation)
			actionStarts = _hopsBorrowStorage(
				preparation.connections.actionStarts, owner: preparation)
			actions = _hopsBorrowStorage(
				preparation.connections.actions, owner: preparation)
			parents = _hopsBorrowStorage(
				preparation.connections.parents, owner: preparation)
			children = _hopsBorrowStorage(
				preparation.connections.children, owner: preparation)
		}
	}
}

// Keep the field accessor's borrow alive until its span has been rebound to
// the immutable owner. All call sites pass storage owned by that exact object.
@_lifetime(borrow owner)
@inlinable
internal func _hopsBorrowStorage<Element: ~Copyable, Owner>(
	_ storage: borrowing UniqueArray<Element>, owner: borrowing Owner
) -> Span<Element> {
	_overrideLifetime(storage.span, borrowing: owner)
}
