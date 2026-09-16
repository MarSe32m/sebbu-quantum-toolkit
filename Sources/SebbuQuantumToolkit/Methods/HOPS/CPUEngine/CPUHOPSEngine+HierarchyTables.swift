// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

extension HOPS.CPUEngine {
	/// Borrow shared damping once per trajectory. Weighted bath connections
	/// are prepared separately for each physical operator.
	@usableFromInline
	internal struct HierarchyTables: ~Escapable {
		@usableFromInline let count: Int
		@usableFromInline let damping: Span<Complex<Double>>

		@_lifetime(borrow hierarchy)
		@inlinable
		init(_ hierarchy: borrowing HOPS.Hierarchy) {
			// Swift 6.3 ends the borrow of a noncopyable class field at its
			// accessor. These immutable allocations actually live as long as
			// their owner, which the returned view is required to borrow.
			count = hierarchy.count
			damping = _hopsBorrowStorage(hierarchy.kWArray, owner: hierarchy)
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
