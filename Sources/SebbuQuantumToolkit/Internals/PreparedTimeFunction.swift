// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Span subscripts of copyable enums can copy their closure contexts.
/// A noncopyable representation forces borrowing, including when the
/// coefficient is a generated function rather than a constant.
@usableFromInline
internal enum _PreparedTimeFunction<Value: Sendable>: ~Copyable, Sendable {
    case constant(Value)
    case generated(@Sendable (Double) -> Value)

    @inlinable
    init(_ source: TimeFunction<Value>) {
        switch source {
        case .constant(let value): self = .constant(value)
        case .generated(let function): self = .generated(function)
        }
    }

    /// Read a constant without evaluating a generated function. Keep the
    /// pattern match on `self`: Swift 6.4 on Windows crashes in SILGen when
    /// switching directly over a UniqueArray subscript with a borrow accessor.
    @inlinable
    borrowing func constantValue(or fallback: Value) -> Value {
        switch self {
        case .constant(let value): value
        case .generated: fallback
        }
    }

    @inlinable
    borrowing func callAsFunction(_ t: Double) -> Value {
        switch self {
        case .constant(let value): value
        case .generated(let function): function(t)
        }
    }
}
