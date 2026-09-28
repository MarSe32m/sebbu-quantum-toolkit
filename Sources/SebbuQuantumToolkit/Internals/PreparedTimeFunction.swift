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

    @inlinable
    borrowing func callAsFunction(_ t: Double) -> Value {
        switch self {
        case .constant(let value): value
        case .generated(let function): function(t)
        }
    }
}
