// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

/// Errors raised while constructing a coherent simulation unit system.
public enum UnitSystemError: Error, Sendable, Equatable, CustomStringConvertible {
    /// A natural-unit reference was NaN or infinite.
    case referenceMustBeFinite

    /// A natural-unit reference was zero or negative.
    case referenceMustBePositive

    /// The reference was valid, but its reciprocal scale overflowed or
    /// underflowed in `Double`.
    case derivedScaleIsNotRepresentable

    public var description: String {
        switch self {
        case .referenceMustBeFinite:
            "The natural-unit reference must be finite."
        case .referenceMustBePositive:
            "The natural-unit reference must be strictly positive."
        case .derivedScaleIsNotRepresentable:
            "The energy/time scale derived from the reference is not representable as a finite positive Double."
        }
    }
}
