// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public enum OutputSchedule: Sendable {
    /// Report every accepted integration step.
    case everyAcceptedStep

    /// Report at exactly these times, using dense output/interpolation if needed.
    case times([Double])

    /// Report at a uniform spacing beginning at `timeSpan.start`.
    case uniform(step: Double)

    /// Report only the final state.
    case final
}
