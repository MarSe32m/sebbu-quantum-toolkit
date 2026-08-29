// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUMCWFEngine: MCWF.Implementation, MCWF.TwoTimeCorrelationImplementation, Sendable {
	@inlinable
	public init() {}

	public enum SolverError: Error, Equatable, Sendable {
		case invalidStateNorm(time: Double)
		case invalidHazard(time: Double)
		case eventNotBracketed(stepStart: Double, stepEnd: Double)
		case eventLocationDidNotConverge(
			stepStart: Double,
			stepEnd: Double,
			iterations: Int
		)
		case invalidJumpWeight(time: Double, channel: Int)
		case noAvailableJump(time: Double)
	}
}
