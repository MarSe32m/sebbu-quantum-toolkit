// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUQSDEngine: QSD.Implementation, QSD.TwoTimeCorrelationImplementation, Sendable {
	@inlinable
	public init() {}

	public enum SolverError: Error, Equatable {
		case invalidStateNorm(time: Double)
	}
}
