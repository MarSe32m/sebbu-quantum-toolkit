// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct IntegrationOptions: Sendable {
	public var minimumStepSize: Double
	public var maximumStepSize: Double
	public var absoluteTolerance: Double
	public var relativeTolerance: Double

	@inlinable
	public init(
		minimumStepSize: Double, maximumStepSize: Double, absoluteTolerance: Double,
		relativeTolerance: Double
	) {
		precondition(
			minimumStepSize.isFinite && minimumStepSize >= .zero,
			"Minimum step size must be finite and nonnegative"
		)
		precondition(
			maximumStepSize.isFinite && maximumStepSize > .zero,
			"Maximum step size must be positive and finite"
		)
		precondition(
			minimumStepSize <= maximumStepSize,
			"Minimum step size cannot exceed maximum step size"
		)
		precondition(
			absoluteTolerance.isFinite && absoluteTolerance >= .zero,
			"Absolute tolerance must be finite and nonnegative"
		)
		precondition(
			relativeTolerance.isFinite && relativeTolerance >= .zero,
			"Relative tolerance must be finite and nonnegative"
		)
		precondition(
			absoluteTolerance > .zero || relativeTolerance > .zero,
			"At least one integration tolerance must be positive"
		)

		self.minimumStepSize = minimumStepSize
		self.maximumStepSize = maximumStepSize
		self.absoluteTolerance = absoluteTolerance
		self.relativeTolerance = relativeTolerance
	}
}
