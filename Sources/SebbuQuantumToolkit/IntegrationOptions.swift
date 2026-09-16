// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

//TODO: Document this type
public struct IntegrationOptions: Sendable {
	public var minimumStepSize: Double
	public var maximumStepSize: Double
	public var absoluteTolerance: Double
	public var relativeTolerance: Double
    /// Maximum tolerated drift of the norm from unity for equations that
    /// analytically preserve normalization.
    ///
    /// If the physical-state norm satisfies
    ///
    ///     abs(norm - 1) > normalizationDriftTolerance
    ///
    /// the state is explicitly renormalized after an accepted integration step.
    ///
    /// `nil` derives a conservative threshold from the integration tolerances.
    /// Explicit normalization is intended only as a safeguard against accumulated
    /// numerical drift. The normalized equation itself preserves the norm.
    public var normalizationDriftTolerance: Double?

	@inlinable
	public init(
		minimumStepSize: Double, maximumStepSize: Double, absoluteTolerance: Double,
        relativeTolerance: Double, normalizationDriftTolerance: Double? = nil
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
        if let normalizationDriftTolerance {
            precondition(
                normalizationDriftTolerance.isFinite && normalizationDriftTolerance >= 0,
                "Normalization drift tolerance must be finite and nonnegative."
            )
        }

		self.minimumStepSize = minimumStepSize
		self.maximumStepSize = maximumStepSize
		self.absoluteTolerance = absoluteTolerance
		self.relativeTolerance = relativeTolerance
        self.normalizationDriftTolerance = normalizationDriftTolerance
	}
}
