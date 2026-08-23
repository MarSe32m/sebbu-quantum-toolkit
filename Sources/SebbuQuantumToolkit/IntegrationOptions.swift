// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

public struct IntegrationOptions {
    public var minimumStepSize: Double
    public var maximumStepSize: Double
    public var absoluteTolerance: Double
    public var relativeTolerance: Double
    
    public init(minimumStepSize: Double, maximumStepSize: Double, absoluteTolerance: Double, relativeTolerance: Double) {
        self.minimumStepSize = minimumStepSize
        self.maximumStepSize = maximumStepSize
        self.absoluteTolerance = absoluteTolerance
        self.relativeTolerance = relativeTolerance
    }
}
