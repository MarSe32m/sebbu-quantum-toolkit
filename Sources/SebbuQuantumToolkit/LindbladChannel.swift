// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import NumericsExtensions
import SebbuScience

public struct LindbladChannel: Sendable {
    public typealias RateFunction = @Sendable (Double) -> Double
    public typealias OperatorFunction = @Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void
    
    public let rate: @Sendable (Double) -> Double
    public let C: OperatorFunction
    public let CdaggerC: OperatorFunction
}
