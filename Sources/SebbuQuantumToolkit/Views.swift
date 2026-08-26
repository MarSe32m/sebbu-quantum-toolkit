// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics

public struct DensityMatrixView: ~Copyable, ~Escapable {
    @usableFromInline
    package let elements: Span<Complex<Double>>
    public let dimension: Int
    
    @_lifetime(copy elements)
    @inlinable
    package init(elements: Span<Complex<Double>>, dimension: Int) {
        self.elements = elements
        self.dimension = dimension
    }
    
    @inlinable
    @inline(always)
    public subscript(row: Int, column: Int) -> Complex<Double> {
        //TODO: This assumes the ADOs are flattened in row-major order. Typically vectorizing a matrix is column-major but maybe we vectorize row-major since that way its more cache friendlier to construct the ADOs / to vectorize them. Check this!
        _read { yield elements[row * dimension + column] }
    }
}

public struct StateVectorView: ~Copyable, ~Escapable {
    @usableFromInline
    package let elements: Span<Complex<Double>>
    
    @_lifetime(copy elements)
    @inlinable
    package init(elements: consuming Span<Complex<Double>>) {
        self.elements = elements
    }
    
    @inlinable
    @inline(always)
    public subscript(index: Int) -> Complex<Double> {
        _read { yield elements[index] }
    }
}
