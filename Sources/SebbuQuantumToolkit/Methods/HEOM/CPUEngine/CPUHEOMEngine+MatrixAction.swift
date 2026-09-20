// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuBLAS

extension HEOM.CPUEngine {
    /// Operates directly on row-major ADO slices. No temporary matrix owners
    /// are constructed in the hierarchy loop. Tiny products avoid BLAS dispatch.
    internal enum MatrixAction {
        @inline(always)
        static func product(
            _ a: UnsafePointer<Complex<Double>>, _ b: UnsafePointer<Complex<Double>>,
            dimension d: Int, adjointA: Bool = false, adjointB: Bool = false,
            scale: Complex<Double> = .one, adding: Bool = true,
            into c: UnsafeMutablePointer<Complex<Double>>
        ) {
            if d <= 4 {
                for i in 0..<d {
                    for j in 0..<d {
                        var value = Complex<Double>.zero
                        for k in 0..<d {
                            let left = adjointA ? a[k * d + i].conjugate : a[i * d + k]
                            let right = adjointB ? b[j * d + k].conjugate : b[k * d + j]
                            value += left * right
                        }
                        c[i * d + j] = (adding ? c[i * d + j] : .zero) + scale * value
                    }
                }
            } else {
                BLAS.zgemm(
                    layout: .rowMajor,
                    transposeA: adjointA ? .conjugateTranspose : .noTranspose,
                    transposeB: adjointB ? .conjugateTranspose : .noTranspose,
                    m: d, n: d, k: d, alpha: scale, a: a, lda: d, b: b, ldb: d,
                    beta: adding ? .one : .zero, c: c, ldc: d)
            }
        }

        @inline(always)
        static func commutator(
            _ op: UnsafePointer<Complex<Double>>, _ rho: UnsafePointer<Complex<Double>>,
            dimension: Int, adjoint: Bool = false, scale: Complex<Double>,
            into output: UnsafeMutablePointer<Complex<Double>>
        ) {
            product(op, rho, dimension: dimension, adjointA: adjoint, scale: scale, into: output)
            product(rho, op, dimension: dimension, adjointB: adjoint, scale: -scale, into: output)
        }
    }
}
