// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    @usableFromInline
    internal struct CollapseView {
        @usableFromInline
        let matrix: UnsafePointer<Complex<Double>>
        @usableFromInline
        let rate: Double
        
        @inlinable
        init(matrix: UnsafePointer<Complex<Double>>, rate: Double) {
            self.matrix = matrix
            self.rate = rate
        }
    }

    /// Borrowed buffers for one synchronous RHS evaluation. Preparation has
    /// finished before publication; all inputs remain read-only until the pool
    /// joins the stage. Each worker writes disjoint ADOs and its own temporary.
    /// No pointer in this value may be used after PersistentWorkerPool.run returns.
    @usableFromInline
    internal struct ADOBatch: @unchecked Sendable {
        @usableFromInline
        let hierarchy: HEOM.Hierarchy
        @usableFromInline
        let dimension: Int
        @usableFromInline
        let centered: Bool
        @usableFromInline
        let workerCount: Int
        @usableFromInline
        let adoCount: Int
        @usableFromInline
        let input: UnsafePointer<Complex<Double>>
        @usableFromInline
        let output: UnsafeMutablePointer<Complex<Double>>
        @usableFromInline
        let hLeft: UnsafePointer<Complex<Double>>
        @usableFromInline
        let hRight: UnsafePointer<Complex<Double>>
        @usableFromInline
        let lambda: UnsafePointer<Complex<Double>>
        @usableFromInline
        let memory: UnsafePointer<Complex<Double>>
        @usableFromInline
        let means: UnsafePointer<Complex<Double>>
        @usableFromInline
        let collapse: UnsafePointer<CollapseView>
        @usableFromInline
        let collapseCount: Int
        @usableFromInline
        let temporaries: UnsafeMutablePointer<Complex<Double>>

        @inlinable
        init(hierarchy: HEOM.Hierarchy, dimension: Int, centered: Bool, workerCount: Int, adoCount: Int, input: UnsafePointer<Complex<Double>>, output: UnsafeMutablePointer<Complex<Double>>, hLeft: UnsafePointer<Complex<Double>>, hRight: UnsafePointer<Complex<Double>>, lambda: UnsafePointer<Complex<Double>>, memory: UnsafePointer<Complex<Double>>, means: UnsafePointer<Complex<Double>>, collapse: UnsafePointer<CollapseView>, collapseCount: Int, temporaries: UnsafeMutablePointer<Complex<Double>>) {
            self.hierarchy = hierarchy
            self.dimension = dimension
            self.centered = centered
            self.workerCount = workerCount
            self.adoCount = adoCount
            self.input = input
            self.output = output
            self.hLeft = hLeft
            self.hRight = hRight
            self.lambda = lambda
            self.memory = memory
            self.means = means
            self.collapse = collapse
            self.collapseCount = collapseCount
            self.temporaries = temporaries
        }
        
        @inlinable
        func evaluate(worker: Int) {
            let d = dimension
            let size = d * d
            let count = hierarchy.count
            let pCount = hierarchy.multiIndexCount / 2
            let chunk = adoCount / workerCount
            let remainder = adoCount % workerCount
            let start = worker * chunk + min(worker, remainder)
            let end = start + chunk + (worker < remainder ? 1 : 0)
            let temporary = temporaries + worker * size

            // Flatten guide and companion hierarchies into the same work list,
            // but keep every hierarchy neighbour within its own block.
            for index in start..<end {
                let ado = index % count
                let base = (index - ado) * size
                let rho = input + index * size
                let derivative = output + index * size
                let damping = hierarchy.kWArray[ado]
                for j in 0..<size { derivative[j] = damping * rho[j] }
                MatrixAction.product(hLeft, rho, dimension: d, scale: -.i, into: derivative)
                MatrixAction.product(rho, hRight, dimension: d, scale: .i, into: derivative)
                for i in 0..<collapseCount where collapse[i].rate != 0 {
                    let op = collapse[i].matrix
                    MatrixAction.product(op, rho, dimension: d, adding: false, into: temporary)
                    MatrixAction.product(
                        temporary, op, dimension: d, adjointB: true,
                        scale: Complex(collapse[i].rate), into: derivative)
                }
                for p in 0..<pCount {
                    let op = lambda + p * size
                    let down = memory + p * size
                    let ket = ado * (2 * pCount) + p
                    let bra = ket + pCount
                    let ketChild = hierarchy.childIndices[ket]
                    let braChild = hierarchy.childIndices[bra]
                    if ketChild >= 0 {
                        MatrixAction.commutator(
                            op, input + base + ketChild * size, dimension: d,
                            adjoint: true, scale: Complex(-hierarchy.childWeights[ket]), into: derivative)
                    }
                    if braChild >= 0 {
                        MatrixAction.commutator(
                            op, input + base + braChild * size, dimension: d,
                            scale: Complex(hierarchy.childWeights[bra]), into: derivative)
                    }
                    let ketParent = hierarchy.parentIndices[ket]
                    let braParent = hierarchy.parentIndices[bra]
                    if ketParent >= 0 {
                        let parent = input + base + ketParent * size
                        let weight = hierarchy.parentWeights[ket]
                        MatrixAction.product(
                            down, parent, dimension: d, scale: Complex(weight), into: derivative)
                        if centered {
                            let value = weight * means[p]
                            for j in 0..<size { derivative[j] -= value * parent[j] }
                        }
                    }
                    if braParent >= 0 {
                        let parent = input + base + braParent * size
                        let weight = hierarchy.parentWeights[bra]
                        MatrixAction.product(
                            parent, down, dimension: d, adjointB: true,
                            scale: Complex(weight), into: derivative)
                        if centered {
                            let value = weight * means[p].conjugate
                            for j in 0..<size { derivative[j] -= value * parent[j] }
                        }
                    }
                }
            }
        }
    }
}
