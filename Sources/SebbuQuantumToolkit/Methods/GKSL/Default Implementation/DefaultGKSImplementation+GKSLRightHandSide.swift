// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
#if swift(<6.5)
import BasicContainers
#else
#warning("Remove swift-collections dependency")
#endif

extension DefaultGKSLImplementation {
    @usableFromInline
    internal struct GKSLRightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable, ODERHSFunction {
        public let hamiltonian: Hamiltonian
        public let constantChannels: UniqueArray<_PreparedConstantMarkovianChannelOperators>
        public let dynamicChannels: UniqueArray<_PreparedDynamicMarkovianChannel>
        
        @usableFromInline
        internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var collapseBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var adjointBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var productBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var temporary1: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var temporary2: UniqueMatrix<Complex<Double>>
        
        @inlinable
        public init(_ problem: DensityMatrixProblem<Hamiltonian>) {
            self.hamiltonian = problem.system.hamiltonian
            let markovianChannels = problem.markovianChannels
            self.constantChannels = .init(capacity: problem.markovianChannels.count) { output in
                for channel in markovianChannels where channel.collapseOperator.isConstant {
                    if let constantChannel = _PreparedConstantMarkovianChannelOperators(channel) {
                        output.append(constantChannel)
                    }
                }
            }
            self.dynamicChannels = .init(capacity: problem.markovianChannels.count) { output in
                for channel in markovianChannels where !channel.collapseOperator.isConstant {
                    output.append(_PreparedDynamicMarkovianChannel(channel))
                }
            }
            
            //TODO: For some reason, these need to be first. Otherwise it crashes...
            self.hamiltonianBuffer = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            self.collapseBuffer = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            self.adjointBuffer = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            self.productBuffer = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            self.temporary1 = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            self.temporary2 = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
            let channelCount = problem.markovianChannels.count
            precondition(channelCount == self.constantChannels.count + self.dynamicChannels.count)
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y: borrowing DensityMatrixState, dy: inout DensityMatrixState) {
            // \dot{\rho} = -i[H, \rho] + \sum_\mu \gamma_\mu (C_\mu \rho C_\mu^\dagger - \frac{1}{2} \{ C_\mu^\dagger C_\mu, \rho \})
            var productBuffer = UniqueMatrix(_unsafeElements: productBuffer.elements, rows: productBuffer.rows, columns: productBuffer.columns)
            var collapseBuffer = UniqueMatrix(_unsafeElements: collapseBuffer.elements, rows: collapseBuffer.rows, columns: collapseBuffer.columns)
            var adjointBuffer = UniqueMatrix(_unsafeElements: adjointBuffer.elements, rows: adjointBuffer.rows, columns: adjointBuffer.columns)
            var temporary = UniqueMatrix(_unsafeElements: temporary1.elements, rows: temporary1.rows, columns: temporary1.columns)
            
            hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
            hamiltonianBuffer.dotBLAS(y.densityMatrix, multiplied: -.i, into: &dy.densityMatrix)
            y.densityMatrix.dotBLAS(hamiltonianBuffer, multiplied: .i, addingInto: &dy.densityMatrix)
            for i in 0..<constantChannels.count {
                let rate = constantChannels[i].rate(t)
                // \gamma_i C \rho C^\dagger
                constantChannels[i].C.dotBLAS(y.densityMatrix, into: &productBuffer)
                productBuffer.dotBLAS(constantChannels[i].Cdagger, multiplied: Complex(rate), addingInto: &dy.densityMatrix)
                // -\gamma_i / 2 {C_i^\dagger C_i, \rho}
                constantChannels[i].CdaggerC.dotBLAS(y.densityMatrix, multiplied: Complex(-0.5 * rate), addingInto: &dy.densityMatrix)
                y.densityMatrix.dotBLAS(constantChannels[i].CdaggerC, multiplied: Complex(-0.5 * rate), addingInto: &dy.densityMatrix)
            }
            for i in 0..<dynamicChannels.count {
                let rate = dynamicChannels[i].rate(t)
                dynamicChannels[i].insert(t: t, into: &collapseBuffer)
                for i in 0..<collapseBuffer.rows {
                    for j in 0..<collapseBuffer.columns {
                        adjointBuffer[j, i] = collapseBuffer[i, j].conjugate
                    }
                }
                adjointBuffer.dotBLAS(collapseBuffer, into: &productBuffer)
                // \gamma_i C \rho C^\dagger
                collapseBuffer.dotBLAS(y.densityMatrix, into: &temporary)
                temporary.dotBLAS(adjointBuffer, multiplied: Complex(rate), addingInto: &dy.densityMatrix)
                // -\gamma_i / 2 {C_i^\dagger C_i, \rho}
                productBuffer.dotBLAS(y.densityMatrix, multiplied: Complex(-0.5 * rate), addingInto: &dy.densityMatrix)
                y.densityMatrix.dotBLAS(productBuffer, multiplied: Complex(-0.5 * rate), addingInto: &dy.densityMatrix)
            }
            let _ = productBuffer.consumeElements()
            let _ = collapseBuffer.consumeElements()
            let _ = adjointBuffer.consumeElements()
            let _ = temporary.consumeElements()
            
        }
    }
}
