// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CPUQSDEngine {
    @usableFromInline
    internal struct QSDRightHandSide<Hamiltonian: HamiltonianFunction>: ~Copyable, SDERHSFunction {
        @usableFromInline
        internal let equationType: QSD.EquationType
        @usableFromInline
        internal let hamiltonian: Hamiltonian
        @usableFromInline
        internal let constantChannels: [_PreparedConstantMarkovianChannel]
        @usableFromInline
        internal let dynamicChannels: [_PreparedDynamicMarkovianChannel]
        @usableFromInline
        internal let noiseProcesses: [StandardGaussianWhiteNoiseProcess]

        @usableFromInline
        internal var hamiltonianBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var collapseBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var adjointBuffer: UniqueMatrix<Complex<Double>>
        @usableFromInline
        internal var productBuffer: UniqueMatrix<Complex<Double>>

        @inlinable
        internal init(_ problem: borrowing PureStateProblem<Hamiltonian>, equationType: QSD.EquationType, noiseProcesses: [StandardGaussianWhiteNoiseProcess]) {
            let dimension = problem.system.dimension
            precondition(dimension > 0, "The quantum-system dimension must be positive")
            let markovianChannelCount = problem.markovianChannels.count
            precondition(markovianChannelCount == noiseProcesses.count, "There must be equal number of markovian channels and noise processes")

            var constantChannels: [_PreparedConstantMarkovianChannel] = []
            var dynamicChannels: [_PreparedDynamicMarkovianChannel] = []
            constantChannels.reserveCapacity(problem.markovianChannels.count)
            dynamicChannels.reserveCapacity(problem.markovianChannels.count)

            for channel in problem.markovianChannels {
                if let prepared = _PreparedConstantMarkovianChannel(
                    channel,
                    dimension: dimension
                ) {
                    constantChannels.append(prepared)
                } else {
                    dynamicChannels.append(
                        _PreparedDynamicMarkovianChannel(
                            channel,
                            dimension: dimension
                        )
                    )
                }
            }
            precondition(constantChannels.count + dynamicChannels.count == noiseProcesses.count)
            self.equationType = equationType
            self.hamiltonian = problem.system.hamiltonian
            self.constantChannels = constantChannels
            self.dynamicChannels = dynamicChannels
            self.noiseProcesses = noiseProcesses
            self.hamiltonianBuffer = .zeros(rows: dimension, columns: dimension)
            self.collapseBuffer = .zeros(rows: dimension, columns: dimension)
            self.adjointBuffer = .zeros(rows: dimension, columns: dimension)
            self.productBuffer = .zeros(rows: dimension, columns: dimension)
        }
        
        @inlinable
        internal mutating func drift(t: Double, y: borrowing StateVector, into dy: inout StateVector) {
            hamiltonian.hamiltonian(t: t, into: &hamiltonianBuffer)
            hamiltonianBuffer.dotBLAS(y.state, multiplied: -.i, into: &dy.state)
            
            for channel in constantChannels {
                // Sample rate
                let rate = channel.rate(t)
                // Apply -rate/2 C^dagger C \phi + dy -> dy
                channel.lossOperator.dotBLAS(y.state, multiplied: Complex(-0.5 * rate), addingInto: &dy.state)
                // Apply Girsanov shift of the noise: rate * <C^dagger> C \phi + dy -> dy
                if equationType == .nonLinear || equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: channel.collapseOperatorAdjoint, y.state) / y.state.normSquared
                    channel.collapseOperator.dotBLAS(y.state, multiplied: rate * expectationValue, addingInto: &dy.state)
                }
                // Apply normalization factor for non-linear normalized QSD: -rate/2 <C^dagger C> \phi + dy -> dy
                if equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: channel.lossOperator, y.state) / y.state.normSquared
                    dy.state.add(y.state, multiplied: -0.5 * rate * expectationValue)
                }
            }
            for channel in dynamicChannels {
                // Sample rate
                let rate = channel.rate(t)
                // Obtain C, C^dagger and C^dagger C
                channel.insert(t: t, into: &collapseBuffer)
                for row in 0..<collapseBuffer.rows {
                    for column in 0..<collapseBuffer.columns {
                        adjointBuffer[unchecked: column, unchecked: row] = collapseBuffer[unchecked: row, unchecked: column].conjugate
                    }
                }
                adjointBuffer.dotBLAS(collapseBuffer, into: &productBuffer)
                // Apply -rate/2 C^dagger C \phi + dy -> dy
                productBuffer.dotBLAS(y.state, multiplied: Complex(-0.5 * rate), addingInto: &dy.state)
                // Apply Girsanov shift of the noise: rate * <C^dagger> C \phi + dy -> dy
                if equationType == .nonLinear || equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: adjointBuffer, y.state) / y.state.normSquared
                    collapseBuffer.dotBLAS(y.state, multiplied: rate * expectationValue, addingInto: &dy.state)
                }
                // Apply normalization factor for non-linear normalized QSD: -rate/2 <C^dagger C> \phi + dy -> dy
                if equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: productBuffer, y.state) / y.state.normSquared
                    dy.state.add(y.state, multiplied: -0.5 * rate * expectationValue)
                }
            }
        }
        
        @inlinable
        internal mutating func diffusion(t: Double, y: borrowing StateVector, channel: Int, into dy: inout StateVector) {
            dy.state.zeroComponents()
            
            for channel in constantChannels {
                let rate = channel.rate(t)
                let coefficient: Double = .sqrt(0.5 * rate)
                // Apply: coefficient * C_i(t) \phi + dy -> dy
                channel.collapseOperator.dotBLAS(y.state, multiplied: Complex(coefficient), addingInto: &dy.state)
                
                // For normalized QSD, apply: -coefficient * <C_i(t)> \phi + dy -> dy
                if equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: channel.collapseOperator, y.state) / y.state.normSquared
                    dy.state.add(y.state, multiplied: -coefficient * expectationValue)
                }
            }
            for channel in dynamicChannels {
                let rate = channel.rate(t)
                let coefficient: Double = .sqrt(0.5 * rate)
                channel.insert(t: t, into: &collapseBuffer)
                // Apply: coefficient * C_i(t) \phi + dy -> dy
                collapseBuffer.dotBLAS(y.state, multiplied: Complex(coefficient), addingInto: &dy.state)
                
                // For normalized QSD, apply: -coefficient * <C_i(t)> \phi + dy -> dy
                if equationType == .nonLinearNormalized {
                    let expectationValue = y.state.inner(metric: collapseBuffer, y.state) / y.state.normSquared
                    dy.state.add(y.state, multiplied: -coefficient * expectationValue)
                }
            }
        }
        
        @inlinable
        internal mutating func sampleNormalizedNoises(t: Double, stepSize: Double, into noises: inout MutableSpan<Complex<Double>>) {
            precondition(noises.count == noiseProcesses.count, "There must be equal count of noises and noise processes")
            for i in noises.indices {
                noises[unchecked: i] = noiseProcesses[i].sample(t)
            }
        }
    }
}
