// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension QSD.CPUEngine {
    @usableFromInline
    internal struct CorrelationRightHandSide: ~Copyable, SDERHSFunction {
        @usableFromInline let equationType: QSD.EquationType
        @usableFromInline var hamiltonian: _PreparedOperator
        @usableFromInline var channels: UniqueArray<_PreparedMarkovianChannel>
        @usableFromInline var randomNumberGenerator: TrajectoryRandomNumberGenerator
        @usableFromInline var scratch: UniqueVector<Complex<Double>>

        @inlinable
        init(
            _ problem: borrowing PureStateProblem,
            equationType: QSD.EquationType,
            seed: UInt64,
            trajectoryID: UInt64,
            ensembleSampling: EnsembleSampling = .independent
        ) {
            self.equationType = equationType
            let d = problem.system.dimension
            hamiltonian = .init(problem.system.hamiltonian, dimension: d)
            channels = _prepareMarkovianChannels(problem.markovianChannels, dimension: d)
            randomNumberGenerator = .init(
                seed: seed, trajectoryID: trajectoryID,
                ensembleSampling: ensembleSampling, purpose: .gaussianWhiteNoise)
            scratch = .zero(d)
        }

        @inlinable
        mutating func drift(t: Double, y: borrowing CorrelationState, into dy: inout CorrelationState) {
            hamiltonian.updateChecked(at: t)
            hamiltonian.action.vector(y.guide.components, into: dy.guide.components, scale: -.i)
            hamiltonian.action.vector(y.ket.components, into: dy.ket.components, scale: -.i)
            hamiltonian.action.vector(y.bra.components, into: dy.bra.components, scale: -.i)
            let norm = equationType == .linear ? 1 : y.guide.normSquared
            precondition(norm.isFinite && norm > 0, "Non-linear QSD requires a finite, nonzero state norm")
            for i in 0..<channels.count {
                let rate = Self.checkedRate(channels[i].rate(t))
                if rate == 0 { continue }
                channels[i].updateChecked(at: t)
                let action = _DiffusiveAction(
                    collapse: channels[i].action, rate: rate,
                    nonlinear: equationType != .linear, normalized: equationType == .nonLinearNormalized,
                    guide: y.guide.components, norm: norm, scratch: scratch.components)
                action.drift(y.guide.components, into: dy.guide.components, scratch: scratch.components,
                    applied: equationType != .linear ? UnsafePointer(scratch.components) : nil)
                action.drift(y.ket.components, into: dy.ket.components, scratch: scratch.components)
                action.drift(y.bra.components, into: dy.bra.components, scratch: scratch.components)
            }
        }

        @inlinable
        mutating func diffusion(
            t: Double, y: borrowing CorrelationState, channel channelIndex: Int,
            into dy: inout CorrelationState
        ) {
            precondition(channelIndex >= 0 && channelIndex < channels.count)
            dy.guide.zeroComponents()
            dy.ket.zeroComponents()
            dy.bra.zeroComponents()
            let rate = Self.checkedRate(channels[channelIndex].rate(t))
            if rate == 0 { return }
            let normalized = equationType == .nonLinearNormalized
            let norm = normalized ? y.guide.normSquared : 1
            precondition(norm.isFinite && norm > 0)
            channels[channelIndex].updateChecked(at: t)
            let action = _DiffusiveAction(
                collapse: channels[channelIndex].action, rate: rate,
                nonlinear: normalized, normalized: normalized,
                guide: y.guide.components, norm: norm, scratch: scratch.components)
            action.diffusion(y.guide.components, into: dy.guide.components,
                applied: normalized ? UnsafePointer(scratch.components) : nil)
            action.diffusion(y.ket.components, into: dy.ket.components)
            action.diffusion(y.bra.components, into: dy.bra.components)
        }

        @inlinable
        mutating func sampleNormalizedNoises(
            t: Double, stepSize: Double, into noises: inout MutableSpan<Complex<Double>>
        ) {
            precondition(noises.count == channels.count)
            for i in noises.indices { noises[unchecked: i] = randomNumberGenerator.nextNormal() }
        }

        @inlinable @inline(always)
        static func checkedRate(_ rate: Double) -> Double {
            precondition(
                rate.isFinite && rate >= 0, "A Markovian-channel rate must be finite and nonnegative")
            return rate
        }
    }
}
