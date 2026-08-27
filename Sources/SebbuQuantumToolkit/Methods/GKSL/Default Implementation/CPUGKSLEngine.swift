// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct CPUGKSLEngine: Sendable {
	@inlinable
	public init() {}
}

extension CPUGKSLEngine: GKSL.Implementation {
    @inlinable
    public func solve<Hamiltonian>(
        problem: borrowing DensityMatrixProblem<Hamiltonian>,
        configuration: GKSL.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        _ = configuration

        let start = propagation.timeSpan.start
        let end = propagation.timeSpan.end
        var outputCursor = OutputCursor(
            timeSpan: propagation.timeSpan,
            schedule: propagation.output
        )
        var state = DensityMatrixState(problem.initialState)

        if let initialTime = outputCursor.takeInitialTime() {
            let control = observer(initialTime, state.densityMatrix)
            if control == .stop {
                return PropagationRunSummary(finalTime: initialTime, endReason: .stoppedByObserver)
            }
        }
        
        guard start < end else {
            return PropagationRunSummary(finalTime: start, endReason: .reachedEndTime)
        }

        let rhs = GKSLRightHandSide(problem)
        var solver = UniqueVerner76Solver(
            t: start,
            dt: Swift.min(
                propagation.integration.maximumStepSize,
                end - start
            ),
            maxStep: propagation.integration.maximumStepSize,
            rhs: rhs,
            y6: .init(dimension: problem.system.dimension),
            k1: .init(dimension: problem.system.dimension),
            k2: .init(dimension: problem.system.dimension),
            k3: .init(dimension: problem.system.dimension),
            k4: .init(dimension: problem.system.dimension),
            k5: .init(dimension: problem.system.dimension),
            k6: .init(dimension: problem.system.dimension),
            k7: .init(dimension: problem.system.dimension),
            k8: .init(dimension: problem.system.dimension),
            k9: .init(dimension: problem.system.dimension),
            k10: .init(dimension: problem.system.dimension),
            temporary: .init(dimension: problem.system.dimension),
            absoluteTolerance: propagation.integration.absoluteTolerance,
            relativeTolerance: propagation.integration.relativeTolerance,
            minimumStep: propagation.integration.minimumStepSize
        )
        var interpolatedState = DensityMatrixState(dimension: problem.system.dimension)

        while solver.t < end {
            let step = try solver.step(y: &state, upTo: end)
            while let outputTime = outputCursor.nextTime(through: step.endTime) {
                let control: PropagationControl
                if outputTime == step.endTime {
                    control = observer(outputTime, state.densityMatrix)
                } else {
                    solver.interpolateLastStep(
                        at: outputTime, into: &interpolatedState)
                    control = observer(outputTime, interpolatedState.densityMatrix)
                }
                if control == .stop {
                    return PropagationRunSummary(finalTime: outputTime, endReason: .stoppedByObserver)
                }
            }
        }
        return PropagationRunSummary(finalTime: end, endReason: .reachedEndTime)
    }
}

extension GKSL {
    @inlinable
    @inline(always)
    @discardableResult
    public static func solve<Hamiltonian>(
        problem: borrowing DensityMatrixProblem<Hamiltonian>,
        configuration: Configuration = .init(),
        propagation: PropagationOptions<CPUGKSLEngine.IntegratorConfiguration>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary where Hamiltonian: HamiltonianFunction {
        let implementation = CPUGKSLEngine()
        return try implementation.solve(
            problem: problem, configuration: configuration, propagation: propagation,
            observing: observer)
    }
    
    @inlinable
    @inline(always)
    @discardableResult
    public static func solve<Hamiltonian: HamiltonianFunction>(
        problem: borrowing PureStateProblem<Hamiltonian>,
        configuration: GKSL.Configuration = .init(),
        propagation: PropagationOptions<CPUGKSLEngine.IntegratorConfiguration>,
        observing observer: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let implementation = CPUGKSLEngine()
        return try implementation.solve(
            problem: problem, configuration: configuration, propagation: propagation,
            observing: observer)
    }
}
