// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension DefaultGKSLImplementation: GKSL.TwoTimeCorrelationImplementation {
    @inlinable
    public func solveTwoTimeCorrelation<Hamiltonian>(problem: borrowing DensityMatrixProblem<Hamiltonian>, configuration: GKSL.Configuration, request: TwoTimeCorrelationRequest, propagation: PropagationOptions<IntegrationOptions>, _ forEach: (Double, Complex<Double>) -> Void) throws where Hamiltonian : HamiltonianFunction {
        _ = configuration
        let start = propagation.timeSpan.start
        let end = propagation.timeSpan.end
        var outputCursor = OutputCursor(
            timeSpan: propagation.timeSpan,
            schedule: propagation.output
        )
        var state = DensityMatrixState(problem.initialState)
        var operatorStorage: UniqueMatrix<Complex<Double>> = .zeros(rows: problem.system.dimension, columns: problem.system.dimension)
        guard start < end else { return }

        let rhs = GKSLRightHandSide(problem)
        var solver = UniqueDOPRISolver(
            t: start,
            dt: Swift.min(
                propagation.integration.maximumStepSize,
                end - start
            ),
            maxStep: propagation.integration.maximumStepSize,
            rhs: rhs,
            y4: .init(dimension: problem.system.dimension),
            k1: .init(dimension: problem.system.dimension),
            k2: .init(dimension: problem.system.dimension),
            k3: .init(dimension: problem.system.dimension),
            k4: .init(dimension: problem.system.dimension),
            k5: .init(dimension: problem.system.dimension),
            k6: .init(dimension: problem.system.dimension),
            k7: .init(dimension: problem.system.dimension),
            temporary: .init(dimension: problem.system.dimension),
            absoluteTolerance: propagation.integration.absoluteTolerance,
            relativeTolerance: propagation.integration.relativeTolerance,
            minimumStep: propagation.integration.minimumStepSize
        )
        var interpolatedState = DensityMatrixState(dimension: problem.system.dimension)
        
        while solver.t < request.insertionTime {
            let step = try solver.step(y: &state, upTo: request.insertionTime)
        }
        switch request.insertion {
        case .left(let B):
            B.insert(t: request.insertionTime, into: &operatorStorage)
            state.densityMatrix = operatorStorage.dotBLAS(state.densityMatrix)
        case .right(let B):
            B.insert(t: request.insertionTime, into: &operatorStorage)
            state.densityMatrix = state.densityMatrix.dotBLAS(operatorStorage)
        }
        solver.restart(at: request.insertionTime)
        while solver.t < end {
            let step = try solver.step(y: &state, upTo: end)
            while let outputTime = outputCursor.nextTime(through: step.endTime) {
                if outputTime == step.endTime {
                    request.observable.insert(t: outputTime, into: &operatorStorage)
                    let correlationFunction = MatrixOperations.trace(state.densityMatrix, operatorStorage)
                    forEach(outputTime, correlationFunction)
                } else {
                    solver.interpolateLastStep(
                        at: outputTime, into: &interpolatedState)
                    request.observable.insert(t: outputTime, into: &operatorStorage)
                    let correlationFunction = MatrixOperations.trace(interpolatedState.densityMatrix, operatorStorage)
                    forEach(outputTime, correlationFunction)
                }
            }
        }
    }
}

extension GKSL {
    @inlinable
    @inline(always)
    public static func solveTwoTimeCorrelation<Hamiltonian>(
        problem: borrowing DensityMatrixProblem<Hamiltonian>,
        configuration: GKSL.Configuration = .init(),
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        _ forEach: (Double, Complex<Double>) -> Void
    ) throws where Hamiltonian: HamiltonianFunction {
        let implementation = DefaultGKSLImplementation()
        try implementation.solveTwoTimeCorrelation(problem: problem, configuration: configuration, request: request, propagation: propagation, forEach)
    }
}
