// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM {
    /// Deterministic, factorially scaled latent-basis HEOM with dense output.
    public struct CPUEngine: Sendable {
        public init() {}

        public enum SolverError: Error, Equatable, Sendable {
            case operatorDimensionMismatch
            case invalidMarkovianRate(time: Double)
            case invalidGuideTrace(time: Double)
            case nonFiniteState(time: Double)
        }
    }
}

extension HEOM.CPUEngine: HEOM.Implementation, HEOM.HierarchyProvidingImplementation {
    @inlinable
    @discardableResult
    public func solve<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let d = problem.system.dimension
        let root = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        return try run(problem: problem, configuration: configuration, propagation: propagation) {
            time, state, offset in
            root.elements.update(from: state.ados.elements + offset, count: d * d)
            return observer(time, root)
        }
    }

    @inlinable
    @discardableResult
    public func solveWithHierarchy<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        return try run(problem: problem, configuration: configuration, propagation: propagation) {
            time, state, _ in
            Self.withHierarchyView(state, dimension: problem.system.dimension) { observer(time, $0) }
        }
    }
}

extension HEOM.CPUEngine: HEOM.TwoTimeCorrelationImplementation, HEOM
        .MultiTimeOrderedCorrelationImplementation
{
    @inlinable
    @discardableResult
    public func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: TwoTimeCorrelationRequest, propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        do {
            return try solveMultiTimeOrderedCorrelation(
                problem: problem, configuration: configuration, request: _multiTimeOrderedRequest(request),
                propagation: propagation, observing: observer)
        } catch let error as MultiTimeOrderedCorrelationError {
            throw _mapMultiTimeOrderedErrorToTwoTime(error)
        }
    }

    @inlinable
    @discardableResult
    public func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: MultiTimeOrderedCorrelationRequest, propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let d = problem.system.dimension
        try _validateMultiTimeOrderedCorrelationRequest(request, timeSpan: propagation.timeSpan, dimension: d)
        var op = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        return try run(
            problem: problem, configuration: configuration, propagation: propagation,
            insertions: request.insertions
        ) { time, state, offset in
            request.observable.insert(t: time, into: &op)
            guard op.rows == d && op.columns == d else {
                throw MultiTimeOrderedCorrelationError.observableDimensionMismatch(
                    expected: d, rows: op.rows, columns: op.columns)
            }
            var value = Complex<Double>.zero
            for i in 0..<d {
                for j in 0..<d { value += op[i, j] * state.ados.elements[offset &+ j &* d &+ i] }
            }
            return observer(time, value)
        }
    }
}

extension HEOM.CPUEngine {
    @inlinable
    internal static func withHierarchyView<Result>(
        _ state: borrowing State, dimension: Int,
        _ body: (borrowing HEOM.HierarchyStateView) -> Result
    ) -> Result {
        // Keep the owner borrowed across the callback, as in HOPS. Swift's
        // borrowed span accessor cannot extend this lifetime through a closure.
        let span = Span(
            _unsafeStart: state.ados.elements,
            count: state.ados.rows * state.ados.columns)
        let view = HEOM.HierarchyStateView(systemDimension: dimension, states: span)
        return body(view)
    }
}
