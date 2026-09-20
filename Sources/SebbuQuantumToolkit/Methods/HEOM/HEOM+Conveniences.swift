// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM {
    @discardableResult
    public static func solve<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solve(
            problem: problem, configuration: configuration,
            propagation: propagation, observing: observer)
    }
    @discardableResult
    public static func solve<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing UniqueMatrix<Complex<Double>>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solve(
            problem: problem, configuration: configuration,
            propagation: propagation, observing: observer)
    }
}

extension HEOM.HierarchyProvidingImplementation {
    @discardableResult
    public func solveWithHierarchy<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try solveWithHierarchy(
            problem: DensityMatrixProblem(problem), configuration: configuration,
            propagation: propagation, observing: observer)
    }
}

extension HEOM {
    @discardableResult
    public static func solveWithHierarchy<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration,
            propagation: propagation, observing: observer)
    }
    @discardableResult
    public static func solveWithHierarchy<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, borrowing HEOM.HierarchyStateView) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveWithHierarchy(
            problem: problem, configuration: configuration,
            propagation: propagation, observing: observer)
    }
}

extension HEOM.TwoTimeCorrelationImplementation {
    @discardableResult
    public func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try solveTwoTimeCorrelation(
            problem: DensityMatrixProblem(problem), configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
}

extension HEOM {
    @discardableResult
    public static func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveTwoTimeCorrelation(
            problem: problem, configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
    @discardableResult
    public static func solveTwoTimeCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: TwoTimeCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveTwoTimeCorrelation(
            problem: problem, configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
}

extension HEOM.MultiTimeOrderedCorrelationImplementation {
    @discardableResult
    public func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: MultiTimeOrderedCorrelationRequest,
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try solveMultiTimeOrderedCorrelation(
            problem: DensityMatrixProblem(problem), configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
}

extension HEOM {
    @discardableResult
    public static func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: MultiTimeOrderedCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveMultiTimeOrderedCorrelation(
            problem: problem, configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
    @discardableResult
    public static func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>, configuration: HEOM.Configuration,
        request: MultiTimeOrderedCorrelationRequest,
        propagation: PropagationOptions<IntegrationOptions>,
        observing observer: (Double, Complex<Double>) -> PropagationControl
    ) throws -> PropagationRunSummary {
        try CPUEngine().solveMultiTimeOrderedCorrelation(
            problem: problem, configuration: configuration,
            request: request,
            propagation: propagation, observing: observer)
    }
}
