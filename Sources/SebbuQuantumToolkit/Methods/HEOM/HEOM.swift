// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public enum HEOM: Sendable {}

extension HEOM {

    public enum ShiftType: Sendable {
        /// Equation (11.5): ordinary latent-basis HEOM.
        case none
        /// Equations (11.8)-(11.9): deterministic mean-field displacement.
        case meanField
    }

    public struct Configuration: Sendable {
        public let hierarchy: Hierarchy
        public var shiftType: ShiftType
        /// Parallelism within each RHS evaluation. Serial execution is the default.
        public var parallelism: Parallelism

        public init(
            hierarchy: Hierarchy,
            shiftType: ShiftType = .none,
            parallelism: Parallelism = .serial
        ) {
            self.hierarchy = hierarchy
            self.shiftType = shiftType
            self.parallelism = parallelism
        }
    }
}

extension HEOM {
    public typealias Environment = BathEnvironment
    public typealias Truncation = BathHierarchyTruncation
}

extension HEOM {
    /// Borrowed, row-major factorially scaled ADOs. In mean-field mode these
    /// are the displaced ADOs; the physical root is unchanged.
    public struct HierarchyStateView: ~Copyable, ~Escapable {
        @usableFromInline
        package let systemDimension: Int

        @usableFromInline
        package let states: Span<Complex<Double>>

        @inlinable
        public var count: Int {
            states.count / (systemDimension &* systemDimension)
        }

        @_lifetime(copy states)
        @inlinable
        package init(systemDimension: Int, states: Span<Complex<Double>>) {
            self.systemDimension = systemDimension
            self.states = states
        }

        @inlinable
        @inline(always)
        public func withPhysicalState<Result>(
            _ body: (
                borrowing DensityMatrixView
            ) -> Result
        ) -> Result {
            withState(at: 0, body)
        }

        @inlinable
        @inline(always)
        public func withState<Result>(
            at index: HEOM.Hierarchy.Index,
            _ body: (
                borrowing DensityMatrixView
            ) -> Result
        ) -> Result {
            precondition(index >= 0 && index < count)
            let span = states.extracting(
                index &* systemDimension &* systemDimension..<(index &+ 1) &* systemDimension
                    &* systemDimension)
            let view = DensityMatrixView(elements: span, dimension: systemDimension)
            return body(view)
        }
    }
}

extension HEOM {
    public protocol Implementation: ~Copyable {
        associatedtype IntegratorConfiguration: Sendable = IntegrationOptions

        @discardableResult
        func solve<Hamiltonian: HamiltonianFunction>(
            problem: DensityMatrixProblem<Hamiltonian>,
            configuration: HEOM.Configuration,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (
                Double,
                borrowing UniqueMatrix<Complex<Double>>
            ) -> PropagationControl
        ) throws -> PropagationRunSummary
    }
}

extension HEOM.Implementation {
    @inlinable
    @discardableResult
    public func solve<Hamiltonian: HamiltonianFunction>(
        problem: PureStateProblem<Hamiltonian>,
        configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegratorConfiguration>,
        observing observer: (
            Double,
            borrowing UniqueMatrix<Complex<Double>>
        ) -> PropagationControl
    ) throws -> PropagationRunSummary {
        let problem = DensityMatrixProblem(problem)
        return try solve(
            problem: problem, configuration: configuration, propagation: propagation, observing: observer)
    }
}

extension HEOM {
    public protocol HierarchyProvidingImplementation: Implementation {
        @discardableResult
        func solveWithHierarchy<Hamiltonian: HamiltonianFunction>(
            problem: DensityMatrixProblem<Hamiltonian>,
            configuration: HEOM.Configuration,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (
                Double,
                borrowing HEOM.HierarchyStateView
            ) -> PropagationControl
        ) throws -> PropagationRunSummary
    }

    /// Insertions act on every ADO, preserving the correlated bath state.
    /// In centered mode, a separate physical guide determines all shifts.
    public protocol TwoTimeCorrelationImplementation: Implementation {
        @discardableResult
        func solveTwoTimeCorrelation<
            Hamiltonian: HamiltonianFunction
        >(
            problem: DensityMatrixProblem<Hamiltonian>,
            configuration: HEOM.Configuration,
            request: TwoTimeCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (
                Double,
                Complex<Double>
            ) -> PropagationControl
        ) throws -> PropagationRunSummary
    }

    public protocol MultiTimeOrderedCorrelationImplementation: Implementation {
        @discardableResult
        func solveMultiTimeOrderedCorrelation<Hamiltonian: HamiltonianFunction>(
            problem: DensityMatrixProblem<Hamiltonian>,
            configuration: HEOM.Configuration,
            request: MultiTimeOrderedCorrelationRequest,
            propagation: PropagationOptions<IntegratorConfiguration>,
            observing observer: (Double, Complex<Double>) -> PropagationControl
        ) throws -> PropagationRunSummary
    }
}
