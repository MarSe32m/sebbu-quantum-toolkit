// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension HEOM.CPUEngine {
    /// One event-aware driver for ordinary propagation and correlations.
    /// Only insertion times constrain steps; output uses Verner dense output.
    @inlinable
    internal func run<Hamiltonian: HamiltonianFunction>(
        problem: DensityMatrixProblem<Hamiltonian>, configuration: HEOM.Configuration,
        propagation: PropagationOptions<IntegrationOptions>,
        insertions: [TimedCorrelationInsertion] = [],
        observing observer: (Double, borrowing State, Int) throws -> PropagationControl
    ) throws -> PropagationRunSummary {
        let progress = propagation.progress.continuous(in: propagation.timeSpan)
        defer { progress?.finish() }
        let start = propagation.timeSpan.start
        let end = propagation.timeSpan.end
        let d = problem.system.dimension
        let count = configuration.hierarchy.count
        let pCount = configuration.hierarchy.multiIndexCount / 2
        let shiftCount = configuration.shiftType == .meanField ? pCount : 0
        let copies = !insertions.isEmpty && shiftCount > 0 ? 2 : 1
        var state = State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies)
        var scratch = State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies)
        let offset = (copies - 1) * count * d * d
        for block in 0..<copies {
            for j in 0..<(d * d) {
                state.ados.elements[block * count * d * d + j] = problem.initialState.elements[j]
            }
        }
        try Self.validate(state, at: start)
        var insertionOperator = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        var insertionScratch = UniqueMatrix<Complex<Double>>.zeros(rows: d, columns: d)
        var cursor = OutputCursor(timeSpan: propagation.timeSpan, schedule: propagation.output)
        if let last = insertions.last { cursor.discardTimes(before: last.time) }
        let failure = Failure()
        let rhs = try RightHandSide(
            problem: problem, configuration: configuration, failure: failure, copies: copies)

        if start == end {
            for index in insertions.indices {
                try Self.apply(
                    insertions[index], index: index, dimension: d, hierarchyCount: count,
                    offset: offset, to: &state, operatorStorage: &insertionOperator,
                    scratch: &insertionScratch)
            }
            if let time = cursor.takeInitialTime() {
                if try observer(time, state, offset) == .stop {
                    progress?.setValue(to: time)
                    return .init(finalTime: time, endReason: .stoppedByObserver)
                }
            }
            progress?.setValue(to: end)
            return .init(finalTime: end, endReason: .reachedEndTime)
        }

        if insertions.isEmpty, let time = cursor.takeInitialTime() {
            if try observer(time, state, offset) == .stop {
                progress?.setValue(to: time)
                return .init(finalTime: time, endReason: .stoppedByObserver)
            }
        }

        var solver = UniqueVerner76Solver(
            t: start, dt: min(propagation.integration.maximumStepSize, end - start),
            maxStep: propagation.integration.maximumStepSize, rhs: rhs,
            y6: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k1: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k2: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k3: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k4: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k5: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k6: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k7: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k8: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k9: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            k10: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            temporary: State(dimension: d, hierarchyCount: count, shiftCount: shiftCount, copies: copies),
            absoluteTolerance: propagation.integration.absoluteTolerance,
            relativeTolerance: propagation.integration.relativeTolerance,
            minimumStep: propagation.integration.minimumStepSize)

        for index in insertions.indices {
            let event = insertions[index]
            while solver.t < event.time {
                do { _ = try solver.step(y: &state, upTo: event.time) } catch {
                    try failure.check()
                    throw error
                }
                try failure.check()
                try Self.validate(state, at: solver.t)
                progress?.setValue(to: solver.t)
            }
            try Self.apply(
                event, index: index, dimension: d, hierarchyCount: count,
                offset: offset, to: &state, operatorStorage: &insertionOperator,
                scratch: &insertionScratch)
            // Invalidate cached derivatives and dense output across the insertion.
            solver.restart(at: event.time)
        }
        if let last = insertions.last {
            if case .everyAcceptedStep = propagation.output {
                // Like GKSL, accepted-step output starts strictly after insertion.
            } else {
                while let time = cursor.nextTime(through: last.time) {
                    if try observer(time, state, offset) == .stop {
                        progress?.setValue(to: time)
                        return .init(finalTime: time, endReason: .stoppedByObserver)
                    }
                }
            }
        }
        while solver.t < end {
            let step: ODEStep
            do { step = try solver.step(y: &state, upTo: end) } catch {
                try failure.check()
                throw error
            }
            try failure.check()
            try Self.validate(state, at: step.endTime)
            while let time = cursor.nextTime(through: step.endTime) {
                let control: PropagationControl
                if time == step.endTime {
                    control = try observer(time, state, offset)
                } else {
                    solver.interpolateLastStep(at: time, into: &scratch)
                    control = try observer(time, scratch, offset)
                }
                if control == .stop {
                    progress?.setValue(to: time)
                    return .init(finalTime: time, endReason: .stoppedByObserver)
                }
            }
            progress?.setValue(to: step.endTime)
        }
        progress?.setValue(to: end)
        return .init(finalTime: end, endReason: .reachedEndTime)
    }

    @inlinable
    internal static func validate(_ state: borrowing State, at time: Double) throws {
        for j in 0..<(state.ados.rows * state.ados.columns) {
            let value = state.ados.elements[j]
            if !value.real.isFinite || !value.imaginary.isFinite {
                throw SolverError.nonFiniteState(time: time)
            }
        }
        for j in 0..<state.shifts.count {
            if !state.shifts[j].real.isFinite || !state.shifts[j].imaginary.isFinite {
                throw SolverError.nonFiniteState(time: time)
            }
        }
    }

    @inlinable
    internal static func apply(
        _ event: TimedCorrelationInsertion, index: Int, dimension d: Int,
        hierarchyCount: Int, offset: Int, to state: inout State,
        operatorStorage: inout UniqueMatrix<Complex<Double>>,
        scratch: inout UniqueMatrix<Complex<Double>>
    ) throws {
        _correlationInsertionOperator(event.insertion).insert(t: event.time, into: &operatorStorage)
        guard operatorStorage.rows == d && operatorStorage.columns == d else {
            throw MultiTimeOrderedCorrelationError.insertionOperatorDimensionMismatch(
                index: index, expected: d, rows: operatorStorage.rows, columns: operatorStorage.columns)
        }
        for ado in 0..<hierarchyCount {
            let input = state.ados.elements + offset + ado * d * d
            switch event.insertion {
            case .left:
                MatrixAction.product(
                    operatorStorage.elements, input, dimension: d,
                    adding: false, into: scratch.elements)
            case .right:
                MatrixAction.product(
                    input, operatorStorage.elements, dimension: d,
                    adding: false, into: scratch.elements)
            }
            input.update(from: scratch.elements, count: d * d)
        }
        // Never insert into the physical guide or its displacement memory.
        try validate(state, at: event.time)
    }
}
