// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Dispatch
import Numerics
import SebbuScience
import Synchronization
import Testing

@testable import SebbuQuantumToolkit

#if canImport(WinSDK)
    import WinSDK
#elseif canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

private func heomThreadID() -> UInt64 {
    #if canImport(WinSDK)
        return UInt64(GetCurrentThreadId())
    #elseif canImport(Darwin)
        return UInt64(UInt(bitPattern: pthread_self()))
    #elseif canImport(Glibc) || canImport(Musl)
        return UInt64(pthread_self())
    #else
        return 0
    #endif
}

@Test("HEOM parallelism defaults to serial and bounds automatic and explicit workers")
func heomParallelWorkerSelection() throws {
    #expect(heomConfiguration().parallelism == .serial)
    func count(_ policy: HEOM.Parallelism, _ d: Int, _ ados: Int, _ cores: Int = 8) -> Int {
        policy.workerCount(
            dimension: d, adoCount: ados, poleCount: 2, collapseCount: 1, availableCores: cores)
    }
    #expect(count(.serial, 128, 100_000) == 1)
    #expect(count(.maximumWorkers(1), 128, 100_000) == 1)
    #expect(count(.maximumWorkers(3), 2, 100) == 3)
    #expect(count(.maximumWorkers(Int.max), 2, 100) == 8)
    #expect(count(.maximumWorkers(8), 64, 3) == 3)
    #expect(count(.automatic, 2, 20) == 1)
    #expect(count(.automatic, 32, 20) > 1)
    #expect(count(.automatic, 128, 2) == 2)
    #expect(count(.automatic, 8, 100) == 1)
    #expect(count(.automatic, 2, 3_000) > 1)
    #expect(count(.automatic, 128, 100_000) == 8)
    #expect(count(.automatic, Int.max, Int.max) == 8)
    #expect(count(.automatic, 128, 100_000, 1) == 1)
    for d in [1, 2, 5, 32, 256] {
        for ados in [1, 3, 37, 1_000] {
            let workers = count(.automatic, d, ados)
            #expect(workers >= 1 && workers <= min(8, ados))
        }
    }
    var config = heomConfiguration(model: .zero(channelCount: 1))
    config.parallelism = .maximumWorkers(8)
    let rhs = try HEOM.CPUEngine.RightHandSide(
        problem: DensityMatrixProblem(hopsProblem()), configuration: config, failure: .init())
    #expect(rhs.workerCount == 1)
    #expect(rhs.pool == nil)
}

@Test("HEOM workers reuse native threads, complete each stage, and join on teardown", arguments: [1, 4])
func heomPersistentWorkers(workers: Int) {
    final class Lifetime: Sendable {}
    weak var lifetime: Lifetime?
    let caller = heomThreadID()
    let history = Mutex(Array(repeating: [(Int, UInt64)](), count: workers))
    do {
        let token = Lifetime()
        lifetime = token
        let pool = PersistentWorkerPool<Int>(workers: workers) { [token] generation, worker in
            withExtendedLifetime(token) {
                let thread = heomThreadID()
                history.withLock { $0[worker].append((generation, thread)) }
            }
        }
        #expect(pool.workerCount == workers)
        for generation in 0..<100 {
            pool.run(generation)
            // The barrier must finish every worker before the next publication.
            #expect(history.withLock { $0.allSatisfy { $0.count == generation + 1 } })
        }
        let records = history.withLock { $0 }
        #expect(records[0].allSatisfy { $0.1 == caller })
        for worker in records {
            #expect(worker.map(\.0) == Array(0..<100))
            #expect(Set(worker.map(\.1)).count == 1)
        }
        #expect(Set(records.compactMap { $0.first?.1 }).count == workers)
    }
    // A retained worker closure would keep the token alive after pool teardown.
    #expect(lifetime == nil)
    // Also exercise shutdown when no stage was ever submitted.
    do {
        let token = Lifetime()
        lifetime = token
        let pool = PersistentWorkerPool<Int>(workers: workers) { [token] _, _ in
            withExtendedLifetime(token) {}
        }
        #expect(pool.workerCount == workers)
    }
    #expect(lifetime == nil)
}

@Test(
    "Parallel HEOM prepares dynamic operators once per stage and matches serial ADOs", arguments: heomShifts,
    [2, 5, 32])
func heomParallelDynamicStages(shift: HEOM.ShiftType, d: Int) throws {
    let caller = heomThreadID()
    let calls = Mutex([String: Int]())
    let callbackThreads = Mutex(Set<UInt64>())
    @Sendable func record(_ name: String) {
        calls.withLock { $0[name, default: 0] += 1 }
        callbackThreads.withLock { _ = $0.insert(heomThreadID()) }
    }
    let h = ClosureHamiltonian { t, out in
        record("H")
        out = .zeros(rows: d, columns: d)
        for i in 0..<d {
            out[i, i] = Complex(Double(i) * (0.1 + t))
            if i + 1 < d {
                out[i, i + 1] = Complex(0.2, 0.1)
                out[i + 1, i] = Complex(0.2, -0.1)
            }
        }
    }
    let coupling = TimeDependentOperator.generatedDense(
        .init { t, out in
            record("bath")
            out = .zeros(rows: d, columns: d)
            for i in 0..<d {
                out[i, i] = Complex(Double(i + 1) * (0.2 + t), 0.1)
                if i + 1 < d { out[i, i + 1] = Complex(0.1, -0.2) }
            }
        })
    let collapse = TimeDependentOperator.generatedDense(
        .init { t, out in
            record("collapse")
            out = .zeros(rows: d, columns: d)
            for i in 1..<d { out[i - 1, i] = Complex(0.5 + t, 0.1) }
        })
    let problem = DensityMatrixProblem(
        initialState: heomIdentity(d), system: .init(dimension: d, hamiltonian: h),
        markovianChannels: [
            .init(
                rate: .generated { t in
                    record("rate")
                    return t == 0.2 ? 0 : 0.3 + t
                }, collapseOperator: collapse),
            .init(rate: .constant(0.17), collapseOperator: .constant(heomIdentity(d))),
        ])
    var config = HEOM.Configuration(
        hierarchy: .init(
            environment: .init(couplingOperator: coupling, bath: heomSingleBath()),
            truncation: .maximumTier(4)), shiftType: shift)
    let failure = HEOM.CPUEngine.Failure()
    var serial = try HEOM.CPUEngine.RightHandSide(
        problem: problem, configuration: config, failure: failure, copies: 2)
    config.parallelism = d == 32 ? .automatic : .maximumWorkers(4)
    var parallel = try HEOM.CPUEngine.RightHandSide(
        problem: problem, configuration: config, failure: failure, copies: 2)
    #expect(parallel.workerCount == min(d == 32 ? 9 : 4, Platform.activeProcessorCount))
    let count = config.hierarchy.count
    let shifts = shift == .meanField ? 1 : 0
    var state = HEOM.CPUEngine.State(dimension: d, hierarchyCount: count, shiftCount: shifts, copies: 2)
    var reference = HEOM.CPUEngine.State(dimension: d, hierarchyCount: count, shiftCount: shifts, copies: 2)
    var actual = HEOM.CPUEngine.State(dimension: d, hierarchyCount: count, shiftCount: shifts, copies: 2)
    let elements = state.ados.rows * d * d
    for i in 0..<elements { state.ados.elements[i] = Complex(0.01 * Double(i + 1), -0.02) }
    if shifts > 0 { state.shifts[0] = Complex(0.13, -0.07) }
    // Include rewound and repeated times, as encountered in rejected RK steps.
    let times = [0.0, 0.2, 0.1, 0.2, 0.4]
    for t in times {
        serial.evaluate(t: t, y: state, dy: &reference)
        parallel.evaluate(t: t, y: state, dy: &actual)
        try failure.check()
        expectHOPSClose(
            (0..<elements).map { actual.ados.elements[$0] },
            (0..<elements).map { reference.ados.elements[$0] }, tolerance: 2e-12)
        for p in 0..<shifts { #expect(actual.shifts[p] == reference.shifts[p]) }
        state.ados.elements[0] += Complex(0.02)
    }
    #expect(calls.withLock { $0 } == ["H": 10, "bath": 10, "collapse": 10, "rate": 10])
    #expect(callbackThreads.withLock { $0 } == [caller])
}

@Test(
    "All HEOM entry points agree across serial and parallel propagation", arguments: heomShifts,
    [false, true])
func heomParallelEntryPoints(shift: HEOM.ShiftType, stopEarly: Bool) throws {
    let problem = hopsProblem(markovian: [
        .init(rate: .constant(0.2), collapseOperator: .constant(hopsMatrix([0, 1, 0, 0])))
    ])
    let configuration = heomConfiguration(shift: shift, depth: 4)
    let b = TimeDependentOperator.constant(hopsMatrix([0, 1, 0, 0]))
    let a = TimeDependentOperator.constant(hopsMatrix([0, 0, 1, 0]))
    let c = TimeDependentOperator.constant(hopsMatrix([1, 0, 0, -1]))
    let propagation = hopsPropagation(
        end: 0.35, maximumStep: 0.09, tolerance: 1e-10,
        output: .times([0, 0.1, 0.17, 0.223, 0.35]))
    for api in 0..<4 {
        func run(_ policy: HEOM.Parallelism) throws -> ([Double], [Complex<Double>], PropagationRunSummary) {
            var config = configuration
            config.parallelism = policy
            var times: [Double] = []
            var values: [Complex<Double>] = []
            func control(_ t: Double) -> PropagationControl {
                times.append(t)
                return stopEarly && t >= 0.223 ? .stop : .proceed
            }
            let result: PropagationRunSummary
            switch api {
            case 0:
                result = try HEOM.solve(problem: problem, configuration: config, propagation: propagation) {
                    t, rho in
                    values.append(contentsOf: (0..<4).map { rho.elements[$0] })
                    return control(t)
                }
            case 1:
                result = try HEOM.solveWithHierarchy(
                    problem: problem, configuration: config, propagation: propagation
                ) { t, view in
                    for i in 0..<view.count {
                        view.withState(at: i) { rho in
                            for row in 0..<2 { for column in 0..<2 { values.append(rho[row, column]) } }
                        }
                    }
                    return control(t)
                }
            case 2:
                result = try HEOM.solveTwoTimeCorrelation(
                    problem: problem, configuration: config,
                    request: .init(insertionTime: 0.1, insertion: .left(b), observable: a),
                    propagation: propagation
                ) { t, value in
                    values.append(value)
                    return control(t)
                }
            default:
                result = try HEOM.solveMultiTimeOrderedCorrelation(
                    problem: problem, configuration: config,
                    request: .init(
                        insertions: [
                            .init(time: 0, insertion: .left(b)), .init(time: 0.17, insertion: .right(c)),
                        ],
                        observable: a), propagation: propagation
                ) { t, value in
                    values.append(value)
                    return control(t)
                }
            }
            return (times, values, result)
        }
        let serial = try run(.serial)
        let parallel = try run(.maximumWorkers(3))
        #expect(serial.0 == parallel.0)
        #expect(serial.2.finalTime == parallel.2.finalTime)
        #expect(serial.2.endReason == parallel.2.endReason)
        expectHOPSClose(serial.1, parallel.1, tolerance: 2e-12)
    }
}

@Test("Independent parallel HEOM solves do not share workers or mutable stage data")
func heomParallelConcurrentSolves() throws {
    var config = heomConfiguration(shift: .meanField, depth: 4)
    let problem = DensityMatrixProblem(hopsProblem())
    let expected = try heomFinal(
        problem: problem, configuration: config, propagation: hopsPropagation(end: 0.2))
    config.parallelism = .maximumWorkers(3)
    let parallelConfig = config
    let results = Mutex([[Complex<Double>]]())
    let errors = Mutex([String]())
    DispatchQueue.concurrentPerform(iterations: 4) { _ in
        do {
            let result = try heomFinal(
                problem: problem, configuration: parallelConfig, propagation: hopsPropagation(end: 0.2))
            results.withLock { $0.append(result) }
        } catch { errors.withLock { $0.append(String(describing: error)) } }
    }
    #expect(errors.withLock { $0.isEmpty })
    let values = results.withLock { $0 }
    #expect(values.count == 4)
    for value in values { expectHOPSClose(value, expected, tolerance: 2e-12) }
}

@Test("Parallel HEOM tears down workers after runtime errors and throwing observers")
func heomParallelFailureCleanup() throws {
    enum ObserverError: Error { case stopped }
    var config = heomConfiguration(shift: .meanField, depth: 3)
    config.parallelism = .maximumWorkers(3)
    let bad = hopsProblem(markovian: [
        .init(
            rate: .generated { t in t > 0.05 ? -1 : 0.2 },
            collapseOperator: .constant(hopsMatrix([0, 1, 0, 0])))
    ])
    #expect(throws: HEOM.CPUEngine.SolverError.self) {
        try HEOM.solve(problem: bad, configuration: config, propagation: hopsPropagation()) { _, _ in .proceed
        }
    }
    #expect(throws: ObserverError.stopped) {
        try HEOM.CPUEngine().run(
            problem: DensityMatrixProblem(hopsProblem()), configuration: config,
            propagation: hopsPropagation(output: .times([0.1, 0.3]))
        ) { _, _, _ in throw ObserverError.stopped }
    }
    let badH = ClosureHamiltonian { t, out in
        out = .zeros(rows: t > 0.05 ? 3 : 2, columns: t > 0.05 ? 3 : 2)
    }
    #expect(throws: HEOM.CPUEngine.SolverError.operatorDimensionMismatch) {
        try HEOM.solve(
            problem: DensityMatrixProblem(
                initialState: heomIdentity(2), system: .init(dimension: 2, hamiltonian: badH)),
            configuration: config, propagation: hopsPropagation()
        ) { _, _ in .proceed }
    }
    // Early returns before the first RHS and normal completion after failures.
    for end in [0.0, 0.1] {
        let result = try HEOM.solve(
            problem: hopsProblem(), configuration: config,
            propagation: hopsPropagation(end: end, output: .times([0]))
        ) { _, _ in .stop }
        #expect(result.endReason == .stoppedByObserver && result.finalTime == 0)
    }
    _ = try heomFinal(problem: DensityMatrixProblem(hopsProblem()), configuration: config)
}
