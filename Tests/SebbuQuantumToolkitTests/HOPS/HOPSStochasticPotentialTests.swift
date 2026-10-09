// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Synchronization
import Testing
@testable import SebbuQuantumToolkit

private struct PrescribedPotential: ~Copyable, MultiChannelNoiseRealization {
    let channelCount = 1
    static func value(_ time: Double) -> Complex<Double> { Complex(0.4 + time, -0.3 + 0.2 * time) }
    mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>) {
        values[0] = Self.value(time)
    }
}

private struct PotentialTestDiffusion: ~Copyable, DiffusiveNoiseRealization {
    let channelCount = 1
    mutating func sampleNormalized(at time: Double, stepSize: Double, into values: inout MutableSpan<Complex<Double>>) {
        values[0] = Complex(0.2 + 0.1 * time, -0.1)
    }
}

private struct PotentialTestGridNoise: ~Copyable, MultiChannelNoiseRealization {
    let channelCount = 1
    let integrationGrid: (origin: Double, step: Double)?
    mutating func sample(_ time: Double, into values: inout MutableSpan<Complex<Double>>) { values[0] = .zero }
}

@Suite("HOPS stochastic potentials", .serialized)
struct HOPSStochasticPotentialTests {
    @Test("Absent potential uses an empty borrowed span and preserves early stop")
    func absentPotentialObservation() throws {
        let run = try HOPS.solveTrajectory(problem: hopsProblem(), configuration: hcConfiguration(0),
            propagation: hopsPropagation(end: 0.2, output: .times([0, 0.1, 0.2])), seed: 5, trajectoryID: 0,
            observingWithNoises: { time, _, noise in
                #expect(noise.bath.count == 1 && noise.potential.isEmpty)
                return time == 0.1 ? .stop : .proceed
            })
        #expect(run.propagation.finalTime == 0.1 && run.propagation.endReason == .stoppedByObserver)
        #expect(run.potentialNoise == nil)
        // An unlabelled three-argument observer must keep resolving to the
        // legacy bath-only overload even if it never inspects the third value.
        _ = try HOPS.solveTrajectory(problem: hopsProblem(), configuration: hcConfiguration(0),
            propagation: hopsPropagation(end: 0.01), seed: 5, trajectoryID: 0) { _, _, _ in .proceed }
    }

    @Test("Potential mesh spacing resolves override, configuration, then integrator")
    func stepPrecedence() throws {
        var config = hcConfiguration(0)
        config.noiseStepSize = nil
        let propagation = hopsPropagation(maximumStep: 0.03)
        config.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: potentialTestModel())
        func resolved(_ configuration: HOPS.Configuration) throws -> Double? {
            let prep = try HCT.Preparation(problem: hopsProblem(), configuration: configuration, propagation: propagation)
            return prep.potentialPath(seed: 0, trajectoryID: 0, ensembleSampling: .independent)?.stepSize
        }
        #expect(try resolved(config) == 0.03)
        config.noiseStepSize = 0.02
        #expect(try resolved(config) == 0.02)
        config.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: potentialTestModel(), noiseStepSize: 0.01)
        #expect(try resolved(config) == 0.01)
    }

    @Test("Every tier and companion sees the uncentered Hermitian potential", arguments: 0..<6, 0..<4)
    func fullDriftReference(variant: Int, representation: Int) throws {
        let d = 4
        let (csr, a) = explicitSparseFixture(d)
        let b = a.conjugateTranspose
        let time = 0.13
        let c = Complex(1 + time, -0.3 * time)
        let e = Complex(0.2, 0.17)
        var literalL = a
        if representation >= 2 {
            for i in 0..<d { for j in 0..<d { literalL[i, j] = c * a[i, j] + e * b[i, j] } }
        }
        let source: TimeDependentOperator
        switch representation {
        case 0: source = .constant(a)
        case 1: source = .constant(csr)
        case 2:
            source = .linearCombination(.init(
                coefficients: [.generated { Complex(1 + $0, -0.3 * $0) }, .constant(e)],
                operators: [.init(csr), .init(b)]))
        default:
            source = .generatedDense(.init { t, output in
                for i in 0..<d { for j in 0..<d { output[i, j] = Complex(1 + t, -0.3 * t) * a[i, j] + e * b[i, j] } }
            })
        }
        let h = Matrix<Complex<Double>>.identity(rows: d)
        let problem = PureStateProblem(initialState: sparseFixtureState(d), system: .init(h),
                                       markovianChannels: [.init(rate: 0.27, collapseOperator: a)])
        let bath = CorrelatedBathModel.independentExponentials(amplitudes: [0.3], poles: [Complex(0.7, 1)])
        let base = hopsConfiguration(variant, model: bath, operators: [a], depth: 2)
        var config = base
        // Potential channels differ from the single physical hierarchy channel.
        config.stochasticPotential = .init(couplingOperators: [source, .constant(b)], covarianceModel: .zero(channelCount: 2))
        let prep = try HCT.Preparation(problem: problem, configuration: config, propagation: hopsPropagation())
        let basePrep = try HCT.Preparation(problem: problem, configuration: base, propagation: hopsPropagation())
        #expect(prep.poles.count == basePrep.poles.count && prep.shiftCount == basePrep.shiftCount)
        #expect(config.hierarchy.count == base.hierarchy.count)
        let y = [Complex<Double>(0.4, -0.3), Complex(-0.2, 0.5)]
        var hPlusV = h
        for i in 0..<d {
            for j in 0..<d {
                hPlusV[i, j] += y[0] * literalL[j, i].conjugate + y[0].conjugate * literalL[i, j]
                    + y[1] * b[j, i].conjugate + y[1].conjugate * b[i, j]
            }
        }
        var rhs = HCT.makeSeededRightHandSide(hamiltonian: .constant(h), preparation: prep, seed: 31, trajectoryID: 2)
        var reference = HCT.makeSeededRightHandSide(hamiltonian: .constant(hPlusV), preparation: basePrep, seed: 31, trajectoryID: 2)
        rhs.physicalNoise[0] = Complex(0.2, -0.1)
        reference.physicalNoise[0] = rhs.physicalNoise[0]
        rhs.potentialCoefficients[0] = y[0]
        rhs.potentialCoefficients[1] = y[1]
        if representation == 1 || representation == 2 {
            #expect(rhs.potentialOperators[0].action.terms[0].matrix.isSparse)
            #expect(rhs.potentialOperators[0].buffer.rows == 0)
        }
        let rows = 3 * config.hierarchy.count
        var state = HCT.State(dimension: d, hierarchyCount: rows, shiftCount: prep.shiftCount)
        var actual = HCT.State(dimension: d, hierarchyCount: rows, shiftCount: prep.shiftCount)
        var expected = HCT.State(dimension: d, hierarchyCount: rows, shiftCount: prep.shiftCount)
        for i in 0..<rows * d { state.amplitudes.elements[i] = Complex(0.1 + 0.013 * Double(i), -0.2 + 0.01 * Double(i % 7)) }
        for i in 0..<prep.shiftCount { state.shifts[i] = Complex(0.1, -0.03 * Double(i + 1)) }
        rhs.evaluateWithCurrentNoise(t: time, y: state, into: &actual)
        reference.evaluateWithCurrentNoise(t: time, y: state, into: &expected)
        expectHOPSClose((0..<rows * d).map { actual.amplitudes.elements[$0] },
                        (0..<rows * d).map { expected.amplitudes.elements[$0] }, tolerance: 2e-12)
        expectHOPSClose((0..<prep.shiftCount).map { actual.shifts[$0] }, (0..<prep.shiftCount).map { expected.shifts[$0] })
        // Potential noise is finite-variation drift, not an extra diffusion channel.
        rhs.diffusion(t: time, y: state, channel: 0, into: &actual)
        reference.diffusion(t: time, y: state, channel: 0, into: &expected)
        expectHOPSClose((0..<rows * d).map { actual.amplitudes.elements[$0] },
                        (0..<rows * d).map { expected.amplitudes.elements[$0] })
    }

    @Test("Custom noncopyable potential overrides its source and matches H plus V", arguments: 0..<6)
    func explicitPropagation(variant: Int) throws {
        let l = hopsFixtureOperators[0]
        let h = hcX
        let calls = Mutex(0)
        var config = hcConfiguration(variant, colored: true, depth: 2)
        config.stochasticPotential = .init(couplingOperator: .constant(l),
            fft: .init(dtMax: 0.1, deltaOmegaMax: 1, omegaMax: 2), powerDensity: { _ in
                calls.withLock { $0 += 1 }; return 1
            })
        var base = config
        base.stochasticPotential = nil
        let markovian = [MarkovianChannel(rate: 0.2, collapseOperator: hcL)]
        let problem = hopsProblem(h, markovian: markovian)
        let reference = PureStateProblem(initialState: problem.initialState,
            system: .init(dimension: 2) { t, output in
                let y = PrescribedPotential.value(t)
                for i in 0..<2 { for j in 0..<2 {
                    output[i, j] = h[i, j] + y * l[j, i].conjugate + y.conjugate * l[i, j]
                } }
            }, markovianChannels: markovian)
        let propagation = hopsPropagation(end: 0.07, maximumStep: 0.005)
        var actual: [Complex<Double>] = []
        let summary = try HOPS.solveWithHierarchy(
            problem: problem, configuration: config, propagation: propagation,
            bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: PotentialTestDiffusion(),
            potentialNoise: PrescribedPotential(), observingWithNoises: { time, hierarchy, noise in
                #expect(noise.potential[0] == PrescribedPotential.value(time))
                for k in 0..<hierarchy.count {
                    hierarchy.withState(at: k) { state in for j in 0..<2 { actual.append(state[j]) } }
                }
                return .proceed
            })
        var expected: [Complex<Double>] = []
        _ = try HOPS.solveWithHierarchy(
            problem: reference, configuration: base, propagation: propagation,
            bathNoise: ZeroMultiChannelNoise(channelCount: 1), markovianNoise: PotentialTestDiffusion()
        ) { _, hierarchy in
            for k in 0..<hierarchy.count {
                hierarchy.withState(at: k) { state in for j in 0..<2 { expected.append(state[j]) } }
            }
            return .proceed
        }
        expectHOPSClose(actual, expected, tolerance: 1e-10)
        #expect(summary.finalTime == 0.07 && calls.withLock { $0 } == 0)
    }

    @Test("Different grid origins and spacings constrain ODE and SDE steps", arguments: [false, true])
    func integrationGrids(stochastic: Bool) throws {
        var config = hcConfiguration(0)
        config.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: .zero(channelCount: 1))
        let problem = hopsProblem(hcZero, markovian: [.init(rate: 0, collapseOperator: hcL)])
        let propagation = hopsPropagation(end: 0.24, start: 0.04, maximumStep: 0.1, output: .everyAcceptedStep)
        var observed: [Double] = []
        func record(_ time: Double, _ state: borrowing UniqueVector<Complex<Double>>) -> PropagationControl {
            observed.append(time); return .proceed
        }
        if stochastic {
            _ = try HOPS.solveTrajectory(problem: problem, configuration: config, propagation: propagation,
                bathNoise: PotentialTestGridNoise(integrationGrid: (0.01, 0.06)), markovianNoise: PotentialTestDiffusion(),
                potentialNoise: PotentialTestGridNoise(integrationGrid: (-0.015, 0.047)), observing: record)
        } else {
            _ = try HOPS.solveTrajectory(problem: problem, configuration: config, propagation: propagation,
                bathNoise: PotentialTestGridNoise(integrationGrid: (0.01, 0.06)), markovianNoise: ZeroDiffusiveNoise(channelCount: 1),
                potentialNoise: PotentialTestGridNoise(integrationGrid: (-0.015, 0.047)), observing: record)
        }
        let boundaries = (1...4).map { 0.01.addingProduct(Double($0), 0.06) }
            + (1...6).map { (-0.015).addingProduct(Double($0), 0.047) }
        for boundary in boundaries where boundary > 0.04 && boundary < 0.24 {
            #expect(observed.contains(boundary))
        }
    }

    @Test("Zero potential preserves propagation, bath provenance and caller RNG words")
    func zeroAndRNGCompatibility() throws {
        let base = hcConfiguration(5, colored: true, depth: 2)
        var zero = base
        zero.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: .zero(channelCount: 1))
        let problem = hopsProblem(markovian: [.init(rate: 0.1, collapseOperator: hcL)])
        let propagation = hopsPropagation(end: 0.04, maximumStep: 0.01)
        var r0 = SplitMix64(seed: 99)
        var rz = SplitMix64(seed: 99)
        var control = SplitMix64(seed: 99)
        _ = control.next(); _ = control.next()
        var a: [Complex<Double>] = []
        var b: [Complex<Double>] = []
        let absent = try HOPS.solveTrajectory(problem: problem, configuration: base, propagation: propagation, rng: &r0) {
            _, state in a = (0..<state.count).map { state[$0] }; return .proceed
        }
        let configured = try HOPS.solveTrajectory(problem: problem, configuration: zero, propagation: propagation, rng: &rz) {
            _, state in b = (0..<state.count).map { state[$0] }; return .proceed
        }
        #expect(a == b)
        let next = control.next()
        #expect(r0.next() == next && rz.next() == next)
        #expect(absent.potentialNoise == nil && configured.potentialNoise != nil)
        #expect(absent.bathNoise.latentCount == configured.bathNoise.latentCount)
        var s0 = absent.bathNoise.makeSampler(windowDuration: 0.1)
        var sz = configured.bathNoise.makeSampler(windowDuration: 0.1)
        var v0 = UniqueVector<Complex<Double>>.zero(1)
        var vz = UniqueVector<Complex<Double>>.zero(1)
        for t in [0.0, 0.02, 0.04] {
            s0.sample(t, into: &v0.mutableSpan); sz.sample(t, into: &vz.mutableSpan)
            #expect(v0[0] == vz[0])
        }
    }

    @Test("Potential changes do not change quantum or Markovian reference streams")
    func independentReferenceStreams() throws {
        let problem = hopsProblem(markovian: [.init(rate: 0.4, collapseOperator: hcL)])
        let config = hcConfiguration(5, colored: true, depth: 2)
        let prep = try HCT.Preparation(problem: problem, configuration: config, propagation: hopsPropagation())
        var a = HCT.makeSeededRightHandSide(hamiltonian: problem.system.hamiltonian, preparation: prep, seed: 7, trajectoryID: 3)
        var modified = config
        modified.stochasticPotential = .init(couplingOperator: .constant(hcX), covarianceModel: potentialTestModel())
        let otherPrep = try HCT.Preparation(problem: problem, configuration: modified, propagation: hopsPropagation())
        var b = HCT.makeSeededRightHandSide(hamiltonian: problem.system.hamiltonian, preparation: otherPrep, seed: 7, trajectoryID: 3)
        var wa = UniqueVector<Complex<Double>>.zero(1)
        var wb = UniqueVector<Complex<Double>>.zero(1)
        for t in [0.0, 0.025, 0.1, 0.09, 0.1, 0.2] {
            a.sampleColoredNoises(at: t); b.sampleColoredNoises(at: t)
            #expect(a.physicalNoise[0] == b.physicalNoise[0])
            a.sampleNormalizedNoises(t: t, stepSize: 0.01, into: &wa.mutableSpan)
            b.sampleNormalizedNoises(t: t, stepSize: 0.01, into: &wb.mutableSpan)
            #expect(wa[0] == wb[0])
        }
    }

    @Test("Observation and replay agree through adaptive retries and nonzero origins")
    func observationReplay() throws {
        var config = hcConfiguration(5, colored: true, depth: 2)
        config.noiseStepSize = 0.021
        config.stochasticPotential = .init(couplingOperator: .constant(hcL), covarianceModel: potentialTestModel(), noiseStepSize: 0.033)
        var observed: [(Double, Complex<Double>, Complex<Double>)] = []
        let problem = hopsProblem(hopsMatrix([0, 15, 15, 0]))
        let run = try HOPS.solveTrajectory(problem: problem, configuration: config,
            propagation: hopsPropagation(end: 1.3, start: 1, maximumStep: 0.3, tolerance: 1e-11,
                output: .times([1, 1.071, 1.19, 1.3])), seed: 73, trajectoryID: 4,
            observingWithNoises: { time, _, noise in
                observed.append((time, noise.bath[0], noise.potential[0])); return .proceed
            })
        var bath = run.bathNoise.makeSampler(windowDuration: 0)
        var potential = try #require(run.potentialNoise).makeSampler(windowDuration: 0)
        var z = UniqueVector<Complex<Double>>.zero(1)
        var y = UniqueVector<Complex<Double>>.zero(1)
        for (t, expectedZ, expectedY) in observed {
            bath.sample(t, into: &z.mutableSpan); potential.sample(t, into: &y.mutableSpan)
            #expect(z[0] == expectedZ && y[0] == expectedY)
        }
        let fine = try hopsFinalState(problem: problem, configuration: config,
            propagation: hopsPropagation(end: 1.3, start: 1, maximumStep: 0.001, tolerance: 1e-11), seed: 73, id: 4)
        let coarse = try hopsFinalState(problem: problem, configuration: config,
            propagation: hopsPropagation(end: 1.3, start: 1, maximumStep: 0.3, tolerance: 1e-11), seed: 73, id: 4)
        expectHOPSClose(fine, coarse, tolerance: 3e-8)
    }

    @Test("Worker counts preserve individual potential trajectories and ensemble averages")
    func parallelism() throws {
        var config = hcConfiguration(5, colored: true, depth: 2)
        config.stochasticPotential = .init(couplingOperator: .constant(hcX), covarianceModel: potentialTestModel())
        let problem = hopsProblem()
        let propagation = hopsPropagation(end: 0.04)
        var previous: [UInt64: [Complex<Double>]]?
        var previousMean: [Complex<Double>]?
        for parallelism: TrajectoryParallelism in [.serial, .maximumConcurrentTasks(3)] {
            let states = Mutex<[UInt64: [Complex<Double>]]>([:])
            let execution = TrajectoryExecution(trajectories: 12, seed: 71, parallelism: parallelism, ensembleSampling: .antithetic)
            let run = try HOPS.solveTrajectories(problem: problem, configuration: config, propagation: propagation, execution: execution) {
                id, _, state in
                let value = (0..<state.count).map { state[$0] }
                states.withLock { $0[id] = value }
            }
            let current = states.withLock { $0 }
            if let previous { #expect(previous == current) }
            previous = current
            #expect(run.potentialNoise?.path(for: 3).trajectoryID == 3)
            var mean: [Complex<Double>] = []
            _ = try HOPS.solveEnsemble(problem: problem, configuration: config, propagation: propagation, execution: execution) {
                _, rho in mean = (0..<4).map { rho.elements[$0] }
            }
            if let previousMean { expectHOPSClose(mean, previousMean, tolerance: 2e-14) }
            previousMean = mean
        }
    }
}
