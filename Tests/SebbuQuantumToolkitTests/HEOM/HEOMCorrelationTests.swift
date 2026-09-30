// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("HEOM Correlation tests")
struct HEOMCorrelationTests {
    @Test("HEOM retains bath memory through left and right insertions", arguments: heomShifts, [false, true])
    func heomTwoTimePseudomode(shift: HEOM.ShiftType, right: Bool) throws {
        let h = hopsMatrix([0, Complex(0.35), Complex(0.35), Complex(0.2)])
        let l = hopsMatrix([0, 0, 0, 1])
        let rho = hopsMatrix([Complex(0.4), Complex(0, 0.2), Complex(0, -0.2), Complex(0.6)])
        let b = hopsMatrix([0, 1, 0, 0])
        let a = hopsMatrix([0, 0, 1, 0])
        let problem = DensityMatrixProblem(initialState: rho, system: QuantumSystem(h))
        let times = [0.0, 0.13, 0.4, 0.67, 1.2]
        let propagation = hopsPropagation(end: 1.2, tolerance: 1e-10, output: .times(times))
        let n = 7
        let insertion: CorrelationInsertion = right ? .right(.constant(b)) : .left(.constant(b))
        let jointB = heomTensor(b, heomIdentity(n))
        let jointInsertion: CorrelationInsertion = right ? .right(.constant(jointB)) : .left(.constant(jointB))
        var expected: [Complex<Double>] = []
        try GKSL.solveTwoTimeCorrelation(
            problem: heomPseudomodeProblem(h: h, coupling: l, rho: rho, oscillatorDimension: n),
            request: .init(
                insertionTime: 0.4, insertion: jointInsertion,
                observable: .constant(heomTensor(a, heomIdentity(n)))),
            propagation: propagation
        ) { _, value in
            expected.append(value)
            return .proceed
        }
        var actual: [Complex<Double>] = []
        var observed: [Double] = []
        let request = TwoTimeCorrelationRequest(
            insertionTime: 0.4, insertion: insertion, observable: .constant(a))
        try HEOM.solveTwoTimeCorrelation(
            problem: problem, configuration: heomConfiguration(shift: shift, depth: 10),
            request: request, propagation: propagation
        ) { t, value in
            actual.append(value)
            observed.append(t)
            return .proceed
        }
        #expect(observed == [0.4, 0.67, 1.2])
        expectHOPSClose(actual, expected, tolerance: 5e-8)
        var multi: [Complex<Double>] = []
        try HEOM.solveMultiTimeOrderedCorrelation(
            problem: problem, configuration: heomConfiguration(shift: shift, depth: 10),
            request: .init(
                insertions: [.init(time: request.insertionTime, insertion: insertion)],
                observable: request.observable),
            propagation: propagation
        ) { _, value in
            multi.append(value)
            return .proceed
        }
        expectHOPSClose(actual, multi, tolerance: 1e-13)
    }
    
    @Test("HEOM ordered mixed-side correlations match an enlarged-system regression", arguments: heomShifts)
    func heomMultiTimePseudomode(shift: HEOM.ShiftType) throws {
        let h = hopsMatrix([0, Complex(0.4), Complex(0.4), Complex(0.2)])
        let l = hopsMatrix([0, 0, 0, 1])
        let rho = hopsMatrix([Complex(0.4), 0, 0, Complex(0.6)])
        let b = hopsMatrix([0, 1, 0, 0])
        let c = hopsMatrix([1, 0, 0, -1])
        let a = hopsMatrix([0, 0, 1, 0])
        let n = 7
        let propagation = hopsPropagation(end: 1.2, tolerance: 1e-10, output: .times([0, 0.3, 0.6, 0.83, 1.2]))
        // The initial insertion is traceless. The physical guide nevertheless has
        // a nonzero <M> and evolves in time: it must supply both later shifts.
        let request = MultiTimeOrderedCorrelationRequest(
            insertions: [
                .init(time: 0, insertion: .left(.constant(b))),
                .init(time: 0.6, insertion: .right(.constant(c))),
            ], observable: .constant(a))
        let jointRequest = MultiTimeOrderedCorrelationRequest(
            insertions: [
                .init(time: 0, insertion: .left(.constant(heomTensor(b, heomIdentity(n))))),
                .init(time: 0.6, insertion: .right(.constant(heomTensor(c, heomIdentity(n))))),
            ], observable: .constant(heomTensor(a, heomIdentity(n))))
        var reference: [Complex<Double>] = []
        try GKSL.solveMultiTimeOrderedCorrelation(
            problem: heomPseudomodeProblem(h: h, coupling: l, rho: rho, oscillatorDimension: n),
            request: jointRequest, propagation: propagation
        ) { _, value in
            reference.append(value)
            return .proceed
        }
        var values: [Complex<Double>] = []
        var times: [Double] = []
        try HEOM.solveMultiTimeOrderedCorrelation(
            problem: .init(initialState: rho, system: QuantumSystem(h)),
            configuration: heomConfiguration(shift: shift, depth: 10), request: request, propagation: propagation
        ) { t, value in
            times.append(t)
            values.append(value)
            return .proceed
        }
        #expect(times == [0.6, 0.83, 1.2])
        expectHOPSClose(values, reference, tolerance: 5e-8)
    }
    
    @Test(
        "Time-dependent insertions and observables use absolute times and preserve all ADOs",
        arguments: heomShifts)
    func heomDynamicCorrelationOperators(shift: HEOM.ShiftType) throws {
        let problem = hopsProblem()
        let config = heomConfiguration(shift: shift)
        let times = [0.4, 0.55, 0.8]
        let propagation = hopsPropagation(end: 0.8, start: 0.2, tolerance: 1e-10, output: .times(times))
        let b = hopsMatrix([0, 1, 0, 0])
        let a = hopsMatrix([0, 0, 1, 0])
        var baseline: [Complex<Double>] = []
        try HEOM.solveTwoTimeCorrelation(
            problem: problem, configuration: config,
            request: .init(insertionTime: 0.4, insertion: .left(.constant(b)), observable: .constant(a)),
            propagation: propagation
        ) { _, value in
            baseline.append(value)
            return .proceed
        }
        let dynamicB = TimeDependentOperator.generatedDense(
            .init { t, out in
                for i in 0..<4 { out.elements[i] = (1 + t) * b.elements[i] }
            })
        let dynamicA = TimeDependentOperator.generatedDense(
            .init { t, out in
                for i in 0..<4 { out.elements[i] = Complex(0, t) * a.elements[i] }
            })
        var index = 0
        try HEOM.CPUEngine().solveTwoTimeCorrelation(
            problem: problem, configuration: config,
            request: .init(insertionTime: 0.4, insertion: .left(dynamicB), observable: dynamicA),
            propagation: propagation
        ) { t, value in
            #expect((value - Complex(0, t) * 1.4 * baseline[index]).length < 1e-8)
            index += 1
            return .proceed
        }
        #expect(index == 3)
    }
    
    @Test("A zero inserted operator stays zero while the centered guide evolves")
    func heomZeroCompanion() throws {
        var callbacks = 0
        try HEOM.solveTwoTimeCorrelation(
            problem: hopsProblem(), configuration: heomConfiguration(shift: .meanField),
            request: .init(
                insertionTime: 0.2, insertion: .left(.constant(hopsMatrix([0, 0, 0, 0]))),
                observable: .constant(hopsMatrix([0, 1, 0, 0]))),
            propagation: hopsPropagation(end: 0.8, output: .uniform(step: 0.1))
        ) { _, value in
            #expect(value == .zero)
            callbacks += 1
            return .proceed
        }
        #expect(callbacks > 0)
    }
}
