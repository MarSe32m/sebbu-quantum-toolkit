// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import NumericsExtensions
import SebbuQuantumToolkit

func NoiseGenerationExample() {
    do {
        try singleBath()
    } catch {
        print("Single bath fitting failed with error:", error)
    }
    
    do {
        try correlatedOUBath(count: 3)
    } catch {
        print("Correlated OU bath fitting failed with error:", error)
    }
    
    do {
        try identialEntriesBath(count: 3, correlation: 0.0)
    } catch {
        print("BCF with identical entries failed with error:", error)
    }
}

fileprivate func singleBath() throws {
    var tau: [Double] = .linearSpace(0, 10, 50)
    let bcf = tau.map { BCF($0) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) } }
    
    plt.figure()
    plt.plot(x: tau, y: bcf.real, label: "Re BCF")
    plt.plot(x: tau, y: bcf.imaginary, label: "Im BCF")
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
    
    let result = try CorrelatedBathFitter.fitBathCorrelation(times: tau, options: .init(maximumPencilPoleCount: 3, maximumFunctionEvaluations: 10000)) { t in
        BCF(t) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) }
    }
    print(result.model.poleCount)
    
    var rng = Philox4x64(seed: 12345)
    let generator = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(model: result.model, windowDuration: 20.0, start: 0.0, step: 0.001)
    
    for _ in 0..<10 {
        tau = .linearSpace(0, 10, 50)
        var noise = generator.generate(generator: &rng)
        var z: [Complex<Double>] = [.zero]
        var samples = tau.map { t in
            var span = z.mutableSpan
            noise.sample(t, into: &span, generator: &rng)
            return span[0]
        }
        plt.figure()
        plt.plot(x: tau, y: samples.map { $0.real }, label: "Re z")
        plt.plot(x: tau, y: samples.map { $0.imaginary }, label: "Im z")
        plt.xlabel("t")
        plt.ylabel("z")
        plt.legend()
        plt.show()
        plt.close()
        
        tau = .linearSpace(0, 10, 10000)
        samples = tau.map { t in
            var span = z.mutableSpan
            noise.sample(t, into: &span, generator: &rng)
            return span[0]
        }
        plt.figure()
        plt.plot(x: tau, y: samples.map { $0.real }, label: "Re z")
        plt.plot(x: tau, y: samples.map { $0.imaginary }, label: "Im z")
        plt.xlabel("t")
        plt.ylabel("z")
        plt.legend()
        plt.show()
        plt.close()
    }
    
    let tSpace: [Double] = .linearSpace(0, 10, 1000)
    plotGaussianNoiseVsBCF(seed: 123355, count: 5000, generator: generator, tSpace: tSpace) { t in
        BCF(t) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) }
    }
}

fileprivate func correlatedOUBath(count: Int) throws {
    precondition(count > 0)
    var kappas: [Complex<Double>] = []
    var gammas: [Double] = []
    var Omegas: [Double] = []
    for i in 1...count {
        kappas.append(Complex(Double(i) + 3, Double(i)))
        gammas.append(Double(i))
        Omegas.append(Double(i))
    }
    
    let tau: [Double] = .linearSpace(0, 10, 50)
    let bcf = tau.map { t in
        var a: Matrix<Complex<Double>> = .zeros(rows: count, columns: count)
        for m in 0..<count {
            for n in 0..<count {
                var result = (gammas[m] * gammas[n] * kappas[m].conjugate * kappas[n])
                result /= Complex(gammas[m] + gammas[n], Omegas[m] - Omegas[n])
                if t == .zero {
                    a[m, n] = result
                } else if t > .zero {
                    a[m, n] = result * .exp(-Complex(gammas[m], Omegas[m]) * t)
                } else {
                    a[m, n] = result * .exp(Complex(gammas[n], -Omegas[n]) * t)
                }
            }
        }
        return a
    }
    
    plt.figure()
    for m in 0..<count {
        for n in 0..<count {
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.real, label: "Re BCF[\(m),\(n)]")
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.imaginary, label: "Im BCF[\(m),\(n)]")
        }
    }
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
    
    let result = try CorrelatedBathFitter.fitBathCorrelation(times: tau, values: bcf, options: .init(maximumPencilPoleCount: 3, maximumFunctionEvaluations: 10000))
    // This should print 3. In a naive elementwise hierarchy construction there would be a total of 27 hierarchy directions
    print(result.model.poleCount)
    
    let generator = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(model: result.model, windowDuration: 10, start: 0, step: 0.001)
    let tSpace: [Double] = .linearSpace(0, 10, 1000)
    for i in 0..<count {
        for j in 0..<count {
            plotGaussianNoiseVsBCF(seed: 123456, i: i, j: j, count: 5000, generator: generator, tSpace: tSpace) { t in
                var a: Matrix<Complex<Double>> = .zeros(rows: count, columns: count)
                for m in 0..<count {
                    for n in 0..<count {
                        var result = (gammas[m] * gammas[n] * kappas[m].conjugate * kappas[n])
                        result /= Complex(gammas[m] + gammas[n], Omegas[m] - Omegas[n])
                        if t == .zero {
                            a[m, n] = result
                        } else if t > .zero {
                            a[m, n] = result * .exp(-Complex(gammas[m], Omegas[m]) * t)
                        } else {
                            a[m, n] = result * .exp(Complex(gammas[n], -Omegas[n]) * t)
                        }
                    }
                }
                return a
            }
        }
    }
}

fileprivate func identialEntriesBath(count: Int, correlation: Double) throws {
    precondition(count > 0)
    precondition(correlation >= 0 && correlation <= 1, "correlation must be between 0 and 1")
    let tau: [Double] = .linearSpace(0, 10, 50)
    let bcf = tau.map { t in
        let _bcf = BCF(t) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) }
        var a: Matrix<Complex<Double>> = .zeros(rows: count, columns: count)
        for m in 0..<count {
            for n in 0..<count {
                a[m, n] = correlation * _bcf + (m != n ? .zero : (1 - correlation) * _bcf)
            }
        }
        return a
    }
    
    plt.figure()
    for m in 0..<count {
        for n in 0..<count {
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.real, label: "Re BCF[\(m),\(n)]")
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.imaginary, label: "Im BCF[\(m),\(n)]")
        }
    }
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
    
    let result = try CorrelatedBathFitter.fitBathCorrelation(times: tau, values: bcf, options: .init(maximumPencilPoleCount: 3, maximumFunctionEvaluations: 10000))
    // This should print either 3 or 9 depending on the correlation parameter.
    // In a elementwise naive hierarchy construction there would be a total of 27 hierarchy directions
    print(result.model.poleCount)
    let generator = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(model: result.model, windowDuration: 10.0, start: 0.0, step: 0.001)
    let tSpace: [Double] = .linearSpace(0, 10, 1000)
    for i in 0..<count {
        for j in 0..<count {
            plotGaussianNoiseVsBCF(seed: 123456, i: i, j: j, count: 5000, generator: generator, tSpace: tSpace) { t in
                let _bcf = BCF(t) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) }
                var a: Matrix<Complex<Double>> = .zeros(rows: count, columns: count)
                for m in 0..<count {
                    for n in 0..<count {
                        a[m, n] = correlation * _bcf + (m != n ? .zero : (1 - correlation) * _bcf)
                    }
                }
                return a
            }
        }
    }
}

fileprivate func _spectralDensity(_ omega: Double, amplitude: Double, cutoff: Double, ohmicity s: Double) -> Double {
    return 0.5 * .pi * amplitude * .pow(cutoff, 1 - s) * .pow(omega, s) * .exp(-omega / cutoff)
}

fileprivate func BCF(_ t: Double, _ spectralDensity: (Double) -> Double) -> Complex<Double> {
    Quad.integrate(a: 0, b: .infinity) { omega in
        Complex(length: spectralDensity(omega), phase: -omega * t)
    }
}

fileprivate func plotGaussianNoiseVsBCF(seed: UInt64, count: Int, generator: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator, tSpace: [Double], bcf: (Double) -> Complex<Double>) {
    plotGaussianNoiseVsBCF(seed: seed, i: 0, j: 0, count: count, generator: generator, tSpace: tSpace) { t in
        return .init(elements: [bcf(t)], rows: 1, columns: 1)
    }
}

fileprivate func plotGaussianNoiseVsBCF(seed: UInt64, i: Int, j: Int, count: Int, generator: UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator, tSpace: [Double], bcf: (Double) -> Matrix<Complex<Double>>) {
    let sIndex = 0
    let generationStart = ContinuousClock.now
    var rng = Philox4x64(seed: seed)
    let noises = (0..<count).map { _ in
        var noise = generator.generate(generator: &rng)
        var samples: [[Complex<Double>]] = []
        for i in tSpace.indices {
            var newSamples: [Complex<Double>] = .init(repeating: .zero, count: noise.channelCount)
            var span = newSamples.mutableSpan
            noise.sample(tSpace[i], into: &span, generator: &rng)
            samples.append(newSamples)
        }
        return samples
    }
    let generationEnd = ContinuousClock.now
    print("Noise generation took:", generationEnd - generationStart)
    let bcf = tSpace.map { bcf($0)[i, j] }
    let meanBCF = tSpace.indices.parallelMap { t in
        var result: Complex<Double> = .zero
        for (_, z) in noises.enumerated() {
            result += z[t][i] * z[sIndex][j].conjugate
        }
        return result / Double(count)
    }
    plt.figure()
    plt.plot(x: tSpace, y: bcf.real, label: "Re alpha(t - s)")
    plt.plot(x: tSpace, y: bcf.imaginary, label: "Im alpha(t - s)")

    plt.plot(x: tSpace, y: meanBCF.real, label: "Re <z(t)z(s)^*>")
    plt.plot(x: tSpace, y: meanBCF.imaginary, label: "Im <z(t)z(s)^*>")

    plt.legend()
    plt.title("BCF[\(i),\(j)]")
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.show()
    plt.close()
}
