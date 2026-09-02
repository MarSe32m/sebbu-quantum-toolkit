// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuPythonKit
import SebbuScience
import Numerics
import NumericsExtensions
import SebbuQuantumToolkit

func BCFFittingExample() {
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
        try identialEntriesBath(count: 3, correlation: 0.33)
    } catch {
        print("BCF with identical entries failed with error:", error)
    }
}

fileprivate func singleBath() throws {
    var tau: [Double] = .linearSpace(0, 10, 50)
    var bcf = tau.map { BCF($0) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) } }
    
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
    tau = .linearSpace(-10, 10, 1000)
    bcf = tau.map { BCF($0) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) } }
    let fittedBCF = tau.map { result.model.bathCorrelation(at: $0)[0, 0] }
    plt.figure()
    plt.plot(x: tau, y: bcf.real, label: "Re BCF")
    plt.plot(x: tau, y: bcf.imaginary, label: "Im BCF")
    plt.plot(x: tau, y: fittedBCF.real, label: "Re BCF", linestyle: "--")
    plt.plot(x: tau, y: fittedBCF.imaginary, label: "Im BCF", linestyle: "--")
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
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
    
    var tau: [Double] = .linearSpace(0, 10, 50)
    var bcf = tau.map { t in
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
    tau = .linearSpace(-10, 10, 1000)
    bcf = tau.map { t in
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
    let fittedBCF = tau.map { result.model.bathCorrelation(at: $0) }
    
    plt.figure()
    for m in 0..<count {
        for n in 0..<count {
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.real, label: "Re BCF[\(m),\(n)]")
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.imaginary, label: "Im BCF[\(m),\(n)]")
            plt.plot(x: tau, y: fittedBCF.map {$0[m, n]}.real, label: "fit Re BCF[\(m),\(n)]", linestyle: "--")
            plt.plot(x: tau, y: fittedBCF.map {$0[m, n]}.imaginary, label: "fit Im BCF[\(m),\(n)]", linestyle: "--")
        }
    }
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
}

fileprivate func identialEntriesBath(count: Int, correlation: Double) throws {
    precondition(count > 0)
    precondition(correlation >= 0 && correlation <= 1, "correlation must be between 0 and 1")
    var tau: [Double] = .linearSpace(0, 10, 50)
    var bcf = tau.map { t in
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
    tau = .linearSpace(-10, 10, 1000)
    bcf = tau.map { t in
        let _bcf = BCF(t) { _spectralDensity($0, amplitude: 0.027, cutoff: 1.447, ohmicity: 3) }
        var a: Matrix<Complex<Double>> = .zeros(rows: count, columns: count)
        for m in 0..<count {
            for n in 0..<count {
                a[m, n] = correlation * _bcf + (m != n ? .zero : (1 - correlation) * _bcf)
            }
        }
        return a
    }
    let fittedBCF = tau.map { result.model.bathCorrelation(at: $0) }
    
    plt.figure()
    for m in 0..<count {
        for n in 0..<count {
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.real, label: "Re BCF[\(m),\(n)]")
            plt.plot(x: tau, y: bcf.map {$0[m, n]}.imaginary, label: "Im BCF[\(m),\(n)]")
            plt.plot(x: tau, y: fittedBCF.map {$0[m, n]}.real, label: "fit Re BCF[\(m),\(n)]", linestyle: "--")
            plt.plot(x: tau, y: fittedBCF.map {$0[m, n]}.imaginary, label: "fit Im BCF[\(m),\(n)]", linestyle: "--")
        }
    }
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.legend()
    plt.show()
    plt.close()
}

func quantumDotSpectralDensity(_ omega: Double, _ A: Double, _ cutoff: Double) -> Double {
    return A * omega * omega * omega * Double.exp(-(omega * omega) / (cutoff * cutoff))
}
fileprivate func _spectralDensity(_ omega: Double, amplitude: Double, cutoff: Double, ohmicity s: Double) -> Double {
    return 0.5 * .pi * amplitude * .pow(cutoff, 1 - s) * .pow(omega, s) * .exp(-omega / cutoff)
}

fileprivate func BCF(_ t: Double, _ spectralDensity: (Double) -> Double) -> Complex<Double> {
    Quad.integrate(a: 0, b: .infinity) { omega in
        Complex(length: spectralDensity(omega), phase: -omega * t)
    }
}
