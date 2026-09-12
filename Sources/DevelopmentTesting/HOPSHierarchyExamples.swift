// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import SebbuQuantumToolkit
import SebbuScience
import Numerics

public func hierarchyExample() {
    do {
        let tau: [Double] = .linearSpace(0, 10, 50)
        let bcf: [Matrix<Complex<Double>>] = tau.map { t in
            let value = 0.5 * .exp(-Complex(0.1, 0.1) * t) + 0.1 * .exp(-Complex(0.5, 0.2) * t)
            return .diagonal(from: [value, 2 * value])
        }
        let L = TimeDependentOperator.constant(Matrix<Complex<Double>>.diagonal(from: [1, -1]))
        let bath = try CorrelatedBathFitter.fitBathCorrelation(times: tau, values: bcf)
        let environment = HOPS.Environment(
            couplingOperators: [L, L],
            bath: bath.model
        )
        let hierarchy = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(8))
        for n in 0..<hierarchy.count {
            hierarchy.childIndices(of: n) { indices in
                hierarchy.childWeights(of: n) { weights in
                    print(n, terminator: ": ")
                    for i in indices.indices {
                        print(indices[i], weights[i], separator: ":", terminator: " ")
                    }
                    print()
                }
            }
        }
        print()
        for n in 0..<hierarchy.count {
            hierarchy.parentIndices(of: n) { indices in
                hierarchy.parentWeights(of: n) { weights in
                    print(n, terminator: ": ")
                    for i in indices.indices {
                        print(indices[i], weights[i], separator: ":", terminator: " ")
                    }
                    print()
                }
            }
        }
        print()
        for n in 0..<hierarchy.count {
            hierarchy.multiIndex(of: n) { multiIndex in
                print(n, terminator: ": ")
                for i in multiIndex.indices {
                    print(multiIndex[i], terminator: " ")
                }
                print()
            }
        }
        print()
        for n in 0..<hierarchy.count {
            print(n, hierarchy.damping(at: n), separator: ": ")
        }
    } catch {
        print("Hierarchy example failed with error:", error)
    }
}
