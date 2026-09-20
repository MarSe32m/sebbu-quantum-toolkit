// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Test("HEOM doubles latent indices, conjugates bra damping and retains factorial weights")
func heomHierarchyTables() {
    let environment = BathEnvironment(
        couplingOperators: hopsFixtureOperators.map { .constant($0) }, bath: hopsFixtureModel())
    let hierarchy = HEOM.Hierarchy(environment: environment, truncation: .maximumTier(3))
    let hops = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(3))
    #expect(hierarchy.multiIndexCount == 6)
    #expect(hierarchy.count == 84)
    #expect(hops.count == 20)
    let indices = heomIndices(hierarchy)
    let ids = Dictionary(uniqueKeysWithValues: indices.enumerated().map { ($0.element, $0.offset) })
    let poles = environment.bath.latentBaths.flatMap(\.poles)
    #expect(indices[0] == [0, 0, 0, 0, 0, 0])
    for index in indices.indices {
        let ns = indices[index]
        #expect(hierarchy.tier(at: index) == ns.reduce(0, +))
        var damping = Complex<Double>.zero
        for p in poles.indices {
            damping -= Double(ns[p]) * poles[p] + Double(ns[p + 3]) * poles[p].conjugate
        }
        #expect((hierarchy.damping(at: index) - damping).length < 1e-13)
        hierarchy.parentIndices(of: index) { parents in
            for p in 0..<6 {
                var n = ns
                n[p] -= 1
                #expect(parents[p] == (ids[n] ?? -1))
            }
        }
        hierarchy.childIndices(of: index) { children in
            for p in 0..<6 {
                var n = ns
                n[p] += 1
                #expect(children[p] == (ids[n] ?? -1))
            }
        }
        hierarchy.parentWeights(of: index) { weights in
            for p in 0..<6 { #expect(weights[p] == Double(ns[p]).squareRoot()) }
        }
        hierarchy.childWeights(of: index) { weights in
            for p in 0..<6 { #expect(weights[p] == Double(ns[p] + 1).squareRoot()) }
        }
    }
}

@Test("HEOM supports finite anisotropic custom truncation and zero baths")
func heomCustomAndEmptyHierarchy() {
    let environment = BathEnvironment(couplingOperator: .constant(heomIdentity(2)), bath: heomSingleBath())
    let hierarchy = HEOM.Hierarchy(
        environment: environment, truncation: .custom { n in n[0] <= 2 && n[1] <= 1 })
    #expect(hierarchy.count == 6)
    #expect(hierarchy.maximumTier == 3)
    #expect(Set(heomIndices(hierarchy)) == Set([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]))
    let empty = HEOM.Hierarchy(
        environment: .init(couplingOperator: .constant(heomIdentity(2)), bath: .zero(channelCount: 1)),
        truncation: .maximumTier(100))
    #expect(empty.count == 1)
    #expect(empty.multiIndexCount == 0)
    #expect(empty.maximumTier == 0)
    #expect(empty.damping(at: 0) == .zero)
    empty.parentIndices(of: 0) { #expect($0.count == 0) }
    empty.childIndices(of: 0) { #expect($0.count == 0) }
    #expect(heomConfiguration(depth: 0).hierarchy.count == 1)
}
