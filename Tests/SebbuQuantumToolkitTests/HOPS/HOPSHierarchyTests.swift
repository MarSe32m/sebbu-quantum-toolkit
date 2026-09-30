// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience
import Testing
@testable import SebbuQuantumToolkit

@Suite("HOPS latent-pole hierarchy")
struct HOPSHierarchyTests {
    @Test("One pole gives a chain with cached Fock weights", arguments: [0, 1, 4, 12])
    func singlePole(depth: Int) {
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(0)),
                                       truncation: .maximumTier(depth))
        #expect(hierarchy.count == depth + 1)
        #expect(hierarchy.multiIndexCount == 1)
        #expect(hierarchy.maximumTier == depth)
        for h in 0..<hierarchy.count {
            let row = hierarchyRow(hierarchy, h)
            #expect(row.n == [h])
            #expect(row.parents == [h == 0 ? -1 : h - 1])
            #expect(row.children == [h == depth ? -1 : h + 1])
            #expect(row.down == [Double(h).squareRoot()])
            #expect(row.up == [Double(h + 1).squareRoot()])
            #expect(hierarchy.tier(at: h) == h)
            #expect((hierarchy.damping(at: h) + Double(h) * Complex(0.7, 1.2)).length < 1e-13)
        }
    }

    @Test("Single, independent, correlated and partially shared baths", arguments: [0, 1, 2, 3])
    func bathLayouts(kind: Int) {
        let model = noiseTestModel(kind)
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(model), truncation: .maximumTier(3))
        let expected = cartesianIndices(directions: model.poleCount, limit: 3) {
            $0.reduce(0, +) <= 3
        }
        #expect(Set(allOccupations(hierarchy)) == Set(expected))
        #expect(hierarchy.count == [4, 10, 10, 35][kind])
        verifyHierarchy(hierarchy)

        let preparation = UniformSlidingWindowCorrelatedOrnsteinUhlenbeckProcessGenerator(
            model: model, windowDuration: 1, step: 0.1)
        #expect(hierarchy.multiIndexCount == preparation.latentCount)
        let rows = allOccupations(hierarchy)
        for a in model.latentBaths.indices {
            for mu in model.latentBaths[a].poles.indices {
                let p = preparation.latentBathRanges[a].lowerBound + mu
                var unit = [Int](repeating: 0, count: hierarchy.multiIndexCount)
                unit[p] = 1
                let h = rows.firstIndex(of: unit)!
                #expect(hierarchy.damping(at: h) == -model.latentBaths[a].poles[mu])
            }
        }
    }

    @Test("Physical coupling count is independent of pole count")
    func rectangularModels() {
        let singleChannel = CorrelatedBathModel(channelCount: 1, latentBaths: [
            noiseTestBath([Complex(1, 2), Complex(2, -1), Complex(3, 0.5)], [[1, 2, 3]])
        ])
        let one = HOPS.Environment(couplingOperator: hierarchyCoupling, bath: singleChannel)
        #expect(HOPS.Hierarchy(environment: one, truncation: .maximumTier(2)).count == 10)

        let shared = CorrelatedBathModel(channelCount: 3, latentBaths: [
            noiseTestBath([Complex(1, 2)], [[1], [Complex(0, 1)], [2]])
        ])
        let three = HOPS.Hierarchy(environment: hierarchyEnvironment(shared), truncation: .maximumTier(2))
        #expect(three.multiIndexCount == 1)
        #expect(three.count == 3)
    }

    @Test("Equal poles in independent baths retain mixed auxiliaries")
    func independentEqualPoles() {
        let model = noiseTestModel(1)
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(model), truncation: .maximumTier(2))
        #expect(hierarchy.multiIndexCount == 2)
        #expect(hierarchy.count == 6)
        #expect(allOccupations(hierarchy).contains([1, 1]))
        verifyHierarchy(hierarchy)
    }

    @Test("Repeated poles within a latent bath preserve the sampler layout")
    func repeatedSharedPoles() {
        let model = CorrelatedBathModel(channelCount: 1, latentBaths: [
            noiseTestBath([Complex(1, 2), Complex(1, 2)], [[1, 2]])
        ])
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(model), truncation: .maximumTier(2))
        #expect(hierarchy.multiIndexCount == 2)
        #expect(hierarchy.count == 6)
        verifyHierarchy(hierarchy)
    }

    @Test("Zero bath retains one root and empty direction spans", arguments: [0, 7, Int.max])
    func zeroBath(depth: Int) {
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(.zero(channelCount: 3)),
                                       truncation: .maximumTier(depth))
        #expect(hierarchy.count == 1)
        #expect(hierarchy.multiIndexCount == 0)
        #expect(hierarchy.maximumTier == 0)
        #expect(hierarchy.tier(at: 0) == 0)
        #expect(hierarchy.damping(at: 0) == .zero)
        let row = hierarchyRow(hierarchy, 0)
        #expect(row.n.isEmpty && row.parents.isEmpty && row.children.isEmpty)
        #expect(row.down.isEmpty && row.up.isEmpty)
        let custom = HOPS.Hierarchy(environment: hierarchy.environment, truncation: .custom { _ in true })
        #expect(custom.count == 1)
    }

    @Test("Custom weighted tiers and box truncations match exhaustive enumeration")
    func customTruncation() {
        let environment = hierarchyEnvironment(noiseTestModel(3))
        let weighted = HOPS.Hierarchy(environment: environment, truncation: .custom { n in
            n[0] + 2 * n[1] + 3 * n[2] + 4 * n[3] <= 5
        })
        let expected = cartesianIndices(directions: 4, limit: 5) { n in
            n[0] + 2 * n[1] + 3 * n[2] + 4 * n[3] <= 5
        }
        #expect(Set(allOccupations(weighted)) == Set(expected))
        #expect(weighted.maximumTier == 5)
        verifyHierarchy(weighted)

        let box = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(2)),
                                 truncation: .custom { n in n[0] <= 2 && n[1] <= 3 })
        #expect(box.count == 12)
        #expect(box.maximumTier == 5)
        #expect(Set(allOccupations(box)) == Set(cartesianIndices(directions: 2, limit: 3) {
            $0[0] <= 2
        }))
        verifyHierarchy(box)
    }

    @Test("Custom root-only truncation and equivalent tier predicates")
    func equivalentTruncations() {
        let environment = hierarchyEnvironment(noiseTestModel(2))
        let rootOnly = HOPS.Hierarchy(environment: environment, truncation: .custom { n in
            for i in 0..<n.count where n[i] != 0 { return false }
            return true
        })
        #expect(rootOnly.count == 1)
        #expect(hierarchyRow(rootOnly, 0).children == [-1, -1])

        let standard = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(4))
        let custom = HOPS.Hierarchy(environment: environment, truncation: .custom { $0[0] + $0[1] <= 4 })
        #expect(allOccupations(standard) == allOccupations(custom))
        for h in 0..<standard.count {
            #expect(hierarchyRow(standard, h) == hierarchyRow(custom, h))
        }
    }

    @Test("BFS order is deterministic and tiers are contiguous")
    func ordering() {
        let environment = hierarchyEnvironment(noiseTestModel(3))
        let a = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(4))
        let b = HOPS.Hierarchy(environment: environment, truncation: .maximumTier(4))
        let rows = allOccupations(a)
        #expect(rows == allOccupations(b))
        #expect(Array(rows.prefix(5)) == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                        [0, 1, 0, 0], [1, 0, 0, 0]])
        for h in 1..<a.count { #expect(a.tier(at: h - 1) <= a.tier(at: h)) }
        #expect(a.count == 70)
    }

    @Test("Cached edge weights implement the factorial similarity transform")
    func factorialScaling() {
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(2)),
                                       truncation: .maximumTier(5))
        let rows = (0..<hierarchy.count).map { hierarchyRow(hierarchy, $0) }
        let normalization = rows.map { row in
            row.n.reduce(1.0) { product, n in
                product * (n == 0 ? 1 : (1...n).reduce(1.0) { $0 * Double($1) })
            }.squareRoot()
        }
        for h in rows.indices {
            for p in 0..<hierarchy.multiIndexCount {
                let parent = rows[h].parents[p]
                let child = rows[h].children[p]
                if parent >= 0 {
                    let transformed = Double(rows[h].n[p]) * normalization[parent] / normalization[h]
                    #expect(abs(transformed - rows[h].down[p]) < 1e-14)
                }
                if child >= 0 {
                    let transformed = normalization[child] / normalization[h]
                    #expect(abs(transformed - rows[h].up[p]) < 1e-14)
                }
            }
        }
    }

    @Test("Immutable hierarchy can be read concurrently")
    func concurrentReaders() async {
        let hierarchy = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(3)),
                                       truncation: .maximumTier(3))
        let expected = allOccupations(hierarchy)
        await withTaskGroup(of: [[Int]].self) { group in
            for _ in 0..<8 { group.addTask { allOccupations(hierarchy) } }
            for await result in group { #expect(result == expected) }
        }
    }

    @Test("Invalid tiers and unrepresentable hierarchy sizes fail before enumeration")
    func invalidSizes() async {
        await #expect(processExitsWith: .failure) {
            _ = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(0)), truncation: .maximumTier(-1))
        }
        await #expect(processExitsWith: .failure) {
            _ = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(0)), truncation: .maximumTier(Int.max))
        }
        await #expect(processExitsWith: .failure) {
            _ = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(3)), truncation: .maximumTier(1_000_000))
        }
    }

    @Test("Custom predicates must retain the root and every required parent")
    func invalidCustomTruncation() async {
        await #expect(processExitsWith: .failure) {
            _ = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(2)), truncation: .custom { _ in false })
        }
        await #expect(processExitsWith: .failure) {
            _ = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(2)), truncation: .custom { n in
                (n[0] == 0 && n[1] <= 1) || (n[0] == 1 && n[1] == 1)
            })
        }
    }

    @Test("Environment validates physical channel count")
    func invalidEnvironment() async {
        await #expect(processExitsWith: .failure) {
            // Two poles, but THREE physical channels; two operators is invalid.
            _ = HOPS.Environment(couplingOperators: [hierarchyCoupling, hierarchyCoupling], bath: noiseTestModel(1))
        }
    }

    @Test("Accessors reject invalid IDs even for empty direction rows")
    func invalidIndices() async {
        await #expect(processExitsWith: .failure) {
            let h = HOPS.Hierarchy(environment: hierarchyEnvironment(.zero(channelCount: 1)), truncation: .maximumTier(0))
            h.multiIndex(of: 1) { _ in }
        }
        await #expect(processExitsWith: .failure) {
            let h = HOPS.Hierarchy(environment: hierarchyEnvironment(noiseTestModel(0)), truncation: .maximumTier(0))
            _ = h.damping(at: -1)
        }
    }
}

private var hierarchyCoupling: TimeDependentOperator { .constant(Matrix<Complex<Double>>.identity(rows: 2)) }

private func hierarchyEnvironment(_ model: CorrelatedBathModel) -> HOPS.Environment {
    .init(couplingOperators: Array(repeating: hierarchyCoupling, count: model.channelCount), bath: model)
}

private struct HierarchyRow: Equatable {
    var n: [Int] = []
    var parents: [Int] = []
    var children: [Int] = []
    var down: [Double] = []
    var up: [Double] = []
}

private func hierarchyRow(_ hierarchy: HOPS.Hierarchy, _ h: Int) -> HierarchyRow {
    var row = HierarchyRow()
    hierarchy.multiIndex(of: h) { s in for i in 0..<s.count { row.n.append(s[i]) } }
    hierarchy.parentIndices(of: h) { s in for i in 0..<s.count { row.parents.append(s[i]) } }
    hierarchy.childIndices(of: h) { s in for i in 0..<s.count { row.children.append(s[i]) } }
    hierarchy.parentWeights(of: h) { s in for i in 0..<s.count { row.down.append(s[i]) } }
    hierarchy.childWeights(of: h) { s in for i in 0..<s.count { row.up.append(s[i]) } }
    return row
}

private func allOccupations(_ hierarchy: HOPS.Hierarchy) -> [[Int]] {
    (0..<hierarchy.count).map { hierarchyRow(hierarchy, $0).n }
}

/// Independent Cartesian-product oracle; intentionally not the BFS constructor.
private func cartesianIndices(directions: Int, limit: Int, accepts: ([Int]) -> Bool) -> [[Int]] {
    var product: [[Int]] = [[]]
    for _ in 0..<directions { product = product.flatMap { prefix in (0...limit).map { prefix + [$0] } } }
    return product.filter(accepts)
}

private func verifyHierarchy(_ hierarchy: HOPS.Hierarchy) {
    let rows = (0..<hierarchy.count).map { hierarchyRow(hierarchy, $0) }
    let ids = Dictionary(uniqueKeysWithValues: rows.enumerated().map { ($0.element.n, $0.offset) })
    let poles = hierarchy.environment.bath.latentBaths.flatMap(\.poles)
    #expect(rows[0].n == Array(repeating: 0, count: poles.count))
    for h in rows.indices {
        let row = rows[h]
        #expect(hierarchy.tier(at: h) == row.n.reduce(0, +))
        var expectedDamping = Complex<Double>.zero
        for p in poles.indices {
            expectedDamping -= Double(row.n[p]) * poles[p]
            var parent = row.n
            parent[p] -= 1
            var child = row.n
            child[p] += 1
            #expect(row.parents[p] == (ids[parent] ?? -1))
            #expect(row.children[p] == (ids[child] ?? -1))
            #expect(row.down[p] == Double(row.n[p]).squareRoot())
            #expect(row.up[p] == Double(row.n[p] + 1).squareRoot())
            if row.parents[p] >= 0 {
                #expect(rows[row.parents[p]].children[p] == h)
                #expect(row.parents[p] < h)
            }
            if row.children[p] >= 0 {
                #expect(rows[row.children[p]].parents[p] == h)
                #expect(row.children[p] > h)
            }
        }
        #expect((hierarchy.damping(at: h) - expectedDamping).length < 1e-13)
    }
}
