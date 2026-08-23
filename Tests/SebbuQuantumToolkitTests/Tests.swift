import Testing
import SebbuQuantumToolkit
import SebbuScience

@Test
func example() throws {
    let system = QuantumSystem(dimension: 2) { t, H in
        H.add(.identity(rows: 2))
    }
    let initialState = UniqueVector<Complex<Double>>([1, 0])
    let hierarchy = HOPS.Hierarchy(bathCorrelationFunctions: .init(elements: [], rows: 0, columns: 0), couplingOperators: []) { _ in
        false
    }
    let integrationOptions = IntegrationOptions(minimumStepSize: 1e-6, maximumStepSize: 1e-2, absoluteTolerance: 1e-8, relativeTolerance: 1e-8)
//    HOPS.solve(start: 0, end: 10, initialState: initialState, system: system, hierarchy: hierarchy, equationType: .linear, shiftType: .meanField, intergation: integrationOptions) { t, state in
//        print(t)
//    }
}
