import SebbuQuantumToolkit
import SebbuScience
import Testing

@Test
func testBasicAPISurface() throws {
	let hamiltonian1 = ConstantHamiltonian(
		Matrix.init(elements: [.zero, .one, .one, .one], rows: 2, columns: 2)
    )
    let hamiltonian2 = ConstantHamiltonian(
        Matrix.init(elements: [.zero, .one, .one, .one], rows: 2, columns: 2)
    )
	let system1 = QuantumSystem(dimension: 2, hamiltonian: hamiltonian1)
	let pureStateProblem = PureStateProblem(
		initialState: .init([.zero, .one]),
		system: system1,
		markovianChannels: [
			.init(
				rate: .constant(0.1),
				collapseOperator: .constant(
					.init(
						Matrix.init(
							elements: [.zero, .one, .zero, .zero],
							rows: 2, columns: 2)))
			)
		]
	)
	let system2 = QuantumSystem(dimension: 2, hamiltonian: hamiltonian2)
	let densityMatrixProblem = DensityMatrixProblem(
		initialState: .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2),
		system: system2,
		markovianChannels: [
			.init(
				rate: .constant(0.1),
				collapseOperator: .constant(
					.init(
						Matrix.init(
							elements: [.zero, .one, .zero, .zero],
							rows: 2, columns: 2)))
			)
		])

	let options = PropagationOptions(
		timeSpan: .init(start: 0, end: 10), output: .uniform(step: 0.01),
		integration: IntegrationOptions(
			minimumStepSize: 0.01, maximumStepSize: 0.1, absoluteTolerance: 1e-8,
			relativeTolerance: 1e-8))
	// GKSL
	let gkslConfiguration = GKSL.Configuration()
	GKSL.solve(
		problem: densityMatrixProblem, configuration: gkslConfiguration,
		propagation: options
	) { _, _ in }

	// HEOM
	let heomHierarchy = HEOM.Hierarchy()
	let heomConfiguration = HEOM.Configuration(hierarchy: heomHierarchy, shiftType: .meanField)
	HEOM.solve(
		problem: densityMatrixProblem, configuration: heomConfiguration,
		propagation: options
	) { _, _ in }

	// HOPS
	let hopsHierarchy = HOPS.Hierarchy()
	let hopsConfiguration = HOPS.Configuration(
		hierarchy: hopsHierarchy,
		equationType: .nonLinear
	)
	HOPS.solve(
		problem: pureStateProblem, configuration: hopsConfiguration, propagation: options,
		seed: 0, trajectoryID: 0
	) { _, _ in }

	// MCWF
	let mcwfConfiguration = MCWF.Configuration(
		jumpAlgorithm: .waitingTime(eventTolerance: 1e-10, maximumEventIterations: 64))
	MCWF.solve(
		problem: pureStateProblem, configuration: mcwfConfiguration, propagation: options,
		seed: 0, trajectoryID: 0
	) { _, _ in }

	// QSD
	let qsdConfiguration = QSD.Configuration(equationType: .nonLinearNormalized)
	QSD.solve(
		problem: pureStateProblem, configuration: qsdConfiguration, propagation: options,
		seed: 0, trajectoryID: 0
	) { _, _ in }
}
