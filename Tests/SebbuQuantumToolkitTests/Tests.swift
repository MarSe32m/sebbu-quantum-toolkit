import SebbuQuantumToolkit
import SebbuScience
import Testing

@Test
func example() throws {
	let hamiltonian = ConstantHamiltonian(
		Matrix.init(elements: [.zero, .one, .one, .one], rows: 2, columns: 2))
	let system = QuantumSystem(dimension: 2, hamiltonian: hamiltonian)
	let problem = PureStateProblem(
		initialState: .init([.zero, .one]),
		system: system,
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
	let hierarchy = HOPS.Hierarchy()
	let configuration = HOPS.Configuration(
		hierarchy: hierarchy,
		equationType: .nonLinear
	)
	let options = PropagationOptions(
		timeSpan: .init(start: 0, end: 10), output: .uniform(step: 0.01),
		integration: IntegrationOptions(
			minimumStepSize: 0.01, maximumStepSize: 0.1, absoluteTolerance: 1e-8,
			relativeTolerance: 1e-8))
	var rng = SystemRandomNumberGenerator()
	HOPS.solve(problem: problem, configuration: configuration, propagation: options, rng: &rng)
	{ t, state in

	}
}
