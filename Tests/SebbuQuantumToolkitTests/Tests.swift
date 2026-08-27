import Numerics
import SebbuQuantumToolkit
import SebbuScience
import Testing

@Test("GKSL reproduces analytic amplitude damping")
func gkslAmplitudeDamping() throws {
	let decayRate = 0.4
	let outputTimes = [0.0, 0.1, 0.5, 1.0, 2.0]
	let problem = amplitudeDampingProblem(
		rate: .constant(decayRate),
		collapseOperator: loweringOperator
	)

	var results: [(time: Double, ground: Double, excited: Double, trace: Double)] = []
	try GKSL.solve(
		problem: problem,
		propagation: propagationOptions(
			end: outputTimes.last!,
			output: .times(outputTimes)
		)
	) { time, densityMatrix in
		results.append(
			(
				time,
				densityMatrix[0, 0].real,
				densityMatrix[1, 1].real,
				densityMatrix[0, 0].real + densityMatrix[1, 1].real
			)
		)
        return .proceed
	}

	#expect(results.count == outputTimes.count)
	for result in results {
		let expectedExcited = Double.exp(-decayRate * result.time)
		#expect(abs(result.excited - expectedExcited) < 2e-8)
		#expect(abs(result.ground - (1 - expectedExcited)) < 2e-8)
		#expect(abs(result.trace - 1) < 2e-10)
	}
}

@Test("Generated collapse operators use the dynamic GKSL path")
func gkslGeneratedCollapseOperator() throws {
	let decayRate = 0.25
	let end = 1.5
	let generatedOperator = TimeDependentOperator.generatedDense(
		DynamicDenseOperator { _, output in
			output[0, 0] = .zero
			output[0, 1] = .one
			output[1, 0] = .zero
			output[1, 1] = .zero
		}
	)
	let problem = amplitudeDampingProblem(
		rate: .generated { _ in decayRate },
		collapseOperator: generatedOperator
	)

	var callbackCount = 0
	var finalExcitedPopulation = Double.nan
	try GKSL.solve(
		problem: problem,
        propagation: propagationOptions(end: end, output: .final),
    ) { time, densityMatrix in
        callbackCount += 1
        #expect(time == end)
        finalExcitedPopulation = densityMatrix[1, 1].real
        return .proceed
    }

	#expect(callbackCount == 1)
	#expect(
		abs(finalExcitedPopulation - .exp(-decayRate * end)) < 2e-8
	)
}

@Test("The pure-state GKSL overload constructs the initial projector")
func gkslPureStateOverload() throws {
	let decayRate = 0.3
	let end = 1.0
	let system = zeroTwoLevelSystem
	let problem = PureStateProblem(
		initialState: Vector<Complex<Double>>([.zero, .one]),
		system: system,
		markovianChannels: [
			MarkovianChannel(
				rate: .constant(decayRate),
				collapseOperator: loweringOperator
			)
		]
	)

	var finalExcitedPopulation = Double.nan
	try GKSL.solve(
		problem: problem,
		propagation: propagationOptions(end: end, output: .final)
	) { _, densityMatrix in
		finalExcitedPopulation = densityMatrix[1, 1].real
        return .proceed
	}

	#expect(
		abs(finalExcitedPopulation - .exp(-decayRate * end)) < 2e-8
	)
}

@Test("Every accepted GKSL step is reported exactly once")
func gkslEveryAcceptedStepSchedule() throws {
	let problem = DensityMatrixProblem(
		initialState: excitedStateDensityMatrix,
		system: zeroTwoLevelSystem
	)
	var outputTimes: [Double] = []

	try GKSL.solve(
		problem: problem,
		propagation: propagationOptions(
			end: 1,
			output: .everyAcceptedStep,
			maximumStep: 0.2
		)
	) { time, _ in
		outputTimes.append(time)
        return .proceed
	}

	#expect(!outputTimes.isEmpty)
	#expect(outputTimes.count <= 6)
	#expect(abs((outputTimes.last ?? .nan) - 1) < 1e-14)
	for index in outputTimes.indices.dropFirst() {
		#expect(outputTimes[index] > outputTimes[index - 1])
	}
}

@Test("Unimplemented trajectory façades retain their API surface")
func testTrajectoryAPISurface() throws {
	let hamiltonian = ConstantHamiltonian(
		Matrix<Complex<Double>>(
			elements: [.zero, .one, .one, .one],
			rows: 2,
			columns: 2
		)
	)
	let system = QuantumSystem(dimension: 2, hamiltonian: hamiltonian)
	let problem = PureStateProblem(
		initialState: Vector<Complex<Double>>([.zero, .one]),
		system: system,
		markovianChannels: [
			MarkovianChannel(
				rate: .constant(0.1),
				collapseOperator: loweringOperator
			)
		]
	)
	let options = propagationOptions(end: 1, output: .final)

	let mcwfConfiguration = MCWF.Configuration(
		jumpAlgorithm: .waitingTime(
			eventTolerance: 1e-10,
			maximumEventIterations: 64
		)
	)
	#expect(throws: ImplementationError.notImplemented) {
		try MCWF.solve(
			problem: problem,
			configuration: mcwfConfiguration,
			propagation: options,
			seed: 0,
			trajectoryID: 0
		) { _, _ in }
	}

	let qsdConfiguration = QSD.Configuration(equationType: .nonLinearNormalized)
	#expect(throws: ImplementationError.notImplemented) {
		try QSD.solve(
			problem: problem,
			configuration: qsdConfiguration,
			propagation: options,
			seed: 0,
			trajectoryID: 0
		) { _, _ in }
	}
}

private var zeroTwoLevelSystem: QuantumSystem<ConstantHamiltonian> {
	QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private var loweringOperator: TimeDependentOperator {
	.constant(
		ConstantOperator(
			Matrix<Complex<Double>>(
				elements: [.zero, .one, .zero, .zero],
				rows: 2,
				columns: 2
			)
		)
	)
}

private var excitedStateDensityMatrix: Matrix<Complex<Double>> {
	Matrix(
		elements: [.zero, .zero, .zero, .one],
		rows: 2,
		columns: 2
	)
}

private func amplitudeDampingProblem(
	rate: ScalarTimeFunction,
	collapseOperator: TimeDependentOperator
) -> DensityMatrixProblem<ConstantHamiltonian> {
	DensityMatrixProblem(
		initialState: excitedStateDensityMatrix,
		system: zeroTwoLevelSystem,
		markovianChannels: [
			MarkovianChannel(
				rate: rate,
				collapseOperator: collapseOperator
			)
		]
	)
}

private func propagationOptions(
	end: Double,
	output: OutputSchedule,
	maximumStep: Double = 0.25
) -> PropagationOptions<IntegrationOptions> {
	PropagationOptions(
		timeSpan: SimulationTimeSpan(start: 0, end: end),
		output: output,
		integration: IntegrationOptions(
			minimumStepSize: 0,
			maximumStepSize: maximumStep,
			absoluteTolerance: 1e-11,
			relativeTolerance: 1e-11
		)
	)
}
