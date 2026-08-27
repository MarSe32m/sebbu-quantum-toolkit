import Numerics
import SebbuQuantumToolkit
import SebbuScience
import Testing

@Test("GKSL QRT reproduces the analytic amplitude-damping correlation")
func gkslTwoTimeAmplitudeDamping() throws {
	let decayRate = 0.4
	let insertionTime = 0.4
	let requestedTimes = [0.0, 0.2, insertionTime, 0.8, 1.4]
	let expectedOutputTimes = Array(requestedTimes.dropFirst(2))
	let problem = correlationAmplitudeDampingProblem(rate: decayRate)

	var results: [(time: Double, value: Complex<Double>)] = []
	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: correlationPropagation(
			start: 0,
			end: requestedTimes.last!,
			output: .times(requestedTimes)
		)
	) { time, value in
		results.append((time, value))
	}

	// Requested times before the insertion are not valid later times and are
	// therefore skipped.
	#expect(results.map { $0.time } == expectedOutputTimes)
	for result in results {
		// <sigma_+(t) sigma_-(s)> = exp[-gamma (t + s) / 2]
		// for an initially excited, undriven two-level system.
		let expected = Double.exp(
			-decayRate * (result.time + insertionTime) / 2
		)
		#expect(abs(result.value.real - expected) < 2e-8)
		#expect(abs(result.value.imaginary) < 2e-8)
	}
}

@Test("GKSL QRT preserves the complex unitary phase")
func gkslTwoTimeUnitaryPhase() throws {
	let frequency = 1.7
	let insertionTime = 0.3
	let outputTimes = [insertionTime, 0.55, 1.0]
	let system = QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, Complex(frequency)],
			rows: 2,
			columns: 2
		)
	)
	let problem = DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: system
	)

	var results: [(time: Double, value: Complex<Double>)] = []
	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: correlationPropagation(
			start: 0,
			end: outputTimes.last!,
			output: .times(outputTimes)
		)
	) { time, value in
		results.append((time, value))
	}

	#expect(results.count == outputTimes.count)
	for result in results {
		let phase = frequency * (result.time - insertionTime)
		let expected = Complex<Double>(Double.cos(phase), Double.sin(phase))
		#expect((result.value - expected).length < 2e-8)
	}
}

@Test("Left and right QRT insertions preserve operator ordering")
func gkslTwoTimeInsertionOrdering() throws {
	let insertionTime = 0.25
	let outputTimes = [insertionTime, 0.5]
	let problem = DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: correlationZeroTwoLevelSystem
	)
	let propagation = correlationPropagation(
		start: 0,
		end: outputTimes.last!,
		output: .times(outputTimes)
	)

	var leftValues: [Complex<Double>] = []
	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: propagation
	) { _, value in
		leftValues.append(value)
	}

	var rightValues: [Complex<Double>] = []
	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .right(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: propagation
	) { _, value in
		rightValues.append(value)
	}

	#expect(leftValues.count == outputTimes.count)
	#expect(rightValues.count == outputTimes.count)
	for value in leftValues {
		#expect((value - .one).length < 1e-12)
	}
	for value in rightValues {
		#expect(value.length < 1e-12)
	}
}

@Test("Time-dependent QRT operators use absolute times")
func gkslTwoTimeAbsoluteOperatorTimesAndPureStateOverload() throws {
	let insertionTime = 0.25
	let scaledLowering = TimeDependentOperator.linearCombination(
		OperatorExpansion(
			coefficients: [.generated { Complex($0) }],
			operators: [correlationLoweringConstantOperator]
		)
	)
	let scaledRaising = TimeDependentOperator.linearCombination(
		OperatorExpansion(
			coefficients: [.generated { Complex(1 + $0) }],
			operators: [correlationRaisingConstantOperator]
		)
	)
	let problem = PureStateProblem(
		initialState: Vector<Complex<Double>>([.zero, .one]),
		system: correlationZeroTwoLevelSystem
	)

	var results: [(time: Double, value: Complex<Double>)] = []
	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(scaledLowering),
			observable: scaledRaising
		),
		propagation: correlationPropagation(
			start: 0,
			end: 1,
			output: .uniform(step: 0.2)
		)
	) { time, value in
		results.append((time, value))
	}

	let expectedTimes = [0.4, 0.6, 0.8, 1.0]
	#expect(results.count == expectedTimes.count)
	for (result, expectedTime) in zip(results, expectedTimes) {
		#expect(abs(result.time - expectedTime) < 1e-14)
		let expected = insertionTime * (1 + result.time)
		#expect(abs(result.value.real - expected) < 1e-11)
		#expect(abs(result.value.imaginary) < 1e-11)
	}
}

@Test("Final QRT output is emitted when the insertion is at the end")
func gkslTwoTimeInsertionAtEnd() throws {
	let decayRate = 0.35
	let end = 1.0
	let problem = correlationAmplitudeDampingProblem(rate: decayRate)
	var results: [(time: Double, value: Complex<Double>)] = []

	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: end,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: correlationPropagation(
			start: 0,
			end: end,
			output: .final
		)
	) { time, value in
		results.append((time, value))
	}

	#expect(results.count == 1)
	#expect(results.first?.time == end)
	#expect(
		abs((results.first?.value.real ?? .nan) - .exp(-decayRate * end))
			< 2e-8
	)
}

@Test("Zero-duration GKSL QRT returns the equal-time correlation")
func gkslTwoTimeZeroDuration() throws {
	let time = 2.0
	let problem = DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: correlationZeroTwoLevelSystem
	)
	var results: [(time: Double, value: Complex<Double>)] = []

	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: time,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: correlationPropagation(
			start: time,
			end: time,
			output: .final
		)
	) { outputTime, value in
		results.append((outputTime, value))
	}

	#expect(results.count == 1)
	#expect(results.first?.time == time)
	#expect(((results.first?.value ?? .zero) - .one).length < 1e-12)
}

@Test("Every accepted QRT step occurs after the insertion")
func gkslTwoTimeEveryAcceptedStep() throws {
	let insertionTime = 0.4
	let end = 1.0
	let problem = DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: correlationZeroTwoLevelSystem
	)
	var outputTimes: [Double] = []

	try GKSL.solveTwoTimeCorrelation(
		problem: problem,
		request: TwoTimeCorrelationRequest(
			insertionTime: insertionTime,
			insertion: .left(correlationLoweringOperator),
			observable: correlationRaisingOperator
		),
		propagation: correlationPropagation(
			start: 0,
			end: end,
			output: .everyAcceptedStep,
			maximumStep: 0.2
		)
	) { time, value in
		outputTimes.append(time)
		#expect((value - .one).length < 1e-12)
	}

	#expect(!outputTimes.isEmpty)
	#expect(outputTimes.allSatisfy { $0 > insertionTime && $0 <= end })
	#expect(abs((outputTimes.last ?? .nan) - end) < 1e-14)
	for index in outputTimes.indices.dropFirst() {
		#expect(outputTimes[index] > outputTimes[index - 1])
	}
}

@Test("GKSL QRT validates insertion times and operator dimensions")
func gkslTwoTimeValidation() throws {
	let problem = DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: correlationZeroTwoLevelSystem
	)
	let propagation = correlationPropagation(
		start: 0,
		end: 1,
		output: .final
	)

	#expect(throws: TwoTimeCorrelationError.nonFiniteInsertionTime) {
		try GKSL.solveTwoTimeCorrelation(
			problem: problem,
			request: TwoTimeCorrelationRequest(
				insertionTime: .nan,
				insertion: .left(correlationLoweringOperator),
				observable: correlationRaisingOperator
			),
			propagation: propagation
		) { _, _ in }
	}

	#expect(
		throws: TwoTimeCorrelationError.insertionTimeOutsideTimeSpan(
			insertionTime: 1.1,
			start: 0,
			end: 1
		)
	) {
		try GKSL.solveTwoTimeCorrelation(
			problem: problem,
			request: TwoTimeCorrelationRequest(
				insertionTime: 1.1,
				insertion: .left(correlationLoweringOperator),
				observable: correlationRaisingOperator
			),
			propagation: propagation
		) { _, _ in }
	}

	let threeByThree = TimeDependentOperator.constant(
		ConstantOperator(
            Matrix.zeros(rows: 3, columns: 3)
		)
	)
	#expect(
		throws: TwoTimeCorrelationError.insertionOperatorDimensionMismatch(
			expected: 2,
			rows: 3,
			columns: 3
		)
	) {
		try GKSL.solveTwoTimeCorrelation(
			problem: problem,
			request: TwoTimeCorrelationRequest(
				insertionTime: 0.5,
				insertion: .left(threeByThree),
				observable: correlationRaisingOperator
			),
			propagation: propagation
		) { _, _ in }
	}

	#expect(
		throws: TwoTimeCorrelationError.observableDimensionMismatch(
			expected: 2,
			rows: 3,
			columns: 3
		)
	) {
		try GKSL.solveTwoTimeCorrelation(
			problem: problem,
			request: TwoTimeCorrelationRequest(
				insertionTime: 0.5,
				insertion: .left(correlationLoweringOperator),
				observable: threeByThree
			),
			propagation: propagation
		) { _, _ in }
	}
}

private var correlationZeroTwoLevelSystem: QuantumSystem<ConstantHamiltonian> {
	QuantumSystem(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private var correlationLoweringConstantOperator: ConstantOperator {
	ConstantOperator(
		Matrix<Complex<Double>>(
			elements: [.zero, .one, .zero, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private var correlationRaisingConstantOperator: ConstantOperator {
	ConstantOperator(
		Matrix<Complex<Double>>(
			elements: [.zero, .zero, .one, .zero],
			rows: 2,
			columns: 2
		)
	)
}

private var correlationLoweringOperator: TimeDependentOperator {
	.constant(correlationLoweringConstantOperator)
}

private var correlationRaisingOperator: TimeDependentOperator {
	.constant(correlationRaisingConstantOperator)
}

private var correlationExcitedStateDensityMatrix: Matrix<Complex<Double>> {
	Matrix(
		elements: [.zero, .zero, .zero, .one],
		rows: 2,
		columns: 2
	)
}

private func correlationAmplitudeDampingProblem(
	rate: Double
) -> DensityMatrixProblem<ConstantHamiltonian> {
	DensityMatrixProblem(
		initialState: correlationExcitedStateDensityMatrix,
		system: correlationZeroTwoLevelSystem,
		markovianChannels: [
			MarkovianChannel(
				rate: .constant(rate),
				collapseOperator: correlationLoweringOperator
			)
		]
	)
}

private func correlationPropagation(
	start: Double,
	end: Double,
	output: OutputSchedule,
	maximumStep: Double = 0.4
) -> PropagationOptions<IntegrationOptions> {
	PropagationOptions(
		timeSpan: SimulationTimeSpan(start: start, end: end),
		output: output,
		integration: IntegrationOptions(
			minimumStepSize: 0,
			maximumStepSize: maximumStep,
			absoluteTolerance: 1e-11,
			relativeTolerance: 1e-11
		)
	)
}
