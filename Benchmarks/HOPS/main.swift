// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Numerics
import SebbuBLAS
import SebbuQuantumToolkit
import SebbuScience

// A fixed physical OU bath avoids including fitting or plotting in the timings.
// swift run -c release HOPSBenchmark --workers 1,8,32 --trajectories 256
let arguments = Array(CommandLine.arguments.dropFirst())
func option(_ name: String, _ fallback: String) -> String {
	guard let index = arguments.firstIndex(of: name) else { return fallback }
	precondition(index + 1 < arguments.count, "Missing value for \(name)")
	return arguments[index + 1]
}
let dimension = Int(option("--dimension", "2"))!
let trajectories = Int(option("--trajectories", "256"))!
let tier = Int(option("--tier", "4"))!
let poleCount = Int(option("--poles", "3"))!
let workers = option("--workers", "1,8").split(separator: ",").map { Int($0)! }
let warmup = Double(option("--warmup", "5"))!
let delay = Double(option("--delay", "5"))!
let step = Double(option("--step", "0.01"))!
let samples = Int(option("--samples", "251"))!
let repeats = Int(option("--repeats", "1"))!
let mixed = arguments.contains("--mixed")
precondition(dimension >= 2 && trajectories > 0 && tier >= 0 && (0...3).contains(poleCount))
precondition(warmup > 0 && delay > 0 && step > 0 && samples >= 2 && repeats > 0)
precondition(workers.allSatisfy { $0 > 0 })

var h = Matrix<Complex<Double>>.zeros(rows: dimension, columns: dimension)
var l = h
var coupling = h
for i in 1..<dimension {
	h[i - 1, i] = Complex(0.5)
	h[i, i - 1] = Complex(0.5)
	l[i - 1, i] = Complex(Double(i).squareRoot())
	coupling[i, i] = .one
}
let poles = [Complex(0.7, 0.9), Complex(1.2, 1.7), Complex(0.9, -0.4)]
let residues = [Complex(0.3, -0.1), Complex(0.2, 0.12), Complex(0.13, -0.2)]
let bath: CorrelatedBathModel =
	poleCount == 0
	? .zero(channelCount: 1)
	: .init(
		channelCount: 1,
		latentBaths: [
			.init(
				poles: Array(poles.prefix(poleCount)),
				residues: Matrix(
					elements: Array(residues.prefix(poleCount)), rows: 1,
					columns: poleCount))
		])
let hierarchy = HOPS.Hierarchy(
	environment: .init(couplingOperator: .constant(coupling), bath: bath),
	truncation: .maximumTier(tier))
let configuration = HOPS.Configuration(
	hierarchy: hierarchy, equationType: .nonLinearNormalized,
	shiftType: .meanField, noiseStepSize: step)
var initial = [Complex<Double>](repeating: .zero, count: dimension)
initial[0] = Complex(0.5.squareRoot())
initial[1] = initial[0]
let problem = PureStateProblem(
	initialState: Vector(initial), system: QuantumSystem(h),
	markovianChannels: [.init(rate: 0.075, collapseOperator: l)])
let integration = IntegrationOptions(
	minimumStepSize: 0, maximumStepSize: step,
	absoluteTolerance: 1e-9, relativeTolerance: 1e-9)
let times = (0..<samples).map { delay * Double($0) / Double(samples - 1) }
var insertions = [
	TimedCorrelationInsertion(time: 0, insertion: .right(.constant(l.conjugateTranspose)))
]
if mixed { insertions.append(.init(time: delay / 2, insertion: .left(.constant(l)))) }
let request = MultiTimeOrderedCorrelationRequest(insertions: insertions, observable: .constant(l))

func seconds(_ duration: Duration) -> Double {
	Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
}
print(
	"# dimension=\(dimension), poles=\(poleCount), hierarchy=\(hierarchy.count), trajectories=\(trajectories), tier=\(tier), warmup=\(warmup), delay=\(delay), step=\(step), samples=\(samples), mixed=\(mixed)"
)
print("phase,workers,repeat,seconds,checksum_real,checksum_imaginary")
for workerCount in workers {
	let execution = TrajectoryExecution(
		trajectories: trajectories, seed: 0xC0FFEE,
		parallelism: .maximumConcurrentTasks(workerCount))
	for repeatIndex in 0..<repeats {
		var checksum = Complex<Double>.zero
		let steadyDuration = try ContinuousClock().measure {
			try HOPS.solveEnsemble(
				problem: problem, configuration: configuration,
				propagation: .init(
					timeSpan: .init(start: 0, end: warmup), output: .final,
					integration: integration),
				execution: execution
			) { _, rho in
				for i in 0..<dimension {
					for j in 0..<dimension {
						checksum +=
							Double(1 + i * dimension + j) * rho[i, j]
					}
				}
			}
		}
		print(
			"steady,\(workerCount),\(repeatIndex),\(seconds(steadyDuration)),\(checksum.real),\(checksum.imaginary)"
		)
		checksum = .zero
		let correlationDuration = try ContinuousClock().measure {
			try HOPS.solveMultiTimeOrderedCorrelation(
				problem: problem, configuration: configuration,
				request: request,
				propagation: .init(
					timeSpan: .init(start: -warmup, end: delay),
					output: .times(times), integration: integration),
				execution: execution
			) { t, value in
				checksum += (1 + t) * value
				return .proceed
			}
		}
		print(
			"\(mixed ? "multi_time" : "two_time"),\(workerCount),\(repeatIndex),\(seconds(correlationDuration)),\(checksum.real),\(checksum.imaginary)"
		)
	}
}
