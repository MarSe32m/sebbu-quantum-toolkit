// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

public struct DefaultGKSLImplementation: Sendable {
	@inlinable
	public init() {}
}

extension DefaultGKSLImplementation: GKSL.Implementation {
	@inlinable
	public func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		_ = configuration

		let start = propagation.timeSpan.start
		let end = propagation.timeSpan.end
		var outputCursor = OutputCursor(
			timeSpan: propagation.timeSpan,
			schedule: propagation.output
		)
		var state = DensityMatrixState(problem.initialState)

		if let initialTime = outputCursor.takeInitialTime() {
			forEach(initialTime, state.densityMatrix)
		}
		guard start < end else { return }

		let rhs = GKSLRightHandSide(problem)
		var solver = UniqueDOPRISolver(
			t: start,
			dt: Swift.min(
				propagation.integration.maximumStepSize,
				end - start
			),
			maxStep: propagation.integration.maximumStepSize,
			rhs: rhs,
			y4: .init(dimension: problem.system.dimension),
			k1: .init(dimension: problem.system.dimension),
			k2: .init(dimension: problem.system.dimension),
			k3: .init(dimension: problem.system.dimension),
			k4: .init(dimension: problem.system.dimension),
			k5: .init(dimension: problem.system.dimension),
			k6: .init(dimension: problem.system.dimension),
			k7: .init(dimension: problem.system.dimension),
			temporary: .init(dimension: problem.system.dimension),
			absoluteTolerance: propagation.integration.absoluteTolerance,
			relativeTolerance: propagation.integration.relativeTolerance,
			minimumStep: propagation.integration.minimumStepSize
		)
		var interpolatedState = DensityMatrixState(dimension: problem.system.dimension)

		while solver.t < end {
			let step = try solver.step(y: &state, upTo: end)
			while let outputTime = outputCursor.nextTime(through: step.endTime) {
				if outputTime == step.endTime {
					forEach(outputTime, state.densityMatrix)
				} else {
					solver.interpolateLastStep(
						at: outputTime, into: &interpolatedState)
					forEach(outputTime, interpolatedState.densityMatrix)
				}
			}
		}
	}
}

extension GKSL {
	@inlinable
	@inline(always)
	public static func solve<Hamiltonian>(
		problem: borrowing DensityMatrixProblem<Hamiltonian>,
		configuration: Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (Double, borrowing UniqueMatrix<Complex<Double>>) -> Void
	) throws where Hamiltonian: HamiltonianFunction {
		let implementation = DefaultGKSLImplementation()
		try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}

	@inlinable
	@inline(always)
	public static func solve<Hamiltonian: HamiltonianFunction>(
		problem: borrowing PureStateProblem<Hamiltonian>,
		configuration: GKSL.Configuration = .init(),
		propagation: PropagationOptions<IntegrationOptions>,
		_ forEach: (
			Double,
			borrowing UniqueMatrix<Complex<Double>>
		) -> Void
	) throws {
		let implementation = DefaultGKSLImplementation()
		try implementation.solve(
			problem: problem, configuration: configuration, propagation: propagation,
			forEach)
	}
}
