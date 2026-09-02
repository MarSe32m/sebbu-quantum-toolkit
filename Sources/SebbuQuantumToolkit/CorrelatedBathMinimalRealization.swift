// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Numerics
import SebbuScience

extension CorrelatedBathModel {
	/// Rank-revealing analysis of the one-sided matrix-exponential BCF
	/// realization.
	///
	/// After poles closer than the requested tolerance have been grouped, the
	/// positive-lag BCF is written as
	///
	/// ```text
	/// alpha(tau) = sum_g A_g exp(-W_g tau).
	/// ```
	///
	/// For simple exponential poles, a minimal matrix realization contains
	/// `rank(A_g)` states at `W_g`. Consequently, `minimalPoleCount` is the
	/// numerical McMillan degree of this grouped realization. When
	/// `poleTolerance` is nonzero this is a tolerance-controlled reduced order,
	/// rather than an exact algebraic invariant.
	public struct MinimalRealizationAnalysis: Sendable {
		public struct PoleGroup: Sendable {
			/// Norm-weighted representative pole of the group.
			public let pole: Complex<Double>

			/// Sum of the one-sided residue matrices in the group.
			public let residue: Matrix<Complex<Double>>

			/// Number of latent pole states contributing to this group.
			public let originalPoleCount: Int

			/// Numerical rank of `residue`.
			public let numericalRank: Int

			/// Singular values of `residue`, in descending order.
			public let singularValues: [Double]

			public init(
				pole: Complex<Double>,
				residue: Matrix<Complex<Double>>,
				originalPoleCount: Int,
				numericalRank: Int,
				singularValues: [Double]
			) {
				self.pole = pole
				self.residue = residue
				self.originalPoleCount = originalPoleCount
				self.numericalRank = numericalRank
				self.singularValues = singularValues
			}
		}

		/// Number of states in the supplied latent realization.
		public let originalPoleCount: Int

		/// Pole groups, including groups whose residue is numerically zero.
		public let poleGroups: [PoleGroup]

		/// Sum of the numerical ranks of all grouped residue matrices.
		public var minimalPoleCount: Int {
			poleGroups.reduce(0) { $0 + $1.numericalRank }
		}

		/// Number of pole groups with a nonzero numerical residue.
		public var observablePoleGroupCount: Int {
			poleGroups.count { $0.numericalRank > 0 }
		}

		public init(
			originalPoleCount: Int,
			poleGroups: [PoleGroup]
		) {
			self.originalPoleCount = originalPoleCount
			self.poleGroups = poleGroups
		}
	}

	/// Analyzes the numerical minimal order of the positive-lag BCF.
	///
	/// - Parameters:
	///   - poleTolerance: Relative pole distance below which poles are treated
	///     as one location. Use zero to merge only exactly equal poles.
	///   - rankTolerance: Singular values no larger than this fraction of the
	///     largest singular value in a pole group are treated as zero. A
	///     dimension-scaled machine-precision floor is always applied.
	/// - Returns: Per-pole residue ranks and their summed minimal order.
	public func analyzeMinimalRealization(
		poleTolerance: Double = 0,
		rankTolerance: Double = 1e-10
	) throws -> MinimalRealizationAnalysis {
		precondition(
			poleTolerance.isFinite && poleTolerance >= 0,
			"The pole tolerance must be finite and nonnegative"
		)
		precondition(
			rankTolerance.isFinite && rankTolerance >= 0,
			"The rank tolerance must be finite and nonnegative"
		)

		struct AccumulatedPoleGroup {
			var pole: Complex<Double>
			var residue: Matrix<Complex<Double>>
			var poleWeight: Double
			var originalPoleCount: Int
		}

		let terms = oneSidedExponentialTerms.sorted {
			let lhsNorm = correlatedBathFrobeniusNorm($0.residue)
			let rhsNorm = correlatedBathFrobeniusNorm($1.residue)
			if lhsNorm != rhsNorm { return lhsNorm > rhsNorm }
			if $0.pole.real != $1.pole.real {
				return $0.pole.real < $1.pole.real
			}
			return $0.pole.imaginary < $1.pole.imaginary
		}

		var accumulated: [AccumulatedPoleGroup] = []
		for term in terms {
			let weight = correlatedBathFrobeniusNorm(term.residue)
			let matchingIndex = accumulated.indices
				.filter {
					correlatedBathPoleDistance(
						accumulated[$0].pole,
						term.pole
					) <= poleTolerance
				}
				.min {
					correlatedBathPoleDistance(accumulated[$0].pole, term.pole)
						< correlatedBathPoleDistance(
							accumulated[$1].pole,
							term.pole
						)
				}

			if let matchingIndex {
				let oldWeight = accumulated[matchingIndex].poleWeight
				let totalWeight = oldWeight + weight
				if totalWeight > 0 {
					accumulated[matchingIndex].pole =
						(oldWeight * accumulated[matchingIndex].pole
							+ weight * term.pole) / totalWeight
				}
				for element in term.residue.elements.indices {
					accumulated[matchingIndex].residue.elements[element] +=
						term.residue.elements[element]
				}
				accumulated[matchingIndex].poleWeight = totalWeight
				accumulated[matchingIndex].originalPoleCount += 1
			} else {
				accumulated.append(
					AccumulatedPoleGroup(
						pole: term.pole,
						residue: term.residue,
						poleWeight: weight,
						originalPoleCount: 1
					)
				)
			}
		}

		var decomposedGroups: [(group: AccumulatedPoleGroup, singularValues: [Double])] = []
		decomposedGroups.reserveCapacity(accumulated.count)
		var globalSingularValueScale = 0.0
		for group in accumulated {
			let decomposition = try MatrixOperations.singularValueDecomposition(
				A: group.residue
			)
			let singularValues = decomposition.singularValues
			globalSingularValueScale = max(
				globalSingularValueScale,
				singularValues.first ?? 0
			)
			decomposedGroups.append((group, singularValues))
		}

		var groups: [MinimalRealizationAnalysis.PoleGroup] = []
		groups.reserveCapacity(decomposedGroups.count)
		for decomposed in decomposedGroups {
			let group = decomposed.group
			let singularValues = decomposed.singularValues
			let numericalRank = correlatedBathNumericalRank(
				singularValues,
				rows: group.residue.rows,
				columns: group.residue.columns,
				relativeTolerance: rankTolerance,
				globalScale: globalSingularValueScale
			)
			groups.append(
				MinimalRealizationAnalysis.PoleGroup(
					pole: group.pole,
					residue: group.residue,
					originalPoleCount: group.originalPoleCount,
					numericalRank: numericalRank,
					singularValues: singularValues
				)
			)
		}
		groups.sort {
			if $0.pole.real != $1.pole.real {
				return $0.pole.real < $1.pole.real
			}
			return $0.pole.imaginary < $1.pole.imaginary
		}
		return MinimalRealizationAnalysis(
			originalPoleCount: poleCount,
			poleGroups: groups
		)
	}
}

private func correlatedBathNumericalRank(
	_ singularValues: [Double],
	rows: Int,
	columns: Int,
	relativeTolerance: Double,
	globalScale: Double
) -> Int {
	guard let largest = singularValues.first, largest > 0 else { return 0 }
	let threshold = max(
		relativeTolerance * largest,
		100 * Double.ulpOfOne * Double(max(rows, columns)) * globalScale
	)
	return singularValues.count { $0 > threshold }
}

private func correlatedBathFrobeniusNorm(
	_ matrix: Matrix<Complex<Double>>
) -> Double {
	matrix.elements.reduce(0) {
		$0 + $1.real * $1.real + $1.imaginary * $1.imaginary
	}.squareRoot()
}

private func correlatedBathPoleDistance(
	_ lhs: Complex<Double>,
	_ rhs: Complex<Double>
) -> Double {
	(lhs - rhs).length / max(1, max(lhs.length, rhs.length))
}
