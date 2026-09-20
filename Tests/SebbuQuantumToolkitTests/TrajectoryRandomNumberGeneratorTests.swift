// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience
import Testing

@testable import SebbuQuantumToolkit

@Suite("Trajectory RNG sampling")
struct TrajectoryRandomNumberGeneratorTests {
	private typealias RNG = TrajectoryRandomNumberGenerator

	@Test(
		"Independent streams retain the original Philox mapping and mixed draw sequence",
		arguments: [UInt64(0), 1, 19, UInt64.max])
	func independentCompatibility(id: UInt64) {
		for purpose: RNG.Purpose in [
			.unspecified, .gaussianWhiteNoise, .coloredNoiseGeneration, .mcwfJumps,
		] {
			for channel: UInt64 in [0, 7] {
				var legacy = Philox4x64(
					seed: 0xA11CE,
					counter: .init(0, channel, id, purpose.rawValue))
				var actual = RNG(
					seed: 0xA11CE, channel: channel, trajectoryID: id,
					purpose: purpose)
				var explicit = RNG(
					seed: 0xA11CE, channel: channel, trajectoryID: id,
					ensembleSampling: .independent, purpose: purpose)
				for _ in 0..<256 {
					let raw = legacy.next()
					#expect(actual.next() == raw && explicit.next() == raw)
					let normal: Complex<Double> = legacy.nextNormal()
					let a: Complex<Double> = actual.nextNormal()
					let b: Complex<Double> = explicit.nextNormal()
					#expect(a == normal && b == normal)
					let uniform = legacy.nextUnitDouble()
					#expect(
						actual.nextUnitDouble() == uniform
							&& explicit.nextUnitDouble() == uniform)
				}
			}
		}
	}

	@Test(
		"Paired mixed draws consume identical underlying words and reconstruct exactly",
		arguments: [UInt64(0), 1, 31, (UInt64.max - 1) / 2])
	func pairedSequences(pair: UInt64) {
		for purpose: RNG.Purpose in [
			.unspecified, .gaussianWhiteNoise, .coloredNoiseGeneration, .mcwfJumps,
		] {
			var a = RNG(
				seed: 91, channel: 3, trajectoryID: 2 * pair,
				ensembleSampling: .antithetic, purpose: purpose)
			var b = RNG(
				seed: 91, channel: 3, trajectoryID: 2 * pair + 1,
				ensembleSampling: .antithetic, purpose: purpose)
			var replayA = RNG(
				seed: 91, channel: 3, trajectoryID: 2 * pair,
				ensembleSampling: .antithetic, purpose: purpose)
			var replayB = RNG(
				seed: 91, channel: 3, trajectoryID: 2 * pair + 1,
				ensembleSampling: .antithetic, purpose: purpose)
			var ordinary = RNG(
				seed: 91, channel: 3, trajectoryID: pair, purpose: purpose)
			#expect(!a.isAntithetic && b.isAntithetic)
			for _ in 0..<2048 {
				let x: Double = a.nextNormal()
				let y: Double = b.nextNormal()
				let rx: Double = replayA.nextNormal()
				let ry: Double = replayB.nextNormal()
				let original: Double = ordinary.nextNormal()
				#expect(x == -y && x == rx && y == ry && x == original)
				let z: Complex<Double> = a.nextNormal(stdev: 0.7)
				let w: Complex<Double> = b.nextNormal(stdev: 0.7)
				let rz: Complex<Double> = replayA.nextNormal(stdev: 0.7)
				let rw: Complex<Double> = replayB.nextNormal(stdev: 0.7)
				let originalZ: Complex<Double> = ordinary.nextNormal(stdev: 0.7)
				#expect(z.real == -w.real && z.imaginary == -w.imaginary)
				#expect(z == rz && w == rw && z == originalZ)
				let u = a.nextUnitDouble()
				let v = b.nextUnitDouble()
				#expect(u + v == 1 - 0x1.0p-53)
				#expect(abs(v - (1 - u)) <= Double.ulpOfOne)
				#expect(u >= 0 && u < 1 && v >= 0 && v < 1)
				#expect(
					u == replayA.nextUnitDouble()
						&& v == replayB.nextUnitDouble())
				#expect(u == ordinary.nextUnitDouble())
				let raw = a.next()
				let antiRaw = b.next()
				#expect(raw == ~antiRaw)
				#expect(raw == replayA.next() && antiRaw == replayB.next())
				#expect(raw == ordinary.next())
			}
		}
	}

	@Test("Scalar and array Gaussian overloads retain means, scales and draw consumption")
	func gaussianOverloads() {
		var a = RNG(seed: 72, trajectoryID: 10, ensembleSampling: .antithetic)
		var b = RNG(seed: 72, trajectoryID: 11, ensembleSampling: .antithetic)
		let pairA = a.nextGaussian()
		let pairB = b.nextGaussian()
		#expect(pairA.0 == -pairB.0 && pairA.1 == -pairB.1)
		for count in [0, 1, 2, 3, 127] {
			let x = a.nextGaussian(count: count)
			let y = b.nextGaussian(count: count)
			#expect(x == y.map { -$0 })
			let doubles: [Double] = a.nextNormal(count: count, stdev: 1.7)
			let antiDoubles: [Double] = b.nextNormal(count: count, stdev: 1.7)
			#expect(doubles == antiDoubles.map { -$0 })
			let floats: [Float] = a.nextNormal(count: count, stdev: Float(0.4))
			let antiFloats: [Float] = b.nextNormal(count: count, stdev: Float(0.4))
			#expect(floats == antiFloats.map { -$0 })
			let complex: [Complex<Double>] = a.nextNormal(count: count)
			let antiComplex: [Complex<Double>] = b.nextNormal(count: count)
			#expect(complex == antiComplex.map { -$0 })
			let small: [Complex<Float>] = a.nextNormal(count: count)
			let antiSmall: [Complex<Float>] = b.nextNormal(count: count)
			#expect(small == antiSmall.map { -$0 })
		}
		let x: Float = a.nextNormal()
		let y: Float = b.nextNormal()
		#expect(x == -y)
		let z: Complex<Float> = a.nextNormal()
		let w: Complex<Float> = b.nextNormal()
		#expect(z == -w)
		let shiftedA: Complex<Double> = a.nextNormal(mean: Complex(2, -3), stdev: 0.25)
		let shiftedB: Complex<Double> = b.nextNormal(mean: Complex(2, -3), stdev: 0.25)
		#expect((shiftedA + shiftedB - Complex(4, -6)).length < 2e-15)
		#expect(a.next() == ~b.next())
	}

	@Test("Signed-grid endpoints preserve polar rejection decisions")
	func polarBoundaries() {
		let limit: UInt64 = 1 << 53
		for m in [0, 1, (1 << 52) - 1, 1 << 52, (1 << 52) + 1, limit - 1] as [UInt64] {
			for low: UInt64 in [0, 1, 2047] {
				let word = m << 11 | low
				let reflected = RNG.GaussianStream.reflectedWord(word)
				#expect(RNG.GaussianStream.reflectedWord(reflected) == word)
				let x = Double(word >> 11) * 0x1.0p-52 - 1
				let y = Double(reflected >> 11) * 0x1.0p-52 - 1
				#expect(m == 0 ? (x == -1 && y == -1) : x == -y)
			}
		}
		// Reject (-1,0), (0,0), and a corner; accept (0.5,-0.25).
		let words: [UInt64] = [
			0, 1 << 63, 1 << 63, 1 << 63, UInt64.max, UInt64.max, 3 << 62, 3 << 61,
		]
		var a = PolarBoundaryGenerator(words: words, reflected: false)
		var b = PolarBoundaryGenerator(words: words, reflected: true)
		let x = a.nextGaussian()
		let y = b.nextGaussian()
		#expect(x.0 == -y.0 && x.1 == -y.1)
		#expect(a.index == 8 && b.index == 8)
	}

	@Test("Open uniforms exclude both endpoints, including rounding at UInt64.max")
	func openUniformBoundaries() {
		for word: UInt64 in [0, 1, 2047, 2048, 1 << 63, UInt64.max - 2048, UInt64.max] {
			let u = RNG.unitDoubleOpen(from: word)
			let v = RNG.unitDoubleOpen(from: ~word)
			#expect(u > 0 && u < 1 && v > 0 && v < 1)
			#expect(abs(u + v - 1) <= Double.ulpOfOne)
		}
		#expect(RNG.unitDoubleOpen(from: 0) == 0x1.0p-54)
		#expect(RNG.unitDoubleOpen(from: UInt64.max) == Double(1).nextDown)
	}

	@Test("Pair IDs, channels and purposes retain distinct streams")
	func streamSeparation() {
		var sequences = Set<[UInt64]>()
		for pair: UInt64 in 0..<8 {
			for channel: UInt64 in 0..<3 {
				for purpose: RNG.Purpose in [
					.unspecified, .gaussianWhiteNoise, .coloredNoiseGeneration,
					.mcwfJumps,
				] {
					var rng = RNG(
						seed: 19, channel: channel, trajectoryID: 2 * pair,
						ensembleSampling: .antithetic, purpose: purpose)
					#expect(
						sequences.insert((0..<4).map { _ in rng.next() })
							.inserted)
				}
			}
		}
	}

	@Test("Ordinary, reflected and pooled marginals retain Gaussian and uniform moments")
	func marginalMoments() {
		var a = RNG(seed: 0x123456, trajectoryID: 0, ensembleSampling: .antithetic)
		var b = RNG(seed: 0x123456, trajectoryID: 1, ensembleSampling: .antithetic)
		var sums = [Double](repeating: 0, count: 8)
		let count = 32768
		for _ in 0..<count {
			let x: Double = a.nextNormal()
			let y: Double = b.nextNormal()
			let u = a.nextUnitDoubleOpen()
			let v = b.nextUnitDoubleOpen()
			#expect(u > 0 && u < 1 && v > 0 && v < 1)
			#expect(abs(u + v - 1) <= Double.ulpOfOne)
			sums[0] += x
			sums[1] += y
			sums[2] += x * x
			sums[3] += y * y
			sums[4] += u
			sums[5] += v
			sums[6] += u * u
			sums[7] += v * v
		}
		for member in 0..<2 {
			let mean = sums[member] / Double(count)
			let variance = sums[2 + member] / Double(count) - mean * mean
			let uniformMean = sums[4 + member] / Double(count)
			let uniformVariance =
				sums[6 + member] / Double(count) - uniformMean * uniformMean
			#expect(abs(mean) < 0.04 && abs(variance - 1) < 0.06)
			#expect(
				abs(uniformMean - 0.5) < 0.015
					&& abs(uniformVariance - 1.0 / 12) < 0.008)
		}
		#expect(sums[0] + sums[1] == 0)
		#expect(abs((sums[2] + sums[3]) / Double(2 * count) - 1) < 0.06)
		#expect(abs((sums[4] + sums[5]) / Double(2 * count) - 0.5) < 1e-13)
	}
}

private struct PolarBoundaryGenerator: RandomNumberGenerator {
	let words: [UInt64]
	let reflected: Bool
	var index = 0
	mutating func next() -> UInt64 {
		let value = words[index]
		index += 1
		return reflected
			? TrajectoryRandomNumberGenerator.GaussianStream.reflectedWord(value)
			: value
	}
}
