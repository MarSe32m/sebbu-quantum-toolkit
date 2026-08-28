// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
import Numerics
import SebbuScience

public struct PreSampledCorrelatedOrnsteinUhlenbeckProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let interpolator: UniformLinearInterpolator<Complex<Double>>
    
    @inlinable
    public init(LC: Matrix<Complex<Double>>, LR: Matrix<Complex<Double>>, F: [Complex<Double>], start: Double, end: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        var random = NumPyRandom(seed: seed)
        var xi: Vector<Complex<Double>> = .zero(F.count)
        var x: Vector<Complex<Double>> = .zero(F.count)
        var xNew: Vector<Complex<Double>> = .zero(F.count)
        //let one: Vector<Complex<Double>> = .init(.init(repeating: .one, count: F.count))
        for i in 0..<xi.count { xi[i] = random.nextNormal(stdev: .sqrt(0.5)) }
        LC._dot(xi, into: &x)
        //var samples: [Complex<Double>] = [one.inner(x)]
        var samples: [Complex<Double>] = [x.components.reduce(.zero, +)]
        samples.reserveCapacity(Int((end - start) / step))
        var t = start
        while t <= end {
            // x_{n+1} = Fx_n + L_R xi_n
            for i in 0..<xi.count { xi[i] = random.nextNormal(stdev: .sqrt(0.5)) }
            for i in 0..<xNew.count { xNew[i] = x[i] * F[i] }
            LR._dot(xi, addingInto: &xNew)
            //samples.append(one.inner(xNew))
            samples.append(xNew.components.reduce(.zero, +))
            swap(&x, &xNew)
            t += step
        }
        self.interpolator = UniformLinearInterpolator(start: start, step: step, y: samples)
    }
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], start: Double, end: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) {
        let (F, LC, LR) = PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator.constructFLCLR(r: r, W: W, dt: step)
        self.init(LC: LC, LR: LR, F: F, start: start, end: end, step: step, seed: seed)
    }
    
    @inlinable
    internal init(_ interpolator: UniformLinearInterpolator<Complex<Double>>) {
        self.interpolator = interpolator
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        interpolator.sample(t)
    }
    
    @inlinable
    @inline(always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(UniformLinearInterpolator(start: interpolator.start, step: interpolator.step, y: interpolator.y.map { -$0 }))
    }
}

public struct PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let LC: Matrix<Complex<Double>>
    
    @usableFromInline
    internal let LR: Matrix<Complex<Double>>
    
    @usableFromInline
    internal let F: [Complex<Double>]
    
    @usableFromInline
    internal let start: Double
    
    @usableFromInline
    internal let end: Double
    
    @usableFromInline
    internal let step: Double
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], start: Double, end: Double, step: Double) {
        precondition(r.count == W.count, "The r and W arrays must have the same size.")
        let (F, LC, LR) = Self.constructFLCLR(r: r, W: W, dt: step)
        self.F = F
        self.LC = LC
        self.LR = LR
        self.start = start
        self.end = end
        self.step = step
    }
    
    @inlinable
    @inline(always)
    public func generate() -> sending PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(LC: LC, LR: LR, F: F, start: start, end: end, step: step)
    }
    
    @inlinable
    @inline(always)
    public func generate(seed: UInt32) -> PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(LC: LC, LR: LR, F: F, start: start, end: end, step: step, seed: seed)
    }
    
    
    @inlinable
    internal static func constructFLCLR(r: [Complex<Double>], W: [Complex<Double>], dt: Double) -> (F: [Complex<Double>], LC: Matrix<Complex<Double>>, LR: Matrix<Complex<Double>>) {
        precondition(!r.isEmpty, "Need at least one OU mode.")
        precondition(r.count == W.count, "The r and W arrays must have the same size.")
        precondition(dt > 0.0, "The timestep must be positive.")

        for w in W {
            precondition(w.real > 0.0, "All W must have positive real part.")
            precondition(w.real.isFinite && w.imaginary.isFinite, "W contains non-finite values.")
        }

        for x in r {
            precondition(x.real.isFinite && x.imaginary.isFinite, "r contains non-finite values.")
        }
        let F: [Complex<Double>] = W.map { .exp(-$0 * dt) }
        var C: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<C.rows {
            for j in 0..<C.columns {
                C[i, j] = r[i] * r[j].conjugate / (W[i] + W[j].conjugate)
            }
        }
        C = MatrixOperations.symmetrizedHermitian(C)
        
        var R: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<R.rows {
            for j in 0..<R.columns {
                // We do it this way since this is more stable for small time-steps
                let z = (W[i] + W[j].conjugate) * dt
                R[i, j] = r[i] * r[j].conjugate * dt * .phiOneMinusExpMinus(z)
            }
        }
        R = MatrixOperations.symmetrizedHermitian(R)
        let LC = MatrixOperations.positiveSemidefiniteSquareRoot(C)
        let LR = MatrixOperations.positiveSemidefiniteSquareRoot(R)
        return (F, LC, LR)
    }
}

