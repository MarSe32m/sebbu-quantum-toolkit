# HOPS propagation benchmark

Run a fixed colored-bath fluorescence workload without fitting, Python or plots:

```sh
swift run -c release HOPSBenchmark --workers 1,8,32 --trajectories 256 --repeats 3
```

For Roihu, compare worker counts within your SLURM CPU allocation. For example,
with at least 128 allocated CPUs:

```sh
swift run -c release HOPSBenchmark --workers 1,8,32,128 --trajectories 8192 --repeats 3
```

The benchmark prints CSV timings and deterministic ensemble checksums for the
steady-state preparation interval and the two-time correlation interval.
The same seed and trajectory IDs are used at every worker count; small changes
in floating-point reductions are expected. The library sets BLAS to one thread
while executing each ensemble.

Defaults are a two-level system, three correlated OU poles, maximum tier four
(35 hierarchy states), 256 trajectories, five units of preparation and five
units of delay, step 0.01 and 251 samples. Options are `--dimension`, `--poles`
(0 through 3), `--tier`, `--trajectories`, `--workers` (comma separated),
`--warmup`, `--delay`, `--step`, `--samples`, and `--repeats`. Add `--mixed` to
benchmark a three-branch, two-insertion correlation instead. A finite warmup
here is a performance workload, not an assertion of physical stationarity.

The fixed OU model is physically valid and exercises the same hierarchy shape
as a three-pole fitted bath. It is not a replacement for convergence tests of
the fitted IBM spectrum. Use the DevelopmentTesting example for those.

## CPU implementation

HOPS now stores operators in their original row-major orientation. The RHS uses
an unrolled two-level action, small direct kernels through dimension four, and
GEMV for larger vectors. Dynamic Markovian loss operators are constructed
column by column without GEMM. No operator transpose buffers are required.

Latent-neighbour weights and valid edges are contracted once in shared
preparation. Each RHS evaluation gathers and applies one row's parents and
children together. The two-level path uses scalar temporaries; other dimensions
use two ket-sized scratch vectors, independent of hierarchy and branch counts.
Guide, ket and bra still have separate hierarchy blocks and share exactly the
same guide means, noise paths and normalization gauge.

This removes GEMM from HOPS propagation, including nonzero baths and dynamic
collapse operators. It does not change other methods or linear-algebra routines
used during bath fitting and noise-generator preparation. Precomputing weighted
connections increases shared immutable preparation storage in exchange for less
work and less scratch storage in each trajectory.

Large dense systems can favor batched GEMM on some backends. This implementation
prioritizes independent trajectory execution without GEMM workspace contention;
benchmark the actual system dimension, hierarchy size and allocated worker
count rather than extrapolating a small-kernel timing to a full simulation.
