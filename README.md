# sebbu-quantum-toolkit

A Swift toolkit for simulating open quantum systems using stochastic trajectory, non-Markovian, and hierarchical methods.

> **Status:** Early development. The public API is expected to evolve and may contain breaking changes.

## Overview

`sebbu-quantum-toolkit` aims to provide a common, high-performance interface for numerical methods used in open quantum dynamics.

The library is designed around a simple idea: different numerical approaches should share the same representation of the underlying quantum system, Markovian dissipation, integration settings and trajectory infrastructure wherever possible, while retaining method-specific representations of non-Markovian environments and auxiliary states.

Planned and developing methods include:

- Monte Carlo wave-function method (MCWF)
- Quantum-state diffusion (QSD)
- Non-Markovian quantum-state diffusion (NMQSD)
- Hierarchy of pure states (HOPS)
- Hierarchical equations of motion (HEOM)
- Hybrid and future trajectory-based open-system methods

The API is intended to make switching between methods straightforward, particularly for benchmarking different descriptions of the same physical system.

## Design

At a high level, simulations are built from a few common components.

### Quantum system

The isolated system is described by its Hamiltonian,

```
H(t)
```

with support for time-dependent systems.

```swift
let system = QuantumSystem(
    dimension: dimension
) { t, H in
    // Construct H(t)
}
```

### Markovian environments

Markovian processes are represented as Lindblad channels,
```markup
\mathcal{D}_j[\rho] = \gamma_j(t) \left(C_j(t)\rho C_j^\dagger(t) - \frac{1}{2} \left\{C_j^\dagger(t)C_j(t),\rho\right\}\right).
```

A channel may therefore contain a time-dependent rate and operator.

```swift
let decay = LindbladChannel(
    rate: { t in
        gamma
    },
    operator: { t, C in
        // Construct C(t)
    }
)
```

The same channels can be used by different unravelings. For example, MCWF can interpret them as jump processes while QSD can interpret them as diffusive processes.

### Antithetic ensembles

Set `ensembleSampling: .antithetic` on `TrajectoryExecution` to pair global
trajectory IDs `(0, 1), (2, 3), ...` in HOPS, QSD and MCWF. Independent sampling
remains the default.

### Optional progress

Set `propagation.progress = .console(style: .bar, label: "HOPS")` to show a
percentage-based progress bar with estimated time remaining. GKSL and HEOM reports accepted
time steps; QSD, MCWF and HOPS report trajectory completions. Reporting is disabled
by default.

## Unified solver interface

Trajectory methods follow a common API shape.

For example:

```swift
var rng = SeededRandomNumberGenerator(seed: 42)

MCWF.solve(
    start: 0.0,
    end: 10.0,
    initialState: initialState,
    on: grid,
    system: system,
    lindbladChannels: channels,
    rng: &rng,
    integration: integration
) { t, psi in
    // Observe the physical state |ψ(t)⟩
}
```

A non-Markovian method adds only the information specific to that method:

```swift
HOPS.solve(
    start: 0.0,
    end: 10.0,
    initialState: initialState,
    on: grid,
    system: system,
    lindbladChannels: channels,
    environment: environment,
    hierarchy: hierarchy,
    noise: noise,
    rng: &rng,
    integration: integration
) { t, psi in
    // Observe the physical HOPS state
}
```

HEOM uses the same `BathEnvironment` and `CorrelatedBathModel` as HOPS:

```swift
let environment = BathEnvironment(couplingOperators: couplingOperators, bath: bathModel)
let hierarchy = HEOM.Hierarchy(environment: environment, truncation: .maximumTier(6))
let configuration = HEOM.Configuration(hierarchy: hierarchy, shiftType: .meanField)

try HEOM.solve(
    problem: densityMatrixProblem,
    configuration: configuration,
    propagation: propagation
) { time, rho in
    // Observe the physical density matrix.
    return .proceed
}
```

`shiftType: .none` (the default) implements ordinary latent-basis HEOM;
`.meanField` integrates the deterministic displacement with the ADOs. Both
support pure-state and density-matrix initial conditions, time-dependent
Hamiltonians and coupling operators, and additional Lindblad channels.
The initial bath is factorized and all nonroot ADOs and displacements start at
zero. No renormalization or positivity projection is applied.

For `P` latent poles, HEOM uses `2P` directions, ordered as all ket occupations
followed by all bra occupations. `.maximumTier(D)` retains `|m| + |n| <= D`
(`binomial(2P + D, D)` ADOs); missing children are zero. This depth is not the
same truncation as a HOPS ket hierarchy. Converge it independently of the
integration tolerances. Custom truncations must be finite and downward closed;
use a ket/bra symmetric set to preserve the ADO adjoint relation.

`solveWithHierarchy` lends a `HEOM.HierarchyStateView` containing row-major,
factorially scaled ADOs (`rho[m,n] / sqrt(m! n!)` relative to the unscaled
hierarchy). In centered mode the view contains displaced ADOs. Its root is the
physical density matrix in either mode. The view cannot escape its callback.

`solveTwoTimeCorrelation` and `solveMultiTimeOrderedCorrelation` accept the
common correlation requests. Every left or right insertion acts on **all**
ADOs, preserving bath memory. Centered correlations carry a separate physical
guide hierarchy to determine the displacement; the inserted companion may be
traceless or non-Hermitian. Fixed schedules include the equal-time value after
insertion and skip earlier outputs. Accepted-step schedules start strictly
after the final insertion. All solvers support observer termination and
optional accepted-step progress reporting.

## Reproducible trajectories

Stochastic methods accept an explicit random-number generator.

```swift
var rng = SeededRandomNumberGenerator(seed: 1234)
```

This makes individual trajectories reproducible and allows deterministic trajectory streams to be constructed for parallel ensemble simulations.

The long-term goal is that a trajectory can be identified by a global seed and trajectory index, independently of whether the ensemble is executed serially, with CPU parallelism, across compute nodes, or on accelerators.

## Numerical methods

The toolkit is intended to support both deterministic and stochastic numerical integration while sharing common infrastructure where appropriate.

Typical solver configuration may include:

```swift
let integration = IntegrationOptions(
    minimumStepSize: 1e-8,
    maximumStepSize: 1e-2,
    absoluteTolerance: 1e-9,
    relativeTolerance: 1e-7
)
```

Method-specific numerical details remain isolated from the common description of the physical system.

## Goals

`sebbu-quantum-toolkit` is being developed with several goals:

- **Consistent APIs** across different open-system methods.
- **Method interoperability**, so the same model can be simulated using multiple approaches.
- **Reproducibility** through explicit random-number-generator control.
- **Performance**, with zero-copy data structures and efficient numerical kernels where possible.
- **Parallelism**, especially over independent stochastic trajectories.
- **Extensibility**, allowing new unravelings and non-Markovian methods to fit naturally into the toolkit.
- **Time-dependent models**, including Hamiltonians, rates and system operators.
- **Research-oriented flexibility** without unnecessarily constraining the underlying numerical methods.

## Planned structure

The exact module structure is still evolving, but the library is expected to contain abstractions for:

```text
QuantumSystem
LindbladChannel
IntegrationOptions

MCWF
QSD
NMQSD
HOPS
HEOM
```

along with common infrastructure for:

```text
trajectory generation
random-number streams
ensemble averaging
noise processes
bath-correlation functions
hierarchy construction and truncation
observables
time integration
```

Method-specific functionality will remain separate where forcing a common abstraction would obscure the underlying mathematics.

## Dependencies

The toolkit is intended to build on the numerical infrastructure provided by the broader `sebbu-*` Swift ecosystem, including libraries for linear algebra and numerical integration.

Specific dependencies will be documented as the package structure stabilizes.

## Installation

The package is currently under active development and is not yet considered API-stable.

Once published as a Swift package, it can be added using Swift Package Manager:

```swift
dependencies: [
    .package(
        url: "https://github.com/MarSe32m/sebbu-quantum-toolkit.git",
        branch: "main"
    )
]
```

and imported with the appropriate module name.

## Development status

This project is currently in its initial implementation phase.

APIs shown in this README illustrate the intended design and may differ from the current implementation. Breaking changes should be expected until the core abstractions have stabilized.

# License

`sebbu-quantum-toolkit` is licensed under the Apache License, Version 2.0.

See the [LICENSE](LICENSE) file for details.
