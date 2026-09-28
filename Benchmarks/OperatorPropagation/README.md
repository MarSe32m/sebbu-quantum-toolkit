# Operator propagation benchmark

Run with one BLAS thread so HEOM worker counts are meaningful
```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 2 --structure dense
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 32 --structure diagonal
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 128 --methods heom --workers 4
```

To isolate HEOM bath storage, hold the Hamiltonian representation fixed and
compare these two runs (the reported checksums should agree)
```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --methods heom --dimensions 32,128 --storage sparse --bath-storage dense
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --methods heom --dimensions 32,128 --storage sparse --bath-storage sparse
```

Add `--shift mean-field` to compare centered propagation, or `--workers 4` to
compare the same bath actions with parallel ADO evaluation.
