# Operator propagation benchmark

Run with one BLAS thread so HEOM worker counts are meaningful
```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 2 --structure dense
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 32 --structure diagonal
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 swift run -c release OperatorPropagationBenchmark --dimensions 128 --methods heom --workers 4
```
