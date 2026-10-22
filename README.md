# Gauss

Eager CPU/CUDA tensors for Encore: `f32`/`f64`, broadcasting, views,
elementwise operations, reductions and batched matrix multiplication.
CPU GEMM uses OpenBLAS; GPU GEMM uses cuBLAS without TF32.

The initial implementation runs on **Linux x86_64** with Encore **0.1.1-neumann**.
Numerical/native checks and public-API memory checks pass with the local compiler
ownership fix; see G-003 in [ISSUES.md](ISSUES.md) for the regression and evidence.
Version `0.1.0` is not published. This is a tensor
foundation, not a complete JAX replacement: autodiff, JIT, vmap, multi-GPU
collectives and DataFrames are outside this version's scope.

## Setup

Requires Encore >=0.1.1, a C11 compiler, pthreads and LP64 OpenBLAS.
Use system OpenBLAS or build the pinned source without installing globally:

```sh
bash scripts/bootstrap-openblas.sh
export GAUSS_BLAS_LIBRARY="$PWD/target/deps/OpenBLAS-0.3.30/libopenblas.so"
```

CUDA is optional and must be selected explicitly. CPU use does not need CUDA.
Build the plugin (default: RTX 5070 Ti / SM 120):

```sh
bash scripts/build-cuda.sh
export GAUSS_CUDA_LIBRARY="$PWD/target/native/libgauss_cuda.so"
```

Set `CUDA_HOME`, `NVCC` and `GAUSS_CUDA_ARCH` for your toolkit/device. The plugin
uses asynchronous allocation and needs a compatible driver/device. An unavailable
CUDA backend is an error, never a silent CPU fallback.

Applications use a local dependency and explicit Linux link arguments:

```toml
[project]
name = "application"
version = "0.1.0"
encore = ">=0.1.1"
dependencies = ["path@../Gauss"]

[target.x86_64-unknown-linux-gnu]
link-args = ["-lm", "-pthread", "-ldl"]
```

Encore discovers Gauss's `runtime.c` through the dependency. Library-only
`encore build` is not currently supported; validate through `encore test` and
executable consumers. See [ISSUES.md](ISSUES.md).

## Examples

Run from the repository root after exporting the backend paths:

```sh
(cd examples/matmul && encore run)             # CPU: [6, 6, 6, 6]
(cd examples/matmul && encore run -- --cuda)   # explicit CUDA device 0
(cd examples/scopes && encore run)            # [9216], 8 tensor bytes survive
```

[Matrix example](examples/matmul/src/main.enq) demonstrates device selection,
matmul and synchronization. [Scope example](examples/scopes/src/main.enq)
demonstrates bounded temporary storage and exporting a result to its parent.
Both handle failures through `GaussError.message()`.

## API

| Owner | Operations |
| --- | --- |
| `Session` | `cpu`, `open(Device)`, `scope`, `finish`, `close`, `allocated_bytes` |
| `Scope`: lifetime | `child`, `export`, `close` |
| `Scope`: creation | `full`, `zeros`, `ones`, `from_f32`, `from_f64`, `to_device` |
| `Scope`: views/copies | `reshape`, `permute`, `transpose`, `broadcast`, `slice`, `contiguous`, `astype` |
| `Scope`: arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `exp`, `log`, `sqrt`, `matmul` |
| `Scope`: reductions | `sum`, `mean`, `min`, `max` |
| `Tensor` | `shape`, `len`, `dtype`, `to_f32`, `to_f64` |

Operations return `Result[..., GaussError]`; shapes and host values use
`core::vec::Vec`. Tensors are immutable; no implicit dtype conversion occurs.

## Lifetime and numerical contract

- Always wrap sessions/scopes in `with` or close them explicitly. Dropping a
  copied Encore descriptor does **not** release these native resources.
- Sessions belong to their creating thread. Create a session inside each worker;
  cross-thread handles are rejected. CPU workers sleep when idle.
  `GAUSS_NUM_THREADS` bounds each pool (default available CPUs, maximum 16);
  `OPENBLAS_NUM_THREADS` independently bounds BLAS workers.
- Children read ancestors. `child.export(tensor)` retains a parent-owned view,
  including its entire backing allocation. Other child handles expire at close.
  Make a compact copy to avoid retaining a large backing view.
- CUDA is asynchronous. Host inputs are copied into owned staging memory before
  returning; staging is freed on synchronization. Use bounded scopes and `finish`
  at result/measurement boundaries, not after every operation.
- `finish`/explicit close report deferred backend errors. Failed sessions reject
  new computation. Automatic cleanup aborts on an unhandled backend failure;
  explicitly close to handle failures through `Result` instead.
- `to_device` copies through synchronized host staging, not peer-to-peer transfer.
  `allocated_bytes` counts tensor backing storage, excluding staging, vendor
  workspaces and cached CUDA allocator reservations.
- Up to 32 dimensions, positive-step slices, contiguous row-major outputs.
  `reshape` is zero-copy and requires contiguous storage. Matmul requires rank
  >=2, row/column-major inner matrix strides and BLAS dimensions <=`INT_MAX`.
  Other layouts require `contiguous`; batch axes may broadcast.
- Reduction axis is nonnegative or `-1` for all axes. Empty sum is zero, empty
  mean is NaN, empty min/max is an error. NaNs propagate. Sums accumulate in f64
  and cast at output. CPU/GPU results are tolerance-equivalent, not promised
  bit-identical. CUDA reduction uses a fixed tree without atomic sums.
- Whole-tensor CPU reductions (`axis = -1`) use the existing worker pool from
  262,144 elements upward. Fixed input partitions and merge order make results
  independent of the configured worker count on the same build/platform.
  Smaller inputs stay serial without waking workers. This does not promise
  identical rounding to the old serial algorithm or to CUDA.

## Verification

NumPy is a **test-only** dependency (validated with 2.5.3), not a Gauss runtime
dependency. Install it in your own test environment.

```sh
encore format --check
encore check
encore lint
encore test --jobs 2
bash scripts/test-native.sh          # CPU oracle, lifetime and failure injection
bash scripts/test-native.sh --cuda   # additionally requires working CUDA
```

The native gate checks every operation against NumPy for both dtypes, scalar/
empty/randomized/broadcast/strided layouts, batched/transposed GEMM, large GPU
reductions, errors, stale handles, deep scopes, allocation rollback and concurrent
independent sessions. Deferred GPU errors use an injected fake backend rather
than deliberately crashing the physical GPU.

See [benchmarks/README.md](benchmarks/README.md) for synchronized measurements
and their limits. No general performance parity with JAX or validation on every
GPU/driver is claimed.
