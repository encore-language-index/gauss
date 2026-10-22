# Active issues

Last reviewed: 2026-09-26. Environment: Linux x86_64, locally built and globally
installed Encore `0.1.1-neumann`, OpenBLAS 0.3.30, CUDA 13.4 / RTX 5070 Ti.
Priorities below describe impact, not authorization to change other repositories.

## G-003 — fixed locally: iterator and temporary-slot ownership

**Owner:** Encore compiler/core runtime, not Gauss's native allocator.
**Status:** fixed in the local `0.1.1-neumann` sources and verified; not published.

**Root cause:** frontend `for` lowering stored fresh iterator values without
releasing their original owners. EHIR slot promotion adds the store's retain,
but the LLVM backend had a name-based exception which suppressed it for physical
slots ending in `_iter_slot`. Ownership therefore differed between promoted
small functions and large/escaping-slot functions. Neither `?` nor Vec's
allocator was the source of the discrepancy.

The fix removes the backend's slot-name exception and explicitly transfers fresh
owners after retaining stores. The same audited mechanism fixes temporary
mutable receivers and `vec!` storage; borrowed `vec!` elements now acquire an
owner before the consuming `push` call. No Gauss API workaround was introduced.

Historical reproduction before the fix:

A `Vec[f64]` returned inside `Result`, extracted with `?`, then traversed with
`for` is not fully released in the reproduction. It contains a conditional early
return inside the loop, though the tested execution reaches the normal return.
100 calls leak **21,600 bytes**: 15,200 direct and 6,400 indirect, in both dev
and release. The same Result/Vec control using `len()` instead of iteration
adds no leaks. This does not prove that every form of Vec iteration leaks.

Gauss's public operations test exposes the same symptom in `uniform`: 4,304
lost bytes from host vectors. Native tensor storage is released, and native
CPU/CUDA sanitizer checks pass. Correct numerical results therefore do not
mean that the complete Encore API passes memory acceptance.

Reproduce without Gauss imports, custom native code or CUDA:

```sh
cd repros/vec-iteration
encore run
valgrind --leak-check=full --errors-for-leak-kinds=definite \
  --error-exitcode=97 target/dev/vec_iteration
encore run --profile release
valgrind --leak-check=full --errors-for-leak-kinds=definite \
  --error-exitcode=97 target/release/vec_iteration
```

Source: [standalone reproduction](repros/vec-iteration/src/main.enq).
Existing local evidence (generated logs, not committed artifacts):

- `repros/vec-iteration/target/valgrind.log`
- `repros/vec-iteration/target/valgrind-release.log`
- `target/native/encore-valgrind.log`

**Verification:** all 264 compiler tests, 8 EHIR tests and 5 LLVM backend tests
pass. New regressions cover normal/early returns, Result extraction and error
propagation, loop transfers, nested owned values, enumerate, returned elements,
temporary receivers/literals and real async suspension. Backend tests cover both
slot names with promotion, escaping addresses and the large-function fallback.
The three new compiler regressions and Gauss's public operations test pass
Valgrind in dev/release/extreme: zero lost allocations and zero access errors,
without suppressions. The runtime retains a reachable 64-byte allocation; this
is not reported as leaked memory.

Fixed-run evidence: `target/native/encore-valgrind-fixed.log` and
`target/ownership-verification/operations/valgrind-{dev,release,extreme}.log`.
Preserve the reproduction and tests. The other open issues below remain separate.

## G-002 — P2 / compiler integration: library artifacts and native link metadata

**Owner:** Encore build/package tooling.
**Status:** observed limitation; Gauss works through executable consumers.

Library-only `encore build` expects `src/main.enq`. In the previous compiler
investigation, `target = "static_lib"` was interpreted as an architecture rather
than an artifact kind. That configuration issue needs its own regression on the
final compiler; Gauss deliberately does not use it as a workaround.

Consumers currently repeat `-lm`, `-pthread`, `-ldl` in their Linux manifest.
This makes dependency setup more fragile than declaring native requirements once
in Gauss. Native-source discovery itself works: the examples compile `runtime.c`.

**Current supported path:** use Gauss as a dependency, retain explicit link args
as shown in [README.md](README.md), validate with `encore test` and both examples.

**Done when:** Encore distinguishes artifact kind from target triple, builds a
library-only fixture without a dummy executable, and reliably propagates native
link requirements to consumers. Add isolated fixtures before removing the
explicit arguments from the examples.

## G-004 — P2 / diagnostics: backend failures lose their original cause

**Owner:** Gauss native bridge and public error API. **Status:** open improvement.

CUDA/cuBLAS failures collapse to `GaussError::Backend`; backend loading failures
become `MissingDependency`. `message()` explains the category, but not the
original operation, vendor status, device or loader error. This makes driver,
allocation and asynchronous execution failures unnecessarily hard to diagnose.

**Done when:** preserve backend/operation/status details per session or error,
expose them safely to Encore, and test immediate and deferred failures plus
missing/incompatible plugins. Cleanup must not overwrite the original cause.

## G-005 — fixed locally: parallel CPU global reductions

**Owner:** Gauss CPU kernels. **Status:** implemented and verified locally.

Previously the pool distributed output elements, so a whole-tensor reduction
with one output stayed serial regardless of `GAUSS_NUM_THREADS`.

Global CPU reductions (`axis = -1`) now distribute 64 fixed input partitions
through the existing sleeping pool for inputs of at least 262,144 elements.
Partials accumulate in f64 and merge in fixed order, independent of worker
count, without heap allocation. Smaller inputs keep the serial path without
waking workers. Axis-specific reductions retain the existing output-parallel
implementation; narrow-output axis reductions are a separate future optimization.

**Verification:** `tests/reductions.c` covers 208 cases at each of 1, 4 and 16
workers, checks bitwise cross-worker parity, actual dispatch/no-wakeup behavior,
both dtypes, threshold boundaries, empty inputs, strided/transposed/broadcast
views and cancellation. Additional cases cover NaN and opposite infinities.
ASan/UBSan with leak detection pass. The expanded NumPy CPU/CUDA oracle passes
2,032 checks, including concurrent independent sessions; both Encore tests and
format/check/lint pass. Allocation rollback/deferred-error tests also pass.

Three local benchmark runs show f32 sum over 1,048,576 elements falling from
0.990 ms before to 0.169 ms with 4 workers (0.086 ms with 16). The updated
one-worker path measures 0.660 ms. Small-input measurements show no regression
in this run; these are machine-specific timings, not universal guarantees.
See [benchmark procedure and measurements](benchmarks/README.md).

## Deliberate contracts, not currently bugs

- Linux x86_64 is the supported initial platform. Other platforms require
  separate build and device validation.
- Sessions are thread-affine and checked at runtime. Use one session per worker;
  do not share handles between threads.
- Resources require `with` or explicit close. Dropping a descriptor is not close.
- Exported views retain their backing storage; long scopes retain results;
  CUDA staging lives until synchronization. Use bounded scopes and compact copies.
- Eager allocation/launch overhead exists. JIT, fusion, autodiff, vmap, multi-GPU
  collectives and DataFrames are outside the initial tensor-library scope.

Publication/CI have not been performed and are not correctness defects. They
require separate authorization after the acceptance blocker is resolved.

## Closed reference

G-001 (`with` return/cleanup miscompilation) was fixed and locally verified in
the installed `0.1.1-neumann` patch. Keep `repros/with-return` as historical
regression evidence; it is not an active Gauss blocker. G-003 is a distinct issue.
