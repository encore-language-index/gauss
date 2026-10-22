#!/usr/bin/env python3
"""Deterministic independent NumPy oracle; CPU always, CUDA with --cuda.

Build target/native/libgauss_test.so from runtime.c before running.
NumPy is a test dependency only, never a dependency of Gauss.
"""
import ctypes as C
from pathlib import Path
import sys
import os
import threading
import warnings
import numpy as np

lib = C.CDLL(str(Path(__file__).resolve().parents[1] / "target/native/libgauss_test.so"))
U, I, Z, P, B = C.c_uint64, C.c_int32, C.c_size_t, C.c_void_p, C.c_bool
signatures = {
    "session": (U, [I]), "scope": (U, [U, B]),
    "close_session": (B, [U]), "close_scope": (B, [U]),
    "finish": (B, [U]), "bytes": (Z, [U]), "error": (C.c_uint32, []),
    "valid": (B, [U]), "export": (U, [U, U]),
    "meta": (Z, [U, I, Z]),
    "from": (U, [U, I, Z, P, P, Z]), "read": (B, [U, I, P, Z]),
    "full": (U, [U, I, Z, P, C.c_double]),
    "map": (U, [U, U, U, I, I]), "reduce": (U, [U, U, I, I, B]),
    "view": (U, [U, U, I, Z, P]), "slice": (U, [U, U, Z, Z, Z, Z]),
    "matmul": (U, [U, U, U]), "to_device": (U, [U, U]),
}
for name, (result, args) in signatures.items():
    fn = getattr(lib, "gauss_" + name)
    fn.restype, fn.argtypes = result, args

checks = 0
def checked(value):
    assert value, f"Gauss error {lib.gauss_error()}"
    return value

def dims(shape):
    return (Z * len(shape))(*shape)

def upload(scope, values):
    values = np.asarray(values)
    dtype = int(values.dtype == np.float64)
    packed = np.ascontiguousarray(values)
    return checked(getattr(lib, "gauss_from")(scope, dtype, values.ndim,
                   dims(values.shape), packed.ctypes.data, values.size))

def equal(handle, expected):
    global checks
    expected = np.asarray(expected)
    rank = lib.gauss_meta(handle, 0, 0)
    assert tuple(lib.gauss_meta(handle, 3, axis) for axis in range(rank)) == expected.shape
    assert lib.gauss_meta(handle, 1, 0) == expected.size
    actual = np.empty(expected.shape, dtype=expected.dtype)
    checked(lib.gauss_read(handle, int(expected.dtype == np.float64),
                           actual.ctypes.data, actual.size))
    tolerance = 3e-5 if expected.dtype == np.float32 else 3e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance,
                               equal_nan=True)
    checks += 1

def view(scope, handle, op, args):
    return checked(lib.gauss_view(scope, handle, op, len(args), dims(args)))

def suite(device):
    rng = np.random.default_rng(142857)
    session = checked(lib.gauss_session(device))
    try:
        for dtype in (np.float32, np.float64):
            for shape in ((), (0,), (7,), (3, 5), (2, 3, 5), (2, 1, 3, 5)):
                scope = checked(lib.gauss_scope(session, False))
                a = rng.uniform(.2, 2, shape).astype(dtype)
                h = upload(scope, a)
                equal(h, a)
                equal(checked(lib.gauss_full(scope, int(dtype == np.float64), len(shape), dims(shape), 2)), np.full(shape, 2, dtype=dtype))
                for op, fn in ((0, lambda x: x), (5, np.negative), (6, np.abs),
                               (7, np.exp), (8, np.log), (9, np.sqrt)):
                    equal(checked(lib.gauss_map(scope, h, 0, op, -1)), fn(a))
                other = np.float64 if dtype == np.float32 else np.float32
                equal(checked(lib.gauss_map(scope, h, 0, 0, int(other == np.float64))), a.astype(other))
                b = np.asarray(1.25, dtype=dtype) if not shape else rng.uniform(.5, 2, (shape[-1],)).astype(dtype)
                hb = upload(scope, b)
                for op, fn in ((1, np.add), (2, np.subtract), (3, np.multiply), (4, np.divide)):
                    equal(checked(lib.gauss_map(scope, h, hb, op, -1)), fn(a, b))
                for axis in [-1] + list(range(a.ndim)):
                    n = a.size if axis == -1 else a.shape[axis]
                    for keep in (False, True):
                        for op, fn in enumerate((np.sum, np.mean, np.min, np.max)):
                            if not n and op >= 2:
                                assert not lib.gauss_reduce(scope, h, axis, op, keep)
                                continue
                            kw = {"axis": None if axis == -1 else axis, "keepdims": keep}
                            if op < 2:
                                kw["dtype"] = np.float64
                            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                                warnings.filterwarnings("ignore", message="Mean of empty slice")
                                expected = np.asarray(fn(a, **kw), dtype=dtype)
                            equal(checked(lib.gauss_reduce(scope, h, axis, op, keep)), expected)
                if a.ndim >= 2:
                    axes = tuple(reversed(range(a.ndim)))
                    t = view(scope, h, 1, axes)
                    equal(t, a.transpose(axes))
                    equal(checked(lib.gauss_map(scope, t, 0, 0, -1)), a.transpose(axes))
                    equal(checked(lib.gauss_reduce(scope, t, 0, 0, False)), a.transpose(axes).sum(axis=0, dtype=np.float64).astype(dtype))
                    sl = checked(lib.gauss_slice(scope, h, a.ndim - 1, 1, shape[-1], 2))
                    equal(sl, a[..., 1::2])
                equal(view(scope, h, 0, (a.size,)), a.reshape(-1))
                scalar = upload(scope, np.asarray(3, dtype=dtype))
                equal(view(scope, scalar, 2, shape), np.full(shape, 3, dtype=dtype))
                checked(lib.gauss_close_scope(scope))
                assert lib.gauss_bytes(session) == 0
            for batch in ((), (2,), (2, 3)):
                for m, k, n in ((1, 1, 1), (3, 7, 5), (17, 9, 13), (2, 0, 3), (0, 3, 2), (3, 2, 0)):
                    for trans_a in (False, True):
                        for trans_b in (False, True):
                            scope = checked(lib.gauss_scope(session, False))
                            a = rng.normal(size=batch + ((k, m) if trans_a else (m, k))).astype(dtype)
                            b = rng.normal(size=((n, k) if trans_b else (k, n))).astype(dtype)
                            ha, hb = upload(scope, a), upload(scope, b)
                            if trans_a:
                                axes = list(range(a.ndim)); axes[-2:] = axes[-2:][::-1]
                                ha = view(scope, ha, 1, axes); a = a.swapaxes(-1, -2)
                            if trans_b:
                                hb = view(scope, hb, 1, (1, 0)); b = b.T
                            equal(checked(lib.gauss_matmul(scope, ha, hb)), a @ b)
                            checked(lib.gauss_close_scope(scope))
            scope = checked(lib.gauss_scope(session, False))
            special = np.asarray([np.nan, np.inf, -np.inf, 0., -0.], dtype=dtype)
            h = upload(scope, special)
            for op in (2, 3):
                equal(checked(lib.gauss_reduce(scope, h, -1, op, False)), np.asarray(np.nan, dtype=dtype))
            checked(lib.gauss_close_scope(scope))
            scope = checked(lib.gauss_scope(session, False))
            for _ in range(12):
                m, n = rng.integers(1, 30, size=2)
                a = rng.uniform(.1, 3, (m, n)).astype(dtype)
                b = rng.uniform(.1, 3, (m, 1)).astype(dtype)
                ha, hb = upload(scope, a), upload(scope, b)
                for op, fn in ((1, np.add), (2, np.subtract), (3, np.multiply), (4, np.divide)):
                    equal(checked(lib.gauss_map(scope, ha, hb, op, -1)), fn(a, b))
                transposed = view(scope, ha, 1, (1, 0))
                for op, fn in ((5, np.negative), (6, np.abs), (7, np.exp), (8, np.log), (9, np.sqrt)):
                    equal(checked(lib.gauss_map(scope, transposed, 0, op, -1)), fn(a.T))
            checked(lib.gauss_close_scope(scope))
            # Exercise CUDA passes and the CPU parallel threshold, including
            # strided views, cancellation, NaNs and f64 accumulation for f32.
            scope = checked(lib.gauss_scope(session, False))
            for size in (4095, 4096, 65537, 262143, 262144, 262145, 1048576):
                values = rng.uniform(-1, 1, size * 2).astype(dtype)
                h = upload(scope, values)
                sliced = checked(lib.gauss_slice(scope, h, 0, 0, size * 2, 2))
                for handle, array in ((h, values), (sliced, values[::2])):
                    for op, fn in enumerate((np.sum, np.mean, np.min, np.max)):
                        options = {"dtype": np.float64} if op < 2 else {}
                        expected = np.asarray(fn(array, **options), dtype=dtype)
                        equal(checked(lib.gauss_reduce(scope, handle, -1, op, False)), expected)
                values[-1] = np.nan
                h = upload(scope, values)
                equal(checked(lib.gauss_reduce(scope, h, -1, 2, False)), np.asarray(np.nan, dtype=dtype))
            checked(lib.gauss_close_scope(scope))
    finally:
        checked(lib.gauss_close_session(session))

def errors_and_threads(device=-1):
    session = checked(lib.gauss_session(device))
    scope = checked(lib.gauss_scope(session, False))
    a = upload(scope, np.ones((2, 3), dtype=np.float32))
    b = upload(scope, np.ones((4,), dtype=np.float32))
    assert not lib.gauss_map(scope, a, b, 1, -1) and lib.gauss_error() == 2
    assert not lib.gauss_reduce(scope, a, 2, 0, False) and lib.gauss_error() == 7
    assert not lib.gauss_full(scope, 0, 2, dims((2**63, 3)), 0) and lib.gauss_error() == 9
    assert not lib.gauss_full(scope, 4, 0, dims(()), 0) and lib.gauss_error() == 3
    assert not lib.gauss_slice(scope, a, 0, 0, 2, 0) and lib.gauss_error() == 2
    t = view(scope, a, 1, (1, 0))
    assert not lib.gauss_view(scope, t, 0, 1, dims((6,))) and lib.gauss_error() == 8
    child = checked(lib.gauss_scope(scope, True))
    temporary = upload(child, np.ones(3, dtype=np.float32))
    assert not lib.gauss_map(scope, temporary, 0, 0, -1) and lib.gauss_error() == 10
    kept = checked(lib.gauss_export(child, temporary))
    checked(lib.gauss_close_scope(child))
    assert not lib.gauss_valid(temporary)
    equal(kept, np.ones(3, dtype=np.float32))
    failures = []
    def thread_work():
        try:
            assert not lib.gauss_valid(a)
            own = checked(lib.gauss_session(device))
            arena = checked(lib.gauss_scope(own, False))
            large = np.arange(262145, dtype=np.float64)
            h = upload(arena, large)
            for _ in range(8):
                equal(checked(lib.gauss_map(arena, h, h, 1, -1)), large + large)
                equal(checked(lib.gauss_reduce(arena, h, -1, 0, False)),
                      np.asarray(large.sum(), dtype=np.float64))
            checked(lib.gauss_close_session(own))
        except BaseException as error:
            failures.append(error)
    threads = [threading.Thread(target=thread_work) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not failures, failures
    assert lib.gauss_valid(a)
    checked(lib.gauss_close_session(session))

def missing_dependencies():
    for device, name in ((-1, "GAUSS_BLAS_LIBRARY"), (0, "GAUSS_CUDA_LIBRARY")):
        old = os.environ.get(name)
        try:
            os.environ[name] = "/gauss-test/nonexistent-backend.so"
            assert not lib.gauss_session(device) and lib.gauss_error() == 11
        finally:
            if old is None: del os.environ[name]
            else: os.environ[name] = old

def transfer():
    cpu, gpu = checked(lib.gauss_session(-1)), checked(lib.gauss_session(0))
    a, b = checked(lib.gauss_scope(cpu, False)), checked(lib.gauss_scope(gpu, False))
    values = np.arange(15, dtype=np.float64).reshape(3, 5)
    h = upload(a, values)
    device = checked(lib.gauss_to_device(b, h))
    checked(lib.gauss_close_session(cpu))
    equal(device, values)
    host = checked(lib.gauss_session(-1))
    scope = checked(lib.gauss_scope(host, False))
    result = checked(lib.gauss_to_device(scope, device))
    checked(lib.gauss_close_session(gpu))
    equal(result, values)
    checked(lib.gauss_close_session(host))

if __name__ == "__main__":
    suite(-1)
    errors_and_threads()
    missing_dependencies()
    if "--cuda" in sys.argv:
        suite(0)
        errors_and_threads(0)
        transfer()
    print(f"Numerical/layout/lifetime/concurrency checks passed: {checks}")
