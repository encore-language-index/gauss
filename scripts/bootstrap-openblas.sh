#!/usr/bin/env bash
set -euo pipefail
gauss_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
deps="$gauss_root/target/deps"
mkdir -p "$deps"
archive="$deps/openblas-0.3.30.tar.gz"
if ! test -f "$archive"; then
    curl --fail --location --retry 2 https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz -o "$archive"
fi
printf '%s  %s\n' 27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d "$archive" | sha256sum -c -
if ! test -d "$deps/OpenBLAS-0.3.30"; then tar -xzf "$archive" -C "$deps"; fi
make -C "$deps/OpenBLAS-0.3.30" -j"${JOBS:-8}" NOFORTRAN=1 NO_LAPACK=1 USE_THREAD=1 NUM_THREADS=64
printf 'export GAUSS_BLAS_LIBRARY=%q\n' "$deps/OpenBLAS-0.3.30/libopenblas.so"
