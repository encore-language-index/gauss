#!/usr/bin/env bash
set -euo pipefail
gauss_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/opt/cuda}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
test -x "$nvcc" || { echo "CUDA toolkit not found; set CUDA_HOME or NVCC" >&2; exit 1; }
mkdir -p "$gauss_root/target/native"
# Explicit target, plus PTX for forward-compatible driver compilation.
"$nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC --fmad=false \
    -gencode "arch=compute_${GAUSS_CUDA_ARCH:-120},code=[sm_${GAUSS_CUDA_ARCH:-120},compute_${GAUSS_CUDA_ARCH:-120}]" \
    "$gauss_root/native/cuda.cu" -lcublas \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    -o "$gauss_root/target/native/libgauss_cuda.so"
printf 'export GAUSS_CUDA_LIBRARY=%q\n' "$gauss_root/target/native/libgauss_cuda.so"
