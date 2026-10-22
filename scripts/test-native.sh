#!/usr/bin/env bash
# A reproducible local gate, with no downloads or global installation.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
mkdir -p target/native
cc=${CC:-cc}
common=(-std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror)
"$cc" "${common[@]}" -O2 -shared -fPIC runtime.c -lm -pthread -ldl -o target/native/libgauss_test.so
"$cc" "${common[@]}" -O2 tests/native.c -lm -pthread -ldl -o target/native/smoke
target/native/smoke "$@"
python3 tests/numerical.py "$@"
"$cc" "${common[@]}" -O2 tests/failures.c -lm -pthread -ldl -o target/native/failures
target/native/failures
"$cc" "${common[@]}" -O2 tests/reductions.c -lm -pthread -ldl -o target/native/reductions-test
target/native/reductions-test
