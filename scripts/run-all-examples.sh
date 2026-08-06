#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$ROOT_DIR/bin/fnacc-flang"

if [[ ! -x "$WRAPPER" ]]; then
  echo "error: wrapper is not executable: $WRAPPER" >&2
  echo "run: chmod +x $WRAPPER" >&2
  exit 1
fi

run_example() {
  local name="$1"
  local dir="$ROOT_DIR/examples/$name"

  echo
  echo "=== building $name ==="

  "$WRAPPER" \
    --kernel-src "$dir/kernel.f90" \
    --main-src "$dir/main.f90" \
    --workdir "$ROOT_DIR/build/$name" \
    --base "$name" \
    -o "$ROOT_DIR/build/$name/$name"

  echo
  echo "=== running $name ==="

  "$ROOT_DIR/build/$name/$name.run"
}

mkdir -p "$ROOT_DIR/build"

run_example vector-add
run_example saxpy
run_example axpby
run_example matrix-add-2d
run_example module-vector-add
run_example assumed-shape-matrix-add-2d

echo
echo "all FNACC examples passed"

