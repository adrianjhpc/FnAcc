#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if (( $# > 1 )); then
  echo "usage: $0 [fnacc-reduction-validation]" >&2
  exit 2
fi

executable="${1:-${FNACC_REDUCTION_TEST_EXECUTABLE:-}}"

if [[ -z "$executable" ]]; then
  candidates=(
    "$PWD/build/tests/reduction/fnacc-reduction-validation"
    "$PWD/build/reduction/fnacc-reduction-validation"
    "$PWD/fnacc-reduction-validation"
    "$script_dir/build/fnacc-reduction-validation"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      executable="$candidate"
      break
    fi
  done
fi

[[ -n "$executable" && -x "$executable" ]] || {
  echo "error: the prebuilt reduction test executable was not found" >&2
  echo "build it with CMake, then pass its path to this script" >&2
  exit 1
}

if [[ "$executable" != /* ]]; then
  executable="$PWD/$executable"
fi

temp_parent="${TMPDIR:-/tmp}"
if [[ ! -d "$temp_parent" || ! -w "$temp_parent" ]]; then
  temp_parent="$PWD"
fi
log_file="$(mktemp "$temp_parent/fnacc-reduction-test.XXXXXX.log")"
trap 'rm -f -- "$log_file"' EXIT

set +e
FNACC_REDUCTION_STATS=1 "$executable" >"$log_file" 2>&1
status=$?
set -e

cat "$log_file"

if (( status != 0 )); then
  echo "error: reduction validation exited with status $status" >&2
  exit "$status"
fi

grep -q '^FNACC reduction validation: PASS$' "$log_file" || {
  echo "error: validation did not report PASS" >&2
  exit 1
}

grep -q '^FNACC reduction workspace:' "$log_file" || {
  echo "error: runtime did not print the requested workspace statistics" >&2
  exit 1
}
