#!/usr/bin/env python3

import argparse
import csv
import os
import random
import re
import subprocess
import sys
from pathlib import Path


CSV_HEADER = [
    "benchmark",
    "backend",
    "target",
    "exe",
    "name",
    "n",
    "m",
    "reps",
    "avg_seconds",
    "rate",
    "errors",
]


def is_executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def runner_target_path(runner: Path):
    """Return path used by `exec "..." "$@"` in a generated .run wrapper."""
    try:
        text = runner.read_text()
    except OSError:
        return None

    match = re.search(r'^\s*exec\s+"([^"]+)"\s+"\$@"', text, re.MULTILINE)
    if not match:
        return None

    return Path(match.group(1))


def runner_is_usable(runner: Path) -> bool:
    if not is_executable(runner):
        return False

    target = runner_target_path(runner)
    if target is None:
        return False

    return is_executable(target)


def find_fnacc_side_files(exe: Path):
    """Find local FNACC PTX/JSON files next to the direct executable.

    fnacc-flang generates files like:

      fnacc-vector-add.kernels.ptx
      fnacc-vector-add.kernels.json

    in the same directory as the executable.
    """
    directory = exe.parent
    stem = exe.name

    ptx = directory / f"{stem}.kernels.ptx"
    json = directory / f"{stem}.kernels.json"

    if ptx.exists() and json.exists():
        return ptx, json

    # Fallback: tolerate alternative naming if needed.
    ptx_candidates = sorted(directory.glob("*.kernels.ptx"))
    json_candidates = sorted(directory.glob("*.kernels.json"))

    if len(ptx_candidates) == 1 and len(json_candidates) == 1:
        return ptx_candidates[0], json_candidates[0]

    return None, None


def choose_exe(path: Path, backend: str):
    """Choose executable path and return metadata.

    For FnACC:
      - prefer .run only if it is not stale;
      - otherwise use the direct executable and set FNACC_PTX/JSON later.
    """
    uses_runner = False

    if backend == "fnacc":
        runner = Path(str(path) + ".run")
        if runner_is_usable(runner):
            return runner, True

        if runner.exists():
            target = runner_target_path(runner)
            print(
                f"warning: ignoring stale FnACC runner {runner}; "
                f"wrapper target is {target}",
                flush=True,
            )

    return path, uses_runner


def format_cmd(cmd):
    return " ".join(str(x) for x in cmd)


def run(cmd, env=None, timeout=None, fail_fast=False):
    print("+", format_cmd(cmd), flush=True)

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"benchmark timed out after {timeout}s: {format_cmd(cmd)}"
        print(f"warning: {msg}", flush=True)

        if exc.stdout:
            if isinstance(exc.stdout, bytes):
                sys.stdout.write(exc.stdout.decode(errors="replace"))
            else:
                sys.stdout.write(exc.stdout)

        if exc.stderr:
            if isinstance(exc.stderr, bytes):
                sys.stderr.write(exc.stderr.decode(errors="replace"))
            else:
                sys.stderr.write(exc.stderr)

        if fail_fast:
            raise RuntimeError(msg)

        return []

    if proc.stdout:
        print(proc.stdout, end="")

    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0:
        msg = (
            f"benchmark failed with exit code {proc.returncode}: "
            + format_cmd(cmd)
        )
        if fail_fast:
            raise RuntimeError(msg)

        print(f"warning: {msg}", flush=True)
        return []

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            continue
        if line.startswith("["):
            continue
        rows.append(line)

    return rows


def parse_sizes(text):
    result = []
    for item in text.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


def parse_matrix_sizes(text):
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        lower = item.lower()
        if "x" in lower:
            lhs, rhs = lower.split("x", 1)
            result.append((int(lhs), int(rhs)))
        else:
            n = int(item)
            result.append((n, n))

    return result


def maybe_add(targets, benchmark, backend, name, exe):
    chosen_exe, uses_runner = choose_exe(exe, backend)

    if is_executable(chosen_exe):
        targets.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "name": name,
                "exe": chosen_exe,
                "direct_exe": exe,
                "uses_runner": uses_runner,
            }
        )
    else:
        print(
            f"warning: missing or non-executable executable for {name}: "
            f"{chosen_exe}"
        )


def collect_targets(
    build,
    include_cuda,
    include_openmp,
    include_openmp_gpu,
    include_openacc,
):
    targets_1d = []
    targets_2d = []

    # ------------------------------------------------------------------ #
    # FnACC
    # ------------------------------------------------------------------ #

    maybe_add(
        targets_1d,
        "vector_add",
        "fnacc",
        "fnacc_vector_add",
        build / "benchmarks/fnacc/fnacc-vector-add/fnacc-vector-add",
    )
    maybe_add(
        targets_1d,
        "saxpy",
        "fnacc",
        "fnacc_saxpy",
        build / "benchmarks/fnacc/fnacc-saxpy/fnacc-saxpy",
    )
    maybe_add(
        targets_1d,
        "axpby",
        "fnacc",
        "fnacc_axpby",
        build / "benchmarks/fnacc/fnacc-axpby/fnacc-axpby",
    )
    maybe_add(
        targets_2d,
        "matrix_add_2d",
        "fnacc",
        "fnacc_matrix_add_2d",
        build / "benchmarks/fnacc/fnacc-matrix-add-2d/fnacc-matrix-add-2d",
    )
    maybe_add(
        targets_2d,
        "matmul_2d",
        "fnacc",
        "fnacc_matmul_2d",
        build / "benchmarks/fnacc/fnacc-matmul-2d/fnacc-matmul-2d",
    )

    # ------------------------------------------------------------------ #
    # CUDA
    # ------------------------------------------------------------------ #

    if include_cuda:
        maybe_add(
            targets_1d,
            "vector_add",
            "cuda",
            "cuda_vector_add",
            build / "benchmarks/cuda/cuda-vector-add",
        )
        maybe_add(
            targets_1d,
            "saxpy",
            "cuda",
            "cuda_saxpy",
            build / "benchmarks/cuda/cuda-saxpy",
        )
        maybe_add(
            targets_1d,
            "axpby",
            "cuda",
            "cuda_axpby",
            build / "benchmarks/cuda/cuda-axpby",
        )
        maybe_add(
            targets_2d,
            "matrix_add_2d",
            "cuda",
            "cuda_matrix_add_2d",
            build / "benchmarks/cuda/cuda-matrix-add-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "cuda",
            "cuda_matmul_2d",
            build / "benchmarks/cuda/cuda-matmul-2d",
        )

    # ------------------------------------------------------------------ #
    # OpenMP CPU
    # ------------------------------------------------------------------ #

    if include_openmp:
        maybe_add(
            targets_1d,
            "vector_add",
            "openmp",
            "openmp_vector_add",
            build / "benchmarks/openmp/openmp-vector-add",
        )
        maybe_add(
            targets_1d,
            "saxpy",
            "openmp",
            "openmp_saxpy",
            build / "benchmarks/openmp/openmp-saxpy",
        )
        maybe_add(
            targets_1d,
            "axpby",
            "openmp",
            "openmp_axpby",
            build / "benchmarks/openmp/openmp-axpby",
        )
        maybe_add(
            targets_2d,
            "matrix_add_2d",
            "openmp",
            "openmp_matrix_add_2d",
            build / "benchmarks/openmp/openmp-matrix-add-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "openmp",
            "openmp_matmul_2d",
            build / "benchmarks/openmp/openmp-matmul-2d",
        )

    # ------------------------------------------------------------------ #
    # OpenMP GPU
    # ------------------------------------------------------------------ #

    if include_openmp_gpu:
        maybe_add(
            targets_1d,
            "vector_add",
            "openmp_gpu",
            "openmp_gpu_vector_add",
            build / "benchmarks/openmp_gpu/openmp-gpu-vector-add",
        )
        maybe_add(
            targets_1d,
            "saxpy",
            "openmp_gpu",
            "openmp_gpu_saxpy",
            build / "benchmarks/openmp_gpu/openmp-gpu-saxpy",
        )
        maybe_add(
            targets_1d,
            "axpby",
            "openmp_gpu",
            "openmp_gpu_axpby",
            build / "benchmarks/openmp_gpu/openmp-gpu-axpby",
        )
        maybe_add(
            targets_2d,
            "matrix_add_2d",
            "openmp_gpu",
            "openmp_gpu_matrix_add_2d",
            build / "benchmarks/openmp_gpu/openmp-gpu-matrix-add-2d",
        )

    # ------------------------------------------------------------------ #
    # OpenACC
    # ------------------------------------------------------------------ #

    if include_openacc:
        maybe_add(
            targets_1d,
            "vector_add",
            "openacc",
            "openacc_vector_add",
            build / "benchmarks/openacc/openacc-vector-add",
        )
        maybe_add(
            targets_1d,
            "saxpy",
            "openacc",
            "openacc_saxpy",
            build / "benchmarks/openacc/openacc-saxpy",
        )
        maybe_add(
            targets_1d,
            "axpby",
            "openacc",
            "openacc_axpby",
            build / "benchmarks/openacc/openacc-axpby",
        )
        maybe_add(
            targets_2d,
            "matrix_add_2d",
            "openacc",
            "openacc_matrix_add_2d",
            build / "benchmarks/openacc/openacc-matrix-add-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "openacc",
            "openacc_matmul_2d",
            build / "benchmarks/openacc/openacc-matmul-2d",
        )

    return targets_1d, targets_2d


def filter_targets(targets, only_backends, only_benchmarks):
    if only_backends:
        allowed = set(only_backends)
        targets = [target for target in targets if target["backend"] in allowed]

    if only_benchmarks:
        allowed = set(only_benchmarks)
        targets = [target for target in targets if target["benchmark"] in allowed]

    return targets


def split_result_row(row):
    parts = [x.strip() for x in next(csv.reader([row]))]
    if len(parts) != 7:
        raise ValueError(
            f"expected 7 CSV columns from benchmark, got {len(parts)}: {row}"
        )
    return parts


def build_tasks(
    targets_1d,
    targets_2d,
    sizes,
    matrix_sizes,
    reps,
    matrix_reps,
    skip_1d,
    skip_2d,
):
    tasks = []

    if not skip_1d:
        for size in sizes:
            for target in targets_1d:
                tasks.append(
                    {
                        "kind": "1d",
                        "target": target,
                        "cmd": [str(target["exe"]), str(size), str(reps)],
                    }
                )

    if not skip_2d:
        for n, m in matrix_sizes:
            for target in targets_2d:
                tasks.append(
                    {
                        "kind": "2d",
                        "target": target,
                        "cmd": [
                            str(target["exe"]),
                            str(n),
                            str(m),
                            str(matrix_reps),
                        ],
                    }
                )

    return tasks


def env_for_target(base_env, target):
    env = base_env.copy()

    # Avoid container locale warnings from stale/incomplete locale setup.
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    if target["backend"] == "fnacc" and not target.get("uses_runner", False):
        ptx, json = find_fnacc_side_files(target["direct_exe"])

        if ptx and json:
            env.setdefault("FNACC_PTX", str(ptx.resolve()))
            env.setdefault("FNACC_KERNELS_JSON", str(json.resolve()))
        else:
            print(
                f"warning: could not find local FNACC PTX/JSON side files "
                f"for {target['direct_exe']}; relying on embedded payload or "
                f"default runtime lookup",
                flush=True,
            )

    return env


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run FnACC/CUDA/OpenMP/OpenMP-GPU/OpenACC benchmarks and collect "
            "CSV results."
        )
    )

    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", default="bench-results.csv")

    parser.add_argument(
        "--sizes",
        default="1024,10000,100000,1000000,10000000",
        help="Comma-separated 1-D sizes.",
    )
    parser.add_argument(
        "--matrix-sizes",
        default="64x64,256x256,1024x1024",
        help=(
            "Comma-separated 2-D sizes, e.g. 64x64,256x256,1024x1024. "
            "A single number means NxN."
        ),
    )
    parser.add_argument("--reps", default="100")
    parser.add_argument("--matrix-reps", default=None)

    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--include-openmp", action="store_true")
    parser.add_argument("--include-openmp-gpu", action="store_true")
    parser.add_argument("--include-openacc", action="store_true")

    parser.add_argument("--skip-1d", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")

    parser.add_argument(
        "--only-backend",
        action="append",
        default=[],
        help=(
            "Run only this backend. May be repeated. "
            "Examples: fnacc, cuda, openmp, openmp_gpu, openacc."
        ),
    )
    parser.add_argument(
        "--only-benchmark",
        action="append",
        default=[],
        help=(
            "Run only this benchmark. May be repeated. "
            "Examples: vector_add, saxpy, axpby, matrix_add_2d, matmul_2d."
        ),
    )

    parser.add_argument("--fnacc-profile", action="store_true")
    parser.add_argument("--fnacc-debug", action="store_true")
    parser.add_argument("--cuda-launch-blocking", action="store_true")

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-command timeout in seconds.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failed benchmark command.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle benchmark task order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List discovered benchmark targets and exit.",
    )

    args = parser.parse_args()

    build = Path(args.build_dir)
    sizes = parse_sizes(args.sizes)
    matrix_sizes = parse_matrix_sizes(args.matrix_sizes)
    reps = int(args.reps)
    matrix_reps = int(args.matrix_reps) if args.matrix_reps else reps

    targets_1d, targets_2d = collect_targets(
        build,
        include_cuda=args.include_cuda,
        include_openmp=args.include_openmp,
        include_openmp_gpu=args.include_openmp_gpu,
        include_openacc=args.include_openacc,
    )

    targets_1d = filter_targets(
        targets_1d,
        only_backends=args.only_backend,
        only_benchmarks=args.only_benchmark,
    )
    targets_2d = filter_targets(
        targets_2d,
        only_backends=args.only_backend,
        only_benchmarks=args.only_benchmark,
    )

    if args.list_targets:
        print("1-D targets:")
        for target in targets_1d:
            mode = "runner" if target.get("uses_runner", False) else "direct"
            print(
                f"  {target['backend']:12s} "
                f"{target['benchmark']:16s} "
                f"{target['name']:28s} "
                f"{mode:8s} "
                f"{target['exe']}"
            )

        print("2-D targets:")
        for target in targets_2d:
            mode = "runner" if target.get("uses_runner", False) else "direct"
            print(
                f"  {target['backend']:12s} "
                f"{target['benchmark']:16s} "
                f"{target['name']:28s} "
                f"{mode:8s} "
                f"{target['exe']}"
            )

        return

    base_env = os.environ.copy()

    # Also set these in the base env so non-FnACC commands inherit the clean
    # locale too.
    base_env["LC_ALL"] = "C"
    base_env["LANG"] = "C"

    if args.fnacc_profile:
        base_env["FNACC_PROFILE"] = "1"

    if args.fnacc_debug:
        base_env["FNACC_DEBUG"] = "1"

    if args.cuda_launch_blocking:
        base_env["CUDA_LAUNCH_BLOCKING"] = "1"

    tasks = build_tasks(
        targets_1d=targets_1d,
        targets_2d=targets_2d,
        sizes=sizes,
        matrix_sizes=matrix_sizes,
        reps=reps,
        matrix_reps=matrix_reps,
        skip_1d=args.skip_1d,
        skip_2d=args.skip_2d,
    )

    if args.shuffle:
        random.Random(args.seed).shuffle(tasks)

    output_rows = []

    for task in tasks:
        target = task["target"]
        env = env_for_target(base_env, target)

        rows = run(
            task["cmd"],
            env=env,
            timeout=args.timeout,
            fail_fast=args.fail_fast,
        )

        for row in rows:
            parts = split_result_row(row)
            output_rows.append(
                [
                    target["benchmark"],
                    target["backend"],
                    target["name"],
                    str(target["exe"]),
                    *parts,
                ]
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(output_rows)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

