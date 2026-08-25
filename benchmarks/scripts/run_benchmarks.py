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

    Supported layouts:

      Legacy single-PTX:
        foo.kernels.ptx
        foo.kernels.json

      Per-kernel PTX:
        foo.kernels.json
        foo.kernels.split/
          fnacc_kernel_0.ptx
          fnacc_kernel_1.ptx
    """
    directory = exe.parent
    stem = exe.name

    json = directory / f"{stem}.kernels.json"
    ptx = directory / f"{stem}.kernels.ptx"
    ptx_dir = directory / f"{stem}.kernels.split"

    if json.exists() and ptx_dir.is_dir():
        return {
            "json": json,
            "ptx": None,
            "ptx_dir": ptx_dir,
        }

    if json.exists() and ptx.exists():
        return {
            "json": json,
            "ptx": ptx,
            "ptx_dir": None,
        }

    json_candidates = sorted(directory.glob("*.kernels.json"))
    ptx_dir_candidates = sorted(
        p for p in directory.glob("*.kernels.split") if p.is_dir()
    )
    ptx_candidates = sorted(directory.glob("*.kernels.ptx"))

    if len(json_candidates) == 1 and len(ptx_dir_candidates) == 1:
        return {
            "json": json_candidates[0],
            "ptx": None,
            "ptx_dir": ptx_dir_candidates[0],
        }

    if len(json_candidates) == 1 and len(ptx_candidates) == 1:
        return {
            "json": json_candidates[0],
            "ptx": ptx_candidates[0],
            "ptx_dir": None,
        }

    return {
        "json": None,
        "ptx": None,
        "ptx_dir": None,
    }


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

def parse_cublas_modes(text):
    result = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in {"tf32", "fp32"}:
            raise ValueError(
                f"invalid cuBLAS mode '{item}'; expected 'tf32' or 'fp32'"
            )
        result.append(item)
    return result

def maybe_add(targets, benchmark, backend, name, exe, extra_args=None):
    chosen_exe, uses_runner = choose_exe(exe, backend)

    if extra_args is None:
        extra_args = []

    if is_executable(chosen_exe):
        targets.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "name": name,
                "exe": chosen_exe,
                "direct_exe": exe,
                "uses_runner": uses_runner,
                "extra_args": list(extra_args),
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
    include_cublas,
    cublas_modes,
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
        "vector_add_f64",
        "fnacc",
        "fnacc_vector_add_f64",
        build / "benchmarks/fnacc/fnacc-vector-add-f64/fnacc-vector-add-f64",
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
        "daxpy",
        "fnacc",
        "fnacc_daxpy",
        build / "benchmarks/fnacc/fnacc-daxpy/fnacc-daxpy",
    )
    maybe_add(
        targets_1d,
        "axpby",
        "fnacc",
        "fnacc_axpby",
        build / "benchmarks/fnacc/fnacc-axpby/fnacc-axpby",
    )
    maybe_add(
        targets_1d,
        "axpby_f64",
        "fnacc",
        "fnacc_axpby_f64",
        build / "benchmarks/fnacc/fnacc-axpby-f64/fnacc-axpby-f64",
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
        "matrix_add_2d_f64",
        "fnacc",
        "fnacc_matrix_add_2d_f64",
        build / "benchmarks/fnacc/fnacc-matrix-add-2d/fnacc-matrix-add-2d-f64",
    )
    maybe_add(
        targets_2d,
        "matmul_2d",
        "fnacc",
        "fnacc_matmul_2d",
        build / "benchmarks/fnacc/fnacc-matmul-2d/fnacc-matmul-2d",
    )
    maybe_add(
        targets_2d,
        "matmul_2d_f64",
        "fnacc",
        "fnacc_matmul_2d_f64",
        build / "benchmarks/fnacc/fnacc-matmul-2d-f64/fnacc-matmul-2d-f64",
    )
    maybe_add(
        targets_1d,
        "reduction_dot",
        "fnacc",
        "fnacc_reduction_dot",
        build / "benchmarks/fnacc/fnacc-reduction-dot/fnacc-reduction-dot",
    )
    maybe_add(
        targets_1d,
        "reduction_sum",
        "fnacc",
        "fnacc_reduction_sum",
        build / "benchmarks/fnacc/fnacc-reduction-sum/fnacc-reduction-sum",
    )
    maybe_add(
        targets_1d,
        "reduction_dot_f64",
        "fnacc",
        "fnacc_reduction_dot_f64",
        build / "benchmarks/fnacc/fnacc-reduction-dot-f64/fnacc-reduction-dot-f64",
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
            "vector_add_f64",
            "cuda",
            "cuda_vector_add_f64",
            build / "benchmarks/cuda/cuda-vector-add-f64",
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
            "daxpy",
            "cuda",
            "cuda_daxpy",
            build / "benchmarks/cuda/cuda-daxpy",
        )
        maybe_add(
            targets_1d,
            "axpby",
            "cuda",
            "cuda_axpby",
            build / "benchmarks/cuda/cuda-axpby",
        )
        maybe_add(
            targets_1d,
            "axpby_f64",
            "cuda",
            "cuda_axpby_f64",
            build / "benchmarks/cuda/cuda-axpby-f64",
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
            "matrix_add_2d_f64",
            "cuda",
            "cuda_matrix_add_2d_f64",
            build / "benchmarks/cuda/cuda-matrix-add-2d-f64",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "cuda",
            "cuda_matmul_2d",
            build / "benchmarks/cuda/cuda-matmul-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d_f64",
            "cuda",
            "cuda_matmul_2d_f64",
            build / "benchmarks/cuda/cuda-matmul-2d-f64",
        )
        maybe_add(
            targets_1d,
            "reduction_sum",
            "cuda",
            "cuda_reduction_sum",
            build / "benchmarks/cuda/cuda-reduction-sum",
        )
        maybe_add(
            targets_1d,
            "reduction_dot",
            "cuda",
            "cuda_reduction_dot",
            build / "benchmarks/cuda/cuda-reduction-dot",
        )
        maybe_add(
            targets_1d,
            "reduction_dot_f64",
            "cuda",
            "cuda_reduction_dot_f64",
            build / "benchmarks/cuda/cuda-reduction-dot-f64",
        )
    # ------------------------------------------------------------------ #
    # CUDA cuBLAS
    # ------------------------------------------------------------------ #

    if include_cublas:
        # FP32 cuBLAS benchmark: tf32/fp32 modes are meaningful.
        for mode in cublas_modes:
            maybe_add(
                targets_2d,
                "matmul_2d",
                "cuda_cublas",
                f"cuda_cublas_matmul_2d_{mode}",
                build / "benchmarks/cuda/cuda-matmul-2d-cublas",
                extra_args=[mode],
            )
         # FP64 cuBLAS benchmark: true DGEMM only. No tf32/fp32 modes.
        maybe_add(
            targets_2d,
            "matmul_2d_f64",
            "cuda_cublas",
            "cuda_cublas_matmul_2d_f64",
            build / "benchmarks/cuda/cuda-matmul-2d-cublas-f64",
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
            "daxpy",
            "openmp",
            "openmp_daxpy",
            build / "benchmarks/openmp/openmp-daxpy",
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
            "matrix_add_2d_f64",
            "openmp",
            "openmp_matrix_add_2d_f64",
            build / "benchmarks/openmp/openmp-matrix-add-2d-f64",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "openmp",
            "openmp_matmul_2d",
            build / "benchmarks/openmp/openmp-matmul-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d_f64",
            "openmp",
            "openmp_matmul_2d_f64",
            build / "benchmarks/openmp/openmp-matmul-2d-f64",
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
            "daxpy",
            "openmp_gpu",
            "openmp_gpu_daxpy",
            build / "benchmarks/openmp_gpu/openmp-gpu-daxpy",
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
        maybe_add(
            targets_2d,
            "matrix_add_2d_f64",
            "openmp_gpu",
            "openmp_gpu_matrix_add_2d_f64",
            build / "benchmarks/openmp_gpu/openmp-gpu-matrix-add-2d-f64",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "openmp_gpu",
            "openmp_gpu_matmul_2d",
            build / "benchmarks/openmp_gpu/openmp-gpu-matmul-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d_f64",
            "openmp_gpu",
            "openmp_gpu_matmul_2d_f64",
            build / "benchmarks/openmp_gpu/openmp-gpu-matmul-2d-f64",
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
            "daxpy",
            "openacc",
            "openacc_daxpy",
            build / "benchmarks/openacc/openacc-daxpy",
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
            "matrix_add_2d_f64",
            "openacc",
            "openacc_matrix_add_2d_f64",
            build / "benchmarks/openacc/openacc-matrix-add-2d-f64",
        )
        maybe_add(
            targets_2d,
            "matmul_2d",
            "openacc",
            "openacc_matmul_2d",
            build / "benchmarks/openacc/openacc-matmul-2d",
        )
        maybe_add(
            targets_2d,
            "matmul_2d_f64",
            "openacc",
            "openacc_matmul_2d_f64",
            build / "benchmarks/openacc/openacc-matmul-2d-f64",
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
                        "cmd": [
                            str(target["exe"]),
                            str(size),
                            str(reps),
                            *target.get("extra_args", []),
                        ],
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
                            *target.get("extra_args", []),
                        ],
                    }
                )

    return tasks


def env_for_target(base_env, target):
    env = base_env.copy()

    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    if target["backend"] == "fnacc" and not target.get("uses_runner", False):
        side = find_fnacc_side_files(target["direct_exe"])

        json = side.get("json")
        ptx = side.get("ptx")
        ptx_dir = side.get("ptx_dir")

        if json and ptx_dir:
            env.setdefault("FNACC_PTX_DIR", str(ptx_dir.resolve()))
            env.setdefault("FNACC_KERNELS_JSON", str(json.resolve()))
        elif json and ptx:
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
    parser.add_argument("--include-cublas", action="store_true")
    parser.add_argument(
        "--cublas-modes",
        default="tf32,fp32",
        help=(
            "Comma-separated cuBLAS matmul modes to run. "
            "Valid values: tf32,fp32. Default: tf32,fp32."
        ),
    )
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
            "Examples: fnacc, cuda, coda_cublas, openmp, openmp_gpu, openacc."
        ),
    )
    parser.add_argument(
        "--only-benchmark",
        action="append",
        default=[],
        help=(
            "Run only this benchmark. May be repeated. "
            "Examples: vector_add, saxpy, daxpy, axpby, matrix_add_2d, matmul_2d."
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
    cublas_modes = parse_cublas_modes(args.cublas_modes)

    targets_1d, targets_2d = collect_targets(
        build,
        include_cuda=args.include_cuda,
        include_cublas=args.include_cublas,
        cublas_modes=cublas_modes,
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
            extra = " ".join(target.get("extra_args", []))
            extra_suffix = f" {extra}" if extra else ""

            print(
              f"  {target['backend']:12s} "
              f"{target['benchmark']:16s} "
              f"{target['name']:32s} "
              f"{mode:8s} "
              f"{target['exe']}{extra_suffix}"
            )

        print("2-D targets:")
        for target in targets_2d:
            mode = "runner" if target.get("uses_runner", False) else "direct"
            extra = " ".join(target.get("extra_args", []))            
            extra_suffix = f" {extra}" if extra else ""

            print(
              f"  {target['backend']:12s} "
              f"{target['benchmark']:16s} "
              f"{target['name']:32s} "
              f"{mode:8s} "
              f"{target['exe']}{extra_suffix}"
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

