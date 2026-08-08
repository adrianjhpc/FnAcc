#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
from pathlib import Path


CSV_HEADER = [
    "benchmark",
    "backend",
    "name",
    "n",
    "m",
    "reps",
    "avg_seconds",
    "rate",
    "errors",
]


def run(cmd, env=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    if proc.stdout:
        print(proc.stdout, end="")

    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0:
        print(
            f"warning: benchmark failed with exit code {proc.returncode}: "
            + " ".join(str(x) for x in cmd),
            flush=True,
        )
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
    return [int(x) for x in text.split(",") if x.strip()]


def parse_matrix_sizes(text):
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        if "x" in item:
            lhs, rhs = item.lower().split("x", 1)
            result.append((int(lhs), int(rhs)))
        else:
            n = int(item)
            result.append((n, n))

    return result


def maybe_add(targets, benchmark, backend, name, exe):
    if exe.exists():
        targets.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "name": name,
                "exe": exe,
            }
        )
    else:
        print(f"warning: missing executable for {name}: {exe}")


def collect_targets(build, include_cuda, include_openmp, include_openacc):
    targets_1d = []
    targets_2d = []

    # FNACC always included.
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


def split_result_row(row):
    parts = [x.strip() for x in row.split(",")]
    if len(parts) != 7:
        raise ValueError(f"expected 7 CSV columns from benchmark, got {len(parts)}: {row}")
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", default="bench-results.csv")
    parser.add_argument("--sizes", default="1024,10000,100000,1000000,10000000")
    parser.add_argument("--matrix-sizes", default="64x64,256x256,1024x1024")
    parser.add_argument("--reps", default="100")
    parser.add_argument("--matrix-reps", default=None)
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--include-openmp", action="store_true")
    parser.add_argument("--include-openacc", action="store_true")
    parser.add_argument("--skip-1d", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--fnacc-profile", action="store_true")
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
        include_openacc=args.include_openacc,
    )

    env = os.environ.copy()
    if args.fnacc_profile:
        env["FNACC_PROFILE"] = "1"

    output_rows = []

    if not args.skip_1d:
        for size in sizes:
            for target in targets_1d:
                rows = run([str(target["exe"]), str(size), str(reps)], env=env)
                for row in rows:
                    parts = split_result_row(row)
                    output_rows.append(
                        [
                            target["benchmark"],
                            target["backend"],
                            *parts,
                        ]
                    )

    if not args.skip_2d:
        for n, m in matrix_sizes:
            for target in targets_2d:
                rows = run(
                    [str(target["exe"]), str(n), str(m), str(matrix_reps)],
                    env=env,
                )
                for row in rows:
                    parts = split_result_row(row)
                    output_rows.append(
                        [
                            target["benchmark"],
                            target["backend"],
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

