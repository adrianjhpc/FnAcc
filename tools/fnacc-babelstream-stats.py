#!/usr/bin/env python3
"""Run BabelStream repeatedly and report robust cross-run statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


KERNEL_ORDER = ("Copy", "Mul", "Add", "Triad", "Dot")
RESULT_RE = re.compile(
    r"^\s*(Copy|Mul|Add|Triad|Dot)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$"
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a BabelStream executable repeatedly and summarize each "
            "kernel's bandwidth, timing spread, and robust outliers."
        )
    )
    parser.add_argument("--arraysize", type=positive_int, default=33554432)
    parser.add_argument("--numtimes", type=positive_int, default=100)
    parser.add_argument("--runs", type=positive_int, default=7)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--device", type=nonnegative_int, default=0,
                        help="FNACC_CUDA_DEVICE value (default: 0)")
    parser.add_argument(
        "--cuda-visible-devices",
        help="optional CUDA_VISIBLE_DEVICES value; otherwise preserve the environment",
    )
    parser.add_argument("--timeout", type=positive_int, default=600,
                        help="per-process timeout in seconds (default: 600)")
    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="also write machine-readable results to FILE",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set an additional environment variable; may be repeated",
    )
    parser.add_argument("executable", type=Path)
    parser.add_argument(
        "executable_args",
        nargs=argparse.REMAINDER,
        help="extra BabelStream arguments (place them after --)",
    )
    args = parser.parse_args()
    if args.executable_args[:1] == ["--"]:
        args.executable_args = args.executable_args[1:]
    return args


def parse_environment(assignments: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --env value {assignment!r}; expected NAME=VALUE")
        result[name] = value
    return result


def parse_results(output: str) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        match = RESULT_RE.match(line)
        if not match:
            continue
        name, bandwidth, minimum, maximum, average = match.groups()
        results[name] = {
            "bandwidth_mb_s": float(bandwidth),
            "min_seconds": float(minimum),
            "max_seconds": float(maximum),
            "average_seconds": float(average),
        }

    missing = [name for name in KERNEL_ORDER if name not in results]
    if missing:
        raise ValueError(
            "could not parse BabelStream result rows for: " + ", ".join(missing)
        )
    return results


def run_benchmark(
    command: list[str], environment: dict[str, str], timeout: int
) -> dict[str, dict[str, float]]:
    try:
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"benchmark timed out after {timeout} seconds") from error

    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        sys.stderr.write(combined)
        raise RuntimeError(
            f"benchmark exited with status {completed.returncode}: "
            f"{shlex.join(command)}"
        )

    try:
        return parse_results(combined)
    except ValueError:
        sys.stderr.write(combined)
        raise


def robust_outlier_indices(values: list[float]) -> list[int]:
    """Return zero-based indices whose modified z-score exceeds 3.5."""
    if len(values) < 3:
        return []

    center = statistics.median(values)
    deviations = [abs(value - center) for value in values]
    mad = statistics.median(deviations)
    if mad > 0.0:
        return [
            index
            for index, value in enumerate(values)
            if abs(0.6744897501960817 * (value - center) / mad) > 3.5
        ]

    if len(values) < 4:
        return []

    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr == 0.0:
        return [index for index, value in enumerate(values) if value != center]

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [
        index for index, value in enumerate(values) if value < lower or value > upper
    ]


def summarize(values: list[float]) -> dict[str, Any]:
    minimum = min(values)
    maximum = max(values)
    median = statistics.median(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    spread = maximum - minimum
    return {
        "count": len(values),
        "min": minimum,
        "median": median,
        "mean": mean,
        "max": maximum,
        "spread": spread,
        "spread_percent_of_median": 100.0 * spread / median if median else math.inf,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation_percent": (
            100.0 * standard_deviation / mean if mean else math.inf
        ),
    }


def build_summary(
    runs: list[dict[str, dict[str, float]]]
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for kernel in KERNEL_ORDER:
        bandwidth = [run[kernel]["bandwidth_mb_s"] for run in runs]
        minimum_time = [run[kernel]["min_seconds"] for run in runs]
        average_time = [run[kernel]["average_seconds"] for run in runs]
        outliers = robust_outlier_indices(bandwidth)
        summary[kernel] = {
            "bandwidth_mb_s": summarize(bandwidth),
            "min_seconds": summarize(minimum_time),
            "average_seconds": summarize(average_time),
            "bandwidth_outlier_runs": [index + 1 for index in outliers],
        }
    return summary


def print_run_table(runs: list[dict[str, dict[str, float]]]) -> None:
    print("\nPer-run bandwidth (MBytes/sec)")
    print(f"{'Run':>4}" + "".join(f" {kernel:>14}" for kernel in KERNEL_ORDER))
    for index, run in enumerate(runs, start=1):
        values = "".join(
            f" {run[kernel]['bandwidth_mb_s']:14.3f}" for kernel in KERNEL_ORDER
        )
        print(f"{index:4d}{values}")


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\nCross-run summary")
    print(
        f"{'Kernel':<8} {'BW min':>13} {'BW median':>13} {'BW mean':>13} "
        f"{'BW max':>13} {'Spread%':>9} {'CV%':>8} "
        f"{'Median ms':>11} {'Outlier runs':>14}"
    )
    for kernel in KERNEL_ORDER:
        kernel_summary = summary[kernel]
        bandwidth = kernel_summary["bandwidth_mb_s"]
        minimum_time = kernel_summary["min_seconds"]
        outliers = kernel_summary["bandwidth_outlier_runs"]
        outlier_text = ",".join(str(value) for value in outliers) or "-"
        print(
            f"{kernel:<8} {bandwidth['min']:13.3f} "
            f"{bandwidth['median']:13.3f} {bandwidth['mean']:13.3f} "
            f"{bandwidth['max']:13.3f} "
            f"{bandwidth['spread_percent_of_median']:9.2f} "
            f"{bandwidth['coefficient_of_variation_percent']:8.2f} "
            f"{minimum_time['median'] * 1000.0:11.4f} {outlier_text:>14}"
        )


def main() -> int:
    args = parse_args()
    executable = args.executable.expanduser().resolve()
    if not executable.is_file():
        print(f"error: executable not found: {executable}", file=sys.stderr)
        return 2
    if not os.access(executable, os.X_OK):
        print(f"error: file is not executable: {executable}", file=sys.stderr)
        return 2

    try:
        additional_environment = parse_environment(args.env)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.update(additional_environment)
    environment["FNACC_CUDA_DEVICE"] = str(args.device)
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    command = [
        str(executable),
        "--arraysize",
        str(args.arraysize),
        "--numtimes",
        str(args.numtimes),
        *args.executable_args,
    ]

    print(f"Command: {shlex.join(command)}")
    print(
        f"Device: FNACC_CUDA_DEVICE={args.device}; warmups={args.warmups}; "
        f"measured runs={args.runs}"
    )

    try:
        for index in range(args.warmups):
            print(f"warmup {index + 1}/{args.warmups}", file=sys.stderr)
            run_benchmark(command, environment, args.timeout)

        runs: list[dict[str, dict[str, float]]] = []
        for index in range(args.runs):
            print(f"measured run {index + 1}/{args.runs}", file=sys.stderr)
            runs.append(run_benchmark(command, environment, args.timeout))
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = build_summary(runs)
    print_run_table(runs)
    print_summary(summary)

    if args.json:
        document = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "environment": {
                "FNACC_CUDA_DEVICE": environment["FNACC_CUDA_DEVICE"],
                "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
            },
            "arraysize": args.arraysize,
            "numtimes": args.numtimes,
            "warmups": args.warmups,
            "runs": runs,
            "summary": summary,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON results to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
