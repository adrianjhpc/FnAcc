#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from collections import defaultdict


KNOWN_TARGET_ORDER = [
    "fnacc_vector_add",
    "cuda_vector_add",
    "openmp_vector_add",
    "openmp_gpu_vector_add",
    "openacc_vector_add",

    "fnacc_saxpy",
    "cuda_saxpy",
    "openmp_saxpy",
    "openmp_gpu_saxpy",
    "openacc_saxpy",

    "fnacc_axpby",
    "cuda_axpby",
    "openmp_axpby",
    "openmp_gpu_axpby",
    "openacc_axpby",

    "fnacc_matrix_add_2d",
    "cuda_matrix_add_2d",
    "openmp_matrix_add_2d",
    "openmp_gpu_matrix_add_2d",
    "openacc_matrix_add_2d",

    "fnacc_matmul_2d",
    "cuda_matmul_2d",
    "cuda_cublas_matmul_2d_fp32",
    "cuda_cublas_matmul_2d_tf32",
    "openmp_matmul_2d",
    "openmp_gpu_matmul_2d",
    "openacc_matmul_2d",
]


KNOWN_BACKEND_ORDER = [
    "fnacc",
    "cuda",
    "cuda_cublas",
    "openmp",
    "openmp_gpu",
    "openacc",
]


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def parse_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def read_rows(path):
    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["n_int"] = parse_int(row.get("n", "0"))
            row["m_int"] = parse_int(row.get("m", "0"))
            row["reps_int"] = parse_int(row.get("reps", "0"))
            row["avg_seconds_float"] = parse_float(row.get("avg_seconds", "nan"))
            row["rate_float"] = parse_float(row.get("rate", "nan"))
            row["errors_int"] = parse_int(row.get("errors", "0"))
            rows.append(row)

    return rows


def size_key(row):
    n = row["n_int"]
    m = row["m_int"]

    if m == 1:
        return f"{n}"
    return f"{n}x{m}"


def size_sort_key(size):
    if "x" in size:
        lhs, rhs = size.split("x", 1)
        return (int(lhs), int(rhs))
    return (int(size), 1)


def target_label(row, mode):
    if mode == "backend":
        return row["backend"]

    if mode == "target":
        return row["target"]

    if mode == "name":
        return row["name"]

    if mode == "backend-target":
        return f"{row['backend']}:{row['target']}"

    raise ValueError(f"unknown label mode: {mode}")


def column_sort_key(label):
    if label in KNOWN_TARGET_ORDER:
        return (0, KNOWN_TARGET_ORDER.index(label), label)

    if label in KNOWN_BACKEND_ORDER:
        return (1, KNOWN_BACKEND_ORDER.index(label), label)

    # Helpful special handling for cuBLAS labels.
    if "cublas" in label and "fp32" in label:
        return (2, 0, label)
    if "cublas" in label and "tf32" in label:
        return (2, 1, label)

    return (9, 0, label)


def aggregate(values, method):
    values = [v for v in values if not math.isnan(v)]

    if not values:
        return math.nan

    if method == "last":
        return values[-1]

    if method == "mean":
        return statistics.mean(values)

    if method == "median":
        return statistics.median(values)

    if method == "min":
        return min(values)

    if method == "max":
        return max(values)

    raise ValueError(f"unknown aggregate method: {method}")


def format_value(value, metric):
    if value is None or math.isnan(value):
        return ""

    if metric == "errors":
        return str(int(round(value)))

    if metric == "avg_seconds":
        return f"{value:.6e}"

    # rate
    if abs(value) >= 1000:
        return f"{value:.2f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def markdown_table(headers, rows):
    out = []

    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def make_pivot_table(rows, benchmark, metric, label_mode, aggregate_method):
    metric_key = {
        "rate": "rate_float",
        "avg_seconds": "avg_seconds_float",
        "errors": "errors_int",
    }[metric]

    # data[(size, label)] -> list of values
    data = defaultdict(list)
    error_data = defaultdict(list)

    labels = set()
    sizes = set()

    for row in rows:
        if row["benchmark"] != benchmark:
            continue

        label = target_label(row, label_mode)
        size = size_key(row)

        labels.add(label)
        sizes.add(size)

        data[(size, label)].append(float(row[metric_key]))
        error_data[(size, label)].append(float(row["errors_int"]))

    labels = sorted(labels, key=column_sort_key)
    sizes = sorted(sizes, key=size_sort_key)

    headers = ["size"] + labels
    table_rows = []

    for size in sizes:
        out_row = [size]

        for label in labels:
            value = aggregate(data.get((size, label), []), aggregate_method)
            out_row.append(format_value(value, metric))

        table_rows.append(out_row)

    return headers, table_rows, labels, sizes, error_data


def make_relative_table(rows, benchmark, label_mode, aggregate_method, baseline_label):
    metric_key = "rate_float"

    data = defaultdict(list)
    labels = set()
    sizes = set()

    for row in rows:
        if row["benchmark"] != benchmark:
            continue

        label = target_label(row, label_mode)
        size = size_key(row)

        labels.add(label)
        sizes.add(size)
        data[(size, label)].append(float(row[metric_key]))

    labels = sorted(labels, key=column_sort_key)
    sizes = sorted(sizes, key=size_sort_key)

    if baseline_label not in labels:
        return None, None

    headers = ["size"] + labels
    table_rows = []

    for size in sizes:
        baseline = aggregate(data.get((size, baseline_label), []), aggregate_method)
        out_row = [size]

        for label in labels:
            value = aggregate(data.get((size, label), []), aggregate_method)

            if math.isnan(value) or math.isnan(baseline) or baseline == 0.0:
                out_row.append("")
            else:
                out_row.append(f"{100.0 * value / baseline:.1f}%")

        table_rows.append(out_row)

    return headers, table_rows


def validation_summary(rows):
    total = len(rows)
    bad = [r for r in rows if r["errors_int"] != 0]

    lines = []
    lines.append("## Validation summary")
    lines.append("")
    lines.append(f"- Total rows: `{total}`")
    lines.append(f"- Rows with nonzero errors: `{len(bad)}`")

    if bad:
        lines.append("")
        lines.append("| benchmark | target | size | errors |")
        lines.append("| --- | --- | --- | --- |")

        for row in bad:
            lines.append(
                f"| {row['benchmark']} | {row['target']} | "
                f"{size_key(row)} | {row['errors_int']} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FnACC benchmark CSV into Markdown summary tables."
    )

    parser.add_argument("csv_file")
    parser.add_argument(
        "--metric",
        choices=["rate", "avg_seconds", "errors"],
        default="rate",
        help="Metric to show in the main tables.",
    )
    parser.add_argument(
        "--label",
        choices=["target", "backend", "name", "backend-target"],
        default="target",
        help="Column label mode. Use 'target' to distinguish cuBLAS fp32/tf32.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["last", "mean", "median", "min", "max"],
        default="last",
        help="How to aggregate duplicate rows.",
    )
    parser.add_argument(
        "--only-benchmark",
        action="append",
        default=[],
        help="Only include this benchmark. May be repeated.",
    )
    parser.add_argument(
        "--only-backend",
        action="append",
        default=[],
        help="Only include this backend. May be repeated.",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help=(
            "Also emit relative-rate tables compared to this column label, "
            "e.g. cuda_saxpy or cuda_cublas_matmul_2d_fp32."
        ),
    )
    parser.add_argument(
        "--no-validation-summary",
        action="store_true",
        help="Do not print the nonzero-error summary.",
    )

    args = parser.parse_args()

    rows = read_rows(args.csv_file)

    if args.only_benchmark:
        allowed = set(args.only_benchmark)
        rows = [r for r in rows if r["benchmark"] in allowed]

    if args.only_backend:
        allowed = set(args.only_backend)
        rows = [r for r in rows if r["backend"] in allowed]

    benchmarks = sorted(set(r["benchmark"] for r in rows))

    if not args.no_validation_summary:
        print(validation_summary(rows))
        print("")

    metric_desc = {
        "rate": "Rate",
        "avg_seconds": "Average seconds",
        "errors": "Errors",
    }[args.metric]

    for benchmark in benchmarks:
        headers, table_rows, labels, sizes, error_data = make_pivot_table(
            rows,
            benchmark=benchmark,
            metric=args.metric,
            label_mode=args.label,
            aggregate_method=args.aggregate,
        )

        if not table_rows:
            continue

        print(f"## {benchmark}")
        print("")
        print(f"**Metric:** `{metric_desc}`")
        print("")
        print(markdown_table(headers, table_rows))
        print("")

        if args.relative_to and args.metric == "rate":
            rel_headers, rel_rows = make_relative_table(
                rows,
                benchmark=benchmark,
                label_mode=args.label,
                aggregate_method=args.aggregate,
                baseline_label=args.relative_to,
            )

            if rel_headers is not None:
                print(f"### {benchmark}: relative to `{args.relative_to}`")
                print("")
                print(markdown_table(rel_headers, rel_rows))
                print("")


if __name__ == "__main__":
    main()

