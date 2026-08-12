#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


TILE_RE = re.compile(
    r"(?P<prefix>!\$fnacc\s+parallel\s+tile\s*)"
    r"\("
    r"[^)]*"
    r"\)",
    re.IGNORECASE,
)


MANIFEST_FIELDS = [
    "config",
    "tile_m",
    "tile_n",
    "tile_k",
    "warps",
    "stages",
    "sm",
    "build_dir",
    "exe",
    "ptx",
    "json",
    "ptx_target",
    "ptx_has_mma",
    "ptx_has_fma",
    "compile_status",
]


RESULT_FIELDS = [
    "config",
    "tile_m",
    "tile_n",
    "tile_k",
    "warps",
    "stages",
    "sm",
    "n",
    "m",
    "reps",
    "trial",
    "name",
    "avg_seconds",
    "rate",
    "errors",
    "exe",
    "ptx",
    "json",
    "returncode",
    "status",
]


SUMMARY_FIELDS = [
    "config",
    "tile_m",
    "tile_n",
    "tile_k",
    "warps",
    "stages",
    "sm",
    "n",
    "m",
    "reps",
    "trials",
    "median_avg_seconds",
    "min_avg_seconds",
    "median_rate",
    "max_rate",
    "all_errors_zero",
    "errors_values",
    "exe",
    "ptx",
    "json",
]


def run_cmd(cmd, cwd=None, env=None, timeout=None, fail=False):
    print("+", " ".join(str(x) for x in cmd), flush=True)

    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if fail and proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: "
            + " ".join(str(x) for x in cmd)
        )

    return proc


def parse_tile_list(text):
    tiles = []

    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue

        parts = item.split("x")
        if len(parts) != 3:
            raise ValueError(f"bad tile shape '{item}', expected MxNxK")

        tiles.append(tuple(int(x) for x in parts))

    return tiles


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def config_name(tile, warps, stages):
    return f"tm{tile[0]}_tn{tile[1]}_tk{tile[2]}_w{warps}_s{stages}"


def patch_tile(source_path: Path, tile):
    text = source_path.read_text()

    replacement = f"\\g<prefix>({tile[0]}, {tile[1]}, {tile[2]})"
    new_text, count = TILE_RE.subn(replacement, text, count=1)

    if count != 1:
        raise RuntimeError(
            f"could not find exactly one !$fnacc parallel tile(...) line in "
            f"{source_path}; found {count}"
        )

    source_path.write_text(new_text)


def find_side_files(build_dir: Path):
    matmul_dir = build_dir / "benchmarks/fnacc/fnacc-matmul-2d"

    ptx_candidates = sorted(matmul_dir.glob("*.kernels.ptx"))
    json_candidates = sorted(matmul_dir.glob("*.kernels.json"))

    ptx = ptx_candidates[0] if ptx_candidates else Path("")
    json = json_candidates[0] if json_candidates else Path("")

    return ptx, json


def inspect_ptx(ptx: Path):
    if not ptx or not ptx.exists():
        return {
            "ptx_target": "",
            "ptx_has_mma": False,
            "ptx_has_fma": False,
        }

    text = ptx.read_text(errors="replace")
    lower = text.lower()

    target = ""
    for line in text.splitlines():
        if ".target" in line:
            target = line.strip()
            break

    return {
        "ptx_target": target,
        "ptx_has_mma": ("mma.sync" in lower or "wgmma" in lower),
        "ptx_has_fma": ("fma.rn.f32" in lower),
    }


def parse_benchmark_output(stdout):
    rows = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            continue
        if line.startswith("["):
            continue

        try:
            parts = [x.strip() for x in next(csv.reader([line]))]
        except Exception:
            continue

        if len(parts) != 7:
            continue

        name, n, m, reps, avg_seconds, rate, errors = parts

        try:
            rows.append(
                {
                    "name": name,
                    "n": int(n),
                    "m": int(m),
                    "reps": int(reps),
                    "avg_seconds": float(avg_seconds),
                    "rate": float(rate),
                    "errors": int(errors),
                    "raw": line,
                }
            )
        except ValueError:
            continue

    if not rows:
        raise RuntimeError("no benchmark CSV row found in stdout")

    return rows[-1]


def parse_path_substitutions(items):
    substitutions = []

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"bad --path-subst '{item}', expected OLD=NEW"
            )
        old, new = item.split("=", 1)
        substitutions.append((old, new))

    return substitutions


def apply_path_substitutions(path_text, substitutions):
    if not path_text:
        return path_text

    for old, new in substitutions:
        if path_text.startswith(old):
            return new + path_text[len(old):]

    return path_text


def command_compile(args):
    root = Path(args.root).resolve()
    source_path = root / args.source
    build_root = Path(args.build_root)

    if not build_root.is_absolute():
        build_root = root / build_root

    build_root = build_root.resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    tiles = parse_tile_list(args.tiles)
    warps_list = parse_int_list(args.warps)
    stages_list = parse_int_list(args.stages)

    if not source_path.exists():
        raise RuntimeError(f"source file not found: {source_path}")

    original_source = source_path.read_text()

    manifest_rows = []

    try:
        for tile in tiles:
            for warps in warps_list:
                for stages in stages_list:
                    cfg = config_name(tile, warps, stages)
                    build_dir = build_root / cfg

                    print("")
                    print("=" * 80)
                    print(f"Compiling {cfg}")
                    print(f"  tile={tile}")
                    print(f"  warps={warps}")
                    print(f"  stages={stages}")
                    print("=" * 80)

                    row = {
                        "config": cfg,
                        "tile_m": tile[0],
                        "tile_n": tile[1],
                        "tile_k": tile[2],
                        "warps": warps,
                        "stages": stages,
                        "sm": args.sm,
                        "build_dir": str(build_dir),
                        "exe": str(
                            build_dir
                            / "benchmarks/fnacc/fnacc-matmul-2d/fnacc-matmul-2d"
                        ),
                        "ptx": "",
                        "json": "",
                        "ptx_target": "",
                        "ptx_has_mma": False,
                        "ptx_has_fma": False,
                        "compile_status": "UNKNOWN",
                    }

                    try:
                        source_path.write_text(original_source)
                        patch_tile(source_path, tile)

                        if args.clean and build_dir.exists():
                            shutil.rmtree(build_dir)

                        build_dir.mkdir(parents=True, exist_ok=True)

                        cmake_cmd = [
                            "cmake",
                            "-S",
                            str(root),
                            "-B",
                            str(build_dir),
                            f"-DFNACC_SM={args.sm}",
                            f"-DFNACC_NUM_WARPS={warps}",
                            "-DFNACC_THREADS_PER_WARP=32",
                            f"-DFNACC_NUM_STAGES={stages}",
                        ]
                        cmake_cmd.extend(args.cmake_arg)

                        run_cmd(
                            cmake_cmd,
                            cwd=root,
                            timeout=args.timeout,
                            fail=True,
                        )

                        run_cmd(
                            [
                                "cmake",
                                "--build",
                                str(build_dir),
                                "--target",
                                "fnacc-matmul-2d",
                                "-j",
                                str(args.jobs),
                            ],
                            cwd=root,
                            timeout=args.timeout,
                            fail=True,
                        )

                        ptx, json = find_side_files(build_dir)
                        ptx_info = inspect_ptx(ptx)

                        row["ptx"] = str(ptx) if ptx else ""
                        row["json"] = str(json) if json else ""
                        row["ptx_target"] = ptx_info["ptx_target"]
                        row["ptx_has_mma"] = ptx_info["ptx_has_mma"]
                        row["ptx_has_fma"] = ptx_info["ptx_has_fma"]
                        row["compile_status"] = "OK"

                    except Exception as exc:
                        print(f"warning: compile failed for {cfg}: {exc}")
                        row["compile_status"] = f"FAILED: {exc}"

                    manifest_rows.append(row)

                    with open(args.manifest, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                        writer.writeheader()
                        writer.writerows(manifest_rows)

    finally:
        source_path.write_text(original_source)

    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("")
    print(f"wrote manifest: {args.manifest}")


def command_run(args):
    substitutions = parse_path_substitutions(args.path_subst)

    n = args.n
    m = args.m if args.m is not None else args.n

    result_rows = []

    with open(args.manifest, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    for row in manifest_rows:
        if row.get("compile_status") != "OK":
            print(f"skipping {row.get('config')}: compile_status={row.get('compile_status')}")
            continue

        exe = apply_path_substitutions(row["exe"], substitutions)
        ptx = apply_path_substitutions(row.get("ptx", ""), substitutions)
        json = apply_path_substitutions(row.get("json", ""), substitutions)

        exe_path = Path(exe)

        if not exe_path.exists() or not os.access(exe_path, os.X_OK):
            print(f"warning: missing/non-executable: {exe_path}")
            continue

        print("")
        print("=" * 80)
        print(f"Running {row['config']}")
        print(f"  exe={exe}")
        print(f"  ptx={ptx}")
        print(f"  json={json}")
        print("=" * 80)

        for trial in range(args.trials):
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            env["LANG"] = "C"

            # Do not let debug printing pollute timing.
            if not args.keep_fnacc_debug:
                env.pop("FNACC_DEBUG", None)

            if args.cuda_launch_blocking:
                env["CUDA_LAUNCH_BLOCKING"] = "1"

            # Ensure the direct executable uses the correct side files.
            if ptx and Path(ptx).exists():
                env["FNACC_PTX"] = str(Path(ptx).resolve())

            if json and Path(json).exists():
                env["FNACC_KERNELS_JSON"] = str(Path(json).resolve())

            cmd = [exe, str(n), str(m), str(args.reps)]

            proc = run_cmd(
                cmd,
                env=env,
                timeout=args.timeout,
                fail=False,
            )

            result = {
                "config": row["config"],
                "tile_m": row["tile_m"],
                "tile_n": row["tile_n"],
                "tile_k": row["tile_k"],
                "warps": row["warps"],
                "stages": row["stages"],
                "sm": row["sm"],
                "n": n,
                "m": m,
                "reps": args.reps,
                "trial": trial,
                "name": "",
                "avg_seconds": "",
                "rate": "",
                "errors": "",
                "exe": exe,
                "ptx": ptx,
                "json": json,
                "returncode": proc.returncode,
                "status": "UNKNOWN",
            }

            if proc.returncode != 0:
                result["status"] = "FAILED"
                result_rows.append(result)
                continue

            try:
                parsed = parse_benchmark_output(proc.stdout)
                result["name"] = parsed["name"]
                result["avg_seconds"] = parsed["avg_seconds"]
                result["rate"] = parsed["rate"]
                result["errors"] = parsed["errors"]
                result["status"] = "OK"
            except Exception as exc:
                result["status"] = f"PARSE_FAILED: {exc}"

            result_rows.append(result)

            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(result_rows)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)

    print("")
    print(f"wrote results: {args.output}")

    if args.summary:
        write_summary(args.output, args.summary)


def write_summary(results_csv, summary_csv):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = {}

    for r in rows:
        if r["status"] != "OK":
            continue

        key = (
            r["config"],
            r["tile_m"],
            r["tile_n"],
            r["tile_k"],
            r["warps"],
            r["stages"],
            r["sm"],
            r["n"],
            r["m"],
            r["reps"],
            r["exe"],
            r["ptx"],
            r["json"],
        )

        groups.setdefault(key, []).append(r)

    summary_rows = []

    for key, rs in groups.items():
        rates = [float(r["rate"]) for r in rs]
        times = [float(r["avg_seconds"]) for r in rs]
        errors = [int(r["errors"]) for r in rs]

        (
            config,
            tile_m,
            tile_n,
            tile_k,
            warps,
            stages,
            sm,
            n,
            m,
            reps,
            exe,
            ptx,
            json,
        ) = key

        summary_rows.append(
            {
                "config": config,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "warps": warps,
                "stages": stages,
                "sm": sm,
                "n": n,
                "m": m,
                "reps": reps,
                "trials": len(rs),
                "median_avg_seconds": statistics.median(times),
                "min_avg_seconds": min(times),
                "median_rate": statistics.median(rates),
                "max_rate": max(rates),
                "all_errors_zero": all(e == 0 for e in errors),
                "errors_values": ";".join(str(e) for e in errors),
                "exe": exe,
                "ptx": ptx,
                "json": json,
            }
        )

    summary_rows.sort(
        key=lambda r: float(r["median_rate"]) if r["median_rate"] != "" else -math.inf,
        reverse=True,
    )

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote summary: {summary_csv}")

    if summary_rows:
        best = summary_rows[0]
        print("")
        print("Best configuration:")
        print(
            f"  config={best['config']} "
            f"tile=({best['tile_m']},{best['tile_n']},{best['tile_k']}) "
            f"warps={best['warps']} stages={best['stages']} "
            f"median_rate={float(best['median_rate']):.2f} "
            f"max_rate={float(best['max_rate']):.2f} "
            f"errors={best['errors_values']}"
        )


def command_summary(args):
    write_summary(args.results, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Split compile/run FnAcc matmul tile/warps sweep."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="Compile sweep configurations.")
    p_compile.add_argument("--root", required=True)
    p_compile.add_argument(
        "--source",
        default="benchmarks/fnacc/matmul_2d_kernel.f90",
        help="FnAcc matmul kernel source relative to --root.",
    )
    p_compile.add_argument(
        "--build-root",
        default="build-sweep-fnacc-matmul",
    )
    p_compile.add_argument(
        "--manifest",
        default="fnacc-matmul-sweep-manifest.csv",
    )
    p_compile.add_argument("--sm", default="80")
    p_compile.add_argument(
        "--tiles",
        default=(
            "16x16x32,"
            "32x32x32,"
            "64x32x32,"
            "32x64x32,"
            "64x64x32,"
            "128x64x32,"
            "64x128x32"
        ),
    )
    p_compile.add_argument("--warps", default="1,2,4,8")
    p_compile.add_argument("--stages", default="3")
    p_compile.add_argument("--cmake-arg", action="append", default=[])
    p_compile.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p_compile.add_argument("--timeout", type=float, default=None)
    p_compile.add_argument("--clean", action="store_true")
    p_compile.set_defaults(func=command_compile)

    p_run = sub.add_parser("run", help="Run previously compiled sweep.")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--output", default="fnacc-matmul-sweep-results.csv")
    p_run.add_argument("--summary", default="fnacc-matmul-sweep-summary.csv")
    p_run.add_argument("--n", type=int, default=2048)
    p_run.add_argument("--m", type=int, default=None)
    p_run.add_argument("--reps", type=int, default=100)
    p_run.add_argument("--trials", type=int, default=3)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument(
        "--path-subst",
        action="append",
        default=[],
        help="Rewrite paths from compile machine to run machine, OLD=NEW.",
    )
    p_run.add_argument("--cuda-launch-blocking", action="store_true")
    p_run.add_argument(
        "--keep-fnacc-debug",
        action="store_true",
        help="Do not remove FNACC_DEBUG from environment.",
    )
    p_run.set_defaults(func=command_run)

    p_summary = sub.add_parser("summary", help="Summarise result CSV.")
    p_summary.add_argument("--results", required=True)
    p_summary.add_argument("--output", default="fnacc-matmul-sweep-summary.csv")
    p_summary.set_defaults(func=command_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

