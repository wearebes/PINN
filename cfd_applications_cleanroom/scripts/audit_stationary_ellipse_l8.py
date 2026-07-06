#!/usr/bin/env python3
"""Audit stationary-ellipse L8 process completion.

This is intentionally read-only.  It reports whether the 256x256 process rows
are complete enough for summary/report use and, for partial rows, the latest
trace progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def latest_complete_summary(case: str, method: str) -> Path | None:
    candidates = sorted(
        RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L8/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == 8
            and data.get("reached_final_time") is True
            and safe_float(data.get("final_tau")) >= 0.999
        ):
            return path
    return None


def latest_trace(case: str, method: str) -> tuple[Path, int, str] | None:
    candidates = sorted(
        RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L8/surface_tension_trace.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        last: dict[str, str] | None = None
        row_count = 0
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    last = row
                    row_count += 1
        except Exception:
            continue
        if last is not None:
            return path, row_count, last.get("tau", "")
    return None


def build_rows() -> tuple[int, list[dict[str, object]]]:
    complete = 0
    rows: list[dict[str, object]] = []
    for case in ("E1", "E2"):
        for method in METHODS:
            summary = latest_complete_summary(case, method)
            if summary is not None:
                complete += 1
                data = json.loads(summary.read_text())
                rows.append({
                    "case": case,
                    "grid_n": 256,
                    "method": method,
                    "status": "complete",
                    "tau": data.get("final_tau", ""),
                    "rows": data.get("trace_row_count", ""),
                    "path": str(summary),
                    "result_dir": str(summary.parent),
                })
                continue
            trace = latest_trace(case, method)
            if trace is None:
                rows.append({
                    "case": case,
                    "grid_n": 256,
                    "method": method,
                    "status": "missing",
                    "tau": "",
                    "rows": "",
                    "path": "",
                    "result_dir": "",
                })
            else:
                path, row_count, tau = trace
                rows.append({
                    "case": case,
                    "grid_n": 256,
                    "method": method,
                    "status": "partial",
                    "tau": tau,
                    "rows": row_count,
                    "path": str(path),
                    "result_dir": str(path.parent),
                })
    return complete, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="print the audit but return success even when L8 rows are still partial",
    )
    parser.add_argument(
        "--complete-result-dirs",
        action="store_true",
        help="print only completed L8 result directories, one per line, for packaging",
    )
    args = parser.parse_args()

    complete, rows = build_rows()
    if args.complete_result_dirs:
        if complete != 8:
            print(f"stationary_ellipse_L8_complete={complete}/8", file=sys.stderr)
            return 1
        for row in rows:
            if row["status"] == "complete":
                print(row["result_dir"])
        return 0

    print(f"stationary_ellipse_L8_complete={complete}/8")
    for row in rows:
        print(
            row["status"].upper(),
            row["case"],
            row["grid_n"],
            row["method"],
            "tau=" + str(row["tau"]),
            "rows=" + str(row["rows"]),
            row["path"],
        )
    return 0 if complete == 8 or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
