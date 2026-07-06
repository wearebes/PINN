#!/usr/bin/env python3
"""Write a reproducible stationary-ellipse completion status snapshot."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "Experiment/report/stationary bubble"
SUMMARY_CSV = REPORT_DIR / "summary.csv"
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
BACKGROUND = ROOT / "cfd_applications_cleanroom/results/background"

STATUS_MD = REPORT_DIR / "stationary_ellipse_completion_status.md"
OPEN_ROWS_CSV = REPORT_DIR / "stationary_ellipse_open_rows.csv"
STATUS_JSON = REPORT_DIR / "stationary_ellipse_completion_status.json"

CASES = ("E1", "E2")
LEVELS = (6, 7, 8)
METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_complete_l8_summary(case: str, method: str) -> Path | None:
    candidates = sorted(
        RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L8/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        data = load_json(path)
        if not data:
            continue
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == 8
            and data.get("reached_final_time") is True
            and safe_float(data.get("final_tau")) >= 0.999
        ):
            return path
    return None


def route_relative(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)
    except ValueError:
        return str(p)


def read_summary_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finalizer_plate_count() -> tuple[int, str]:
    latest_path = BACKGROUND / "stationary_ellipse_finalizer_latest.txt"
    if not latest_path.exists():
        return 0, ""
    log_dir = Path(latest_path.read_text().strip())
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    manifest = load_json(log_dir / "figure_manifest.json")
    if not manifest:
        return 0, str(log_dir)
    plates = manifest.get("plates", [])
    valid = 0
    for plate in plates:
        png = Path(str(plate.get("png", "")))
        if not png.is_absolute():
            png = ROOT / png
        if png.exists() and png.stat().st_size > 0:
            valid += 1
    return valid, str(log_dir)


def build_status() -> dict[str, Any]:
    rows = read_summary_rows()
    diagnostic = [
        row for row in rows
        if row.get("benchmark_family") == "stationary_ellipse"
        and row.get("tier") == "curvature-diagnostic"
    ]
    process = [
        row for row in rows
        if row.get("benchmark_family") == "stationary_ellipse"
        and row.get("tier") == "curvature-process"
    ]
    open_rows = [row for row in process if row.get("report_ready") != "true"]

    complete_l8 = []
    missing_l8 = []
    for case in CASES:
        for method in METHODS:
            summary = latest_complete_l8_summary(case, method)
            if summary:
                complete_l8.append({"case": case, "method": method, "summary_json": route_relative(summary)})
            else:
                missing_l8.append({"case": case, "method": method})

    plate_count, finalizer_dir = finalizer_plate_count()
    diagnostic_ready = sum(row.get("report_ready") == "true" for row in diagnostic)
    process_ready = sum(row.get("report_ready") == "true" for row in process)
    final_ready = (
        diagnostic_ready == 12
        and len(diagnostic) == 12
        and process_ready == 24
        and len(process) == 24
        and len(complete_l8) == 8
        and plate_count == 6
    )
    return {
        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_csv": route_relative(SUMMARY_CSV),
        "diagnostic_ready": diagnostic_ready,
        "diagnostic_expected": len(diagnostic),
        "process_ready": process_ready,
        "process_expected": len(process),
        "l8_process_ready": len(complete_l8),
        "l8_process_expected": 8,
        "process_png_plates_ready": plate_count,
        "process_png_plates_expected": 6,
        "finalizer_dir": route_relative(finalizer_dir),
        "final_delivery_status": "ready" if final_ready else "not_ready",
        "open_rows": [
            {
                "case_id": row.get("case_id", ""),
                "grid_n": row.get("grid_n", ""),
                "level": row.get("level", ""),
                "method": row.get("method", ""),
                "status": row.get("status", ""),
                "evidence_status": row.get("evidence_status", ""),
                "report_ready": row.get("report_ready", ""),
                "final_tau_or_current_tau": row.get("final_tau_or_current_tau", ""),
                "progress_percent": row.get("progress_percent", ""),
                "trace_row_count": row.get("trace_row_count", ""),
                "surface_tension_trace_csv": row.get("surface_tension_trace_csv", ""),
                "summary_json": row.get("summary_json", ""),
            }
            for row in open_rows
        ],
        "missing_l8": missing_l8,
    }


def write_open_rows_csv(status: dict[str, Any]) -> None:
    fields = [
        "case_id", "grid_n", "level", "method", "status", "evidence_status",
        "report_ready", "final_tau_or_current_tau", "progress_percent",
        "trace_row_count", "surface_tension_trace_csv", "summary_json",
    ]
    with OPEN_ROWS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in status["open_rows"]:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_status_md(status: dict[str, Any]) -> None:
    lines = [
        "# Stationary Ellipse Completion Status",
        "",
        f"- updated_utc: `{status['updated_utc']}`",
        f"- summary_csv: `{status['summary_csv']}`",
        f"- diagnostic_ready: `{status['diagnostic_ready']}/{status['diagnostic_expected']}`",
        f"- process_ready: `{status['process_ready']}/{status['process_expected']}`",
        f"- l8_process_ready: `{status['l8_process_ready']}/{status['l8_process_expected']}`",
        f"- process_png_plates_ready: `{status['process_png_plates_ready']}/{status['process_png_plates_expected']}`",
        f"- final_delivery_status: `{status['final_delivery_status']}`",
    ]
    if status["finalizer_dir"]:
        lines.append(f"- finalizer_dir: `{status['finalizer_dir']}`")
    lines.extend([
        "",
        "## Current Missing Rows",
        "",
        "| case | grid | method | status | tau/progress | trace rows |",
        "|---|---:|---|---|---:|---:|",
    ])
    if not status["open_rows"]:
        lines.append("| none |  |  |  |  |  |")
    for row in status["open_rows"]:
        lines.append(
            "| {case} | {grid} | {method} | {status_text} | {tau} ({progress}%) | {count} |".format(
                case=row.get("case_id", ""),
                grid=row.get("grid_n", ""),
                method=row.get("method", ""),
                status_text=row.get("status", ""),
                tau=row.get("final_tau_or_current_tau", ""),
                progress=row.get("progress_percent", ""),
                count=row.get("trace_row_count", ""),
            )
        )
    lines.extend([
        "",
        "## Required Completion Step",
        "",
        "Run on Windows/WSL from the repository root:",
        "",
        "```bash",
        "MAX_JOBS=4 bash cfd_applications_cleanroom/scripts/run_and_pack_stationary_ellipse_l8_wsl.sh",
        "```",
        "",
        "After copying the generated package back to this workspace, run:",
        "",
        "```bash",
        "bash cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh \\",
        "  cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_*.tar.gz",
        "```",
        "",
        "Final acceptance command:",
        "",
        "```bash",
        "python cfd_applications_cleanroom/scripts/verify_stationary_ellipse_delivery.py",
        "```",
        "",
        "Final acceptance requires `process_ready=24/24`, `stationary_ellipse_l8_complete=8/8`, and six generated process PNG plates.",
    ])
    STATUS_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    status = build_status()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    write_open_rows_csv(status)
    write_status_md(status)
    print(f"status_md={route_relative(STATUS_MD)}")
    print(f"open_rows_csv={route_relative(OPEN_ROWS_CSV)}")
    print(f"status_json={route_relative(STATUS_JSON)}")
    print(f"final_delivery_status={status['final_delivery_status']}")
    print(f"process_ready={status['process_ready']}/{status['process_expected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
