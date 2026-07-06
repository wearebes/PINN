#!/usr/bin/env python3
"""Verify the final stationary-ellipse CSV and PNG delivery gates."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "Experiment/report/stationary bubble/summary.csv"
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
BACKGROUND = ROOT / "cfd_applications_cleanroom/results/background"
RENDER_SOURCE = ROOT / "cfd_applications_cleanroom/cfd_apps/curvature_diagnostic_shared.py"

CASES = ("E1", "E2")
LEVELS = (6, 7, 8)
PROCESS_METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
DIAG_METHODS = ("NN27_RAW", "NN27_D4")


def fail(message: str) -> int:
    print(f"FAIL {message}")
    return 1


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


def read_summary_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(str(SUMMARY_CSV))
    with SUMMARY_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_summary(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    ellipse = [row for row in rows if row.get("benchmark_family") == "stationary_ellipse"]
    diag = [row for row in ellipse if row.get("tier") == "curvature-diagnostic"]
    proc = [row for row in ellipse if row.get("tier") == "curvature-process"]

    expected_diag = {
        (case, str(1 << level), method)
        for case in CASES
        for level in LEVELS
        for method in DIAG_METHODS
    }
    expected_proc = {
        (case, str(1 << level), method)
        for case in CASES
        for level in LEVELS
        for method in PROCESS_METHODS
    }
    actual_diag = {(row.get("case_id", ""), row.get("grid_n", ""), row.get("method", "")) for row in diag}
    actual_proc = {(row.get("case_id", ""), row.get("grid_n", ""), row.get("method", "")) for row in proc}

    if actual_diag != expected_diag:
        errors.append(f"diagnostic_matrix_mismatch expected={len(expected_diag)} actual={len(actual_diag)}")
    if actual_proc != expected_proc:
        errors.append(f"process_matrix_mismatch expected={len(expected_proc)} actual={len(actual_proc)}")

    open_diag = [row for row in diag if row.get("report_ready") != "true"]
    open_proc = [row for row in proc if row.get("report_ready") != "true"]
    if open_diag:
        errors.append(f"diagnostic_not_ready={len(open_diag)}/{len(diag)}")
    if open_proc:
        detail = ";".join(
            f"{row.get('case_id')}:{row.get('grid_n')}:{row.get('method')}:{row.get('status')}:"
            f"{row.get('final_tau_or_current_tau')}"
            for row in open_proc
        )
        errors.append(f"process_not_ready={len(open_proc)}/{len(proc)} open={detail}")
    return errors


def verify_l8_raw_completion() -> list[str]:
    errors: list[str] = []
    for case in CASES:
        for method in PROCESS_METHODS:
            summary = latest_complete_l8_summary(case, method)
            if summary is None:
                errors.append(f"missing_complete_l8_summary {case} {method}")
    return errors


def verify_figures() -> list[str]:
    errors: list[str] = []
    latest_path = BACKGROUND / "stationary_ellipse_finalizer_latest.txt"
    if not latest_path.exists():
        return ["missing_finalizer_latest"]
    log_dir = Path(latest_path.read_text().strip())
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    manifest_path = log_dir / "figure_manifest.json"
    done_path = log_dir / "DONE"
    if not done_path.exists():
        errors.append(f"missing_finalizer_done {done_path}")
    manifest = load_json(manifest_path)
    if not manifest:
        errors.append(f"missing_or_unreadable_figure_manifest {manifest_path}")
        return errors
    plates = manifest.get("plates", [])
    expected = {(case, 1 << level) for case in CASES for level in LEVELS}
    actual = {(plate.get("case"), int(plate.get("grid_n", -1))) for plate in plates}
    if actual != expected:
        errors.append(f"figure_plate_matrix_mismatch expected={len(expected)} actual={len(actual)}")
    for plate in plates:
        png = Path(str(plate.get("png", "")))
        if not png.is_absolute():
            png = ROOT / png
        if not png.exists() or png.stat().st_size <= 0:
            errors.append(f"missing_or_empty_png {png}")
    return errors


def verify_camax_label_source() -> list[str]:
    if not RENDER_SOURCE.exists():
        return [f"missing_render_source {RENDER_SOURCE}"]
    text = RENDER_SOURCE.read_text()
    bad_lines = []
    for i, line in enumerate(text.splitlines(), start=1):
        match = re.search(r"label\s*=\s*(?:fr|rf|f|r)?([\"'])(.*?)\1", line)
        if not match:
            continue
        rendered_literal = re.sub(r"\{[^{}]*\}", "", match.group(2))
        lower = rendered_literal.lower()
        if "ca_" in lower and "final" in lower:
            bad_lines.append(f"{i}:{line.strip()}")
    if bad_lines:
        return ["camax_label_contains_final " + " | ".join(bad_lines)]
    if not re.search(r"label=.*Ca_\{\{?max", text):
        return ["camax_label_not_found_in_render_source"]
    return []


def main() -> int:
    try:
        rows = read_summary_rows()
    except FileNotFoundError as exc:
        return fail(f"missing_summary_csv {exc}")

    errors: list[str] = []
    errors.extend(verify_summary(rows))
    errors.extend(verify_l8_raw_completion())
    errors.extend(verify_figures())
    errors.extend(verify_camax_label_source())

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    print(f"PASS summary_csv={SUMMARY_CSV}")
    print("PASS stationary_ellipse_diagnostic_ready=12/12")
    print("PASS stationary_ellipse_process_ready=24/24")
    print("PASS stationary_ellipse_l8_complete=8/8")
    print("PASS stationary_ellipse_process_png_plates=6/6")
    print("PASS camax_labels_without_final=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
