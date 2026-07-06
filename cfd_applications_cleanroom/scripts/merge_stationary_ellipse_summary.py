#!/usr/bin/env python3
"""Merge stationary-ellipse raw evidence into the report summary CSV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "Experiment/report/stationary bubble/summary.csv"
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
MANIFESTS = ROOT / "cfd_applications_cleanroom/results/manifests"
FIGURES = ROOT / "cfd_applications_cleanroom/results/figures"
SOURCE_DATA = ROOT / "cfd_applications_cleanroom/results/source_data"

CASES = {
    "E1": {"a": 0.4472136, "b": 0.3577709, "aspect_ratio": "5:4"},
    "E2": {"a": 0.4898979, "b": 0.3265986, "aspect_ratio": "3:2"},
}
LEVELS = (6, 7, 8)
PROCESS_METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
DIAG_METHODS = ("NN27_RAW", "NN27_D4")
PUBLIC_LABEL = {
    "NN_DISABLE": "native",
    "NN_PROBE_ONLY": "probe_only",
    "NN27_RAW": "NN",
    "NN27_D4": "NND4",
}


def rel(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        if p.is_absolute():
            return str(p.relative_to(ROOT))
        return str(p)
    except ValueError:
        return str(p)


def abspath(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def exists_flag(path: str | Path | None) -> str:
    p = abspath(path)
    return "true" if p is not None and p.exists() else "false"


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    p = abspath(path)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def put(row: dict[str, str], key: str, value: Any) -> None:
    row[key] = clean(value)


def last_csv_row(path: str | Path | None) -> tuple[dict[str, str] | None, int]:
    p = abspath(path)
    if p is None or not p.exists():
        return None, 0
    last = None
    count = 0
    with p.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            last = record
            count += 1
    return last, count


def latest_complete_process_summary(case: str, level: int, method: str) -> Path | None:
    candidates = sorted(RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/summary.json"))
    for path in reversed(candidates):
        data = load_json(path)
        if not data:
            continue
        try:
            final_tau = float(data.get("final_tau", 0.0))
        except Exception:
            final_tau = 0.0
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == level
            and data.get("reached_final_time") is True
            and final_tau >= 0.999
        ):
            return path
    return None


def latest_partial_process_trace(case: str, level: int, method: str) -> Path | None:
    candidates = list(RAW.glob(
        f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/surface_tension_trace.csv"
    ))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_diag_summary(case: str, level: int, method: str) -> Path | None:
    candidates = sorted(RAW.glob(f"stationary_ellipse_curvature_{case}_*/{method}_L{level}/summary.json"))
    return candidates[-1] if candidates else None


def fill_band(row: dict[str, str], prefix: str, band: dict[str, Any]) -> None:
    for key in (
        "row_count", "mean_kappa_native", "mean_kappa_nn", "mean_kappa_ratio_to_native",
        "mean_hk_ratio_to_native", "mean_delta_hk", "mean_delta_kappa", "std_delta_hk",
        "std_delta_kappa", "std_kappa_native", "std_kappa_nn", "max_abs_delta_hk",
        "p95_abs_delta_hk", "p99_abs_delta_hk", "rmse_direct", "rmse_negated",
        "tail_mean_abs_delta_hk", "scale_identity_max_abs", "sign_agreement_fraction",
    ):
        put(row, f"{prefix}_{key}", band.get(key, ""))


def fill_snapshot(row: dict[str, str], index: str, snapshot: dict[str, Any]) -> None:
    prefix = f"snapshot_{index}"
    put(row, f"{prefix}_fraction", snapshot.get("snapshot_fraction", ""))
    put(row, f"{prefix}_t", snapshot.get("t", ""))
    put(row, f"{prefix}_tau", snapshot.get("tau", ""))
    put(row, f"{prefix}_row_count", snapshot.get("row_count", ""))
    fill_band(row, f"{prefix}_force", snapshot.get("force_band", {}))
    fill_band(row, f"{prefix}_interface", snapshot.get("interface", {}))


def common_ellipse_row(case: str, level: int, method: str, tier: str) -> dict[str, str]:
    meta = CASES[case]
    row: dict[str, str] = {}
    put(row, "grid_n", 1 << level)
    put(row, "level", level)
    put(row, "method", method)
    put(row, "benchmark_family", "stationary_ellipse")
    put(row, "case_id", case)
    put(row, "geometry_family", "ellipse")
    put(row, "geometry_type", "single_stationary_axis_aligned_ellipse")
    put(row, "geometry_motion", "stationary")
    put(row, "geometry_parameters", f"a={meta['a']}; b={meta['b']}; aspect_ratio={meta['aspect_ratio']}; area_match=a*b≈0.16")
    put(row, "geometry_role", "area_matched_ellipse_stress_case_for_stationary_bubble_route")
    put(row, "tier", tier)
    put(row, "case_variant", case)
    put(row, "method_public_label", PUBLIC_LABEL.get(method, method))
    put(row, "is_d4", method == "NN27_D4")
    put(row, "ellipse_a", meta["a"])
    put(row, "ellipse_b", meta["b"])
    put(row, "ellipse_aspect_ratio", meta["aspect_ratio"])
    put(row, "expected_matrix_row", "true")
    return row


def diagnostic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in CASES:
        for level in LEVELS:
            for method in DIAG_METHODS:
                row = common_ellipse_row(case, level, method, "curvature-diagnostic")
                summary_path = latest_diag_summary(case, level, method)
                data = load_json(summary_path)
                if data:
                    run_id = data.get("run_id", "")
                    put(row, "run_id", run_id)
                    put(row, "status", "complete_manifest")
                    put(row, "record_kind", "diagnostic_summary")
                    put(row, "evidence_status", "manifest_backed_complete")
                    put(row, "report_ready", "true")
                    put(row, "summary_json", rel(summary_path))
                    put(row, "summary_json_currently_present", exists_flag(summary_path))
                    put(row, "manifest_json", rel(MANIFESTS / f"{run_id}.manifest.json"))
                    put(row, "manifest_currently_present", exists_flag(MANIFESTS / f"{run_id}.manifest.json"))
                    put(row, "curvature_trace_csv", rel(data.get("raw_csv", "")))
                    put(row, "curvature_field_csv", rel(data.get("field_csv", "")))
                    put(row, "curvature_trace_currently_present", exists_flag(data.get("raw_csv", "")))
                    put(row, "field_csv_currently_present", exists_flag(data.get("field_csv", "")))
                    put(row, "raw_row_count", data.get("raw_row_count", ""))
                    put(row, "raw_schema_valid", data.get("raw_schema_valid", ""))
                    put(row, "source_hashes_unchanged", data.get("source_hashes_unchanged", ""))
                    put(row, "official_source_manifest_sha256", data.get("official_source_manifest_sha256", ""))
                    put(row, "resolved_config_sha256", data.get("resolved_config_sha256", ""))
                    put(row, "binary_sha256", data.get("binary_sha256", ""))
                    put(row, "sign_verdict", data.get("sign_scale", {}).get("sign_verdict", ""))
                    fill_band(row, "diagnostic_force", data.get("force_band", {}))
                    fill_band(row, "diagnostic_interface", data.get("interface", {}))
                else:
                    put(row, "status", "missing_expected_run")
                    put(row, "record_kind", "diagnostic_summary")
                    put(row, "evidence_status", "missing")
                    put(row, "report_ready", "false")
                rows.append(row)
    return rows


def process_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in CASES:
        for level in LEVELS:
            for method in PROCESS_METHODS:
                row = common_ellipse_row(case, level, method, "curvature-process")
                summary_path = latest_complete_process_summary(case, level, method)
                data = load_json(summary_path)
                if data:
                    put(row, "run_id", data.get("run_id", ""))
                    put(row, "status", "complete_raw_summary")
                    put(row, "record_kind", "process_summary")
                    put(row, "evidence_status", "complete")
                    put(row, "report_ready", "true")
                    put(row, "summary_json", rel(summary_path))
                    put(row, "summary_json_currently_present", exists_flag(summary_path))
                    put(row, "surface_tension_trace_csv", rel(data.get("surface_tension_trace_csv", "")))
                    put(row, "curvature_process_csv", rel(data.get("curvature_process_csv", "")))
                    put(row, "surface_tension_trace_currently_present", exists_flag(data.get("surface_tension_trace_csv", "")))
                    put(row, "curvature_process_csv_currently_present", exists_flag(data.get("curvature_process_csv", "")))
                    put(row, "reached_final_time", data.get("reached_final_time", ""))
                    put(row, "final_tau_or_current_tau", data.get("final_tau", ""))
                    put(row, "progress_percent", float(data.get("final_tau", 0.0)) * 100.0)
                    put(row, "final_t_or_current_t", data.get("final_t", ""))
                    put(row, "expected_tmax", data.get("expected_tmax", ""))
                    put(row, "trace_row_count", data.get("trace_row_count", ""))
                    put(row, "Ca_final_or_current", data.get("Ca_final", ""))
                    put(row, "Ca_max", data.get("Ca_max", ""))
                    put(row, "Ca_tail_mean", data.get("Ca_tail_mean", ""))
                    put(row, "Ca_tail_max", data.get("Ca_tail_max", ""))
                    put(row, "finite_Ca", data.get("finite_Ca", ""))
                    put(row, "raw_schema_valid", data.get("raw_schema_valid", ""))
                    put(row, "source_hashes_unchanged", data.get("source_hashes_unchanged", ""))
                    put(row, "official_source_manifest_sha256", data.get("official_source_manifest_sha256", ""))
                    put(row, "resolved_config_sha256", data.get("resolved_config_sha256", ""))
                    put(row, "binary_sha256", data.get("binary_sha256", ""))
                    snapshots = data.get("snapshots", {})
                    put(row, "snapshot_count", len(snapshots))
                    put(row, "snapshot_indices", ";".join(sorted(snapshots, key=lambda x: int(x))))
                    for index, snapshot in snapshots.items():
                        fill_snapshot(row, str(index), snapshot)
                else:
                    trace_path = latest_partial_process_trace(case, level, method)
                    last, count = last_csv_row(trace_path)
                    if trace_path and last:
                        put(row, "run_id", last.get("run_id", ""))
                        put(row, "status", "partial_trace_no_summary")
                        put(row, "record_kind", "partial_trace")
                        put(row, "evidence_status", "partial_trace_no_summary")
                        put(row, "report_ready", "false")
                        put(row, "surface_tension_trace_csv", rel(trace_path))
                        put(row, "surface_tension_trace_currently_present", exists_flag(trace_path))
                        put(row, "trace_row_count", count)
                        put(row, "final_tau_or_current_tau", last.get("tau", ""))
                        put(row, "progress_percent", float(last.get("tau", 0.0)) * 100.0)
                        put(row, "final_t_or_current_t", last.get("t", ""))
                        put(row, "Ca_final_or_current", last.get("Ca", ""))
                        for key in ("i", "t", "dt", "tau", "Ca", "mass", "kappa_linf", "dc"):
                            put(row, f"partial_last_{key}", last.get(key, ""))
                    else:
                        put(row, "status", "missing_expected_run")
                        put(row, "record_kind", "missing")
                        put(row, "evidence_status", "missing")
                        put(row, "report_ready", "false")
                        put(row, "progress_percent", "0")
                rows.append(row)
    return rows


def read_existing_circle_rows() -> tuple[list[str], list[dict[str, str]]]:
    if not SUMMARY_CSV.exists():
        return [], []
    with SUMMARY_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = [dict(row) for row in reader if row.get("geometry_family") != "ellipse"]
    return columns, rows


def main() -> int:
    existing_columns, circle_rows = read_existing_circle_rows()
    rows = circle_rows + diagnostic_rows() + process_rows()
    columns: list[str] = []
    seen = set()
    preferred = existing_columns + [
        "tier", "record_kind", "evidence_status", "case_variant", "method_public_label",
        "ellipse_a", "ellipse_b", "ellipse_aspect_ratio", "manifest_json",
        "curvature_trace_csv", "curvature_field_csv", "raw_row_count",
        "snapshot_count", "snapshot_indices",
    ]
    for column in preferred:
        if column and column not in seen:
            columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean(row.get(column, "")) for column in columns})

    schema = SUMMARY_CSV.parent / "summary_schema.md"
    schema.write_text(
        "\n".join([
            "# Stationary Bubble / Ellipse Summary CSV Schema",
            "",
            f"- CSV: `{rel(SUMMARY_CSV)}`",
            f"- Rows: {len(rows)} = {len(circle_rows)} circle rows + {len(rows) - len(circle_rows)} ellipse rows",
            "- Existing circle rows are retained; ellipse rows are regenerated from raw summaries/traces.",
            "- Use `geometry_family`, `tier`, and `evidence_status` to filter plot-ready records.",
            "- Complete ellipse process rows have `evidence_status=complete` and `report_ready=true`.",
            "- Partial traces and missing rows are retained only for progress/accounting.",
            "",
        ]) + "\n"
    )
    print(f"summary_csv={rel(SUMMARY_CSV)}")
    print(f"rows={len(rows)}")
    print(f"columns={len(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
