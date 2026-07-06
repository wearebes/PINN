"""Stationary-ellipse cleanroom runner and summarizer.

New, additive diagnostic family -- see
docs/superpowers/plans/2026-07-03-stationary-ellipse-curvature-stress-diagnostic.md.
Reuses cfd_apps/curvature_diagnostic_shared.py (the same compile/run/hash-guard
and curvature-statistics code the circle route uses) rather than duplicating
it; only the ellipse-specific geometry compile step and the row-relabeling
adapter for the analytic-reference comparison (plan section 8, Phase 4) are
new here.
"""

from __future__ import annotations

import json
import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.hashes import sha256_file, tree_manifest
from cfd_applications_cleanroom.cfd_apps.manifest import stale_output_guard
from cfd_applications_cleanroom.cfd_apps.paths import MANIFESTS, RAW, ROOT, route_relative
from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import (
    BUILD_SRC,
    VENDOR_SRC,
    StationaryGateError,
    compile_curvature_binary,
    compile_curvature_process_binary,
    curvature_band_stats,
    load_required_source_audit,
    run_curvature_binary_one,
    run_curvature_process_binary_one,
    write_curvature_diagnostic_report,
    write_curvature_process_report,
)


CASE_DIR = ROOT / "cfd_applications_cleanroom/cases/stationary_ellipse"
CURVATURE_FIELD_SOURCE = CASE_DIR / "stationary_ellipse_curvature_field.c"
CURVATURE_PROCESS_SOURCE = CASE_DIR / "stationary_ellipse_curvature_process.c"
GEOMETRY_SOURCE = CASE_DIR / "ellipse_geometry_clean.c"
BENCHMARK = "stationary_ellipse"

PAPER_DEPLOYABLE_METHODS = ("NN27_RAW", "NN27_D4")
CURVATURE_DIAGNOSTIC_METHODS = PAPER_DEPLOYABLE_METHODS
CURVATURE_PROCESS_METHODS = ("NN_DISABLE", "NN_PROBE_ONLY") + PAPER_DEPLOYABLE_METHODS
CURVATURE_DIAGNOSTIC_LEVELS = (6, 7, 8)
CURVATURE_PROCESS_LEVELS = (6, 7, 8)

# Qcc's whole-program dimensional-constraint solver cannot see through the
# separately-compiled ellipse_geometry_clean.o and mis-infers a length
# dimension for the analytic h*kappa; downgrade to a warning rather than
# disabling the checker file-wide. See plan section 7 "empirical diagnosis".
ELLIPSE_EXTRA_QCC_FLAGS = ("-Wdimensions",)


@dataclass(frozen=True)
class EllipseCase:
    case_id: str
    a: float
    b: float


ELLIPSE_CASES = {
    "E1": EllipseCase("E1", a=0.4472136, b=0.3577709),
    "E2": EllipseCase("E2", a=0.4898979, b=0.3265986),
}


def _resolve_case(case: str) -> EllipseCase:
    if case not in ELLIPSE_CASES:
        raise StationaryGateError(f"stationary_ellipse_unknown_case:{case}")
    return ELLIPSE_CASES[case]


def _compile_ellipse_geometry_object(*, ellipse_case: EllipseCase, compile_dir: Path) -> dict[str, Any]:
    object_path = compile_dir / "ellipse_geometry_clean.o"
    compile_log = compile_dir / "ellipse_geometry_compile.log"
    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    cmd = [
        os.environ.get("CC", "cc"),
        "-std=c99",
        "-O2",
        f"-I{BUILD_SRC}",
        f"-I{CASE_DIR}",
        f"-DELLIPSE_A={ellipse_case.a}",
        f"-DELLIPSE_B={ellipse_case.b}",
        "-c",
        str(GEOMETRY_SOURCE),
        "-o",
        str(object_path),
    ]
    compile_result = subprocess.run(
        cmd,
        cwd=compile_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    compile_log.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={compile_dir}\nreturncode={compile_result.returncode}\n\n"
        + compile_result.stdout
    )
    vendor_after = tree_manifest(VENDOR_SRC, source_only=True)
    build_after = tree_manifest(BUILD_SRC, source_only=True)
    if compile_result.returncode != 0 or not object_path.exists():
        raise StationaryGateError(f"ellipse_geometry_object_compile_failed:{route_relative(compile_log)}")
    if vendor_before != vendor_after:
        raise StationaryGateError("route_vendor_source_mutated_during_ellipse_geometry_compile")
    if build_before != build_after:
        raise StationaryGateError("route_build_source_mutated_during_ellipse_geometry_compile")
    return {
        "geometry_object": object_path,
        "geometry_object_sha256": sha256_file(object_path),
        "geometry_source_sha256": sha256_file(GEOMETRY_SOURCE),
        "geometry_compile_command": " ".join(cmd),
        "geometry_compile_log": str(compile_log),
    }


def _extra_qcc_flags_for(ellipse_case: EllipseCase) -> tuple[str, ...]:
    return (
        *ELLIPSE_EXTRA_QCC_FLAGS,
        f"-DELLIPSE_A={ellipse_case.a}",
        f"-DELLIPSE_B={ellipse_case.b}",
    )


def _relabel_rows_for_analytic_reference(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Route analytic-vs-value comparisons through the unmodified shared
    curvature_band_stats by relabeling rows so its native_kappa/hk_native/
    delta_kappa/delta_hk keys carry the analytic reference instead. No
    change to curvature_band_stats itself (plan section 8, Phase 4)."""
    relabeled = []
    for row in records:
        new_row = dict(row)
        new_row["native_kappa"] = row["kappa_analytic"]
        new_row["hk_native"] = row["hk_analytic"]
        new_row["delta_kappa"] = row["delta_kappa_nn_analytic"]
        new_row["delta_hk"] = row["delta_hk_nn_analytic"]
        relabeled.append(new_row)
    return relabeled


def run_curvature_diagnostic(*, case: str, methods: list[str], levels: list[int]) -> dict[str, Any]:
    ellipse_case = _resolve_case(case)
    source_audit = load_required_source_audit()
    methods = methods or list(PAPER_DEPLOYABLE_METHODS)
    unknown = sorted(set(methods) - set(CURVATURE_DIAGNOSTIC_METHODS))
    if unknown:
        raise StationaryGateError(f"stationary_ellipse_curvature_unknown_methods:{','.join(unknown)}")
    unknown_levels = sorted(set(levels) - set(CURVATURE_DIAGNOSTIC_LEVELS))
    if unknown_levels:
        raise StationaryGateError(f"stationary_ellipse_curvature_unknown_levels:{unknown_levels}")

    run_id = f"stationary_ellipse_curvature_{case}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / BENCHMARK / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "benchmark": BENCHMARK,
        "tier": "curvature-diagnostic",
        "case": case,
        "a": ellipse_case.a,
        "b": ellipse_case.b,
        "levels": levels,
        "methods": methods,
        "case_source": route_relative(CURVATURE_FIELD_SOURCE),
        "field_contract": {
            "initial_level_set": "DistancePointEllipse(ELLIPSE_A,ELLIPSE_B,x,y,&qx,&qy) (vendor distance_point_ellipse.h)",
            "native_hk": "Delta*distance_curvature(point,d)",
            "analytic_hk": "Delta*ab/(a^2 sin^2(theta)+b^2 cos^2(theta))^1.5",
            "model_output": "h*kappa",
            "solver_conversion": "kappa=h*kappa/Delta",
            "force_band": "|d|<=2*Delta",
        },
        "not_a_stationary_bubble_gate": True,
        "not_an_ellipse_regression_benchmark": True,
    }
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in levels:
        geometry_compile_dir = run_root / "build" / "geometry" / f"L{level}"
        geometry_compile_dir.mkdir(parents=True, exist_ok=True)
        geometry_evidence = _compile_ellipse_geometry_object(
            ellipse_case=ellipse_case, compile_dir=geometry_compile_dir
        )
        for method in methods:
            compile_evidence = compile_curvature_binary(
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                case_source=CURVATURE_FIELD_SOURCE,
                case_dir=CASE_DIR,
                binary_stem=f"stationary_ellipse_curvature_{case.lower()}",
                extra_qcc_flags=_extra_qcc_flags_for(ellipse_case),
                extra_link_objects=(geometry_evidence["geometry_object"],),
            )
            row, summary = run_curvature_binary_one(
                run_id=run_id,
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                compile_evidence=compile_evidence,
                resolved_config_path=resolved_config_path,
                resolved_config_sha256=resolved_config_sha256,
                benchmark=BENCHMARK,
                case_id=f"stationary_ellipse_curvature_field_{case}",
            )
            summary["case"] = case
            summary["a"] = ellipse_case.a
            summary["b"] = ellipse_case.b
            summary["geometry_object_sha256"] = geometry_evidence["geometry_object_sha256"]
            summary["geometry_source_sha256"] = geometry_evidence["geometry_source_sha256"]
            summary["analytic_reference"] = _analytic_reference_summary(Path(summary["field_csv"]))

            rows.append(row)
            summaries.append(summary)

    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": BENCHMARK,
        "tier": "curvature-diagnostic",
        "case": case,
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)
    report = write_curvature_diagnostic_report(
        run_id=run_id,
        methods=methods,
        levels=levels,
        summaries=summaries,
        manifest_path=manifest_path,
        benchmark=BENCHMARK,
    )
    report["case"] = case
    report["roughness_comparison"] = _roughness_comparison(summaries)
    return report


def _analytic_reference_summary(field_csv: Path) -> dict[str, Any]:
    with field_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    relabeled = _relabel_rows_for_analytic_reference(records)
    interface_relabeled = [row for row in relabeled if float(row["abs_d_over_delta"]) <= 1.0]
    return {
        "force_band": curvature_band_stats(relabeled, band_name="force_band"),
        "interface": curvature_band_stats(interface_relabeled, band_name="interface"),
    }


def _roughness_comparison(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {summary["method"]: summary for summary in summaries}
    if "NN27_RAW" not in by_method or "NN27_D4" not in by_method:
        return {}
    raw = by_method["NN27_RAW"]["analytic_reference"]["force_band"]
    d4 = by_method["NN27_D4"]["analytic_reference"]["force_band"]
    fields = ("std_delta_kappa", "max_abs_delta_hk", "p95_abs_delta_hk")
    ratios = {}
    losses = 0
    for field in fields:
        ratio = d4[field] / raw[field] if raw[field] else float("nan")
        ratios[f"{field}_ratio"] = ratio
        if ratio > 1.2:
            losses += 1
    ratios["fields_lost"] = losses
    ratios["roughness_diagnostic_failure"] = losses >= 2
    ratios["comparison"] = "NN27_D4_vs_NN27_RAW"
    return ratios


def run_curvature_process(*, case: str, methods: list[str], levels: list[int]) -> dict[str, Any]:
    ellipse_case = _resolve_case(case)
    source_audit = load_required_source_audit()
    methods = methods or ["NN_DISABLE", "NN_PROBE_ONLY", *PAPER_DEPLOYABLE_METHODS]
    unknown = sorted(set(methods) - set(CURVATURE_PROCESS_METHODS))
    if unknown:
        raise StationaryGateError(f"stationary_ellipse_curvature_process_unknown_methods:{','.join(unknown)}")
    unknown_levels = sorted(set(levels) - set(CURVATURE_PROCESS_LEVELS))
    if unknown_levels:
        raise StationaryGateError(f"stationary_ellipse_curvature_process_unknown_levels:{unknown_levels}")

    diameter_equiv = 2.0 * (ellipse_case.a * ellipse_case.b) ** 0.5
    laplace = 12000.0
    mu = (diameter_equiv / laplace) ** 0.5
    expected_tmax = diameter_equiv**2 / mu

    run_id = f"stationary_ellipse_curvature_process_{case}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / BENCHMARK / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "benchmark": BENCHMARK,
        "tier": "curvature-process",
        "case": case,
        "a": ellipse_case.a,
        "b": ellipse_case.b,
        "levels": levels,
        "methods": methods,
        "case_source": route_relative(CURVATURE_PROCESS_SOURCE),
        "snapshot_fractions": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        "diameter_equiv": diameter_equiv,
        "expected_tmax": expected_tmax,
        "not_a_stationary_bubble_gate": True,
    }
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    native_trace_csv_by_level: dict[int, Path] = {}
    for level in levels:
        geometry_compile_dir = run_root / "build" / "geometry" / f"L{level}"
        geometry_compile_dir.mkdir(parents=True, exist_ok=True)
        geometry_evidence = _compile_ellipse_geometry_object(
            ellipse_case=ellipse_case, compile_dir=geometry_compile_dir
        )
        ordered_methods = [m for m in ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4") if m in methods]
        for method in ordered_methods:
            compile_evidence = compile_curvature_process_binary(
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                case_source=CURVATURE_PROCESS_SOURCE,
                case_dir=CASE_DIR,
                binary_stem=f"stationary_ellipse_curvature_process_{case.lower()}",
                extra_qcc_flags=_extra_qcc_flags_for(ellipse_case),
                extra_link_objects=(geometry_evidence["geometry_object"],),
            )
            row, summary = run_curvature_process_binary_one(
                run_id=run_id,
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                compile_evidence=compile_evidence,
                resolved_config_path=resolved_config_path,
                resolved_config_sha256=resolved_config_sha256,
                benchmark=BENCHMARK,
                expected_tmax=expected_tmax,
            )
            summary["case"] = case
            summary["a"] = ellipse_case.a
            summary["b"] = ellipse_case.b
            if method == "NN_DISABLE":
                native_trace_csv_by_level[level] = Path(summary["surface_tension_trace_csv"])
            rows.append(row)
            summaries.append(summary)

    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": BENCHMARK,
        "tier": "curvature-process",
        "case": case,
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)

    primary_level = levels[0]
    native_trace_override = native_trace_csv_by_level.get(primary_level)
    report = write_curvature_process_report(
        run_id=run_id,
        methods=methods,
        levels=levels,
        summaries=summaries,
        manifest_path=manifest_path,
        benchmark=BENCHMARK,
        artifact_prefix=f"stationary_ellipse_curvature_process_{case.lower()}",
        native_trace_csv_override=native_trace_override,
        trace_yscale="log",
    )
    report["case"] = case
    return report
