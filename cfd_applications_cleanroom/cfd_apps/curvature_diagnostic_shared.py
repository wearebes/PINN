"""Shared compile/run/hash-guard and curvature-statistics helpers for the
stationary-bubble curvature-diagnostic and curvature-process tiers.

Extracted from `stationary.py` (behavior-preserving refactor, see
`docs/superpowers/plans/2026-07-03-stationary-ellipse-curvature-stress-diagnostic.md`
Rev 3, section 12.1) so both the circle route (`stationary.py`) and the
ellipse route (`stationary_ellipse.py`) call one implementation instead of
maintaining two copies of the source-immutability guard and the curvature
error statistics. Every function here is generic over `case_source`,
`case_dir`, `benchmark`, `case_id`; the circle-specific values are bound by
thin wrappers in `stationary.py`, not hardcoded here.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.hashes import (
    manifest_digest,
    sha256_file,
    tree_manifest,
)
from cfd_applications_cleanroom.cfd_apps.paths import (
    FIGURES,
    RAW,
    REPORTS,
    ROOT,
    SOURCE_DATA,
    route_relative,
)
from cfd_applications_cleanroom.cfd_apps.schema import MANIFEST_COLUMNS, validate_no_duplicate_time_rows


class StationaryGateError(RuntimeError):
    """Raised when manifest-backed evidence is absent or unsafe for this route.

    Defined here (not in stationary.py) so both the circle and ellipse
    routes raise/catch the same exception identity.
    """


BUILD_SRC = ROOT / "cfd_applications_cleanroom/build/basilisk_arm64/src"
VENDOR_SRC = ROOT / "cfd_applications_cleanroom/vendor/basilisk_clean/src"
SOURCE_AUDIT = ROOT / "cfd_applications_cleanroom/results/manifests/source_audit.json"
NN_DIR = ROOT / "cfd_applications_cleanroom/nn"
NN_EXPORT_MANIFEST = NN_DIR / "generated/export_manifest.json"
NN_FORWARD_SOURCE = NN_DIR / "nn_forward_clean.c"
NN_MODE_VALUES = {
    "NN27_RAW": 2,
}
NN_LABELS = {
    "NN27_RAW": "nn27_r128",
}
METHOD_DISPLAY_LABELS = {
    "NN27_RAW": "NN",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


# ==== _compile_curvature_binary (516-605) ====
def compile_curvature_binary(
    *,
    method: str,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
    case_source: Path,
    case_dir: Path,
    binary_stem: str,
    extra_qcc_flags: tuple[str, ...] = (),
    extra_link_objects: tuple[Path, ...] = (),
) -> dict[str, Any]:
    compile_dir = run_root / "build" / method / f"L{level}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    binary = compile_dir / f"{binary_stem}_{method.lower()}_L{level}"
    compile_log = compile_dir / "compile.log"
    source_hashes_json = compile_dir / "source_hashes_compile.json"
    nn_metadata = canary_nn_metadata(method)

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    case_sha = sha256_file(case_source)
    nn_object_evidence = compile_nn_forward_object(
        compile_dir=compile_dir,
        nn_metadata=nn_metadata,
    )
    cmd = [
        str(BUILD_SRC / "qcc"),
        *extra_qcc_flags,
        "-autolink",
        "-O2",
        f"-DLEVEL={level}",
        f"-DNN_MODE={NN_MODE_VALUES[method]}",
        f"-DMETHOD_ID=\\\"{method}\\\"",
        f"-I{NN_DIR}",
        case_source.name,
        str(nn_object_evidence["nn_object"]),
        *[str(obj) for obj in extra_link_objects],
        "-o",
        str(binary),
        "-lm",
    ]
    env = os.environ.copy()
    env["BASILISK"] = str(BUILD_SRC)
    compile_result = subprocess.run(
        cmd,
        cwd=case_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    compile_log.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={case_dir}\nreturncode={compile_result.returncode}\n\n"
        + compile_result.stdout
    )
    vendor_after_compile = tree_manifest(VENDOR_SRC, source_only=True)
    build_after_compile = tree_manifest(BUILD_SRC, source_only=True)
    source_hashes = {
        "case_source_sha256": case_sha,
        "nn_forward_source_sha256": nn_object_evidence.get("nn_forward_source_sha256", ""),
        "nn_object_sha256": nn_object_evidence.get("nn_object_sha256", ""),
        "nn_object_source_hashes_unchanged": nn_object_evidence.get("source_hashes_unchanged", ""),
        "vendor_source_hash_before": manifest_digest(vendor_before),
        "vendor_source_hash_after_compile": manifest_digest(vendor_after_compile),
        "vendor_source_hashes_unchanged": vendor_before == vendor_after_compile,
        "build_source_hash_before": manifest_digest(build_before),
        "build_source_hash_after_compile": manifest_digest(build_after_compile),
        "build_source_hashes_unchanged": build_before == build_after_compile,
    }
    source_hashes_json.write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    if compile_result.returncode != 0 or not binary.exists():
        raise StationaryGateError(f"stationary_curvature_compile_failed:{route_relative(compile_log)}")
    if vendor_before != vendor_after_compile:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_curvature_compile")
    if build_before != build_after_compile:
        raise StationaryGateError("route_build_source_mutated_during_stationary_curvature_compile")
    return {
        "level": level,
        "method": method,
        "binary": binary,
        "binary_sha256": sha256_file(binary),
        "compile_command": " ".join(cmd),
        "nn_object_compile_command": nn_object_evidence.get("nn_object_compile_command", ""),
        "nn_object_compile_log": nn_object_evidence.get("nn_object_compile_log", ""),
        "compile_log": compile_log,
        "compile_returncode": compile_result.returncode,
        "case_source": case_source,
        "case_source_sha256": case_sha,
        "source_hashes_compile_json": source_hashes_json,
        "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
        "checkpoint_path": nn_metadata.get("checkpoint_path", ""),
        "checkpoint_sha256": nn_metadata.get("checkpoint_sha256", ""),
        "nn_weights_header_sha256": nn_metadata.get("weights_header_sha256", ""),
    }


# ==== _compile_curvature_process_binary (607-698) ====
def compile_curvature_process_binary(
    *,
    method: str,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
    case_source: Path,
    case_dir: Path,
    binary_stem: str,
    extra_qcc_flags: tuple[str, ...] = (),
    extra_link_objects: tuple[Path, ...] = (),
) -> dict[str, Any]:
    compile_dir = run_root / "build" / method / f"L{level}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    binary = compile_dir / f"{binary_stem}_{method.lower()}_L{level}"
    compile_log = compile_dir / "compile.log"
    source_hashes_json = compile_dir / "source_hashes_compile.json"
    nn_metadata = canary_nn_metadata(method)

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    case_sha = sha256_file(case_source)
    nn_object_evidence = compile_nn_forward_object(
        compile_dir=compile_dir,
        nn_metadata=nn_metadata,
    )
    cmd = [
        str(BUILD_SRC / "qcc"),
        *extra_qcc_flags,
        "-autolink",
        "-O2",
        f"-DLEVEL={level}",
        f"-DNN_MODE={NN_MODE_VALUES[method]}",
        f"-DMETHOD_ID=\\\"{method}\\\"",
        f"-I{NN_DIR}",
        case_source.name,
        str(nn_object_evidence["nn_object"]),
        *[str(obj) for obj in extra_link_objects],
        "-o",
        str(binary),
        "-lm",
    ]
    env = os.environ.copy()
    env["BASILISK"] = str(BUILD_SRC)
    compile_result = subprocess.run(
        cmd,
        cwd=case_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    compile_log.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={case_dir}\nreturncode={compile_result.returncode}\n\n"
        + compile_result.stdout
    )
    vendor_after_compile = tree_manifest(VENDOR_SRC, source_only=True)
    build_after_compile = tree_manifest(BUILD_SRC, source_only=True)
    source_hashes = {
        "case_source_sha256": case_sha,
        "nn_forward_source_sha256": nn_object_evidence.get("nn_forward_source_sha256", ""),
        "nn_object_sha256": nn_object_evidence.get("nn_object_sha256", ""),
        "nn_object_source_hashes_unchanged": nn_object_evidence.get("source_hashes_unchanged", ""),
        "vendor_source_hash_before": manifest_digest(vendor_before),
        "vendor_source_hash_after_compile": manifest_digest(vendor_after_compile),
        "vendor_source_hashes_unchanged": vendor_before == vendor_after_compile,
        "build_source_hash_before": manifest_digest(build_before),
        "build_source_hash_after_compile": manifest_digest(build_after_compile),
        "build_source_hashes_unchanged": build_before == build_after_compile,
    }
    source_hashes_json.write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    if compile_result.returncode != 0 or not binary.exists():
        raise StationaryGateError(
            f"stationary_curvature_process_compile_failed:{route_relative(compile_log)}"
        )
    if vendor_before != vendor_after_compile:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_curvature_process_compile")
    if build_before != build_after_compile:
        raise StationaryGateError("route_build_source_mutated_during_stationary_curvature_process_compile")
    return {
        "level": level,
        "method": method,
        "binary": binary,
        "binary_sha256": sha256_file(binary),
        "compile_command": " ".join(cmd),
        "nn_object_compile_command": nn_object_evidence.get("nn_object_compile_command", ""),
        "nn_object_compile_log": nn_object_evidence.get("nn_object_compile_log", ""),
        "compile_log": compile_log,
        "compile_returncode": compile_result.returncode,
        "case_source": case_source,
        "case_source_sha256": case_sha,
        "source_hashes_compile_json": source_hashes_json,
        "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
        "checkpoint_path": nn_metadata.get("checkpoint_path", ""),
        "checkpoint_sha256": nn_metadata.get("checkpoint_sha256", ""),
        "nn_weights_header_sha256": nn_metadata.get("weights_header_sha256", ""),
    }


# ==== _compile_nn_forward_object (700-749) ====
def compile_nn_forward_object(*, compile_dir: Path, nn_metadata: dict[str, str]) -> dict[str, Any]:
    object_path = compile_dir / "nn_forward_clean.o"
    compile_log = compile_dir / "nn_forward_compile.log"
    weights_header = str(nn_metadata.get("weights_header", ""))
    if not weights_header:
        raise StationaryGateError("nn_forward_object_missing_weights_header")
    weights_header_relative = "generated/" + Path(weights_header).name
    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    cmd = [
        os.environ.get("CC", "cc"),
        "-std=c99",
        "-O2",
        f"-I{NN_DIR}",
        f'-DNN_WEIGHTS_HEADER="{weights_header_relative}"',
        "-c",
        str(NN_FORWARD_SOURCE),
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
        raise StationaryGateError(f"nn_forward_object_compile_failed:{route_relative(compile_log)}")
    if vendor_before != vendor_after:
        raise StationaryGateError("route_vendor_source_mutated_during_nn_forward_object_compile")
    if build_before != build_after:
        raise StationaryGateError("route_build_source_mutated_during_nn_forward_object_compile")
    return {
        "nn_object": object_path,
        "nn_object_sha256": sha256_file(object_path),
        "nn_forward_source_sha256": sha256_file(NN_FORWARD_SOURCE),
        "nn_object_compile_command": " ".join(cmd),
        "nn_object_compile_log": str(compile_log),
        "source_hashes_unchanged": True,
    }


# ==== _run_curvature_binary_one (990-1119) ====
def run_curvature_binary_one(
    *,
    run_id: str,
    method: str,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
    compile_evidence: dict[str, Any],
    resolved_config_path: Path,
    resolved_config_sha256: str,
    benchmark: str,
    case_id: str,
    extra_summary_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{method}_L{level}"
    out_dir = run_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(compile_evidence["binary"])
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    status_file = out_dir / "status.json"
    field_csv = out_dir / "curvature_field.csv"
    raw_csv = out_dir / "curvature_trace.csv"
    summary_json = out_dir / "summary.json"
    source_hashes_json = out_dir / "source_hashes.json"

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)

    run_env = os.environ.copy()
    run_env["BASILISK"] = str(BUILD_SRC)
    run_env["CLEANROOM_RUN_ID"] = run_id
    run_result = subprocess.run(
        [str(binary)],
        cwd=out_dir,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_log.write_text(run_result.stdout)
    stderr_log.write_text(run_result.stderr)
    vendor_after_run = tree_manifest(VENDOR_SRC, source_only=True)
    build_after_run = tree_manifest(BUILD_SRC, source_only=True)
    source_hashes = {
        "compile_source_hashes_json": str(compile_evidence["source_hashes_compile_json"]),
        "vendor_source_hash_before_run": manifest_digest(vendor_before),
        "vendor_source_hash_after_run": manifest_digest(vendor_after_run),
        "vendor_source_hashes_unchanged": vendor_before == vendor_after_run,
        "build_source_hash_before_run": manifest_digest(build_before),
        "build_source_hash_after_run": manifest_digest(build_after_run),
        "build_source_hashes_unchanged": build_before == build_after_run,
    }
    source_hashes_json.write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    if vendor_before != vendor_after_run:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_curvature_run")
    if build_before != build_after_run:
        raise StationaryGateError("route_build_source_mutated_during_stationary_curvature_run")
    if run_result.returncode != 0 or not field_csv.exists():
        raise StationaryGateError(f"stationary_curvature_run_failed:{route_relative(stderr_log)}")

    summary = summarize_curvature_field_csv(
        field_csv,
        run_id=run_id,
        method=method,
        level=level,
        benchmark=benchmark,
    )
    raw_csv.write_text(
        "run_id,repeat_id,benchmark,case_id,method,deployable,evidence_level,level,grid_n,i,t,dt,primary_metric_name,primary_metric_value\n"
        f"{run_id},0,{benchmark},{case_id},{method},false,controlled_diagnostic,"
        f"{level},{1 << level},0,0,0,curvature_field_rows,{summary['raw_row_count']}\n"
    )
    summary.update(
        {
            "resolved_config_sha256": resolved_config_sha256,
            "resolved_config_json": str(resolved_config_path),
            "binary_sha256": compile_evidence["binary_sha256"],
            "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
            "source_hashes_unchanged": source_hashes["vendor_source_hashes_unchanged"]
            and source_hashes["build_source_hashes_unchanged"],
            "raw_schema_valid": True,
            "raw_csv": str(raw_csv),
            "field_csv": str(field_csv),
            "status_file": str(status_file),
        }
    )
    if extra_summary_fields:
        summary.update(extra_summary_fields)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    status = {
        "status": "curvature_diagnostic",
        "method": method,
        "run_returncode": run_result.returncode,
        "compile_returncode": compile_evidence["compile_returncode"],
    }
    status_file.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "repeat_id": "0",
            "benchmark": benchmark,
            "method": method,
            "level": str(level),
            "grid_n": str(1 << level),
            "case_source_path": route_relative(Path(compile_evidence["case_source"])),
            "case_source_sha256": compile_evidence["case_source_sha256"],
            "official_source_root": route_relative(VENDOR_SRC),
            "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
            "compile_command": compile_evidence["compile_command"],
            "run_command": str(binary),
            "compile_log": str(compile_evidence["compile_log"]),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "status_file": str(status_file),
            "raw_csv": str(raw_csv),
            "field_csv": str(field_csv),
            "summary_json": str(summary_json),
            "checkpoint_path": compile_evidence["checkpoint_path"],
            "checkpoint_sha256": compile_evidence["checkpoint_sha256"],
            "nn_weights_header_sha256": compile_evidence["nn_weights_header_sha256"],
            "result_status": "curvature_diagnostic",
            "evidence_level": "controlled_diagnostic",
            "resolved_config_json": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "binary": str(binary),
            "binary_sha256": compile_evidence["binary_sha256"],
            "source_hashes_json": str(source_hashes_json),
            "source_hashes_compile_json": str(compile_evidence["source_hashes_compile_json"]),
        }
    )
    return row, summary


# ==== _run_curvature_process_binary_one (1121-1252) ====
def run_curvature_process_binary_one(
    *,
    run_id: str,
    method: str,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
    compile_evidence: dict[str, Any],
    resolved_config_path: Path,
    resolved_config_sha256: str,
    benchmark: str,
    expected_tmax: float,
    extra_summary_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{method}_L{level}"
    out_dir = run_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(compile_evidence["binary"])
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    status_file = out_dir / "status.json"
    process_csv = out_dir / "curvature_process.csv"
    trace_csv = out_dir / "surface_tension_trace.csv"
    summary_json = out_dir / "summary.json"
    source_hashes_json = out_dir / "source_hashes.json"

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)

    run_env = os.environ.copy()
    run_env["BASILISK"] = str(BUILD_SRC)
    run_env["CLEANROOM_RUN_ID"] = run_id
    run_env["CLEANROOM_REPEAT_ID"] = "0"
    run_result = subprocess.run(
        [str(binary)],
        cwd=out_dir,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_log.write_text(run_result.stdout)
    stderr_log.write_text(run_result.stderr)
    vendor_after_run = tree_manifest(VENDOR_SRC, source_only=True)
    build_after_run = tree_manifest(BUILD_SRC, source_only=True)
    source_hashes = {
        "compile_source_hashes_json": str(compile_evidence["source_hashes_compile_json"]),
        "vendor_source_hash_before_run": manifest_digest(vendor_before),
        "vendor_source_hash_after_run": manifest_digest(vendor_after_run),
        "vendor_source_hashes_unchanged": vendor_before == vendor_after_run,
        "build_source_hash_before_run": manifest_digest(build_before),
        "build_source_hash_after_run": manifest_digest(build_after_run),
        "build_source_hashes_unchanged": build_before == build_after_run,
    }
    source_hashes_json.write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    if vendor_before != vendor_after_run:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_curvature_process_run")
    if build_before != build_after_run:
        raise StationaryGateError("route_build_source_mutated_during_stationary_curvature_process_run")
    if run_result.returncode != 0 or not process_csv.exists() or not trace_csv.exists():
        raise StationaryGateError(
            f"stationary_curvature_process_run_failed:{route_relative(stderr_log)}"
        )

    validate_no_duplicate_time_rows(trace_csv)
    summary = summarize_curvature_process_csvs(
        process_csv=process_csv,
        trace_csv=trace_csv,
        run_id=run_id,
        method=method,
        level=level,
        benchmark=benchmark,
        expected_tmax=expected_tmax,
    )
    summary.update(
        {
            "resolved_config_sha256": resolved_config_sha256,
            "resolved_config_json": str(resolved_config_path),
            "binary_sha256": compile_evidence["binary_sha256"],
            "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
            "source_hashes_unchanged": source_hashes["vendor_source_hashes_unchanged"]
            and source_hashes["build_source_hashes_unchanged"],
            "raw_schema_valid": True,
            "curvature_process_csv": str(process_csv),
            "surface_tension_trace_csv": str(trace_csv),
            "status_file": str(status_file),
        }
    )
    if extra_summary_fields:
        summary.update(extra_summary_fields)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    status = {
        "status": "curvature_process_diagnostic",
        "method": method,
        "run_returncode": run_result.returncode,
        "compile_returncode": compile_evidence["compile_returncode"],
        "evidence_level": "controlled_diagnostic",
    }
    status_file.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "repeat_id": "0",
            "benchmark": benchmark,
            "method": method,
            "level": str(level),
            "grid_n": str(1 << level),
            "case_source_path": route_relative(Path(compile_evidence["case_source"])),
            "case_source_sha256": compile_evidence["case_source_sha256"],
            "official_source_root": route_relative(VENDOR_SRC),
            "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
            "compile_command": compile_evidence["compile_command"],
            "run_command": str(binary),
            "compile_log": str(compile_evidence["compile_log"]),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "status_file": str(status_file),
            "raw_csv": str(trace_csv),
            "curvature_process_csv": str(process_csv),
            "surface_tension_trace_csv": str(trace_csv),
            "summary_json": str(summary_json),
            "checkpoint_path": compile_evidence["checkpoint_path"],
            "checkpoint_sha256": compile_evidence["checkpoint_sha256"],
            "nn_weights_header_sha256": compile_evidence["nn_weights_header_sha256"],
            "result_status": "curvature_process_diagnostic",
            "evidence_level": "controlled_diagnostic",
            "resolved_config_json": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "binary": str(binary),
            "binary_sha256": compile_evidence["binary_sha256"],
            "source_hashes_json": str(source_hashes_json),
            "source_hashes_compile_json": str(compile_evidence["source_hashes_compile_json"]),
        }
    )
    return row, summary


# ==== _summarize_curvature_field_csv (1254-1298) ====
def summarize_curvature_field_csv(
    raw_csv: Path,
    *,
    run_id: str,
    method: str,
    level: int,
    benchmark: str,
) -> dict[str, Any]:
    with raw_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise StationaryGateError(f"stationary_curvature_empty_raw_csv:{route_relative(raw_csv)}")
    if any(record["run_id"] != run_id for record in records):
        raise StationaryGateError("stationary_curvature_run_id_mismatch")
    if any(record["method"] != method for record in records):
        raise StationaryGateError("stationary_curvature_method_mismatch")

    force_band = curvature_band_stats(records, band_name="force_band")
    interface_records = [row for row in records if float(row["abs_d_over_delta"]) <= 1.0]
    interface_band = curvature_band_stats(interface_records, band_name="interface")
    raw_prefilter = raw_prefilter_stats(records)

    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "tier": "curvature-diagnostic",
        "method": method,
        "method_role": _method_role(method),
        "level": level,
        "grid_n": 1 << level,
        "raw_row_count": len(records),
        "force_band": force_band,
        "interface": interface_band,
        "raw_prefilter": raw_prefilter,
        "sign_scale": {
            "model_output": "h*kappa",
            "solver_conversion": "kappa=h*kappa/Delta",
            "scale_identity_max_abs": force_band["scale_identity_max_abs"],
            "sign_agreement_fraction": force_band["sign_agreement_fraction"],
            "rmse_direct": force_band["rmse_direct"],
            "rmse_negated": force_band["rmse_negated"],
            "sign_verdict": "same_sign"
            if force_band["rmse_direct"] <= force_band["rmse_negated"]
            else "negated_sign_closer",
        },
        "raw_csv": str(raw_csv),
    }


def raw_prefilter_stats(records: list[dict[str, str]]) -> dict[str, Any]:
    if not records or "hk_nn_raw" not in records[0]:
        return {"status": "absent"}
    delta_raw = [float(row["delta_hk_raw"]) for row in records]
    abs_raw = [abs(value) for value in delta_raw]
    return {
        "status": "present",
        "mean_delta_hk_raw": mean_(delta_raw),
        "p95_abs_delta_hk_raw": percentile(abs_raw, 0.95),
        "max_abs_delta_hk_raw": max(abs_raw),
    }


# ==== _summarize_curvature_process_csvs (1300-1367) ====
def summarize_curvature_process_csvs(
    *,
    process_csv: Path,
    trace_csv: Path,
    run_id: str,
    method: str,
    level: int,
    benchmark: str,
    expected_tmax: float,
) -> dict[str, Any]:
    with process_csv.open(newline="") as handle:
        process_records = list(csv.DictReader(handle))
    with trace_csv.open(newline="") as handle:
        trace_records = list(csv.DictReader(handle))
    if not process_records:
        raise StationaryGateError(f"stationary_curvature_process_empty_csv:{route_relative(process_csv)}")
    if not trace_records:
        raise StationaryGateError(f"stationary_curvature_process_empty_trace:{route_relative(trace_csv)}")
    if any(record["run_id"] != run_id for record in process_records + trace_records):
        raise StationaryGateError("stationary_curvature_process_run_id_mismatch")
    if any(record["method"] != method for record in process_records + trace_records):
        raise StationaryGateError("stationary_curvature_process_method_mismatch")

    snapshots: dict[str, Any] = {}
    for snapshot_index in sorted({int(row["snapshot_index"]) for row in process_records}):
        rows = [row for row in process_records if int(row["snapshot_index"]) == snapshot_index]
        interface_rows = [row for row in rows if float(row["abs_d_over_delta"]) <= 1.0]
        snapshots[str(snapshot_index)] = {
            "snapshot_index": snapshot_index,
            "snapshot_fraction": float(rows[0]["snapshot_fraction"]),
            "t": float(rows[0]["t"]),
            "tau": float(rows[0]["tau"]),
            "row_count": len(rows),
            "force_band": curvature_band_stats(rows, band_name="force_band"),
            "interface": curvature_band_stats(interface_rows, band_name="interface"),
            "raw_prefilter": raw_prefilter_stats(rows),
        }
    if sorted(snapshots) != ["0", "1", "2", "3"]:
        raise StationaryGateError("stationary_curvature_process_missing_snapshots")

    ca = [float(record["Ca"]) for record in trace_records]
    tail_start = max(0, int(len(ca) * 0.8))
    tail = ca[tail_start:] or ca
    final_t = float(trace_records[-1]["t"])
    final_tau = float(trace_records[-1]["tau"])

    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "tier": "curvature-process",
        "method": method,
        "level": level,
        "grid_n": 1 << level,
        "snapshot_fractions": [snapshots[str(idx)]["snapshot_fraction"] for idx in range(4)],
        "snapshots": snapshots,
        "trace_row_count": len(trace_records),
        "final_t": final_t,
        "final_tau": final_tau,
        "expected_tmax": expected_tmax,
        "reached_final_time": final_t >= expected_tmax * 0.999,
        "finite_Ca": all(math.isfinite(value) for value in ca),
        "Ca_max": max(ca),
        "Ca_tail_max": max(tail),
        "Ca_tail_mean": sum(tail) / len(tail),
        "Ca_final": ca[-1],
        "sigma": float(trace_records[0]["sigma"]),
        "curvature_process_csv": str(process_csv),
        "surface_tension_trace_csv": str(trace_csv),
    }


# ==== _write_curvature_process_report (1369-1396) ====
def write_curvature_process_report(
    *,
    run_id: str,
    methods: list[str],
    levels: list[int],
    summaries: list[dict[str, Any]],
    manifest_path: Path,
    benchmark: str,
    artifact_prefix: str,
    native_trace_csv_override: Path | None = None,
    trace_yscale: str = "linear",
) -> dict[str, Any]:
    figures = render_curvature_process_plate(
        run_id=run_id,
        summaries=summaries,
        artifact_prefix=artifact_prefix,
        native_trace_csv_override=native_trace_csv_override,
        trace_yscale=trace_yscale,
    )
    report = {
        "run_id": run_id,
        "benchmark": benchmark,
        "tier": "curvature-process",
        "levels": levels,
        "methods": {summary["method"]: summary for summary in summaries},
        "manifest": route_relative(manifest_path),
        "figures": figures,
        "overall_status": "PASS",
        "evidence_level": "controlled_diagnostic",
        "primary_metric": "Ca=mu*Umax/sigma",
        "primary_report_gate_metric": "Ca_tail_max",
    }
    report_path = REPORTS / f"{run_id}_curvature_process_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path = REPORTS / f"{run_id}_curvature_process_summary.md"
    md_path.write_text(render_curvature_process_report(report))
    return report


# ==== _render_curvature_process_report (1398-1426) ====
def render_curvature_process_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stationary Curvature Process Diagnostic",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        f"- primary_metric: `{report['primary_metric']}`",
        f"- report_gate_metric: `{report['primary_report_gate_metric']}`",
        f"- figure_png: `{report['figures']['png']}`",
        f"- source_data: `{report['figures']['source_data_csv']}`",
        "",
        "| method | Ca max | Ca tail max | reached final time | sigma |",
        "|---|---:|---:|---|---:|",
    ]
    for method, summary in report["methods"].items():
        lines.append(
            f"| {method} | {summary['Ca_max']:.6e} | {summary['Ca_tail_max']:.6e} | "
            f"{summary['reached_final_time']} | {summary['sigma']:.6g} |"
        )
    lines.extend(["", "## Snapshot Curvature Error", "", "| method | fraction | tau | max |Δ(hκ)| | stdΓ κ_NN |", "|---|---:|---:|---:|---:|"])
    for method, summary in report["methods"].items():
        for snapshot in summary["snapshots"].values():
            force = snapshot["force_band"]
            lines.append(
                f"| {method} | {snapshot['snapshot_fraction']:.6g} | {snapshot['tau']:.6g} | "
                f"{force['max_abs_delta_hk']:.6e} | {force['force_band_std_kappa_nn']:.6e} |"
            )
    return "\n".join(lines) + "\n"


# ==== _render_curvature_process_plate (1428-1571) ====
def render_curvature_process_plate(
    *,
    run_id: str,
    summaries: list[dict[str, Any]],
    artifact_prefix: str,
    native_trace_csv_override: Path | None = None,
    trace_yscale: str = "linear",
    curvature_ylim_quantiles: tuple[float, float] | None = None,
    curvature_ylim_scope: str = "global",
    curvature_ytick_format: str | None = None,
) -> dict[str, str]:
    try:
        mpl_config_dir = ROOT / "cfd_applications_cleanroom/results/tmp/matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise StationaryGateError(f"python_plot_runtime_missing:{exc}") from exc

    if not summaries:
        raise StationaryGateError("stationary_curvature_process_no_summaries_for_figure")
    primary = next((summary for summary in summaries if summary["method"] == "NN27_RAW"), summaries[0])
    process_csv = Path(primary["curvature_process_csv"])
    trace_csv = Path(primary["surface_tension_trace_csv"])
    with process_csv.open(newline="") as handle:
        process_records = list(csv.DictReader(handle))
    with trace_csv.open(newline="") as handle:
        trace_records = list(csv.DictReader(handle))
    clsvof_trace_csv = native_trace_csv_override if native_trace_csv_override is not None else find_clsvof_trace_for_process(primary)
    clsvof_trace_records: list[dict[str, str]] = []
    if clsvof_trace_csv is not None:
        with clsvof_trace_csv.open(newline="") as handle:
            clsvof_trace_records = list(csv.DictReader(handle))

    source_data_csv = SOURCE_DATA / f"{run_id}_{artifact_prefix}_plate_source_data.csv"
    expanded_rows = write_curvature_process_source_data(
        source_data_csv=source_data_csv,
        process_records=process_records,
        trace_records=trace_records,
        clsvof_trace_records=clsvof_trace_records,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.1, 5.45), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[0.16, 1.0, 1.0, 0.9])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = [fig.add_subplot(grid[1 + idx // 2, idx % 2]) for idx in range(4)]
    trace_ax = fig.add_subplot(grid[3, :])
    colors = {"native": "#4c566a", "nn": "#0072b2"}
    labels = {
        0: "t/T = 0",
        1: "t/T = 1/3",
        2: "t/T = 2/3",
        3: "t/T = 1",
    }
    def _curvature_ylim(rows: list[dict[str, str]]) -> tuple[float, float]:
        y_values = [float(row["hk_native"]) for row in rows] + [float(row["hk_nn"]) for row in rows]
        if not y_values:
            return (-0.1, 0.1)
        if curvature_ylim_quantiles is not None:
            sorted_y = sorted(y_values)
            lo_q, hi_q = curvature_ylim_quantiles
            lo_index = min(len(sorted_y) - 1, max(0, int(lo_q * (len(sorted_y) - 1))))
            hi_index = min(len(sorted_y) - 1, max(0, int(hi_q * (len(sorted_y) - 1))))
            y_min, y_max = sorted_y[lo_index], sorted_y[hi_index]
        else:
            y_min, y_max = min(y_values), max(y_values)
        y_pad = max((y_max - y_min) * 0.08, 1e-4)
        return (y_min - y_pad, y_max + y_pad)

    curvature_rows = [row for row in expanded_rows if row["row_type"] == "curvature"]
    y_lim = _curvature_ylim(curvature_rows)
    if curvature_ylim_scope not in {"global", "snapshot"}:
        raise StationaryGateError(f"invalid_curvature_ylim_scope:{curvature_ylim_scope}")
    for snapshot_index, ax in enumerate(axes):
        rows = [
            row for row in expanded_rows
            if row["row_type"] == "curvature" and int(row["snapshot_index"]) == snapshot_index
        ]
        rows.sort(key=lambda row: float(row["theta_deg_360"]))
        theta = [float(row["theta_deg_360"]) for row in rows]
        native = [float(row["hk_native"]) for row in rows]
        nn = [float(row["hk_nn"]) for row in rows]
        ax.scatter(theta, native, s=4, color=colors["native"], alpha=0.45, linewidths=0, label="CLSVOF-LS")
        ax.scatter(theta, nn, s=4, color=colors["nn"], alpha=0.55, linewidths=0, label="NN")
        ax.set_xlim(0, 360)
        ax.set_ylim(*(_curvature_ylim(rows) if curvature_ylim_scope == "snapshot" else y_lim))
        ax.set_xticks([0, 90, 180, 270, 360])
        if curvature_ytick_format:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(curvature_ytick_format))
            if curvature_ytick_format == "%.3f":
                y_lo, y_hi = ax.get_ylim()
                tick_step = 0.005 if (y_hi - y_lo) > 0.01 else 0.001
                tick_start = math.ceil(y_lo / tick_step) * tick_step
                ticks = []
                tick = tick_start
                while tick <= y_hi + tick_step * 0.5:
                    ticks.append(round(tick, 3))
                    tick += tick_step
                if ticks:
                    ax.set_yticks(ticks)
        ax.set_title(labels[snapshot_index])
        if snapshot_index in (2, 3):
            ax.set_xlabel("degree")
        if snapshot_index in (0, 2):
            ax.set_ylabel("h*kappa")
        if snapshot_index == 0:
            handles, handle_labels = ax.get_legend_handles_labels()
            legend_ax.legend(
                handles,
                handle_labels,
                loc="center",
                ncol=2,
                markerscale=2,
                handlelength=1.0,
                handletextpad=0.4,
                columnspacing=1.2,
            )

    def _final_value(values: list[float]) -> float:
        return values[-1] if values else math.nan

    def _trace_label(method: str, tau_values: list[float], ca_values: list[float]) -> str:
        # final = LAST plotted value (Ca_final semantics, or "last recorded"
        # for an incomplete trace); max = true peak of the plotted curve.
        # Legend used to print the final value mislabeled "Ca_max" until
        # 2026-07-08, which contradicted the archive's own Ca_max column.
        final_value = _final_value(ca_values)
        max_value = max(ca_values) if ca_values else math.nan
        tau_value = _final_value(tau_values)
        if math.isfinite(tau_value) and tau_value < 0.999:
            return (fr"{method} $Ca$={final_value:.2e} at $t/T$={tau_value:.2f} (incomplete), "
                    fr"$Ca_{{max}}$={max_value:.2e}")
        return fr"{method} $Ca_{{final}}$={final_value:.2e}, $Ca_{{max}}$={max_value:.2e}"

    trace_rows = [row for row in expanded_rows if row["row_type"] == "surface_tension_trace"]
    tau = [float(row["tau"]) for row in trace_rows]
    ca = [float(row["Ca"]) for row in trace_rows]
    trace_ax.plot(tau, ca, color="#0072b2", linewidth=1.1, label=_trace_label("NN", tau, ca))
    clsvof_rows = [row for row in expanded_rows if row["row_type"] == "clsvof_ls_trace"]
    if clsvof_rows:
        clsvof_tau = [float(row["tau"]) for row in clsvof_rows]
        clsvof_ca = [float(row["Ca"]) for row in clsvof_rows]
        trace_ax.plot(
            clsvof_tau, clsvof_ca, color="#4c566a", linewidth=1.1,
            label=_trace_label("CLSVOF-LS", clsvof_tau, clsvof_ca),
        )
    for frac in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        trace_ax.axvline(frac, color="#7f8c8d", linewidth=0.7, alpha=0.45)
    trace_ax.set_xlabel("t/T")
    trace_ax.set_ylabel(r"$Ca$")
    trace_ax.set_xlim(0, max(1.0, max(tau) if tau else 1.0))
    if trace_yscale != "linear":
        # Ellipse traces span a real relaxation transient (~1e-3) down to a
        # settled residual floor (~1e-5) -- on a linear axis the floor is
        # visually indistinguishable from zero. Log scale silently drops the
        # single exact-zero point at t=0 (Umax=0 there); everything else is
        # unaffected. Not applied to the circle plate: its Ca stays in one
        # narrow band throughout (no large transient), so linear is correct
        # there and changing it would disturb the already-validated
        # pixel-identical circle output for no benefit.
        trace_ax.set_yscale(trace_yscale)
    trace_ax.legend(loc="best")

    stem = FIGURES / f"{run_id}_{artifact_prefix}_plate"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest_path = SOURCE_DATA / f"{run_id}_{artifact_prefix}_plate_manifest.json"
    manifest = {
        "run_id": run_id,
        "figure": f"{artifact_prefix}_plate",
        "curvature_process_csv": str(process_csv),
        "surface_tension_trace_csv": str(trace_csv),
        "clsvof_ls_trace_csv": str(clsvof_trace_csv) if clsvof_trace_csv is not None else "",
        "source_data_csv": str(source_data_csv),
        "angle_expansion": "quadrant data mirrored to 0-360 degrees",
        "exports": {
            "svg": f"{stem}.svg",
            "pdf": f"{stem}.pdf",
            "png": f"{stem}.png",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "svg": str(stem) + ".svg",
        "pdf": str(stem) + ".pdf",
        "png": str(stem) + ".png",
        "source_data_csv": str(source_data_csv),
        "source_data_manifest": str(manifest_path),
    }


# ==== _write_curvature_process_source_data (1573-1657) ====
def write_curvature_process_source_data(
    *,
    source_data_csv: Path,
    process_records: list[dict[str, str]],
    trace_records: list[dict[str, str]],
    clsvof_trace_records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in process_records:
        if float(record["abs_d_over_delta"]) > 1.0:
            continue
        theta = float(record["theta_deg"])
        mirrored = (theta, 180.0 - theta, 180.0 + theta, 360.0 - theta)
        for mirror_index, theta360 in enumerate(mirrored):
            rows.append(
                {
                    "row_type": "curvature",
                    "mirror_index": mirror_index,
                    "snapshot_index": int(record["snapshot_index"]),
                    "snapshot_fraction": float(record["snapshot_fraction"]),
                    "t": float(record["t"]),
                    "tau": float(record["tau"]),
                    "theta_deg_360": theta360,
                    "hk_native": float(record["hk_native"]),
                    "hk_nn": float(record["hk_nn"]),
                    "delta_hk": float(record["delta_hk"]),
                    "Ca": float(record["Ca"]),
                    "sigma": float(record["sigma"]),
                }
            )
    for record in trace_records:
        rows.append(
            {
                "row_type": "surface_tension_trace",
                "mirror_index": "",
                "snapshot_index": "",
                "snapshot_fraction": "",
                "t": float(record["t"]),
                "tau": float(record["tau"]),
                "theta_deg_360": "",
                "hk_native": "",
                "hk_nn": "",
                "delta_hk": "",
                "Ca": float(record["Ca"]),
                "sigma": float(record["sigma"]),
            }
        )
    for record in clsvof_trace_records:
        rows.append(
            {
                "row_type": "clsvof_ls_trace",
                "mirror_index": "",
                "snapshot_index": "",
                "snapshot_fraction": "",
                "t": float(record["t"]),
                "tau": float(record["tau"]),
                "theta_deg_360": "",
                "hk_native": "",
                "hk_nn": "",
                "delta_hk": "",
                "Ca": float(record["Ca"]),
                "sigma": "",
            }
        )
    fieldnames = [
        "row_type",
        "mirror_index",
        "snapshot_index",
        "snapshot_fraction",
        "t",
        "tau",
        "theta_deg_360",
        "hk_native",
        "hk_nn",
        "delta_hk",
        "Ca",
        "sigma",
    ]
    source_data_csv.parent.mkdir(parents=True, exist_ok=True)
    with source_data_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ==== _find_clsvof_trace_for_process (1659-1668) ====
def find_clsvof_trace_for_process(summary: dict[str, Any]) -> Path | None:
    level = int(summary["level"])
    candidates = sorted((RAW / "stationary").glob(
        f"stationary_canary_*/CLSVOF_LS_NATIVE_L{level}_r0/stationary_trace.csv"
    ))
    candidates += sorted((RAW / "stationary").glob(
        f"stationary_clsvof_baseline_*/CLSVOF_LS_NATIVE_L{level}_r0/stationary_trace.csv"
    ))
    return candidates[-1] if candidates else None


# ==== _curvature_band_stats (1670-1738) ====
def curvature_band_stats(records: list[dict[str, str]], *, band_name: str) -> dict[str, Any]:
    if not records:
        return {
            "band": band_name,
            "row_count": 0,
            "mean_delta_hk": math.nan,
            "mean_delta_kappa": math.nan,
            "mean_kappa_native": math.nan,
            "mean_kappa_nn": math.nan,
            "std_kappa_native": math.nan,
            "std_kappa_nn": math.nan,
            "interface_std_kappa_nn": math.nan,
            "force_band_std_kappa_nn": math.nan,
            "std_delta_hk": math.nan,
            "std_delta_kappa": math.nan,
            "max_abs_delta_hk": math.nan,
            "p95_abs_delta_hk": math.nan,
            "p99_abs_delta_hk": math.nan,
            "tail_mean_abs_delta_hk": math.nan,
            "sign_agreement_fraction": math.nan,
            "rmse_direct": math.nan,
            "rmse_negated": math.nan,
            "scale_identity_max_abs": math.nan,
            "mean_hk_ratio_to_native": math.nan,
            "mean_kappa_ratio_to_native": math.nan,
        }

    hk_native = [float(row["hk_native"]) for row in records]
    hk_nn = [float(row["hk_nn"]) for row in records]
    delta_hk = [float(row["delta_hk"]) for row in records]
    native_kappa = [float(row["native_kappa"]) for row in records]
    kappa_nn = [float(row["kappa_nn"]) for row in records]
    delta_kappa = [float(row["delta_kappa"]) for row in records]
    scale_identity_error = [abs(float(row["scale_identity_error"])) for row in records]
    abs_delta_hk = [abs(value) for value in delta_hk]
    direct_sq = [(nn - native) ** 2 for nn, native in zip(kappa_nn, native_kappa)]
    negated_sq = [(nn + native) ** 2 for nn, native in zip(kappa_nn, native_kappa)]
    sign_agreement = [
        1.0 if nn * native > 0.0 else 0.0 for nn, native in zip(kappa_nn, native_kappa)
    ]
    hk_ratios = [nn / native for nn, native in zip(hk_nn, hk_native) if abs(native) > 1e-300]
    kappa_ratios = [
        nn / native for nn, native in zip(kappa_nn, native_kappa) if abs(native) > 1e-300
    ]
    std_kappa_nn = sample_std(kappa_nn)
    stats = {
        "band": band_name,
        "row_count": len(records),
        "mean_delta_hk": mean_(delta_hk),
        "mean_delta_kappa": mean_(delta_kappa),
        "mean_kappa_native": mean_(native_kappa),
        "mean_kappa_nn": mean_(kappa_nn),
        "std_kappa_native": sample_std(native_kappa),
        "std_kappa_nn": std_kappa_nn,
        "interface_std_kappa_nn": std_kappa_nn if band_name == "interface" else math.nan,
        "force_band_std_kappa_nn": std_kappa_nn if band_name == "force_band" else math.nan,
        "std_delta_hk": sample_std(delta_hk),
        "std_delta_kappa": sample_std(delta_kappa),
        "max_abs_delta_hk": max(abs_delta_hk),
        "p95_abs_delta_hk": percentile(abs_delta_hk, 0.95),
        "p99_abs_delta_hk": percentile(abs_delta_hk, 0.99),
        "tail_mean_abs_delta_hk": tail_mean(abs_delta_hk, 0.10),
        "sign_agreement_fraction": mean_(sign_agreement),
        "rmse_direct": math.sqrt(mean_(direct_sq)),
        "rmse_negated": math.sqrt(mean_(negated_sq)),
        "scale_identity_max_abs": max(scale_identity_error),
        "mean_hk_ratio_to_native": mean_(hk_ratios),
        "mean_kappa_ratio_to_native": mean_(kappa_ratios),
    }
    return stats


# ==== _mean (1740-1744) ====
def mean_(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


# ==== _sample_std (1746-1751) ====
def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    mean = mean_(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


# ==== _percentile (1753-1764) ====
def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


# ==== _tail_mean (1766-1771) ====
def tail_mean(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    count = max(1, math.ceil(len(values) * fraction))
    return mean_(sorted(values, reverse=True)[:count])


def _method_role(method: str) -> str:
    return "raw_reference_branch"


def _method_display_label(method: str) -> str:
    return METHOD_DISPLAY_LABELS.get(method, method)


# ==== _write_curvature_diagnostic_report (2270-2318) ====
def write_curvature_diagnostic_report(
    *,
    run_id: str,
    methods: list[str],
    levels: list[int],
    summaries: list[dict[str, Any]],
    manifest_path: Path,
    benchmark: str,
) -> dict[str, Any]:
    methods_by_name = {summary["method"]: summary for summary in summaries}
    report = {
        "run_id": run_id,
        "benchmark": benchmark,
        "tier": "curvature-diagnostic",
        "levels": levels,
        "methods": methods_by_name,
        "comparison": {},
        "manifest": route_relative(manifest_path),
        "overall_status": "PASS",
        "evidence_level": "controlled_diagnostic",
        "primary_branch": "NN27_RAW",
        "paper_model_line": "baseline_hgradient",
    }
    report_path = REPORTS / f"{run_id}_curvature_field_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path = REPORTS / f"{run_id}_curvature_field_summary.md"
    md_path.write_text(render_curvature_diagnostic_report(report))
    return report


# ==== _render_curvature_diagnostic_report (2320-2355) ====
def render_curvature_diagnostic_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stationary Curvature-Field Diagnostic",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        f"- primary_branch: `{report['primary_branch']}`",
        f"- paper_model_line: `{report.get('paper_model_line', '')}`",
        "",
        "| method | rows | mean Δ(hκ) | stdΓ κ_NN | max |Δ(hκ)| | p95 |Δ(hκ)| | sign | scale max |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for method, summary in report["methods"].items():
        force = summary["force_band"]
        sign_scale = summary["sign_scale"]
        display_method = _method_display_label(method)
        lines.append(
            f"| {display_method} | {force['row_count']} | {force['mean_delta_hk']:.6e} | "
            f"{force['force_band_std_kappa_nn']:.6e} | {force['max_abs_delta_hk']:.6e} | "
            f"{force['p95_abs_delta_hk']:.6e} | {sign_scale['sign_verdict']} | "
            f"{sign_scale['scale_identity_max_abs']:.6e} |"
        )
    if report["comparison"]:
        comparison_key = next(iter(report["comparison"]))
        cmp_row = report["comparison"][comparison_key]
        lines.extend(
            [
                "",
                f"## {comparison_key}",
                "",
                f"- force_band_std_kappa_nn_ratio: `{cmp_row['force_band_std_kappa_nn_ratio']:.6e}`",
                f"- max_abs_delta_hk_ratio: `{cmp_row['max_abs_delta_hk_ratio']:.6e}`",
                f"- tail_mean_abs_delta_hk_ratio: `{cmp_row['tail_mean_abs_delta_hk_ratio']:.6e}`",
                f"- interpretation: `{cmp_row['interpretation']}`",
            ]
        )
    return "\n".join(lines) + "\n"


# ==== _canary_nn_metadata (2555-2573) ====
def canary_nn_metadata(method: str) -> dict[str, str]:
    if method == "CLSVOF_LS_NATIVE":
        return {}
    if not NN_EXPORT_MANIFEST.exists():
        raise StationaryGateError("nn_export_manifest_missing")
    manifest = json.loads(NN_EXPORT_MANIFEST.read_text())
    expected_label = NN_LABELS.get(method)
    if not expected_label:
        raise StationaryGateError(f"nn_export_manifest_no_method_label:{method}")
    for row in manifest.get("rows", []):
        if row.get("label") == expected_label:
            return {
                "checkpoint_path": str(row["checkpoint_path"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "weights_header": str(row["weights_header"]),
                "weights_header_sha256": str(row["weights_header_sha256"]),
            }
    raise StationaryGateError(f"nn_export_manifest_missing_{expected_label}")


# ==== _load_required_source_audit (2575-2583) ====
def load_required_source_audit() -> dict[str, Any]:
    if not SOURCE_AUDIT.exists():
        raise StationaryGateError("source_audit_missing")
    audit = json.loads(SOURCE_AUDIT.read_text())
    if audit.get("overall_status") != "PASS":
        raise StationaryGateError("source_audit_not_pass")
    if not (BUILD_SRC / "qcc").exists():
        raise StationaryGateError("route_local_qcc_missing")
    return audit
