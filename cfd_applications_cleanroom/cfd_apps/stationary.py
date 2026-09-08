"""Stationary-bubble cleanroom runner and summarizer."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.hashes import (
    manifest_digest,
    sha256_file,
    tree_manifest,
)
from cfd_applications_cleanroom.cfd_apps.manifest import stale_output_guard
from cfd_applications_cleanroom.cfd_apps.paths import (
    MANIFESTS,
    RAW,
    REPORTS,
    ROOT,
    route_relative,
)
from cfd_applications_cleanroom.cfd_apps.schema import MANIFEST_COLUMNS, validate_no_duplicate_time_rows
from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import (
    BUILD_SRC,
    NN_DIR,
    NN_MODE_VALUES,
    VENDOR_SRC,
    StationaryGateError,
    canary_nn_metadata as _canary_nn_metadata,
    compile_curvature_binary as _shared_compile_curvature_binary,
    compile_curvature_process_binary as _shared_compile_curvature_process_binary,
    compile_nn_forward_object as _compile_nn_forward_object,
    load_required_source_audit as _load_required_source_audit,
    run_curvature_binary_one as _shared_run_curvature_binary_one,
    run_curvature_process_binary_one as _shared_run_curvature_process_binary_one,
    summarize_curvature_field_csv as _shared_summarize_curvature_field_csv,
    summarize_curvature_process_csvs as _shared_summarize_curvature_process_csvs,
    write_curvature_diagnostic_report as _shared_write_curvature_diagnostic_report,
    write_curvature_process_report as _shared_write_curvature_process_report,
)


CASE_DIR = ROOT / "cfd_applications_cleanroom/cases/stationary"
CASE_SOURCE = CASE_DIR / "stationary_vof_hf_stock_single.c"
NATIVE_CLSVOF_SOURCE = CASE_DIR / "stationary_clsvof_native.c"
NN_CLSVOF_SOURCE = CASE_DIR / "stationary_clsvof_nn.c"
CURVATURE_FIELD_SOURCE = CASE_DIR / "stationary_curvature_field.c"
CURVATURE_PROCESS_SOURCE = CASE_DIR / "stationary_curvature_process.c"
CONFIG_PATH = ROOT / "cfd_applications_cleanroom/configs/stationary_bubble.yaml"
INERT_CANARY_METHODS = ("CLSVOF_LS_NATIVE",)
PAPER_DEPLOYABLE_METHODS = ("NN27_RAW",)
DEPLOYABLE_CANARY_METHODS = PAPER_DEPLOYABLE_METHODS
SUPPORTED_DEPLOYABLE_METHODS = PAPER_DEPLOYABLE_METHODS
CURVATURE_DIAGNOSTIC_METHODS = SUPPORTED_DEPLOYABLE_METHODS
CURVATURE_PROCESS_METHODS = SUPPORTED_DEPLOYABLE_METHODS
CURVATURE_PROCESS_LEVELS = (4, 5, 6, 7, 8, 9)
CANARY_METHODS = INERT_CANARY_METHODS + SUPPORTED_DEPLOYABLE_METHODS
STATIONARY_TMAX_EXPECTED = 0.8 * 0.8 / ((0.8 / 12000.0) ** 0.5)


def _compile_curvature_binary(*, method, level, run_root, source_audit):
    return _shared_compile_curvature_binary(
        method=method,
        level=level,
        run_root=run_root,
        source_audit=source_audit,
        case_source=CURVATURE_FIELD_SOURCE,
        case_dir=CASE_DIR,
        binary_stem="stationary_curvature",
    )


def _compile_curvature_process_binary(*, method, level, run_root, source_audit):
    return _shared_compile_curvature_process_binary(
        method=method,
        level=level,
        run_root=run_root,
        source_audit=source_audit,
        case_source=CURVATURE_PROCESS_SOURCE,
        case_dir=CASE_DIR,
        binary_stem="stationary_curvature_process",
    )


def _run_curvature_binary_one(**kwargs):
    return _shared_run_curvature_binary_one(
        benchmark="stationary",
        case_id="stationary_curvature_field",
        **kwargs,
    )


def _run_curvature_process_binary_one(**kwargs):
    return _shared_run_curvature_process_binary_one(
        benchmark="stationary",
        expected_tmax=STATIONARY_TMAX_EXPECTED,
        **kwargs,
    )


def _summarize_curvature_field_csv(*args, **kwargs):
    return _shared_summarize_curvature_field_csv(*args, benchmark="stationary", **kwargs)


def _summarize_curvature_process_csvs(**kwargs):
    return _shared_summarize_curvature_process_csvs(
        benchmark="stationary",
        expected_tmax=STATIONARY_TMAX_EXPECTED,
        **kwargs,
    )


def _write_curvature_diagnostic_report(**kwargs):
    return _shared_write_curvature_diagnostic_report(benchmark="stationary", **kwargs)


def _write_curvature_process_report(**kwargs):
    return _shared_write_curvature_process_report(
        benchmark="stationary",
        artifact_prefix="stationary_curvature_process",
        **kwargs,
    )


def run_stock_reference(*, levels: list[int], repeat: int) -> dict[str, Any]:
    source_audit = _load_required_source_audit()
    if repeat != 3:
        raise StationaryGateError("stationary_stock_reference_requires_repeat_3")
    run_id = "stationary_stock_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / "stationary" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = _resolved_stationary_config(levels=levels, repeat=repeat)
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for level in levels:
        compile_evidence = _compile_stock_binary(
            level=level,
            run_root=run_root,
            source_audit=source_audit,
        )
        for repeat_index in range(repeat):
            row, summary = _run_stock_binary_one(
                run_id=run_id,
                level=level,
                repeat_index=repeat_index,
                run_root=run_root,
                source_audit=source_audit,
                compile_evidence=compile_evidence,
                resolved_config_path=resolved_config_path,
                resolved_config_sha256=resolved_config_sha256,
            )
            rows.append(row)
            summaries.append(summary)

    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "stock-reference",
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)

    return _summarize_stock_gate(
        run_id=run_id,
        summaries=summaries,
        manifest_path=manifest_path,
        expected_repeat=repeat,
    )


def run_canary(
    *,
    methods: list[str],
    levels: list[int],
    repeat: int,
) -> dict[str, Any]:
    source_audit = _load_required_source_audit()
    methods = methods or list(CANARY_METHODS)
    unknown = sorted(set(methods) - set(CANARY_METHODS))
    if unknown:
        raise StationaryGateError(f"stationary_canary_unknown_methods:{','.join(unknown)}")
    run_kind = _canary_run_kind(methods)
    parity_report = _load_latest_inert_parity_report() if run_kind == "deployable" else None
    if repeat != 3:
        raise StationaryGateError("stationary_canary_requires_repeat_3")
    resolved_config = _resolved_stationary_canary_config(
        methods=methods,
        levels=levels,
        repeat=repeat,
        run_kind=run_kind,
        parity_report=parity_report,
    )
    run_id = "stationary_canary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / "stationary" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in levels:
        for method in methods:
            compile_evidence = _compile_canary_binary(
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
            )
            for repeat_index in range(repeat):
                row, summary = _run_canary_binary_one(
                    run_id=run_id,
                    method=method,
                    level=level,
                    repeat_index=repeat_index,
                    run_root=run_root,
                    source_audit=source_audit,
                    compile_evidence=compile_evidence,
                    resolved_config_path=resolved_config_path,
                    resolved_config_sha256=resolved_config_sha256,
                )
                rows.append(row)
                summaries.append(summary)

    if run_kind == "inert":
        report = _evaluate_canary_gates(
            run_id=run_id,
            summaries=summaries,
            rows=rows,
            methods=methods,
            levels=levels,
            expected_repeat=repeat,
        )
    else:
        assert parity_report is not None
        report = _evaluate_nn_canary_gates(
            run_id=run_id,
            summaries=summaries,
            rows=rows,
            methods=methods,
            levels=levels,
            expected_repeat=repeat,
            parity_report=parity_report,
        )
    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "canary",
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)
    report["manifest"] = route_relative(manifest_path)
    return _write_canary_report(report)


def run_curvature_diagnostic(
    *,
    methods: list[str],
    levels: list[int],
) -> dict[str, Any]:
    source_audit = _load_required_source_audit()
    methods = methods or list(PAPER_DEPLOYABLE_METHODS)
    unknown = sorted(set(methods) - set(CURVATURE_DIAGNOSTIC_METHODS))
    if unknown:
        raise StationaryGateError(f"stationary_curvature_unknown_methods:{','.join(unknown)}")
    if levels != [6]:
        raise StationaryGateError("stationary_curvature_diagnostic_level6_only")

    run_id = "stationary_curvature_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / "stationary" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "benchmark": "stationary",
        "tier": "curvature-diagnostic",
        "levels": levels,
        "methods": methods,
        "case_source": route_relative(CURVATURE_FIELD_SOURCE),
        "paper_model_line": "baseline_hgradient",
        "field_contract": {
            "initial_level_set": "d=sqrt(x^2+y^2)-RADIUS",
            "native_hk": "Delta*distance_curvature(point,d)",
            "model_output": "h*kappa",
            "solver_conversion": "kappa=h*kappa/Delta",
            "force_band": "|d|<=2*Delta",
            "interface_band": "|d|<=Delta",
        },
    }
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in levels:
        for method in methods:
            compile_evidence = _compile_curvature_binary(
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
            )
            row, summary = _run_curvature_binary_one(
                run_id=run_id,
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                compile_evidence=compile_evidence,
                resolved_config_path=resolved_config_path,
                resolved_config_sha256=resolved_config_sha256,
            )
            rows.append(row)
            summaries.append(summary)

    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "curvature-diagnostic",
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)
    return _write_curvature_diagnostic_report(
        run_id=run_id,
        methods=methods,
        levels=levels,
        summaries=summaries,
        manifest_path=manifest_path,
    )


def run_curvature_process(
    *,
    methods: list[str],
    levels: list[int],
) -> dict[str, Any]:
    source_audit = _load_required_source_audit()
    methods = methods or list(PAPER_DEPLOYABLE_METHODS)
    unknown = sorted(set(methods) - set(CURVATURE_PROCESS_METHODS))
    if unknown:
        raise StationaryGateError(f"stationary_curvature_process_unknown_methods:{','.join(unknown)}")
    unknown_levels = sorted(set(levels) - set(CURVATURE_PROCESS_LEVELS))
    if unknown_levels:
        raise StationaryGateError(f"stationary_curvature_process_unknown_levels:{unknown_levels}")

    run_id = "stationary_curvature_process_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RAW / "stationary" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "benchmark": "stationary",
        "tier": "curvature-process",
        "levels": levels,
        "methods": methods,
        "paper_model_line": "baseline_hgradient",
        "case_source": route_relative(CURVATURE_PROCESS_SOURCE),
        "snapshot_fractions": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        "angle_domain": "0-360 degrees by quadrant symmetry expansion from the quadrant stationary bubble",
        "field_contract": {
            "native_hk": "Delta*distance_curvature(point,d)",
            "model_output": "h*kappa",
            "solver_conversion": "kappa=h*kappa/Delta",
            "force_band": "|d|<=2*Delta",
            "interface_band": "|d|<=Delta",
            "surface_tension_metric": "Ca=mu*Umax/sigma",
            "sigma": 1.0,
        },
    }
    resolved_config_path = run_root / "config.resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in levels:
        for method in methods:
            compile_evidence = _compile_curvature_process_binary(
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
            )
            row, summary = _run_curvature_process_binary_one(
                run_id=run_id,
                method=method,
                level=level,
                run_root=run_root,
                source_audit=source_audit,
                compile_evidence=compile_evidence,
                resolved_config_path=resolved_config_path,
                resolved_config_sha256=resolved_config_sha256,
            )
            rows.append(row)
            summaries.append(summary)

    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    manifest = {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "curvature-process",
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    stale_output_guard(manifest_path)
    return _write_curvature_process_report(
        run_id=run_id,
        methods=methods,
        levels=levels,
        summaries=summaries,
        manifest_path=manifest_path,
    )


def _compile_stock_binary(
    *,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    compile_dir = run_root / "build" / f"L{level}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    binary = compile_dir / "stationary_vof_hf_stock_single"
    compile_log = compile_dir / "compile.log"
    source_hashes_json = compile_dir / "source_hashes_compile.json"

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    case_sha = sha256_file(CASE_SOURCE)
    cmd = [
        str(BUILD_SRC / "qcc"),
        "-autolink",
        "-O2",
        f"-DLEVEL={level}",
        CASE_SOURCE.name,
        "-o",
        str(binary),
        "-lm",
    ]
    env = os.environ.copy()
    env["BASILISK"] = str(BUILD_SRC)
    compile_result = subprocess.run(
        cmd,
        cwd=CASE_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    compile_log.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={CASE_DIR}\nreturncode={compile_result.returncode}\n\n"
        + compile_result.stdout
    )
    vendor_after_compile = tree_manifest(VENDOR_SRC, source_only=True)
    build_after_compile = tree_manifest(BUILD_SRC, source_only=True)
    source_hashes = {
        "case_source_sha256": case_sha,
        "vendor_source_hash_before": manifest_digest(vendor_before),
        "vendor_source_hash_after_compile": manifest_digest(vendor_after_compile),
        "vendor_source_hashes_unchanged": vendor_before == vendor_after_compile,
        "build_source_hash_before": manifest_digest(build_before),
        "build_source_hash_after_compile": manifest_digest(build_after_compile),
        "build_source_hashes_unchanged": build_before == build_after_compile,
    }
    source_hashes_json.write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    if compile_result.returncode != 0 or not binary.exists():
        raise StationaryGateError(f"stationary_stock_compile_failed:{route_relative(compile_log)}")
    if vendor_before != vendor_after_compile:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_compile")
    if build_before != build_after_compile:
        raise StationaryGateError("route_build_source_mutated_during_stationary_compile")
    return {
        "level": level,
        "binary": binary,
        "binary_sha256": sha256_file(binary),
        "compile_command": " ".join(cmd),
        "compile_log": compile_log,
        "compile_returncode": compile_result.returncode,
        "case_source_sha256": case_sha,
        "source_hashes_compile_json": source_hashes_json,
        "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
    }


def _compile_canary_binary(
    *,
    method: str,
    level: int,
    run_root: Path,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    compile_dir = run_root / "build" / method / f"L{level}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    binary = compile_dir / f"stationary_{method.lower()}_L{level}"
    compile_log = compile_dir / "compile.log"
    source_hashes_json = compile_dir / "source_hashes_compile.json"
    case_source = _canary_case_source(method)
    nn_metadata = _canary_nn_metadata(method)

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)
    case_sha = sha256_file(case_source)
    nn_object_evidence: dict[str, Any] = {}
    if method in NN_MODE_VALUES:
        nn_object_evidence = _compile_nn_forward_object(
            compile_dir=compile_dir,
            nn_metadata=nn_metadata,
        )
    cmd = [
        str(BUILD_SRC / "qcc"),
        "-autolink",
        f"-DLEVEL={level}",
    ]
    if method in NN_MODE_VALUES:
        cmd.append(f"-DNN_MODE={NN_MODE_VALUES[method]}")
        cmd.extend([f"-DMETHOD_ID=\\\"{method}\\\"", f"-I{NN_DIR}"])
    cmd.append(case_source.name)
    if nn_object_evidence:
        cmd.append(str(nn_object_evidence["nn_object"]))
    cmd.extend(["-o", str(binary), "-lm"])
    env = os.environ.copy()
    env["BASILISK"] = str(BUILD_SRC)
    compile_result = subprocess.run(
        cmd,
        cwd=CASE_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    compile_log.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={CASE_DIR}\nreturncode={compile_result.returncode}\n\n"
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
        raise StationaryGateError(f"stationary_canary_compile_failed:{route_relative(compile_log)}")
    if vendor_before != vendor_after_compile:
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_canary_compile")
    if build_before != build_after_compile:
        raise StationaryGateError("route_build_source_mutated_during_stationary_canary_compile")
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





def _run_stock_binary_one(
    *,
    run_id: str,
    level: int,
    repeat_index: int,
    run_root: Path,
    source_audit: dict[str, Any],
    compile_evidence: dict[str, Any],
    resolved_config_path: Path,
    resolved_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"L{level}_r{repeat_index}"
    out_dir = run_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(compile_evidence["binary"])
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    status_file = out_dir / "status.json"
    raw_csv = out_dir / "stationary_trace.csv"
    summary_json = out_dir / "summary.json"
    source_hashes_json = out_dir / "source_hashes.json"

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)

    run_env = os.environ.copy()
    run_env["BASILISK"] = str(BUILD_SRC)
    run_env["CLEANROOM_RUN_ID"] = run_id
    run_env["CLEANROOM_REPEAT_ID"] = str(repeat_index)
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
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_run")
    if build_before != build_after_run:
        raise StationaryGateError("route_build_source_mutated_during_stationary_run")
    if run_result.returncode != 0 or not raw_csv.exists():
        raise StationaryGateError(f"stationary_stock_run_failed:{route_relative(stderr_log)}")

    validate_no_duplicate_time_rows(raw_csv)
    summary = _summarize_raw_csv(raw_csv, run_id=run_id, level=level, repeat_index=repeat_index)
    summary.update(
        {
            "resolved_config_sha256": resolved_config_sha256,
            "resolved_config_json": str(resolved_config_path),
            "binary_sha256": compile_evidence["binary_sha256"],
            "official_source_manifest_sha256": source_audit["clean_source"]["source_manifest_sha256"],
            "source_hashes_unchanged": source_hashes["vendor_source_hashes_unchanged"]
            and source_hashes["build_source_hashes_unchanged"],
            "raw_schema_valid": True,
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    accepted = summary["gate"] == "PASS"
    status = {
        "status": "accepted_stock_reference" if accepted else "failed_stock_reference",
        "run_returncode": run_result.returncode,
        "compile_returncode": compile_evidence["compile_returncode"],
    }
    status_file.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "repeat_id": str(repeat_index),
            "benchmark": "stationary",
            "method": "VOF_HF_NATIVE",
            "level": str(level),
            "grid_n": str(1 << level),
            "case_source_path": route_relative(CASE_SOURCE),
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
            "summary_json": str(summary_json),
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "nn_weights_header_sha256": "",
            "result_status": status["status"],
            "evidence_level": "controlled_benchmark" if accepted else "runtime_smoke",
            "resolved_config_json": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "binary": str(binary),
            "binary_sha256": compile_evidence["binary_sha256"],
            "source_hashes_json": str(source_hashes_json),
            "source_hashes_compile_json": str(compile_evidence["source_hashes_compile_json"]),
        }
    )
    return row, summary


def _run_canary_binary_one(
    *,
    run_id: str,
    method: str,
    level: int,
    repeat_index: int,
    run_root: Path,
    source_audit: dict[str, Any],
    compile_evidence: dict[str, Any],
    resolved_config_path: Path,
    resolved_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{method}_L{level}_r{repeat_index}"
    out_dir = run_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(compile_evidence["binary"])
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    status_file = out_dir / "status.json"
    raw_csv = out_dir / "stationary_trace.csv"
    summary_json = out_dir / "summary.json"
    source_hashes_json = out_dir / "source_hashes.json"

    vendor_before = tree_manifest(VENDOR_SRC, source_only=True)
    build_before = tree_manifest(BUILD_SRC, source_only=True)

    run_env = os.environ.copy()
    run_env["BASILISK"] = str(BUILD_SRC)
    run_env["CLEANROOM_RUN_ID"] = run_id
    run_env["CLEANROOM_REPEAT_ID"] = str(repeat_index)
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
        raise StationaryGateError("route_vendor_source_mutated_during_stationary_canary_run")
    if build_before != build_after_run:
        raise StationaryGateError("route_build_source_mutated_during_stationary_canary_run")
    if run_result.returncode != 0 or not raw_csv.exists():
        raise StationaryGateError(f"stationary_canary_run_failed:{route_relative(stderr_log)}")

    validate_no_duplicate_time_rows(raw_csv)
    summary = _summarize_canary_raw_csv(
        raw_csv,
        run_id=run_id,
        method=method,
        level=level,
        repeat_index=repeat_index,
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
            "status_file": str(status_file),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    status = {
        "status": "ran_canary_trace",
        "method": method,
        "run_returncode": run_result.returncode,
        "compile_returncode": compile_evidence["compile_returncode"],
    }
    status_file.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "repeat_id": str(repeat_index),
            "benchmark": "stationary",
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
            "summary_json": str(summary_json),
            "checkpoint_path": compile_evidence["checkpoint_path"],
            "checkpoint_sha256": compile_evidence["checkpoint_sha256"],
            "nn_weights_header_sha256": compile_evidence["nn_weights_header_sha256"],
            "result_status": "pending_canary_gate",
            "evidence_level": "runtime_smoke",
        }
    )
    return row, summary
















def _summarize_raw_csv(raw_csv: Path, *, run_id: str, level: int, repeat_index: int) -> dict[str, Any]:
    with raw_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise StationaryGateError(f"stationary_empty_raw_csv:{route_relative(raw_csv)}")
    if any(record["run_id"] != run_id for record in records):
        raise StationaryGateError("stationary_run_id_mismatch")
    ca = [float(record["Ca"]) for record in records]
    tail_start = max(0, int(len(ca) * 0.8))
    tail = ca[tail_start:] or ca
    final_t = float(records[-1]["t"])
    final_tau = float(records[-1]["tau"])
    expected_tmax = 0.8 * 0.8 / ((0.8 / 12000.0) ** 0.5)
    reached_final_time = final_t >= expected_tmax * 0.999
    finite = all(value == value and value not in (float("inf"), float("-inf")) for value in ca)
    gate = max(tail) <= 1e-12 and reached_final_time and finite
    return {
        "run_id": run_id,
        "repeat_id": str(repeat_index),
        "benchmark": "stationary",
        "method": "VOF_HF_NATIVE",
        "level": level,
        "grid_n": 1 << level,
        "row_count": len(records),
        "final_t": final_t,
        "final_tau": final_tau,
        "expected_tmax": expected_tmax,
        "reached_final_time": reached_final_time,
        "finite_Ca": finite,
        "Ca_tail_max": max(tail),
        "Ca_tail_mean": sum(tail) / len(tail),
        "Ca_final": ca[-1],
        "gate": "PASS" if gate else "FAIL",
    }


def _summarize_canary_raw_csv(
    raw_csv: Path,
    *,
    run_id: str,
    method: str,
    level: int,
    repeat_index: int,
) -> dict[str, Any]:
    with raw_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise StationaryGateError(f"stationary_canary_empty_raw_csv:{route_relative(raw_csv)}")
    if any(record["run_id"] != run_id for record in records):
        raise StationaryGateError("stationary_canary_run_id_mismatch")
    if any(record["method"] != method for record in records):
        raise StationaryGateError("stationary_canary_method_mismatch")
    ca = [float(record["Ca"]) for record in records]
    mass = [float(record["mass"]) for record in records]
    kappa_linf = [float(record["kappa_linf"]) for record in records]
    kappa_probe_linf = [float(record["kappa_probe_linf"]) for record in records]
    tail_start = max(0, int(len(ca) * 0.8))
    tail = ca[tail_start:] or ca
    kappa_tail = kappa_linf[tail_start:] or kappa_linf
    probe_tail = kappa_probe_linf[tail_start:] or kappa_probe_linf
    final_t = float(records[-1]["t"])
    final_tau = float(records[-1]["tau"])
    expected_tmax = 0.8 * 0.8 / ((0.8 / 12000.0) ** 0.5)
    reached_final_time = final_t >= expected_tmax * 0.999
    finite_ca = all(math.isfinite(value) for value in ca)
    finite_mass = all(math.isfinite(value) for value in mass)
    finite_kappa = all(math.isfinite(value) for value in kappa_linf)
    finite_gate = reached_final_time and finite_ca and finite_mass and finite_kappa
    return {
        "run_id": run_id,
        "repeat_id": str(repeat_index),
        "benchmark": "stationary",
        "method": method,
        "level": level,
        "grid_n": 1 << level,
        "row_count": len(records),
        "final_t": final_t,
        "final_tau": final_tau,
        "expected_tmax": expected_tmax,
        "reached_final_time": reached_final_time,
        "finite_Ca": finite_ca,
        "finite_mass": finite_mass,
        "finite_kappa_linf": finite_kappa,
        "Ca_tail_max": max(tail),
        "Ca_tail_mean": sum(tail) / len(tail),
        "Ca_final": ca[-1],
        "mass_final": mass[-1],
        "kappa_linf_tail_max": max(kappa_tail),
        "kappa_linf_tail_mean": sum(kappa_tail) / len(kappa_tail),
        "kappa_probe_linf_tail_max": max(probe_tail),
        "finite_gate": "PASS" if finite_gate else "FAIL",
    }


def _evaluate_canary_gates(
    *,
    run_id: str,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    methods: list[str],
    levels: list[int],
    expected_repeat: int,
) -> dict[str, Any]:
    by_key = {
        (str(summary["method"]), int(summary["level"]), str(summary["repeat_id"])): summary
        for summary in summaries
    }
    missing = [
        (method, level, repeat_index)
        for method in methods
        for level in levels
        for repeat_index in range(expected_repeat)
        if (method, level, str(repeat_index)) not in by_key
    ]
    if missing:
        raise StationaryGateError(f"stationary_canary_missing_repeats:{missing}")

    repeat_consistency = _canary_repeat_consistency(
        summaries=summaries,
        methods=methods,
        levels=levels,
        expected_repeat=expected_repeat,
    )
    native_summaries = [
        by_key[("CLSVOF_LS_NATIVE", level, str(repeat_index))]
        for level in levels
        for repeat_index in range(expected_repeat)
        if "CLSVOF_LS_NATIVE" in methods
    ]
    sb_g2_pass = bool(native_summaries) and all(
        summary["finite_gate"] == "PASS" for summary in native_summaries
    )
    method_pass = {
        "CLSVOF_LS_NATIVE": sb_g2_pass,
    }
    for row in rows:
        method = str(row["method"])
        level = int(row["level"])
        repeat_ok = repeat_consistency.get(method, {}).get(str(level), {}).get("pass", False)
        accepted = method_pass.get(method, False) and repeat_ok
        if method == "CLSVOF_LS_NATIVE":
            row["result_status"] = "accepted_clsvof_native" if accepted else "failed_clsvof_native"
            row["evidence_level"] = "controlled_diagnostic" if accepted else "runtime_smoke"
        _update_status_file(row)

    repeated_all_pass = all(
        level_data["pass"]
        for method_data in repeat_consistency.values()
        for level_data in method_data.values()
    )
    overall_pass = sb_g2_pass and repeated_all_pass
    return {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "canary",
        "methods": methods,
        "levels": levels,
        "SB-G2": "PASS" if sb_g2_pass else "FAIL",
        "repeat_consistency": repeat_consistency,
        "overall_pass": overall_pass,
        "evidence_level": "controlled_diagnostic" if overall_pass else "runtime_smoke",
        "summaries": summaries,
    }


def _evaluate_nn_canary_gates(
    *,
    run_id: str,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    methods: list[str],
    levels: list[int],
    expected_repeat: int,
    parity_report: dict[str, Any],
) -> dict[str, Any]:
    by_key = {
        (str(summary["method"]), int(summary["level"]), str(summary["repeat_id"])): summary
        for summary in summaries
    }
    baseline = _native_baseline_from_parity_report(parity_report)
    missing = [
        (method, level, repeat_index)
        for method in methods
        for level in levels
        for repeat_index in range(expected_repeat)
        if (method, level, str(repeat_index)) not in by_key
    ]
    if missing:
        raise StationaryGateError(f"stationary_nn_canary_missing_repeats:{missing}")

    repeat_consistency = _canary_repeat_consistency(
        summaries=summaries,
        methods=methods,
        levels=levels,
        expected_repeat=expected_repeat,
    )
    ratios: dict[str, Any] = {}
    method_labels: dict[str, str] = {}
    for method in methods:
        method_ratios = []
        for level in levels:
            level_ratios = []
            for repeat_index in range(expected_repeat):
                repeat_id = str(repeat_index)
                summary = by_key[(method, level, repeat_id)]
                baseline_summary = baseline[(level, repeat_id)]
                native_ca = float(baseline_summary["Ca_tail_max"])
                if native_ca <= 0. or not math.isfinite(native_ca):
                    raise StationaryGateError("stationary_nn_canary_invalid_native_baseline")
                ratio = float(summary["Ca_tail_max"]) / native_ca
                level_ratios.append(
                    {
                        "level": level,
                        "repeat_id": repeat_id,
                        "Ca_tail_max": float(summary["Ca_tail_max"]),
                        "native_Ca_tail_max": native_ca,
                        "Ca_ratio_to_native": ratio,
                        "pass": ratio <= 1.10 and summary["finite_gate"] == "PASS",
                    }
                )
            values = [entry["Ca_ratio_to_native"] for entry in level_ratios]
            spread = max(values) - min(values)
            level_pass = all(entry["pass"] for entry in level_ratios) and spread <= 0.01
            method_ratios.append(
                {
                    "level": level,
                    "repeat_ratios": level_ratios,
                    "max_Ca_ratio_to_native": max(values),
                    "min_Ca_ratio_to_native": min(values),
                    "spread_Ca_ratio_to_native": spread,
                    "pass": level_pass,
                }
            )
        method_pass = all(level["pass"] for level in method_ratios)
        method_labels[method] = (
            "stationary_nn_not_worse" if method_pass else "stationary_nn_worse_than_native"
        )
        ratios[method] = method_ratios

    for row in rows:
        label = method_labels[str(row["method"])]
        row["result_status"] = label
        row["evidence_level"] = "controlled_diagnostic"
        _update_status_file(row)

    overall_pass = all(label == "stationary_nn_not_worse" for label in method_labels.values())
    return {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "canary",
        "run_kind": "deployable",
        "methods": methods,
        "levels": levels,
        "requires_parity_run_id": parity_report["run_id"],
        "method_labels": method_labels,
        "ratios": ratios,
        "repeat_consistency": repeat_consistency,
        "SB-G6": "PASS" if overall_pass else "FAIL",
        "overall_pass": overall_pass,
        "evidence_level": "controlled_diagnostic",
        "summaries": summaries,
    }


def _native_baseline_from_parity_report(
    parity_report: dict[str, Any]
) -> dict[tuple[int, str], dict[str, Any]]:
    baseline: dict[tuple[int, str], dict[str, Any]] = {}
    for summary in parity_report.get("summaries", []):
        if summary.get("method") == "CLSVOF_LS_NATIVE":
            baseline[(int(summary["level"]), str(summary["repeat_id"]))] = summary
    if not baseline:
        raise StationaryGateError("stationary_blocked_by_parity")
    return baseline


def _canary_repeat_consistency(
    *,
    summaries: list[dict[str, Any]],
    methods: list[str],
    levels: list[int],
    expected_repeat: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in methods:
        method_data: dict[str, Any] = {}
        for level in levels:
            rows = [
                summary for summary in summaries
                if summary["method"] == method and int(summary["level"]) == level
            ]
            ca_tail = [float(row["Ca_tail_max"]) for row in rows] or [float("nan")]
            verdicts = {str(row["finite_gate"]) for row in rows}
            consistency = {
                "expected_repeat": expected_repeat,
                "repeat_count": len(rows),
                "same_source_hashes": len({row["official_source_manifest_sha256"] for row in rows}) == 1
                and all(bool(row["source_hashes_unchanged"]) for row in rows),
                "same_config_hash": len({row["resolved_config_sha256"] for row in rows}) == 1,
                "same_binary_hash": len({row["binary_sha256"] for row in rows}) == 1,
                "same_raw_schema": all(bool(row["raw_schema_valid"]) for row in rows),
                "verdict_unchanged_across_repeats": len(verdicts) == 1,
                "Ca_tail_max_spread": max(ca_tail) - min(ca_tail),
            }
            consistency["pass"] = (
                consistency["repeat_count"] == expected_repeat
                and consistency["same_source_hashes"]
                and consistency["same_config_hash"]
                and consistency["same_binary_hash"]
                and consistency["same_raw_schema"]
                and consistency["verdict_unchanged_across_repeats"]
            )
            method_data[str(level)] = consistency
        result[method] = method_data
    return result


def _canary_overlay_gate(
    *,
    native_by_key: dict[tuple[str, int, str], dict[str, Any]],
    overlay_method: str,
    levels: list[int],
    expected_repeat: int,
    gate: float,
    methods: list[str],
) -> dict[str, Any]:
    if overlay_method not in methods:
        return {"pass": False, "reason": "method_not_requested"}
    comparisons = []
    for level in levels:
        for repeat_index in range(expected_repeat):
            repeat_id = str(repeat_index)
            native = native_by_key[("CLSVOF_LS_NATIVE", level, repeat_id)]
            overlay = native_by_key[(overlay_method, level, repeat_id)]
            comparison = _compare_ca_raw(
                Path(native["raw_csv"]),
                Path(overlay["raw_csv"]),
                level=level,
                repeat_id=repeat_id,
                overlay_method=overlay_method,
            )
            comparisons.append(comparison)
    max_abs = max((entry["max_abs_Ca_diff"] for entry in comparisons), default=float("inf"))
    aligned = all(entry["row_count_equal"] and entry["time_aligned"] for entry in comparisons)
    return {
        "pass": aligned and max_abs <= gate,
        "threshold": gate,
        "max_abs_Ca_diff": max_abs,
        "all_row_counts_equal": all(entry["row_count_equal"] for entry in comparisons),
        "all_time_aligned": all(entry["time_aligned"] for entry in comparisons),
        "comparisons": comparisons,
    }


def _compare_ca_raw(
    native_csv: Path,
    overlay_csv: Path,
    *,
    level: int,
    repeat_id: str,
    overlay_method: str,
) -> dict[str, Any]:
    native = _read_raw_trace(native_csv)
    overlay = _read_raw_trace(overlay_csv)
    row_count_equal = len(native) == len(overlay)
    max_abs = 0.0
    time_aligned = row_count_equal
    for left, right in zip(native, overlay, strict=False):
        if abs(float(left["t"]) - float(right["t"])) > 1e-14:
            time_aligned = False
        diff = abs(float(left["Ca"]) - float(right["Ca"]))
        if diff > max_abs:
            max_abs = diff
    if not row_count_equal:
        max_abs = float("inf")
    return {
        "level": level,
        "repeat_id": repeat_id,
        "overlay_method": overlay_method,
        "native_csv": str(native_csv),
        "overlay_csv": str(overlay_csv),
        "native_rows": len(native),
        "overlay_rows": len(overlay),
        "row_count_equal": row_count_equal,
        "time_aligned": time_aligned,
        "max_abs_Ca_diff": max_abs,
    }


def _read_raw_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _update_status_file(row: dict[str, Any]) -> None:
    status_file = Path(str(row["status_file"]))
    data = json.loads(status_file.read_text())
    data["gate_status"] = row["result_status"]
    data["evidence_level"] = row["evidence_level"]
    status_file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_canary_report(report: dict[str, Any]) -> dict[str, Any]:
    report_path = REPORTS / f"{report['run_id']}_canary_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path = REPORTS / f"{report['run_id']}_canary_summary.md"
    md_path.write_text(_render_canary_report(report))
    if not report["overall_pass"] and report.get("run_kind") != "deployable":
        raise StationaryGateError(f"STATIONARY_CANARY_GATE=FAIL:{route_relative(report_path)}")
    return report


def _render_canary_report(report: dict[str, Any]) -> str:
    if report.get("run_kind") == "deployable":
        return _render_nn_canary_report(report)
    lines = [
        "# Stationary CLSVOF Canary",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- SB-G2: `{report['SB-G2']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        "",
        "| method | level | repeats | Ca_tail_max spread | repeat consistency |",
        "|---|---:|---:|---:|---|",
    ]
    for method, levels in report["repeat_consistency"].items():
        for level, consistency in levels.items():
            lines.append(
                f"| {method} | {level} | {consistency['repeat_count']} | "
                f"{consistency['Ca_tail_max_spread']:.6e} | {consistency['pass']} |"
            )
    return "\n".join(lines) + "\n"


def _render_nn_canary_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stationary NN Canary",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- requires_parity_run_id: `{report['requires_parity_run_id']}`",
        f"- SB-G6: `{report['SB-G6']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        "",
        "| method | label | level | max ratio | spread | pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for method, levels in report["ratios"].items():
        label = report["method_labels"][method]
        for data in levels:
            lines.append(
                f"| {method} | {label} | {data['level']} | "
                f"{data['max_Ca_ratio_to_native']:.6e} | "
                f"{data['spread_Ca_ratio_to_native']:.6e} | {data['pass']} |"
            )
    return "\n".join(lines) + "\n"




def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def _summarize_stock_gate(
    *,
    run_id: str,
    summaries: list[dict[str, Any]],
    manifest_path: Path,
    expected_repeat: int,
) -> dict[str, Any]:
    by_level: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_level.setdefault(int(summary["level"]), []).append(summary)
    levels = {}
    for level, rows in sorted(by_level.items()):
        ca_tail = [float(row["Ca_tail_max"]) for row in rows]
        verdicts = {str(row["gate"]) for row in rows}
        repeat_consistency = {
            "expected_repeat": expected_repeat,
            "repeat_count": len(rows),
            "same_source_hashes": len({row["official_source_manifest_sha256"] for row in rows}) == 1
            and all(bool(row["source_hashes_unchanged"]) for row in rows),
            "same_config_hash": len({row["resolved_config_sha256"] for row in rows}) == 1,
            "same_binary_hash": len({row["binary_sha256"] for row in rows}) == 1,
            "same_raw_schema": all(bool(row["raw_schema_valid"]) for row in rows),
            "verdict_unchanged_across_repeats": len(verdicts) == 1,
            "Ca_tail_max_spread": max(ca_tail) - min(ca_tail),
        }
        repeat_consistency["pass"] = (
            repeat_consistency["repeat_count"] == expected_repeat
            and repeat_consistency["same_source_hashes"]
            and repeat_consistency["same_config_hash"]
            and repeat_consistency["same_binary_hash"]
            and repeat_consistency["same_raw_schema"]
            and repeat_consistency["verdict_unchanged_across_repeats"]
        )
        levels[str(level)] = {
            "repeat_count": len(rows),
            "max_Ca_tail_max": max(ca_tail),
            "all_reached_final_time": all(bool(row["reached_final_time"]) for row in rows),
            "all_finite_Ca": all(bool(row["finite_Ca"]) for row in rows),
            "all_pass": all(row["gate"] == "PASS" for row in rows),
            "repeat_consistency": repeat_consistency,
        }
    overall_pass = all(
        value["all_pass"] and value["repeat_consistency"]["pass"] for value in levels.values()
    )
    report = {
        "run_id": run_id,
        "benchmark": "stationary",
        "tier": "stock-reference",
        "manifest": route_relative(manifest_path),
        "levels": levels,
        "VOF_HF_REFERENCE_GATE": "PASS" if overall_pass else "FAIL",
        "evidence_level": "controlled_benchmark" if overall_pass else "runtime_smoke",
        "summaries": summaries,
    }
    report_path = REPORTS / f"{run_id}_stock_reference_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path = REPORTS / f"{run_id}_stock_reference_summary.md"
    md_path.write_text(_render_stock_report(report))
    if not overall_pass:
        raise StationaryGateError(f"VOF_HF_REFERENCE_GATE=FAIL:{route_relative(report_path)}")
    return report


def _render_stock_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stationary Stock VOF-HF Reference",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- VOF_HF_REFERENCE_GATE: `{report['VOF_HF_REFERENCE_GATE']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        "",
        "| level | repeats | max Ca_tail_max | spread | reached full time | finite Ca | repeat consistency | pass |",
        "|---:|---:|---:|---:|---|---|---|---|",
    ]
    for level, data in report["levels"].items():
        consistency = data["repeat_consistency"]
        lines.append(
            f"| {level} | {data['repeat_count']} | {data['max_Ca_tail_max']:.6e} | "
            f"{consistency['Ca_tail_max_spread']:.6e} | {data['all_reached_final_time']} | "
            f"{data['all_finite_Ca']} | {consistency['pass']} | {data['all_pass']} |"
        )
    return "\n".join(lines) + "\n"


def _resolved_stationary_config(*, levels: list[int], repeat: int) -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise StationaryGateError("stationary_config_missing")
    try:
        base = json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise StationaryGateError("stationary_config_not_json_yaml_subset") from exc
    stock = base.get("stock_reference", {})
    if not isinstance(stock, dict):
        raise StationaryGateError("stationary_config_missing_stock_reference")
    base["resolved"] = {
        "tier": "stock-reference",
        "levels": levels,
        "repeat": repeat,
        "method": "VOF_HF_NATIVE",
        "case_source": route_relative(CASE_SOURCE),
        "source_config": route_relative(CONFIG_PATH),
    }
    if float(stock.get("ca_tail_max_gate", 0.0)) != 1e-12:
        raise StationaryGateError("stationary_config_gate_mismatch")
    return base


def _resolved_stationary_canary_config(
    *,
    methods: list[str],
    levels: list[int],
    repeat: int,
    run_kind: str,
    parity_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise StationaryGateError("stationary_config_missing")
    try:
        base = json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise StationaryGateError("stationary_config_not_json_yaml_subset") from exc
    canary = base.get("canary", {})
    if not isinstance(canary, dict):
        raise StationaryGateError("stationary_config_missing_canary")
    declared_methods = list(canary.get("methods", []))
    if declared_methods != list(INERT_CANARY_METHODS):
        raise StationaryGateError("stationary_canary_methods_config_mismatch")
    deployable_methods = list(canary.get("deployable_methods", []))
    if deployable_methods != list(DEPLOYABLE_CANARY_METHODS):
        raise StationaryGateError("stationary_deployable_canary_methods_config_mismatch")
    base["resolved"] = {
        "tier": "canary",
        "run_kind": run_kind,
        "levels": levels,
        "repeat": repeat,
        "methods": methods,
        "requires_parity_run_id": parity_report.get("run_id") if parity_report else "",
        "case_sources": {
            "CLSVOF_LS_NATIVE": route_relative(NATIVE_CLSVOF_SOURCE),
            "NN27_RAW": route_relative(NN_CLSVOF_SOURCE),
        },
        "source_config": route_relative(CONFIG_PATH),
    }
    return base


def _canary_run_kind(methods: list[str]) -> str:
    requested = set(methods)
    if requested.issubset(set(INERT_CANARY_METHODS)):
        if "CLSVOF_LS_NATIVE" not in requested:
            raise StationaryGateError("stationary_inert_canary_requires_clsvof_native")
        return "inert"
    if requested.issubset(set(SUPPORTED_DEPLOYABLE_METHODS)):
        return "deployable"
    raise StationaryGateError("stationary_canary_mixes_inert_and_deployable_methods")


def _load_latest_inert_parity_report() -> dict[str, Any]:
    candidates = sorted(
        REPORTS.glob("stationary_canary_*_canary_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        data = json.loads(path.read_text())
        if (
            data.get("tier") == "canary"
            and set(data.get("methods", [])) == set(INERT_CANARY_METHODS)
            and data.get("SB-G2") == "PASS"
            and data.get("overall_pass") is True
        ):
            return data
    raise StationaryGateError("stationary_blocked_by_parity")


def _canary_case_source(method: str) -> Path:
    if method == "CLSVOF_LS_NATIVE":
        return NATIVE_CLSVOF_SOURCE
    if method in NN_MODE_VALUES:
        return NN_CLSVOF_SOURCE
    raise StationaryGateError(f"stationary_canary_unknown_method:{method}")
