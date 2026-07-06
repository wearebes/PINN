from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest

from cfd_applications_cleanroom.cfd_apps.manifest import (
    ManifestError,
    STALE_OUTPUT_ERROR,
    stale_output_guard,
)
from cfd_applications_cleanroom.cfd_apps.schema import (
    COMMON_TRACE_COLUMNS,
    MANIFEST_COLUMNS,
    SchemaError,
    validate_manifest_row,
    validate_trace_header,
)


def test_common_trace_schema_columns_are_pinned() -> None:
    assert COMMON_TRACE_COLUMNS == (
        "run_id",
        "repeat_id",
        "benchmark",
        "case_id",
        "method",
        "deployable",
        "evidence_level",
        "level",
        "grid_n",
        "i",
        "t",
        "dt",
        "primary_metric_name",
        "primary_metric_value",
    )


def test_manifest_schema_columns_are_pinned() -> None:
    assert "raw_csv" in MANIFEST_COLUMNS
    assert "official_source_manifest_sha256" in MANIFEST_COLUMNS
    assert "nn_weights_header_sha256" in MANIFEST_COLUMNS


def test_trace_header_validation_rejects_missing_common_column(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text("run_id,t\nr1,0\n")

    with pytest.raises(SchemaError):
        validate_trace_header(trace)


def test_manifest_row_validation_rejects_missing_required_field() -> None:
    row = {column: "" for column in MANIFEST_COLUMNS if column != "raw_csv"}

    with pytest.raises(SchemaError):
        validate_manifest_row(row)


def test_stale_guard_accepts_manifested_fresh_trace(tmp_path: Path) -> None:
    compile_log = tmp_path / "compile.log"
    raw_csv = tmp_path / "raw.csv"
    compile_log.write_text("compile\n")
    time.sleep(0.01)
    _write_trace(raw_csv, run_id="run-a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [_manifest_row(raw_csv, compile_log, "run-a")]}))

    rows = stale_output_guard(manifest)

    assert len(rows) == 1


def test_stale_guard_rejects_duplicate_manifest_reference(tmp_path: Path) -> None:
    compile_log = tmp_path / "compile.log"
    raw_csv = tmp_path / "raw.csv"
    compile_log.write_text("compile\n")
    time.sleep(0.01)
    _write_trace(raw_csv, run_id="run-a")
    row = _manifest_row(raw_csv, compile_log, "run-a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [row, dict(row)]}))

    with pytest.raises(ManifestError, match=STALE_OUTPUT_ERROR):
        stale_output_guard(manifest)


def test_stale_guard_rejects_raw_older_than_compile_log(tmp_path: Path) -> None:
    compile_log = tmp_path / "compile.log"
    raw_csv = tmp_path / "raw.csv"
    _write_trace(raw_csv, run_id="run-a")
    time.sleep(0.01)
    compile_log.write_text("compile\n")
    os.utime(compile_log, None)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [_manifest_row(raw_csv, compile_log, "run-a")]}))

    with pytest.raises(ManifestError, match=STALE_OUTPUT_ERROR):
        stale_output_guard(manifest)


def test_stale_guard_rejects_run_id_mismatch(tmp_path: Path) -> None:
    compile_log = tmp_path / "compile.log"
    raw_csv = tmp_path / "raw.csv"
    compile_log.write_text("compile\n")
    time.sleep(0.01)
    _write_trace(raw_csv, run_id="run-b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [_manifest_row(raw_csv, compile_log, "run-a")]}))

    with pytest.raises(ManifestError, match=STALE_OUTPUT_ERROR):
        stale_output_guard(manifest)


def _write_trace(path: Path, *, run_id: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMMON_TRACE_COLUMNS) + ["Ca"])
        writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "repeat_id": "0",
                "benchmark": "stationary",
                "case_id": "stationary-smoke",
                "method": "CLSVOF_LS_NATIVE",
                "deployable": "true",
                "evidence_level": "contract_validated",
                "level": "6",
                "grid_n": "64",
                "i": "0",
                "t": "0.0",
                "dt": "0.0",
                "primary_metric_name": "Ca",
                "primary_metric_value": "0.0",
                "Ca": "0.0",
            }
        )


def _manifest_row(raw_csv: Path, compile_log: Path, run_id: str) -> dict[str, str]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "repeat_id": "0",
            "benchmark": "stationary",
            "method": "CLSVOF_LS_NATIVE",
            "level": "6",
            "grid_n": "64",
            "compile_log": str(compile_log),
            "raw_csv": str(raw_csv),
            "result_status": "accepted_contract",
            "evidence_level": "contract_validated",
        }
    )
    return row
