"""Schema contracts for cleanroom trace and manifest evidence."""

from __future__ import annotations

import csv
from pathlib import Path


COMMON_TRACE_COLUMNS = (
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

MANIFEST_COLUMNS = (
    "run_id",
    "repeat_id",
    "benchmark",
    "method",
    "level",
    "grid_n",
    "case_source_path",
    "case_source_sha256",
    "official_source_root",
    "official_source_manifest_sha256",
    "compile_command",
    "run_command",
    "compile_log",
    "stdout_log",
    "stderr_log",
    "status_file",
    "raw_csv",
    "summary_json",
    "checkpoint_path",
    "checkpoint_sha256",
    "nn_weights_header_sha256",
    "result_status",
    "evidence_level",
)


class SchemaError(ValueError):
    """Raised when a cleanroom artifact violates its declared schema."""


def validate_trace_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SchemaError(f"empty_trace_csv:{path}") from exc
    missing = [column for column in COMMON_TRACE_COLUMNS if column not in header]
    if missing:
        raise SchemaError(f"trace_schema_missing_columns:{','.join(missing)}")
    return header


def validate_no_duplicate_time_rows(path: Path) -> None:
    validate_trace_header(path)
    seen: set[tuple[str, str, str, str, str]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                row["run_id"],
                row["repeat_id"],
                row["benchmark"],
                row["method"],
                row["t"],
            )
            if key in seen:
                raise SchemaError(f"duplicate_time_row:{path}:{key}")
            seen.add(key)


def validate_manifest_row(row: dict[str, object]) -> None:
    missing = [column for column in MANIFEST_COLUMNS if column not in row]
    if missing:
        raise SchemaError(f"manifest_schema_missing_columns:{','.join(missing)}")
