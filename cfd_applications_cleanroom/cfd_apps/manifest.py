"""Manifest loading and stale-output guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.schema import (
    SchemaError,
    validate_manifest_row,
    validate_no_duplicate_time_rows,
    validate_trace_header,
)


STALE_OUTPUT_ERROR = "stale_or_unmanifested_output"


class ManifestError(ValueError):
    """Raised when manifest-backed evidence is absent or unsafe to summarize."""


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ManifestError(STALE_OUTPUT_ERROR)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "rows" not in data:
        raise ManifestError(STALE_OUTPUT_ERROR)
    if not isinstance(data["rows"], list):
        raise ManifestError(STALE_OUTPUT_ERROR)
    for row in data["rows"]:
        if not isinstance(row, dict):
            raise ManifestError(STALE_OUTPUT_ERROR)
        try:
            validate_manifest_row(row)
        except SchemaError as exc:
            raise ManifestError(STALE_OUTPUT_ERROR) from exc
    return data


def stale_output_guard(manifest_path: Path, *, accepted_only: bool = True) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    rows = manifest["rows"]
    if accepted_only:
        rows = [row for row in rows if _is_accepted(row)]
    if not rows:
        return []

    raw_counts: dict[str, int] = {}
    for row in rows:
        raw_csv = str(row["raw_csv"])
        raw_counts[raw_csv] = raw_counts.get(raw_csv, 0) + 1
    if any(count != 1 for count in raw_counts.values()):
        raise ManifestError(STALE_OUTPUT_ERROR)

    for row in rows:
        _validate_manifested_raw_csv(row)
    return rows


def _validate_manifested_raw_csv(row: dict[str, Any]) -> None:
    raw_csv = Path(str(row["raw_csv"]))
    compile_log = Path(str(row["compile_log"]))
    if not raw_csv.exists() or not compile_log.exists():
        raise ManifestError(STALE_OUTPUT_ERROR)
    if raw_csv.stat().st_mtime < compile_log.stat().st_mtime:
        raise ManifestError(STALE_OUTPUT_ERROR)
    try:
        validate_trace_header(raw_csv)
        validate_no_duplicate_time_rows(raw_csv)
    except SchemaError as exc:
        raise ManifestError(STALE_OUTPUT_ERROR) from exc
    with raw_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_row in reader:
            if csv_row["run_id"] != str(row["run_id"]):
                raise ManifestError(STALE_OUTPUT_ERROR)


def _is_accepted(row: dict[str, Any]) -> bool:
    result_status = str(row.get("result_status", ""))
    evidence_level = str(row.get("evidence_level", ""))
    return result_status.startswith("accepted") or evidence_level in {
        "contract_validated",
        "parity_validated",
        "controlled_diagnostic",
        "controlled_benchmark",
        "publication_candidate",
        "accepted_evidence",
    }
