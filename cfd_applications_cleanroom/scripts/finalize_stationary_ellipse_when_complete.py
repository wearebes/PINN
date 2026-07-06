#!/usr/bin/env python3
"""Wait for the stationary-ellipse process matrix, then merge CSV and figures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
BACKGROUND = ROOT / "cfd_applications_cleanroom/results/background"
PYTHON = sys.executable

CASES = ("E1", "E2")
LEVELS = (6, 7, 8)
METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
EXPECTED_ROWS = [(case, level, method) for case in CASES for level in LEVELS for method in METHODS]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_complete_summary(case: str, level: int, method: str) -> Path | None:
    candidates = sorted(RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/summary.json"))
    for path in reversed(candidates):
        data = load_json(path)
        if not data:
            continue
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == level
            and data.get("reached_final_time") is True
            and _safe_float(data.get("final_tau")) >= 0.999
        ):
            return path
    return None


def completion_snapshot() -> dict[str, Any]:
    rows = []
    complete_count = 0
    for case, level, method in EXPECTED_ROWS:
        summary_path = latest_complete_summary(case, level, method)
        complete = summary_path is not None
        complete_count += int(complete)
        row: dict[str, Any] = {
            "case": case,
            "level": level,
            "grid_n": 1 << level,
            "method": method,
            "complete": complete,
            "summary_json": str(summary_path) if summary_path is not None else "",
        }
        if summary_path is not None:
            data = load_json(summary_path) or {}
            row["final_tau"] = data.get("final_tau", "")
            row["reached_final_time"] = data.get("reached_final_time", "")
            row["trace_row_count"] = data.get("trace_row_count", "")
        rows.append(row)
    return {
        "utc": _utc(),
        "complete_count": complete_count,
        "expected_count": len(EXPECTED_ROWS),
        "all_complete": complete_count == len(EXPECTED_ROWS),
        "rows": rows,
    }


def wait_for_completion(log_dir: Path, poll_seconds: int) -> dict[str, Any]:
    status_path = log_dir / "completion_status.jsonl"
    while True:
        snapshot = completion_snapshot()
        with status_path.open("a") as handle:
            handle.write(json.dumps({
                "utc": snapshot["utc"],
                "complete_count": snapshot["complete_count"],
                "expected_count": snapshot["expected_count"],
                "all_complete": snapshot["all_complete"],
            }) + "\n")
        (log_dir / "latest_completion_audit.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        if snapshot["all_complete"]:
            return snapshot
        time.sleep(poll_seconds)


def run_merge(log_dir: Path) -> None:
    log_path = log_dir / "merge_stationary_ellipse_summary.log"
    command = [PYTHON, str(ROOT / "cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py")]
    with log_path.open("w") as log:
        log.write(f"start_utc={_utc()}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"end_utc={_utc()}\n")
        log.write(f"returncode={result.returncode}\n")
    if result.returncode != 0:
        raise SystemExit(f"merge_failed:log={log_path}")


def render_process_plates(log_dir: Path) -> dict[str, Any]:
    from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import render_curvature_process_plate

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: dict[str, Any] = {
        "utc": _utc(),
        "plates": [],
    }
    for case in CASES:
        for level in LEVELS:
            summaries = []
            for method in METHODS:
                summary_path = latest_complete_summary(case, level, method)
                if summary_path is None:
                    raise SystemExit(f"missing_complete_summary_for_figure:{case}:L{level}:{method}")
                data = load_json(summary_path)
                if not data:
                    raise SystemExit(f"unreadable_summary_for_figure:{summary_path}")
                summaries.append(data)
            native_summary = next(summary for summary in summaries if summary["method"] == "NN_DISABLE")
            run_id = f"stationary_ellipse_process_plate_{case.lower()}_l{level}_{run_tag}"
            artifact_prefix = f"stationary_ellipse_curvature_process_{case.lower()}_l{level}"
            figures = render_curvature_process_plate(
                run_id=run_id,
                summaries=summaries,
                artifact_prefix=artifact_prefix,
                native_trace_csv_override=Path(native_summary["surface_tension_trace_csv"]),
                trace_yscale="log",
            )
            png = Path(figures["png"])
            if not png.exists() or png.stat().st_size <= 0:
                raise SystemExit(f"missing_or_empty_png:{png}")
            manifest["plates"].append({
                "case": case,
                "level": level,
                "grid_n": 1 << level,
                **figures,
            })
    out_path = log_dir / "figure_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = BACKGROUND / f"stationary_ellipse_finalizer_{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (BACKGROUND / "stationary_ellipse_finalizer_latest.txt").write_text(str(log_dir) + "\n")

    snapshot = wait_for_completion(log_dir, args.poll_seconds)
    (log_dir / "completion_audit.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    run_merge(log_dir)
    figure_manifest = render_process_plates(log_dir)
    done = {
        "utc": _utc(),
        "completion_audit": str(log_dir / "completion_audit.json"),
        "summary_csv": str(ROOT / "Experiment/report/stationary bubble/summary.csv"),
        "figure_manifest": str(log_dir / "figure_manifest.json"),
        "plate_count": len(figure_manifest["plates"]),
    }
    (log_dir / "DONE").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
