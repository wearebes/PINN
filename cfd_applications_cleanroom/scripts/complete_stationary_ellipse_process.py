#!/usr/bin/env python3
"""Complete stationary-ellipse process rows one case/level/method at a time.

This script is intended for long-running local recovery after an interrupted
matrix run. It treats any existing full-time summary.json as complete and only
runs rows that are missing or partial.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def complete_summary_for(case: str, level: int, method: str) -> Path | None:
    candidates = sorted(RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/summary.json"))
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == level
            and data.get("reached_final_time") is True
            and _safe_float(data.get("final_tau")) >= 0.999
        ):
            return path
    return None


def run_row(case: str, level: int, method: str, log_dir: Path) -> None:
    log_path = log_dir / f"{case}_L{level}_{method}.log"
    command = [
        PYTHON,
        "-m",
        "cfd_applications_cleanroom.cfd_apps.cli",
        "reproduce",
        "--benchmark",
        "stationary_ellipse",
        "--tier",
        "curvature-process",
        "--cases",
        case,
        "--levels",
        str(level),
        "--methods",
        method,
    ]
    with log_path.open("w") as log:
        log.write(f"start_utc={_utc()}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"end_utc={_utc()}\n")
        log.write(f"returncode={result.returncode}\n")
    if result.returncode != 0:
        raise SystemExit(f"row_failed:{case}:L{level}:{method}:log={log_path}")


def main() -> int:
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = BACKGROUND / f"stationary_ellipse_row_completion_{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (BACKGROUND / "stationary_ellipse_row_completion_latest.txt").write_text(str(log_dir) + "\n")
    status_path = log_dir / "status.jsonl"

    with status_path.open("w") as status:
        for case, level, method in EXPECTED_ROWS:
            existing = complete_summary_for(case, level, method)
            if existing is not None:
                status.write(json.dumps({
                    "utc": _utc(),
                    "case": case,
                    "level": level,
                    "method": method,
                    "status": "skip_existing_complete",
                    "summary_json": str(existing),
                }) + "\n")
                status.flush()
                continue
            status.write(json.dumps({
                "utc": _utc(),
                "case": case,
                "level": level,
                "method": method,
                "status": "run_start",
            }) + "\n")
            status.flush()
            run_row(case, level, method, log_dir)
            completed = complete_summary_for(case, level, method)
            status.write(json.dumps({
                "utc": _utc(),
                "case": case,
                "level": level,
                "method": method,
                "status": "run_complete",
                "summary_json": str(completed) if completed is not None else "",
            }) + "\n")
            status.flush()

    merge_log = log_dir / "merge_stationary_ellipse_summary.log"
    merge_command = [PYTHON, str(ROOT / "cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py")]
    with merge_log.open("w") as log:
        log.write(f"start_utc={_utc()}\n")
        log.write("command=" + " ".join(merge_command) + "\n")
        log.flush()
        result = subprocess.run(merge_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"end_utc={_utc()}\n")
        log.write(f"returncode={result.returncode}\n")
    if result.returncode != 0:
        raise SystemExit(f"merge_failed:log={merge_log}")

    (log_dir / "DONE").write_text(f"complete_utc={_utc()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
