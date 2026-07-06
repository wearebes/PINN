#!/usr/bin/env python3
"""Run selected stationary-ellipse process rows with per-row completion checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
BACKGROUND = ROOT / "cfd_applications_cleanroom/results/background"
PYTHON = sys.executable


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


def parse_row(spec: str) -> tuple[str, int, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"row must be CASE:LEVEL:METHOD, got {spec!r}")
    case, level_text, method = parts
    if case not in {"E1", "E2"}:
        raise argparse.ArgumentTypeError(f"unsupported case {case!r}")
    try:
        level = int(level_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid level {level_text!r}") from exc
    if level not in {6, 7, 8}:
        raise argparse.ArgumentTypeError(f"unsupported level {level!r}")
    if method not in {"NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4"}:
        raise argparse.ArgumentTypeError(f"unsupported method {method!r}")
    return case, level, method


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--row", action="append", required=True, type=parse_row)
    args = parser.parse_args()

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.label)
    log_dir = BACKGROUND / f"stationary_ellipse_subset_{safe_label}_{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.jsonl"

    with status_path.open("w") as status:
        for case, level, method in args.row:
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

    (log_dir / "DONE").write_text(f"complete_utc={_utc()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
