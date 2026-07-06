#!/usr/bin/env python3
"""Queue selected stationary-ellipse process rows until simulator capacity opens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    level = int(level_text)
    if case not in {"E1", "E2"} or level not in {6, 7, 8}:
        raise argparse.ArgumentTypeError(f"unsupported row {spec!r}")
    if method not in {"NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4"}:
        raise argparse.ArgumentTypeError(f"unsupported method {method!r}")
    return case, level, method


def active_simulator_count() -> int:
    try:
        output = subprocess.check_output(["ps", "-axo", "command"], text=True)
    except subprocess.CalledProcessError:
        return 0
    count = 0
    for line in output.splitlines():
        if (
            "/results/raw/stationary_ellipse/" in line
            and "/build/" in line
            and "stationary_ellipse_curvature_process_" in line
        ):
            count += 1
    return count


def run_row(case: str, level: int, method: str, log_dir: Path) -> int:
    log_path = log_dir / f"{case}_L{level}_{method}.log"
    command = [
        PYTHON,
        str(ROOT / "cfd_applications_cleanroom/scripts/run_stationary_ellipse_process_subset.py"),
        "--label",
        f"queued_{case.lower()}_l{level}_{method.lower()}",
        "--row",
        f"{case}:{level}:{method}",
    ]
    with log_path.open("w") as log:
        log.write(f"start_utc={_utc()}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"end_utc={_utc()}\n")
        log.write(f"returncode={result.returncode}\n")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--row", action="append", required=True, type=parse_row)
    parser.add_argument("--max-active-simulators", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.label)
    log_dir = BACKGROUND / f"stationary_ellipse_queue_{safe_label}_{run_tag}"
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

            while active_simulator_count() >= args.max_active_simulators:
                status.write(json.dumps({
                    "utc": _utc(),
                    "case": case,
                    "level": level,
                    "method": method,
                    "status": "waiting_for_capacity",
                    "active_simulators": active_simulator_count(),
                }) + "\n")
                status.flush()
                time.sleep(args.poll_seconds)

            status.write(json.dumps({
                "utc": _utc(),
                "case": case,
                "level": level,
                "method": method,
                "status": "run_start",
                "active_simulators": active_simulator_count(),
            }) + "\n")
            status.flush()
            returncode = run_row(case, level, method, log_dir)
            completed = complete_summary_for(case, level, method)
            status.write(json.dumps({
                "utc": _utc(),
                "case": case,
                "level": level,
                "method": method,
                "status": "run_complete" if returncode == 0 else "run_failed",
                "returncode": returncode,
                "summary_json": str(completed) if completed is not None else "",
            }) + "\n")
            status.flush()
            if returncode != 0:
                return returncode

    (log_dir / "DONE").write_text(f"complete_utc={_utc()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
