#!/usr/bin/env python3
"""Stage stationary-ellipse process runs to avoid CPU oversubscription.

The campaign can have many already-started L8 simulator binaries.  This helper
keeps short D4 gaps moving first, then resumes L8 rows in a bounded batch.  It
only sends SIGSTOP/SIGCONT to simulator binaries; it does not terminate runs or
modify raw result files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"

PROCESS_METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
PROCESS_ROWS = tuple(
    (case, level, method)
    for case in ("E1", "E2")
    for level in (6, 7, 8)
    for method in PROCESS_METHODS
)

HIGH_PRIORITY_ROWS = (
    ("E1", 7, "NN27_D4"),
    ("E2", 6, "NN27_D4"),
    ("E2", 7, "NN27_D4"),
)

METHOD_FROM_BINARY = {
    "nn_disable": "NN_DISABLE",
    "nn_probe_only": "NN_PROBE_ONLY",
    "nn27_raw": "NN27_RAW",
    "nn27_d4": "NN27_D4",
}

SIM_RE = re.compile(r"stationary_ellipse_curvature_process_(e[12])_(.+)_L([678])$")


@dataclass(frozen=True)
class Simulator:
    pid: int
    state: str
    command: str
    row: tuple[str, int, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def complete_summary(row: tuple[str, int, str]) -> bool:
    case, level, method = row
    for path in sorted(
        RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/summary.json"),
        reverse=True,
    ):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        try:
            final_tau = float(data.get("final_tau", 0.0))
        except Exception:
            final_tau = 0.0
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == level
            and data.get("reached_final_time") is True
            and final_tau >= 0.999
        ):
            return True
    return False


def simulator_row(command: str) -> tuple[str, int, str] | None:
    name = Path(command.split()[0]).name
    match = SIM_RE.search(name)
    if not match:
        return None
    case = match.group(1).upper()
    method = METHOD_FROM_BINARY.get(match.group(2))
    if not method:
        return None
    return case, int(match.group(3)), method


def list_simulators() -> list[Simulator]:
    output = subprocess.check_output(["ps", "-axo", "pid,state,command"], text=True)
    simulators: list[Simulator] = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            pid_text, state, command = line.split(maxsplit=2)
        except ValueError:
            continue
        row = simulator_row(command)
        if row is None:
            continue
        simulators.append(Simulator(int(pid_text), state, command, row))
    return simulators


def choose_allowed(max_l8_running: int) -> set[tuple[str, int, str]]:
    incomplete = [row for row in PROCESS_ROWS if not complete_summary(row)]
    high_incomplete = [row for row in HIGH_PRIORITY_ROWS if row in incomplete]
    if high_incomplete:
        return set(HIGH_PRIORITY_ROWS)

    l8_rows = [row for row in incomplete if row[1] == 8]
    l8_rows.sort(key=lambda row: (row[0], row[2]))
    return set(l8_rows[:max_l8_running])


def set_state(sim: Simulator, should_run: bool, dry_run: bool) -> None:
    stopped = sim.state.startswith("T")
    if should_run and stopped:
        log(f"CONT pid={sim.pid} row={sim.row}")
        if not dry_run:
            os.kill(sim.pid, signal.SIGCONT)
    elif not should_run and not stopped:
        log(f"STOP pid={sim.pid} row={sim.row}")
        if not dry_run:
            os.kill(sim.pid, signal.SIGSTOP)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-l8-running", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log(
        "start "
        f"poll_seconds={args.poll_seconds} max_l8_running={args.max_l8_running} dry_run={args.dry_run}"
    )
    while True:
        allowed = choose_allowed(args.max_l8_running)
        simulators = list_simulators()
        active_rows = {sim.row for sim in simulators if not sim.state.startswith("T")}
        stopped_rows = {sim.row for sim in simulators if sim.state.startswith("T")}
        complete_count = sum(1 for row in PROCESS_ROWS if complete_summary(row))
        log(
            "status "
            f"complete={complete_count}/24 allowed={sorted(allowed)} "
            f"active={sorted(active_rows)} stopped={sorted(stopped_rows)}"
        )
        for sim in simulators:
            set_state(sim, sim.row in allowed, args.dry_run)
        if complete_count == len(PROCESS_ROWS):
            log("all_complete")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
