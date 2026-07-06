#!/usr/bin/env python3
"""Pause, resume, or inspect local stationary-ellipse L8 simulator binaries."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


METHOD_FROM_BINARY = {
    "nn_disable": "NN_DISABLE",
    "nn_probe_only": "NN_PROBE_ONLY",
    "nn27_raw": "NN27_RAW",
    "nn27_d4": "NN27_D4",
}
SIM_RE = re.compile(r"stationary_ellipse_curvature_process_(e[12])_(.+)_L8$")


@dataclass(frozen=True)
class Simulator:
    pid: int
    state: str
    row: tuple[str, str]
    command: str


def simulator_row(command: str) -> tuple[str, str] | None:
    exe = command.split()[0]
    name = Path(exe).name
    match = SIM_RE.search(name)
    if not match:
        return None
    method = METHOD_FROM_BINARY.get(match.group(2))
    if method is None:
        return None
    return match.group(1).upper(), method


def list_l8_simulators() -> list[Simulator]:
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
        simulators.append(Simulator(int(pid_text), state, row, command))
    return sorted(simulators, key=lambda sim: (sim.row[0], sim.row[1], sim.pid))


def print_status(simulators: list[Simulator]) -> None:
    if not simulators:
        print("no_stationary_ellipse_l8_simulators")
        return
    for sim in simulators:
        state = "paused" if sim.state.startswith("T") else "running"
        print(f"{state} pid={sim.pid} case={sim.row[0]} method={sim.row[1]} state={sim.state}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "pause", "resume"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    simulators = list_l8_simulators()
    if args.action == "status":
        print_status(simulators)
        return 0

    changed = []
    for sim in simulators:
        paused = sim.state.startswith("T")
        if args.action == "pause" and not paused:
            changed.append(sim)
            if not args.dry_run:
                os.kill(sim.pid, signal.SIGSTOP)
        elif args.action == "resume" and paused:
            changed.append(sim)
            if not args.dry_run:
                os.kill(sim.pid, signal.SIGCONT)

    print(f"{args.action}_changed={len(changed)} dry_run={args.dry_run}")
    for sim in changed:
        print(f"{args.action.upper()} pid={sim.pid} case={sim.row[0]} method={sim.row[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
