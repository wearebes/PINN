#!/usr/bin/env bash
set -euo pipefail

# Import a completed stationary-ellipse L8 transfer package, then verify,
# merge the report summary, and render final process PNG plates.

if [ "$#" -ne 1 ]; then
  echo "usage: $0 cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_*.tar.gz" >&2
  exit 2
fi

PACKAGE="$1"
if [ ! -f "$PACKAGE" ]; then
  echo "package_not_found: $PACKAGE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/cfd_applications_cleanroom/results/background/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cfd_applications_cleanroom/results/background/xdg_cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/validate_stationary_ellipse_l8_package.py "$PACKAGE"
tar -xzf "$PACKAGE"

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py
"$PYTHON_BIN" cfd_applications_cleanroom/scripts/write_stationary_ellipse_status.py

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

raw = Path("cfd_applications_cleanroom/results/raw/stationary_ellipse")
summary_csv = Path("Experiment/report/stationary bubble/summary.csv")
methods = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")

missing_complete = []
for case in ("E1", "E2"):
    for method in methods:
        complete = False
        for summary in sorted(
            raw.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L8/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(summary.read_text())
            except Exception:
                continue
            final_tau = float(data.get("final_tau", 0.0) or 0.0)
            if (
                data.get("method") == method
                and int(data.get("level", -1)) == 8
                and data.get("reached_final_time") is True
                and final_tau >= 0.999
            ):
                complete = True
                break
        if not complete:
            missing_complete.append((case, method))

with summary_csv.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
proc = [
    row for row in rows
    if row.get("benchmark_family") == "stationary_ellipse"
    and row.get("tier") == "curvature-process"
]
open_rows = [row for row in proc if row.get("report_ready") != "true"]

print(f"stationary_ellipse_process_ready={len(proc) - len(open_rows)}/{len(proc)}")
if missing_complete or open_rows:
    print("import_not_complete")
    for case, method in missing_complete:
        print(f"MISSING_COMPLETE_L8 {case} {method}")
    for row in open_rows:
        print(
            "OPEN_SUMMARY",
            row.get("case_id"),
            row.get("grid_n"),
            row.get("method"),
            row.get("status"),
            row.get("final_tau_or_current_tau"),
            row.get("progress_percent"),
        )
    raise SystemExit(3)
PY

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/finalize_stationary_ellipse_when_complete.py --poll-seconds 1
"$PYTHON_BIN" cfd_applications_cleanroom/scripts/write_stationary_ellipse_status.py
"$PYTHON_BIN" cfd_applications_cleanroom/scripts/verify_stationary_ellipse_delivery.py

echo "import_complete"
echo "summary_csv=Experiment/report/stationary bubble/summary.csv"
latest="cfd_applications_cleanroom/results/background/stationary_ellipse_finalizer_latest.txt"
if [ -f "$latest" ]; then
  echo "finalizer_dir=$(cat "$latest")"
fi
