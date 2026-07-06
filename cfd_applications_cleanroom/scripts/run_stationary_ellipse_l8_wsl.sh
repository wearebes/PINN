#!/usr/bin/env bash
set -euo pipefail

# Run the full stationary-ellipse 256x256 process matrix.
#
# Intended use on Windows is inside WSL/Ubuntu from the repository root:
#
#   cd /path/to/PINN
#   MAX_JOBS=4 bash cfd_applications_cleanroom/scripts/run_stationary_ellipse_l8_wsl.sh
#
# Optional filters:
#   CASES="E1 E2"
#   METHODS="NN_DISABLE NN_PROBE_ONLY NN27_RAW NN27_D4"
#   MAX_JOBS=4

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CASES="${CASES:-E1 E2}"
METHODS="${METHODS:-NN_DISABLE NN_PROBE_ONLY NN27_RAW NN27_D4}"
MAX_JOBS="${MAX_JOBS:-4}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid_MAX_JOBS=${MAX_JOBS}" >&2
  exit 2
fi

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="cfd_applications_cleanroom/results/background/stationary_ellipse_l8_wsl_${RUN_TAG}"
mkdir -p "$LOG_DIR"
FAILED=0

echo "repo=$ROOT"
echo "log_dir=$LOG_DIR"
echo "cases=$CASES"
echo "methods=$METHODS"
echo "max_jobs=$MAX_JOBS"

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [ "$(active_jobs)" -ge "$MAX_JOBS" ]; do
    if ! wait -n; then
      FAILED=1
    fi
  done
}

run_row() {
  local case_id="$1"
  local method="$2"
  local label="wsl_l8_${case_id}_${method}"
  local log_path="${LOG_DIR}/${case_id}_L8_${method}.log"
  echo "START ${case_id}:8:${method} log=${log_path}"
  "$PYTHON_BIN" cfd_applications_cleanroom/scripts/run_stationary_ellipse_process_subset.py \
    --label "$label" \
    --row "${case_id}:8:${method}" \
    >"$log_path" 2>&1
  echo "DONE ${case_id}:8:${method}"
}

for case_id in $CASES; do
  for method in $METHODS; do
    wait_for_slot
    run_row "$case_id" "$method" &
  done
done

while [ "$(active_jobs)" -gt 0 ]; do
  if ! wait -n; then
    FAILED=1
  fi
done

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py

"$PYTHON_BIN" - <<'PY' | tee "${LOG_DIR}/open_rows.txt"
import csv
from pathlib import Path

summary = Path("Experiment/report/stationary bubble/summary.csv")
with summary.open(newline="") as f:
    rows = list(csv.DictReader(f))

open_rows = [
    row for row in rows
    if row.get("benchmark_family") == "stationary_ellipse"
    and row.get("tier") == "curvature-process"
    and row.get("report_ready") != "true"
]

print(f"summary_csv={summary}")
print(f"stationary_ellipse_process_open_rows={len(open_rows)}")
for row in open_rows:
    print(
        "OPEN",
        row.get("case_id"),
        row.get("grid_n"),
        row.get("method"),
        row.get("status"),
        row.get("final_tau_or_current_tau"),
        row.get("progress_percent"),
    )
PY

if [ "$FAILED" -ne 0 ]; then
  echo "run_status=failed" | tee "${LOG_DIR}/FAILED"
  exit 1
fi

echo "run_status=complete" | tee "${LOG_DIR}/DONE"
