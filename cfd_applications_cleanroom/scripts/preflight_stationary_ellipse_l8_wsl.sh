#!/usr/bin/env bash
set -euo pipefail

# Lightweight preflight for the stationary-ellipse 256x256 WSL run.
# This does not launch any solver rows.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_JOBS="${MAX_JOBS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/cfd_applications_cleanroom/results/background/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cfd_applications_cleanroom/results/background/xdg_cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid_MAX_JOBS=${MAX_JOBS}" >&2
  exit 2
fi

echo "repo=$ROOT"
echo "python_bin=$PYTHON_BIN"
echo "max_jobs=$MAX_JOBS"

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m cfd_applications_cleanroom.cfd_apps.cli --help >/dev/null
"$PYTHON_BIN" -m cfd_applications_cleanroom.cfd_apps.cli audit
"$PYTHON_BIN" -m cfd_applications_cleanroom.cfd_apps.cli figures --check-python-runtime

bash -n cfd_applications_cleanroom/scripts/run_stationary_ellipse_l8_wsl.sh
bash -n cfd_applications_cleanroom/scripts/run_and_pack_stationary_ellipse_l8_wsl.sh
bash -n cfd_applications_cleanroom/scripts/pack_stationary_ellipse_l8_results.sh
bash -n cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh

"$PYTHON_BIN" -m py_compile \
  cfd_applications_cleanroom/scripts/audit_stationary_ellipse_l8.py \
  cfd_applications_cleanroom/scripts/run_stationary_ellipse_process_subset.py \
  cfd_applications_cleanroom/scripts/validate_stationary_ellipse_l8_package.py \
  cfd_applications_cleanroom/scripts/write_stationary_ellipse_status.py \
  cfd_applications_cleanroom/scripts/verify_stationary_ellipse_delivery.py

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/audit_stationary_ellipse_l8.py --allow-incomplete

echo "preflight_status=PASS"
