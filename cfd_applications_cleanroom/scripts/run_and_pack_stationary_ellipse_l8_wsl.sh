#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for Windows/WSL:
#   1. Run the stationary-ellipse L8 process matrix.
#   2. Audit that all eight L8 rows reached final time.
#   3. Package raw outputs and logs for transfer back to the Mac workspace.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "step=preflight"
bash cfd_applications_cleanroom/scripts/preflight_stationary_ellipse_l8_wsl.sh

echo "step=run_l8"
bash cfd_applications_cleanroom/scripts/run_stationary_ellipse_l8_wsl.sh

echo "step=audit_l8"
"$PYTHON_BIN" cfd_applications_cleanroom/scripts/audit_stationary_ellipse_l8.py

echo "step=pack_l8"
bash cfd_applications_cleanroom/scripts/pack_stationary_ellipse_l8_results.sh
