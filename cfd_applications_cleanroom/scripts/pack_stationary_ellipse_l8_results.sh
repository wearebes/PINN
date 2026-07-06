#!/usr/bin/env bash
set -euo pipefail

# Package stationary-ellipse L8 (256x256) process outputs after a WSL run.
# Run from the repository root after run_stationary_ellipse_l8_wsl.sh finishes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="cfd_applications_cleanroom/results/transfer"
PACKAGE="${OUT_DIR}/stationary_ellipse_l8_results_${RUN_TAG}.tar.gz"
MANIFEST="${OUT_DIR}/stationary_ellipse_l8_results_${RUN_TAG}.manifest.txt"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/audit_stationary_ellipse_l8.py

PATH_LIST="${OUT_DIR}/stationary_ellipse_l8_results_${RUN_TAG}.paths.txt"
"$PYTHON_BIN" cfd_applications_cleanroom/scripts/audit_stationary_ellipse_l8.py \
  --complete-result-dirs \
  | sed "s#^${ROOT}/##" >"$PATH_LIST"

{
  echo "package_created_utc=${RUN_TAG}"
  echo "repo_root=${ROOT}"
  echo "included_paths:"
  cat "$PATH_LIST"
  find cfd_applications_cleanroom/results/background \
    -maxdepth 1 \
    -type d \
    -name 'stationary_ellipse_l8_wsl_*' \
    | sort
} >"$MANIFEST"

tar -czf "$PACKAGE" \
  --files-from "$PATH_LIST" \
  cfd_applications_cleanroom/results/background/stationary_ellipse_l8_wsl_* \
  "$MANIFEST" \
  "$PATH_LIST"

"$PYTHON_BIN" cfd_applications_cleanroom/scripts/validate_stationary_ellipse_l8_package.py "$PACKAGE"

echo "package=$PACKAGE"
echo "manifest=$MANIFEST"
echo "path_list=$PATH_LIST"
