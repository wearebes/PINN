#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="cfd_applications_cleanroom/results/background/stationary_ellipse_completion_${run_tag}"
mkdir -p "$log_dir"
printf '%s\n' "$log_dir" > cfd_applications_cleanroom/results/background/stationary_ellipse_completion_latest.txt

run_step() {
  local name="$1"
  shift
  local log_file="${log_dir}/${name}.log"
  {
    printf 'step=%s\n' "$name"
    printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } | tee "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
  printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_file"
}

run_step d4_l6_l7 \
  python -m cfd_applications_cleanroom.cfd_apps.cli reproduce \
  --benchmark stationary_ellipse \
  --tier curvature-process \
  --cases E1 E2 \
  --levels 6 7 \
  --methods NN27_D4

run_step l8_all_methods \
  python -m cfd_applications_cleanroom.cfd_apps.cli reproduce \
  --benchmark stationary_ellipse \
  --tier curvature-process \
  --cases E1 E2 \
  --levels 8 \
  --methods NN_DISABLE NN_PROBE_ONLY NN27_RAW NN27_D4

if [[ -f cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py ]]; then
  run_step merge_summary \
    python cfd_applications_cleanroom/scripts/merge_stationary_ellipse_summary.py
fi

printf 'complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${log_dir}/DONE"
