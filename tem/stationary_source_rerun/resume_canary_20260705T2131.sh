#!/usr/bin/env bash
set -euo pipefail

cd /Users/jcy/research/PINN
export PYTHONPATH=/Users/jcy/research/PINN${PYTHONPATH:+:$PYTHONPATH}
export MPLCONFIGDIR=/Users/jcy/research/PINN/cfd_applications_cleanroom/results/tmp/matplotlib

/opt/anaconda3/bin/python3 - <<'PY'
from cfd_applications_cleanroom.cfd_apps.stationary import run_canary
from tem.stationary_source_rerun.run_full_stationary_source_pipeline import build_plot_data

process_report = {"run_id": "stationary_curvature_process_20260705T043136Z"}
native_report = run_canary(methods=["CLSVOF_LS_NATIVE"], levels=[6, 7, 8], repeat=3)
build_plot_data(process_report, native_report)
PY
