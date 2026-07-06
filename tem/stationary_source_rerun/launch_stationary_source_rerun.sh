#!/usr/bin/env bash
set -euo pipefail

cd /Users/jcy/research/PINN
mkdir -p tem/stationary_source_rerun
exec >> tem/stationary_source_rerun/rerun_678_launchctl.log 2>&1
export PYTHONPATH=/Users/jcy/research/PINN${PYTHONPATH:+:$PYTHONPATH}

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 tem/stationary_source_rerun/run_full_stationary_source_pipeline.py
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
