#!/usr/bin/env bash
set -euo pipefail

cd /Users/jcy/research/PINN
mkdir -p tem/stationary_source_rerun

python3 -m cfd_applications_cleanroom.cfd_apps.cli reproduce \
  --benchmark stationary \
  --tier curvature-process \
  --methods NN27_RAW \
  --levels 6 7 8
