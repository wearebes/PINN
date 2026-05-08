#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_DATASET=${TRAIN_DATASET:-dataset/256.h5}
TRAIN_MODEL=${TRAIN_MODEL:-out/base256.pt}
TRAIN_NORM=${TRAIN_NORM:-out/base256_phi9.csv}
FLOWER_RHO_MODEL=${FLOWER_RHO_MODEL:-256}
FLOWER_DATA=${FLOWER_DATA:-test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN_rho${FLOWER_RHO_MODEL}.h5}
USE_SWANLAB=${USE_SWANLAB:-0}
SWANLAB_MODE=${SWANLAB_MODE:-cloud}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-PINN}

SWANLAB_ARGS=()
if [[ "$USE_SWANLAB" == "1" ]]; then
  SWANLAB_ARGS=(--use-swanlab --swanlab-mode "$SWANLAB_MODE" --swanlab-project "$SWANLAB_PROJECT")
fi

python -m model.train \
  --dataset-output "$TRAIN_DATASET" \
  --output-model "$TRAIN_MODEL" \
  "${SWANLAB_ARGS[@]}"

python -m evaluate.split \
  --data "$TRAIN_DATASET" \
  --split test \
  --model-path "$TRAIN_MODEL" \
  --normalization-csv "$TRAIN_NORM"

python -m testdata_generate.generate \
  --rho-model "$FLOWER_RHO_MODEL" \
  --output "$FLOWER_DATA"

python -m evaluate.flower \
  --data "$FLOWER_DATA" \
  --model-path "$TRAIN_MODEL" \
  --normalization-csv "$TRAIN_NORM"
