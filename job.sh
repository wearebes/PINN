#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Runtime
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

# Training dataset
TRAIN_DATASET="${TRAIN_DATASET:-dataset/train_0.5_setting1.h5}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-dataset}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-$(basename "$TRAIN_DATASET")}"

# Test dataset
TEST_DATASET="${TEST_DATASET:-test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN.h5}"

# Model outputs
CHECKPOINT_PATH="${CHECKPOINT_PATH:-best_reinit_pinn.pt}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-out/pinn-1.pt}"

# Train / eval knobs
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-81920}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

# SwanLab
USE_SWANLAB="${USE_SWANLAB:-0}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-PINN}"
TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME:-pinn-train}"
TRAIN_SWANLAB_TAGS="${TRAIN_SWANLAB_TAGS:-train}"
EVAL_EXPERIMENT_NAME="${EVAL_EXPERIMENT_NAME:-curvature-eval}"
EVAL_SWANLAB_TAGS="${EVAL_SWANLAB_TAGS:-curvature,eval,hkappa}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-swanlog}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

run_cmd() {
  echo
  echo "==> $*"
  "$@"
}

TRAIN_SWANLAB_ARGS=()
EVAL_SWANLAB_ARGS=()
if [[ "$USE_SWANLAB" == "1" ]]; then
  TRAIN_SWANLAB_ARGS=(
    --use-swanlab
    --swanlab-project "$SWANLAB_PROJECT"
    --swanlab-experiment-name "$TRAIN_EXPERIMENT_NAME"
    --swanlab-tags "$TRAIN_SWANLAB_TAGS"
    --swanlab-logdir "$SWANLAB_LOGDIR"
    --swanlab-mode "$SWANLAB_MODE"
  )
  EVAL_SWANLAB_ARGS=(
    --use-swanlab
    --swanlab-project "$SWANLAB_PROJECT"
    --swanlab-experiment-name "$EVAL_EXPERIMENT_NAME"
    --swanlab-tags "$EVAL_SWANLAB_TAGS"
    --swanlab-logdir "$SWANLAB_LOGDIR"
    --swanlab-mode "$SWANLAB_MODE"
  )
fi

echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Device: $DEVICE"
echo "Train dataset: $TRAIN_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output model: $OUTPUT_MODEL_PATH"

mkdir -p "$(dirname "$TRAIN_DATASET")" "$(dirname "$TEST_DATASET")" "$(dirname "$OUTPUT_MODEL_PATH")"

run_cmd \
  "$PYTHON_BIN" -m traingenerate.generate \
  --output-dir "$TRAIN_OUTPUT_DIR" \
  --dataset-name "$TRAIN_DATASET_NAME"

run_cmd \
  "$PYTHON_BIN" -m testdata_generate.generate \
  --output "$TEST_DATASET"

run_cmd \
  "$PYTHON_BIN" -m model.train \
  --dataset-output "$TRAIN_DATASET" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-model "$OUTPUT_MODEL_PATH" \
  --device "$DEVICE" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  "${TRAIN_SWANLAB_ARGS[@]}"

EVAL_CMD=(
  "$PYTHON_BIN" -m evaluate
  --test-data "$TEST_DATASET"
  --model "$OUTPUT_MODEL_PATH"
  --device "$DEVICE"
  --batch-size "$EVAL_BATCH_SIZE"
  "${EVAL_SWANLAB_ARGS[@]}"
)

if [[ "$MAX_EVAL_SAMPLES" != "0" ]]; then
  EVAL_CMD+=(--max-samples "$MAX_EVAL_SAMPLES")
fi

run_cmd "${EVAL_CMD[@]}"

echo
echo "Pipeline finished."
echo "Best checkpoint: $CHECKPOINT_PATH"
echo "Output model: $OUTPUT_MODEL_PATH"
