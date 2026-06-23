#!/usr/bin/env bash
#SBATCH --job-name=geometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

# ---------- 0. 锁定值 ----------
CUDA_MODULE="${CUDA_MODULE:-cuda12.6}"
CONDA_ENV="${CONDA_ENV:-pinn}"

fatal() {
  local msg="$1"
  printf '%s\n' "[FATAL] $msg" | tee -a "$ERROR_LOG" >&2
  exit 1
}

warn() {
  local msg="$1"
  printf '%s\n' "[WARN] $msg" | tee -a "$ERROR_LOG" >&2
}

log_startup() {
  printf '%s\n' "$*" | tee -a "$STARTUP_LOG"
}

log_train() {
  printf '%s\n' "$*" | tee -a "$TRAIN_LOG"
}

# ---------- 1. repo 根 + 路径骨架 ----------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -d dataset && -d model ]] || { echo "[FATAL] 不在 PINN repo 根: $PWD"; exit 2; }

JOB_ID="${SLURM_JOB_ID:-local$$}"
RUN_ROOT="$PWD/logs/$JOB_ID"
SWAN_ROOT="$PWD/swanlog/$JOB_ID"
OUT_ROOT="$PWD/out/$JOB_ID"
TMPDIR="$PWD/tmp/$JOB_ID"
STARTUP_LOG="$RUN_ROOT/startup.log"
TRAIN_LOG="$RUN_ROOT/train.log"
ERROR_LOG="$RUN_ROOT/error.log"
mkdir -p "$PWD/logs" "$PWD/out" "$PWD/tmp"
mkdir -p "$RUN_ROOT" "$OUT_ROOT" "$TMPDIR"
: >"$STARTUP_LOG"
: >"$TRAIN_LOG"
: >"$ERROR_LOG"
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT

touch "$OUT_ROOT/.write_test" "$RUN_ROOT/.write_test" 2>>"$ERROR_LOG" \
  || fatal "没有权限写入 OUT_ROOT=$OUT_ROOT 或 RUN_ROOT=$RUN_ROOT"
rm -f "$OUT_ROOT/.write_test" "$RUN_ROOT/.write_test"

ENABLE_SWANLAB=1
mkdir -p "$SWAN_ROOT" 2>>"$ERROR_LOG" || ENABLE_SWANLAB=0
if [[ $ENABLE_SWANLAB -eq 1 ]]; then
  touch "$SWAN_ROOT/.write_test" 2>>"$ERROR_LOG" || ENABLE_SWANLAB=0
  rm -f "$SWAN_ROOT/.write_test"
fi
if [[ $ENABLE_SWANLAB -eq 0 ]]; then
  warn "SwanLab offline 目录不可写，训练会继续，但不会记录 SwanLab 离线日志。SWAN_ROOT=$SWAN_ROOT"
fi

log_startup "[startup] job_id=$JOB_ID"
log_startup "[startup] run_root=$RUN_ROOT"
log_startup "[startup] swan_root=$SWAN_ROOT"
log_startup "[startup] out_root=$OUT_ROOT"
log_startup "[startup] swanlab_enabled=$ENABLE_SWANLAB"

# ---------- 2. 环境 ----------
module purge 2>/dev/null || true
module load "$CUDA_MODULE" 2>>"$ERROR_LOG" || fatal "module load $CUDA_MODULE 失败"
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
  2>>"$ERROR_LOG" || fatal "conda 未就绪"
conda activate "$CONDA_ENV" 2>>"$ERROR_LOG" || fatal "conda activate $CONDA_ENV 失败"
log_startup "[startup] cuda_module=$CUDA_MODULE"
log_startup "[startup] conda_env=$CONDA_ENV"

# ---------- 3. CUDA preflight ----------
python - <<'PY' >>"$STARTUP_LOG" 2>>"$ERROR_LOG" || fatal "CUDA preflight 失败"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
n = torch.cuda.device_count()
assert n >= 2, f"need 2 gpus, got {n}"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"[preflight] torch={torch.__version__} cuda={torch.version.cuda} "
          f"dev[{i}]={p.name} mem={p.total_memory/1e9:.1f}GB cc={p.major}.{p.minor}")
PY

# ---------- 4. GPU mapping ----------
IFS=',' read -r -a CUDA_VISIBLE_DEVICES_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU0="${CUDA_VISIBLE_DEVICES_LIST[0]}"
GPU1="${CUDA_VISIBLE_DEVICES_LIST[1]:-}"
[[ -n "$GPU1" ]] || fatal "need 2 gpus; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log_startup "[startup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} gpu0=$GPU0 gpu1=$GPU1"

# ---------- 5. 公共环境 ----------
export SWANLAB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- 6. DCTS (part2) 数据集：转换成 model.train 能吃的单文件格式 ----------
DATASET_TAG="${DATASET_TAG:-main}"
DCTS_PROCESSED_DIR="dataset/part2_dcts/${DATASET_TAG}/processed"
TRAIN_H5="dataset/part2_dcts/${DATASET_TAG}/training/dcts_${DATASET_TAG}_v7.h5"

if [[ ! -f "$TRAIN_H5" ]]; then
  [[ -f "$DCTS_PROCESSED_DIR/train.h5" ]] \
    || fatal "DCTS split 文件不存在: $DCTS_PROCESSED_DIR/train.h5（先跑 python -m train_generate.part2.dcts --main 生成）"
  log_startup "[startup] 转换 DCTS -> v7 训练 HDF5: $TRAIN_H5"
  python -m train_generate.part2.dcts.to_training_hdf5 --tag "$DATASET_TAG" \
    >>"$STARTUP_LOG" 2>>"$ERROR_LOG" || fatal "to_training_hdf5 转换失败"
fi

# ---------- 7. Parallel training (two GPUs, weight_decay sweep) ----------
WD_GPU0=("0" "1e-4")
WD_GPU1=("1e-6" "1e-8")
ALL_WD=("${WD_GPU0[@]}" "${WD_GPU1[@]}")
FAILED=0
PIDS=()

launch_train() {
  local wd="$1"
  local gpu="$2"
  local tag="wd${wd}"
  local SWAN_DIR="$SWAN_ROOT/$tag"
  local -a swan_args=()
  local swan_enabled_for_tag=0

  if [[ $ENABLE_SWANLAB -eq 1 ]]; then
    if mkdir -p "$SWAN_DIR" 2>>"$ERROR_LOG" && touch "$SWAN_DIR/.write_test" 2>>"$ERROR_LOG"; then
      rm -f "$SWAN_DIR/.write_test"
      swan_enabled_for_tag=1
      swan_args=(
        --use-swanlab
        --swanlab-mode offline
        --swanlab-logdir "$SWAN_DIR"
        --swanlab-project PINN
        --swanlab-experiment-name "dcts_${DATASET_TAG}_${tag}"
        --swanlab-tags dcts_wd_sweep
      )
    else
      warn "wd=${wd} 无法写入 SwanLab 离线目录，跳过 SwanLab。SWAN_DIR=$SWAN_DIR"
    fi
  fi

  log_train "[launch] wd=${wd} gpu=${gpu} swanlog_enabled=${swan_enabled_for_tag} swanlog=$SWAN_DIR out=$OUT_ROOT start=$(date -Is)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/${tag}/.torchinductor"
    export TRITON_CACHE_DIR="$TMPDIR/${tag}/.triton"
    export XDG_CACHE_HOME="$TMPDIR/${tag}/.cache"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

    if python -m model.train \
        --dataset-output       "$TRAIN_H5" \
        --output-model         "$OUT_ROOT/dcts_${DATASET_TAG}_${tag}.pt" \
        --normalization-csv    "$OUT_ROOT/dcts_${DATASET_TAG}_${tag}.csv" \
        --optimizer-type adamw \
        --l2-reg "$wd" \
        "${swan_args[@]}" \
      >> "$TRAIN_LOG" 2>> "$ERROR_LOG"; then
      log_train "[done]   wd=${wd} OK end=$(date -Is)"
    else
      rc=$?
      log_train "[FAIL]   wd=${wd} exit=$rc end=$(date -Is)"
      printf '%s\n' "[FAIL] wd=${wd} exit=$rc" >>"$ERROR_LOG"
      exit "$rc"
    fi
  ) &
  PIDS+=($!)
}

for wd in "${WD_GPU0[@]}"; do
  launch_train "$wd" "$GPU0"
done

for wd in "${WD_GPU1[@]}"; do
  launch_train "$wd" "$GPU1"
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED+1))
  fi
done

# ---------- 8. 汇总收尾 ----------
TOTAL=${#ALL_WD[@]}
if [[ $ENABLE_SWANLAB -eq 1 ]] && find "$SWAN_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  SWAN_SYNC_ARGS=()
  for wd in "${ALL_WD[@]}"; do
    SWAN_SYNC_ARGS+=("$SWAN_ROOT/wd${wd}")
  done
  echo "[summary] swanlab sync: swanlab sync ${SWAN_SYNC_ARGS[*]}"
else
  echo "[summary] swanlab offline logs not available"
fi

echo "[summary] failures=$FAILED / $TOTAL   swanlog: $SWAN_ROOT   models: $OUT_ROOT"
exit $FAILED
