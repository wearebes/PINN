#!/usr/bin/env bash
#SBATCH --job-name=geometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
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
p = torch.cuda.get_device_properties(0)
print(f"[preflight] torch={torch.__version__} cuda={torch.version.cuda} "
      f"dev={p.name} mem={p.total_memory/1e9:.1f}GB cc={p.major}.{p.minor}")
PY

# ---------- 4. 公共环境 ----------
export SWANLAB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- 5. 串行训练 ----------
RES=(256 266 276)
FAILED=0

for r in "${RES[@]}"; do
  SWAN_DIR="$SWAN_ROOT/$r"
  swan_args=()
  swan_enabled_for_r=0
  if [[ $ENABLE_SWANLAB -eq 1 ]]; then
    if mkdir -p "$SWAN_DIR" 2>>"$ERROR_LOG" && touch "$SWAN_DIR/.write_test" 2>>"$ERROR_LOG"; then
      rm -f "$SWAN_DIR/.write_test"
      swan_enabled_for_r=1
      swan_args=(
        --use-swanlab
        --swanlab-mode offline
        --swanlab-logdir "$SWAN_DIR"
        --swanlab-project PINN
        --swanlab-experiment-name "baseline2_${r}_${JOB_ID}"
        --swanlab-tags baseline
      )
    else
      warn "res=${r} 无法写入 SwanLab 离线目录，跳过 SwanLab。SWAN_DIR=$SWAN_DIR"
    fi
  fi

  log_train "[launch] res=${r} swanlog_enabled=${swan_enabled_for_r} swanlog=$SWAN_DIR out=$OUT_ROOT start=$(date -Is)"
  if (
    export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/${r}/.torchinductor"
    export TRITON_CACHE_DIR="$TMPDIR/${r}/.triton"
    export XDG_CACHE_HOME="$TMPDIR/${r}/.cache"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

    python -m model.train \
        --dataset-output       "dataset/${r}.h5" \
        --output-model         "$OUT_ROOT/baseline2_${r}.pt" \
        --normalization-csv    "$OUT_ROOT/baseline2_${r}.csv" \
        --disable-compile \
        "${swan_args[@]}" \
      >> "$TRAIN_LOG" 2>> "$ERROR_LOG"
  ); then
    log_train "[done]   res=${r} OK end=$(date -Is)"
  else
    rc=$?
    log_train "[FAIL]   res=${r} exit=$rc end=$(date -Is)"
    printf '%s\n' "[FAIL] res=${r} exit=$rc" >>"$ERROR_LOG"
    FAILED=$((FAILED+1))
  fi
done

# ---------- 6. 汇总收尾 ----------
if [[ $ENABLE_SWANLAB -eq 1 ]] && find "$SWAN_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  echo "[summary] swanlab sync: swanlab sync $SWAN_ROOT/256 $SWAN_ROOT/266 $SWAN_ROOT/276"
else
  echo "[summary] swanlab offline logs not available"
fi

echo "[summary] failures=$FAILED / 3   swanlog: $SWAN_ROOT   models: $OUT_ROOT"
exit $FAILED
