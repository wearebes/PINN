#!/usr/bin/env bash
#SBATCH --job-name=part2_dataset
#SBATCH --partition=gpu

#SBATCH --gres=gpu:4090:2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail

# Volume x field-mode experiment, modeled on job.sh's structure. wd is fixed
# at 0 only (the wd sweep in job.sh was found degenerate -- 4 bit-identical
# checkpoints, see memory part2-training-data-contract.md -- so there is no
# point repeating it here). Checkpoints land in a STABLE dataset-tag-named
# directory (not out/$JOB_ID/), since with 5 datasets across one job, an
# opaque numeric job id tells you nothing about which checkpoint is which.
#
# GPU0 (one card): the two large sdf+nonsdf datasets, run SEQUENTIALLY.
# GPU1 (one card): the other three datasets, run SEQUENTIALLY.
#   main           (1x, sdf-only)        was already trained as run 7367 -- skipped here.

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
OUT_ROOT="$PWD/out/dcts_volume_experiment"   # stable, NOT tied to JOB_ID -- self-describing filenames live here
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
log_startup "[startup] out_root=$OUT_ROOT (stable, shared across all 5 datasets)"
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

# ---------- 6. dataset tags + conversion (DCTS processed/ -> single v7 training HDF5) ----------
GPU0_TAGS=("main_2x_sdfnonsdf" "main_3x_sdfnonsdf")
GPU1_TAGS=("main_sdfnonsdf" "main_2x" "main_3x")
ALL_TAGS=("${GPU0_TAGS[@]}" "${GPU1_TAGS[@]}")

for tag in "${ALL_TAGS[@]}"; do
  processed_dir="dataset/part2_dcts/${tag}/processed"
  train_h5="dataset/part2_dcts/${tag}/training/dcts_${tag}_v7.h5"
  if [[ ! -f "$train_h5" ]]; then
    [[ -f "$processed_dir/train.h5" ]] \
      || fatal "DCTS split 文件不存在: $processed_dir/train.h5（先跑 python -m train_generate.part2.dcts --main ... 生成 tag=$tag）"
    log_startup "[startup] 转换 DCTS -> v7 训练 HDF5: $train_h5"
    python -m train_generate.part2.dcts.to_training_hdf5 --tag "$tag" \
      >>"$STARTUP_LOG" 2>>"$ERROR_LOG" || fatal "to_training_hdf5 转换失败 (tag=$tag)"
  fi
done

# ---------- 7. per-GPU SEQUENTIAL training, wd=0 only ----------
# Self-describing checkpoint names: dcts_<tag>.pt / dcts_<tag>.csv -- no wd
# suffix (always 0 here) and no job-id-only path, so any checkpoint's name
# alone tells you exactly which dataset variant produced it.
FAILED=0
PIDS=()

launch_group() {
  local gpu="$1"; shift
  local tags=("$@")
  (
    set -uo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    for tag in "${tags[@]}"; do
      train_h5="dataset/part2_dcts/${tag}/training/dcts_${tag}_v7.h5"
      model_name="dcts_${tag}"

      cache_dir="$TMPDIR/${tag}"
      export TORCHINDUCTOR_CACHE_DIR="$cache_dir/.torchinductor"
      export TRITON_CACHE_DIR="$cache_dir/.triton"
      export XDG_CACHE_HOME="$cache_dir/.cache"
      mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

      swan_args=()
      swan_enabled_for_tag=0
      if [[ $ENABLE_SWANLAB -eq 1 ]]; then
        swan_dir="$SWAN_ROOT/$tag"
        if mkdir -p "$swan_dir" 2>>"$ERROR_LOG" && touch "$swan_dir/.write_test" 2>>"$ERROR_LOG"; then
          rm -f "$swan_dir/.write_test"
          swan_enabled_for_tag=1
          swan_args=(
            --use-swanlab
            --swanlab-mode offline
            --swanlab-logdir "$swan_dir"
            --swanlab-project PINN
            --swanlab-experiment-name "$model_name"
            --swanlab-tags dcts_volume_experiment
          )
        else
          warn "tag=${tag} 无法写入 SwanLab 离线目录，跳过 SwanLab。SWAN_DIR=$swan_dir"
        fi
      fi

      log_train "[launch] tag=${tag} gpu=${gpu} model=${model_name} swanlab=${swan_enabled_for_tag} start=$(date -Is)"
      if python -m model.train \
          --dataset-output    "$train_h5" \
          --output-model      "$OUT_ROOT/${model_name}.pt" \
          --normalization-csv "$OUT_ROOT/${model_name}.csv" \
          --optimizer-type adamw \
          --l2-reg 0 \
          "${swan_args[@]}" \
        >> "$TRAIN_LOG" 2>> "$ERROR_LOG"; then
        log_train "[done]   tag=${tag} OK end=$(date -Is)"
      else
        rc=$?
        log_train "[FAIL]   tag=${tag} exit=$rc end=$(date -Is)"
        printf '%s\n' "[FAIL] tag=${tag} exit=$rc" >>"$ERROR_LOG"
        exit "$rc"
      fi
    done
  ) &
  PIDS+=($!)
}

launch_group "$GPU0" "${GPU0_TAGS[@]}"
launch_group "$GPU1" "${GPU1_TAGS[@]}"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED+1))
  fi
done

# ---------- 8. 汇总收尾 ----------
TOTAL=${#ALL_TAGS[@]}
done_count=$(grep -c '^\[done\]' "$TRAIN_LOG" 2>/dev/null || true)
if [[ $ENABLE_SWANLAB -eq 1 ]] && find "$SWAN_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  SWAN_SYNC_ARGS=()
  for tag in "${ALL_TAGS[@]}"; do
    SWAN_SYNC_ARGS+=("$SWAN_ROOT/$tag")
  done
  echo "[summary] swanlab sync: swanlab sync ${SWAN_SYNC_ARGS[*]}"
else
  echo "[summary] swanlab offline logs not available"
fi

echo "[summary] task_failures=$FAILED/$TOTAL task_done=${done_count}/$TOTAL  swanlog: $SWAN_ROOT   models: $OUT_ROOT"
exit $FAILED
