#!/usr/bin/env bash
#SBATCH --job-name=PINN_3x
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -uo pipefail

# ---------- 0. 锁定值 ----------
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
CONDA_ENV="${CONDA_ENV:-pinn}"

# ---------- 1. repo 根 + 路径骨架 ----------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -d dataset && -d model ]] || { echo "[FATAL] 不在 PINN repo 根: $PWD"; exit 2; }

JOB_ID="${SLURM_JOB_ID:-local$$}"
RUN_ROOT="$PWD/logs/$JOB_ID"
SWAN_ROOT="$PWD/swanlog/$JOB_ID"
OUT_ROOT="$PWD/out/$JOB_ID"
TMPDIR="$PWD/tmp/$JOB_ID"
mkdir -p "$RUN_ROOT" "$SWAN_ROOT" "$OUT_ROOT" "$TMPDIR" out logs
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT

# ---------- 2. 环境 ----------
module purge 2>/dev/null || true
module load "$CUDA_MODULE"
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
  || { echo "[FATAL] conda 未就绪"; exit 4; }
conda activate "$CONDA_ENV"

# ---------- 3. CUDA preflight ----------
python - <<'PY' || { echo "[FATAL] CUDA preflight 失败"; exit 3; }
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

# ---------- 5. 三进程并行训练（错峰启动） ----------
RES=(256 266 276)
PIDS=()
NAMES=()

for r in "${RES[@]}"; do
  RUN_DIR="$RUN_ROOT/$r"
  SWAN_DIR="$SWAN_ROOT/$r"
  mkdir -p "$RUN_DIR" "$SWAN_DIR"

  (
    export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/.torchinductor"
    export TRITON_CACHE_DIR="$RUN_DIR/.triton"
    export XDG_CACHE_HOME="$RUN_DIR/.cache"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

    /usr/bin/time -p -o "$RUN_DIR/time.txt" \
      python -m model.train \
        --dataset-output       "dataset/${r}.h5" \
        --output-model         "$OUT_ROOT/baseline_${r}.pt" \
        --normalization-csv    "$OUT_ROOT/baseline_${r}.csv" \
        --phase-log-path       "$RUN_DIR/phase.jsonl" \
        --disable-compile \
        --use-swanlab \
        --swanlab-mode         offline \
        --swanlab-logdir       "$SWAN_DIR" \
        --swanlab-project      PINN \
        --swanlab-experiment-name "baseline_${r}_${JOB_ID}" \
        --swanlab-tags         baseline \
      > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
  ) &
  PIDS+=("$!")
  NAMES+=("$r")
  echo "[launch] res=${r} pid=$! logs=$RUN_DIR swanlog=$SWAN_DIR out=$OUT_ROOT"
  sleep 2
done

# ---------- 6. 汇总收尾 ----------
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done]   res=${NAMES[$i]} OK"
  else
    rc=$?
    echo "[FAIL]   res=${NAMES[$i]} exit=$rc  see $RUN_ROOT/${NAMES[$i]}/stderr.log"
    FAILED=$((FAILED+1))
  fi
done

# ---------- 7. SwanLab 打包 + 三通道 sync 提示 ----------
TARBALL="$PWD/swanlog/${JOB_ID}.tar.gz"
tar -czf "$TARBALL" -C "$PWD/swanlog" "$JOB_ID" 2>/dev/null \
  && echo "[pack]   $TARBALL"

cat <<EOF

============== SwanLab 同步提示（选一条可行的） ==============
[A] 登录节点能联网:
    swanlab sync $SWAN_ROOT/256 $SWAN_ROOT/266 $SWAN_ROOT/276

[B] 登录节点不能联网，本地能联网:
    # 本地终端:
    scp <user>@<cluster>:$TARBALL  ./
    tar -xzf ${JOB_ID}.tar.gz
    swanlab sync ${JOB_ID}/256 ${JOB_ID}/266 ${JOB_ID}/276

[C] 集群有传输节点 / 外网分区:
    在该节点 cd $PWD && swanlab sync $SWAN_ROOT/{256,266,276}
==============================================================
EOF

echo "[summary] failures=$FAILED / 3   swanlog: $SWAN_ROOT   models: $OUT_ROOT"
exit $FAILED
