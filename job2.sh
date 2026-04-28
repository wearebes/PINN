#!/usr/bin/env bash
#SBATCH -J pinn_train_2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.results
#SBATCH -e slurm-%j.err

set -euo pipefail

# Load modules
module load cuda12.6
module load gcc12

# Activate conda environment
source activate pinn

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Project root: $ROOT_DIR"
echo "Python: $(which python)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# 离线 SwanLab 训练
python -m model.train \
  --dataset-output dataset/train_0.5_setting1.h5 \
  --output-model out/pinn-4_curvature001.pt \
  --batch-size 204800 \
  --lambda-curvature 0.001 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name version1_Curvature_weighted0.001 \
  --swanlab-tags Curvature,stage2 \
  --swanlab-mode offline &

python -m model.train \
  --dataset-output dataset/train_0.5_setting1.h5 \
  --output-model out/pinn-4_curvature00001.pt \
  --batch-size 204800 \
  --lambda-curvature 0.00001 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name version1_Curvature_weighted0.00001 \
  --swanlab-tags Curvature,stage2 \
  --swanlab-mode offline &

echo
echo "Training finished."
