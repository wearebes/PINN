#!/usr/bin/env bash
#SBATCH -J pinn_train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 24:00:00
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
  --batch-size 81920 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name version1_Curvature_weighted0.001 \
  --swanlab-tags Curvature,stage2 \
  --swanlab-mode offline

echo
echo "Training finished."
