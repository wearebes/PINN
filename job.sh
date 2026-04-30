#!/usr/bin/env bash
#SBATCH -J stencil_hkappa
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

module load cuda12.6
module load gcc12
source activate pinn

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python -m traingenerate.generate

python -m model.train \
  --dataset-output dataset/train_stencil_setting1.h5 \
  --batch-size 8192 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name stencil-hkappa \
  --swanlab-tags stencil,hkappa \
  --swanlab-mode offline
