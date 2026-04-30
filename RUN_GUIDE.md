# Run Guide

## Pipeline

This repo now keeps only one training task:

- input: `3x3` local `phi` stencil
- output: `h*kappa`
- loss: `MSE(hk_pred, hk_true)`

The initial field can be either:

- `sdf`
- `nonsdf`

## 1. Generate Data

Edit defaults in `traingenerate/config.py`, then run:

```bash
python -m traingenerate.generate
```

Common overrides:

```bash
python -m traingenerate.generate \
  --initial-field-types sdf,nonsdf \
  --n-samples-per-circle 4096 \
  --output-dir dataset \
  --dataset-name circle256.h5
```

## 2. Train

Edit defaults in `model/config.py`, then run:

```bash
python -m model.train \
  --dataset-output dataset/circle256.h5 \
  --batch-size 8192 \
  --output-model out/best_stencil_hkappa.pt
```

Training automatically standardizes the 9 `phi` entries column-wise using only the training split, then applies the same statistics to train/val/test. The normalization parameters are saved as a CSV next to the model in `out/`.

With SwanLab:

```bash
python -m model.train \
  --dataset-output dataset/train_stencil_setting1.h5 \
  --batch-size 8192 \
  --output-model out/best_stencil_hkappa.pt \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name stencil-hkappa \
  --swanlab-tags stencil,hkappa
```
