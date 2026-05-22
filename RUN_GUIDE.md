# Run Guide

This is the recommended end-to-end flow for the 256-resolution baseline.

V1 uses the legacy `phi9` pipeline and writes a `*_phi9.csv` normalization sidecar. V2 uses the `phi9 + nx9 + ny9` feature layout, fits standardization plus optional PCA whitening, and writes a `*_feature_stats.npz` sidecar.

## 1. Generate V1 Training Data

```bash
python -m train_generate.generate \
  --resolutions 256 \
  --geometry-seed 42 \
  --variations 12 \
  --initial-field-types sdf,nonsdf \
  --augment-sign-flip \
  --shape-types circle,ellipse \
  --output dataset/256.h5
```

Outputs:

- `dataset/256.h5`

`feature_version=1` keeps the legacy `phi9 -> h*kappa` pipeline. Sign-flip augmentation is only active in V1.

## 2. Generate V2 Training Data

```bash
python -m train_generate.generate \
  --resolutions 256 \
  --geometry-seed 42 \
  --variations 12 \
  --initial-field-types sdf,nonsdf \
  --feature-version 2 \
  --gradient-epsilon 1e-8 \
  --shape-types circle,ellipse \
  --feature-stats stats/256.npz \
  --output dataset/256_v2.h5

python -m train_generate.generate \
  --resolutions 266 \
  --geometry-seed 42 \
  --variations 12 \
  --initial-field-types sdf,nonsdf \
  --feature-version 2 \
  --gradient-epsilon 1e-8 \
  --feature-stats stats/266.npz \
  --shape-types circle,ellipse \
  --output dataset/266_v2.h5
python -m train_generate.generate \
  --resolutions 276 \
  --geometry-seed 42 \
  --variations 12 \
  --initial-field-types sdf,nonsdf \
  --feature-version 2 \
  --feature-stats stats/276.npz \
  --gradient-epsilon 1e-8 \
  --shape-types circle,ellipse \
  --output dataset/276_v2.h5
```
Outputs:

- `dataset/256_v2.h5`

`feature_version=2` stores raw `features` with shape `(N, 27)` in block order `phi9 + nx9 + ny9` and also keeps `phi9` for numeric baselines.

## 3. Train V1
```bash
/usr/bin/time -p python -m model.train \
  --dataset-output dataset/266.h5 \
  --output-model out/baseline_266.pt \
  --normalization-csv out/baseline_266.csv \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline_266 \
  --swanlab-tags baseline

/usr/bin/time -p python -m model.train \
  --dataset-output dataset/276.h5 \
  --output-model out/baseline_276.pt \
  --normalization-csv out/baseline_276.csv \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline_276 \
  --swanlab-tags baseline

## 4. Train V2
```bash
/usr/bin/time -p python -m model.train \
  --dataset-output dataset/256_v2.h5 \
  --output-model out/baseline_256_v2.pt \
  --feature-stats out/baseline_256_v2_feature_stats.npz \
  --pca-enabled \
  --pca-dim 18 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline-256-v2
```

V2 fits standardization + PCA whitening on the train split only, saves the transform into the checkpoint, and also writes the sidecar `.npz` file.

## 5. Evaluate Train/Val/Test Split


```bash
python -m evaluate.split \
  --data dataset/256_v2.h5 \
  --split test \
  --model-path out/baseline_256_v2.pt \
  --feature-stats out/baseline_256_v2_feature_stats.npz \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name split-test-256-v2
```

## 6. Generate Flower Test Data

V1 example:

```bash
python -m testdata_generate.generate \
  --rho-model 128 \
  --feature-version 1 \
  --output test_data/rho128_v1.h5
```



V2 example:

```bash
python -m testdata_generate.generate \
  --rho-model 256 \
  --feature-version 2 \
  --output test_data/test_flower_rho256_v2.h5
```

Outputs:

- `test_data/test_flower_rho256_v2.h5`

This generates the flower test set for the 256-resolution model. If you want a different resolution, change `--rho-model` and use the matching model/data pair everywhere else.

## 7. Evaluate Flower Test Data

```bash
python -m evaluate.flower \
  --data test_data/test_flower_rho256_v2.h5 \
  --model-path out/baseline_256_v2.pt \
  --feature-stats out/baseline_256_v2_feature_stats.npz \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name flower-rho256-v2
```

V1 example:

```bash
python -m evaluate.flower \
  --data test_data/test_flower_rho256_v1.h5 \
  --model-path out/baseline_256.pt \
  --normalization-csv out/baseline_256_phi9.csv \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name flower-rho256 \
  --swanlab-tag baseline
```

If you are evaluating a legacy V1 checkpoint, replace the model and data paths accordingly and use `--normalization-csv out/baseline_256_phi9.csv` instead of `--feature-stats`.

For SwanLab, the flower evaluator currently logs the overall metrics only, so the plot view will show the global numeric/model comparison series. The per-iter, per-case, and per-rho breakdowns are printed to the console, not logged as separate SwanLab series.