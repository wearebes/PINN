# Run Guide

This repo is now V1-only.

Canonical pipeline:

`phi9 -> 9D standardized input -> h*kappa`

Supported artifacts:

- training dataset: HDF5 with `phi9`, `features`, `hkappa_target`, where `features` is always 9D `phi9`
- model checkpoint: `.pt`
- normalization sidecar: `*_phi9.csv` or compatible `*.csv`
- flower test dataset: HDF5 with V1-only `phi9` / `features`

Removed workflows:

- `--feature-version 2`
- `phi9 + nx9 + ny9`
- `--feature-stats ...npz`
- `--pca-enabled`
- `--pca-dim`

## 1. V1 Target

The only supported runtime path is:

1. Generate V1 training data.
2. Train a V1 model with CSV normalization.
3. Evaluate `train/val/test` with the same V1 checkpoint + CSV.
4. Generate V1 flower test data, either from built-in `--rho-model` scenarios or V1-only `--scenario-config`.
5. Evaluate flower test data with the same V1 checkpoint + CSV, optionally logging to SwanLab.

## 2. Generate V1 Training Data

256 baseline example:

```bash
python -m train_generate.generate \
  --output dataset/256.h5 \
  --output-dir dataset \
  --dataset-name 256.h5 \
  --resolutions 256 \
  --geometry-seed 42 \
  --variations 12 \
  --initial-field-types sdf,nonsdf \
  --augment-sign-flip \
  --shape-types circle,ellipse \
  --train-fraction 0.7 \
  --val-fraction 0.15 \
  --ellipse-num-a 48 \
  --ellipse-variations-per-a 124 \
  --ellipse-axis-ratio-min 0.5 \
  --ellipse-axis-ratio-max 0.9 \
  --ellipse-rotation-min 0.0 \
  --ellipse-rotation-max 3.141592653589793 \
  --ellipse-a-min-factor 8.0 \
  --ellipse-sdf-newton-max-iter 30 \
  --ellipse-sdf-newton-tol 1e-12 \
  --ellipse-hp-dps 80 \
  --ellipse-hp-newton-max-iter 100 \
  --num-workers 24 \
  --generation-batch-size 10240
```

To build the 266 or 276 dataset, replace:

- `--resolutions 256`
- `--output dataset/256.h5`
- `--dataset-name 256.h5`

with the matching resolution.

## 3. Train V1 Model

256 baseline example with full SwanLab arguments:

```bash
/usr/bin/time -p python -m model.train \
  --model-type mlp \
  --dataset-output dataset/256.h5 \
  --output-model out/baseline_256.pt \
  --normalization-csv out/baseline_256.csv \
  --device "" \
  --hidden-units 128 \
  --kernel-size 3 \
  --padding 1 \
  --lr 1e-4 \
  --max-epochs 1000 \
  --patience 30 \
  --batch-size 256 \
  --seed 42 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline_256 \
  --swanlab-description "V1 phi9 baseline training for rho256" \
  --swanlab-tags baseline,v1,rho256 \
  --swanlab-group baseline-train \
  --swanlab-workspace your-workspace \
  --swanlab-logdir swanlog \
  --swanlab-mode offline \
  --phase-log-path out/baseline_256_phase_log.jsonl \
  --compile-mode reduce-overhead
```

For 266 or 276, replace the dataset, checkpoint, CSV, experiment name, description, and tags with the matching resolution.

## 4. Evaluate Train/Val/Test Split

256 test split example with full SwanLab arguments:

```bash
python -m evaluate.split \
  --data dataset/256.h5 \
  --split test \
  --model-path out/baseline_256.pt \
  --normalization-csv out/baseline_256.csv \
  --device "" \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name split_test_256 \
  --swanlab-description "V1 split evaluation for rho256 baseline checkpoint" \
  --swanlab-tags split,v1,rho256,test \
  --swanlab-group split-eval \
  --swanlab-workspace your-workspace \
  --swanlab-logdir swanlog \
  --swanlab-mode offline
```

## 5. Generate V1 Flower Test Data

Built-in legacy scenario example for `rho_model=256`:

```bash
python -m testdata_generate.generate \
  --rho-model 256 \
  --test-iters 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --output test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN_rho256.h5
```

The same command shape works for `266` and `276`; only replace `--rho-model` and the output filename.

## 6. Generate Arbitrary-Resolution V1 Flower Data

`--scenario-config` is still supported, but only for V1 phi9 flower generation.

Example config:

```json
{
  "dataset_name": "flower_custom_v1.h5",
  "output_dir": "test_data",
  "test_iters": [1, 5, 10, 20],
  "scenarios": [
    {
      "exp_id": "smooth_300",
      "experiment_type": "smooth",
      "rho_model": 300,
      "L": 0.2072,
      "N": 121,
      "a": 0.05,
      "b": 0.15,
      "p": 3
    }
  ]
}
```

Generation command:

```bash
python -m testdata_generate.generate \
  --scenario-config test_data/flower_custom_v1.json \
  --test-iters 1,5,10,20 \
  --output test_data/flower_custom_v1.h5
```

Notes:

- `h` is derived from `L` and `N`
- only `dataset_name`, `output_dir`, `test_iters`, and `scenarios` are accepted at the top level
- generated `features` are always 9D `phi9`

## 7. Evaluate V1 Flower Test Data

256 built-in flower evaluation with full SwanLab arguments:

```bash
python -m evaluate.flower \
  --data test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN_rho256.h5 \
  --model-path out/baseline_256.pt \
  --normalization-csv out/baseline_256.csv \
  --device "" \
  --representative-case-id smooth_256 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name flower_rho256 \
  --swanlab-description "V1 flower evaluation for rho256 baseline checkpoint" \
  --swanlab-tags flower,v1,rho256,baseline \
  --swanlab-group flower-eval \
  --swanlab-workspace your-workspace \
  --swanlab-logdir swanlog \
  --swanlab-mode offline
```

SwanLab output now includes:

- overall summary metrics under `flower_eval/*`
- `flower_eval/representative_curve`
- `flower_eval/case_summary_table`
- step-series scalar logs under:
  - `flower_eval/by_iter/...`
  - `flower_eval/by_case/<case_label>/...`

## 8. Artifact Pairing Rule

Always keep these three artifacts aligned:

1. V1 dataset
2. V1 checkpoint
3. V1 normalization CSV

For example, a 256 run should stay on:

- `dataset/256.h5`
- `out/baseline_256.pt`
- `out/baseline_256.csv`

Do not mix:

- old V2 `.npz` feature stats
- 27D historical datasets
- checkpoints trained from non-V1 feature layouts

If an old command still contains `--feature-stats`, `--pca-enabled`, `--pca-dim`, or `--feature-version 2`, it is stale and should be replaced with the V1 commands above.
