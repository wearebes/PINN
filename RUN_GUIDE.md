# Run Guide

## Pipeline Summary

This repo keeps one supervised task:

- input: `3x3` local `phi` stencil (`phi9`)
- output: `h*kappa`
- loss: `MSE(hk_pred, hk_true)`

Training data and flower test data are different pipelines:

- training data: generated from `circle + ellipse` blueprints
- flower data: held-out test only

## 1. Generate Training Data

Edit defaults in `train_generate/config.py`, then run:

```bash
python -m train_generate.generate
```

The generated HDF5 contains `train/val/test` splits with `phi9` and `hkappa_target`.

## 2. Train

```bash
python -m model.train \
  --dataset-output dataset/256.h5 \
  --output-model out/base256.pt
```

Training computes `phi9` normalization from the training split only and writes a CSV next to the checkpoint.
With the current code path, the normalization file name is `<model_stem>_phi9.csv`.

The checkpoint payload contains:

- `checkpoint_format_version`
- `model_type`
- `model_config`
- `state_dict`

If you want SwanLab logging during training:

```bash
/usr/bin/time -p python -m model.train \
  --dataset-output dataset/256.h5 \
  --output-model out/base256.pt \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project PINN \
  --swanlab-experiment-name base256
```

## 3. Evaluate Train/Val/Test Splits

```bash
python -m evaluate.split \
  --data dataset/256.h5 \
  --split test \
  --model-path out/base256.pt \
  --normalization-csv out/base256_phi9.csv
```

The split evaluator reports:

- `numeric_vs_analytic`
- `model_vs_analytic`
- `model_vs_numeric`

Optional SwanLab logging is available through the same `--use-swanlab` and `--swanlab-*` flags.

## 4. Flower Function Construction

Flower testing is a separate held-out pipeline under `testdata_generate/`.
It does not contribute training samples.

The analytical flower initial field is:

```text
phi0(x, y) = sqrt(x^2 + y^2) - a * cos(p * theta) - b
theta = atan2(y, x)
```

For each configured flower scenario:

1. Build the analytical `phi0`.
2. Reinitialize it step by step with the configured numerical method.
3. At each saved iteration, find the current interface nodes from sign changes of the current `phi`.
4. Extract a `3x3` stencil around each current interface node and encode it as `phi9`.
5. Project the current node coordinates back to the analytical flower curve.
6. Compute analytical `h*kappa` on that projected point and store it as `hkappa_target`.

The default flower test configuration is encoded in the dataset file name:

- `DynSign`: dynamic sign field during reinitialization
- `CFL0.5`
- `EPS2.5`
- `RK3`
- `WENO5`
- `CIN`: current-interface-node sampling
- `ST9`: `3x3` stencil encoded as `phi9`
- `APCN`: analytic projection of current nodes

## 5. Generate Flower Test Data

Generate one flower HDF5 per `rho_model`, and pass that file explicitly when you evaluate.
The `rho256` example is:

```bash
python -m testdata_generate.generate \
  --rho-model 256 \
  --output test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN_rho256.h5
```

This writes a flat HDF5 under `test_data/` with fields:

- `phi9`
- `xy`
- `phi0_center`
- `hkappa_target`
- `case_id`
- `iter`
- `rho_model`
- `h`

Each file keeps both `smooth` and `acute` flower cases for the selected resolution.

## 6. What The Model Is Actually Being Tested On

The model under test is still the trained `phi9 -> h*kappa` predictor from the supervised training pipeline.

Flower evaluation feeds that model with `phi9` sampled from reinitialized flower interfaces.
The truth target is always the analytical `hkappa_target` obtained by projection back to the analytical flower curve.

The evaluator also computes a numeric baseline from the same `phi9` using central differences.

So flower testing compares three objects:

- analytical truth: `hkappa_target`
- numeric baseline from `phi9`
- model prediction from `phi9`

## 7. Evaluate Flower Test Data

```bash
python -m evaluate.flower \
  --data test_data/test_FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN_rho256.h5 \
  --model-path out/base256.pt \
  --normalization-csv out/base256_phi9.csv
```

The flower evaluator prints:

- overall metrics
- grouped metrics by `iter`
- grouped metrics by `case_id`
- grouped metrics by `rho_model`

## 8. How To Read Flower Outputs

The main question is not "did one scalar pass a threshold?".
The main question is whether the model is closer to the analytical `h*kappa` than the numeric baseline, and where that is true or false.

Treat these as the primary comparisons:

- `numeric_vs_analytic`
- `model_vs_analytic`

Treat this as an auxiliary comparison:

- `model_vs_numeric`

Use the following metrics on the two primary comparisons:

- `MSE`
- `MAE`
- `MaxAE`

Interpretation:

- `MSE`: average squared error, sensitive to larger deviations
- `MAE`: average absolute error, easier to compare across cases
- `MaxAE`: worst-point absolute error, used to catch localized failures hidden by averages

The practical reading order is:

1. Compare `model_vs_analytic` against `numeric_vs_analytic` in the overall table.
2. Check `by_iter` to see whether the model stays better or worse as reinitialization evolves.
3. Check `by_case_id` to see whether the behavior differs between `smooth` and `acute` flowers.
4. Use `MaxAE` to detect whether a seemingly good average hides a few bad interface points.

For another resolution, regenerate the flower HDF5 with the matching `--rho-model`, then pass that file together with the corresponding model checkpoint and normalization CSV.
