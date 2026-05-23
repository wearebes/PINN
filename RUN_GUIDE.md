# Run Guide

This repo is now V1-only.

Canonical pipeline:

`phi9 -> 9D standardized input -> h*kappa`



## 1. V1 Target

256 baseline example:

```bash
python -m train_generate.generate \
  --output dataset/256.h5 \
  --output-dir dataset \
  --dataset-name 256.h5 \
  --resolutions 256 
```

## 3. Train V1 Model

256 baseline example with full SwanLab arguments:

```bash
/usr/bin/time -p python -m model.train \
  --model-type mlp \
  --dataset-output dataset/256.h5 \
  --output-model out/baseline_256.pt \
  --normalization-csv out/baseline_256.csv \
  --batch-size 256 \
  --use-swanlab \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline_256 \
  --swanlab-tags baseline,v1,rho256 \
  --swanlab-logdir swanlog \
  --swanlab-mode offline \
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
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project PINN \
  --swanlab-experiment-name split_test_256 \
  --swanlab-tags split,v1,rho256,test \
  
```

## 5. Generate V1 Flower Test Data

```bash
python -m testdata_generate.generate \
  --rho-model 276 \
  --test-iters 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --output test_data/rho276.h5
```

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

## 7. Evaluate V1 Flower Test Data

```bash
python -m evaluate.flower \
  --data test_data/rho276.h5 \
  --model-path out/baseline_276.pt \
  --normalization-csv out/baseline_276.csv \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project PINN \
  --swanlab-experiment-name flower_rho276 \
  --swanlab-tags baseline 
```

