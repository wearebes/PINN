# Run Guide
## 1. V1 Target

256 baseline example:

```bash
python -m train_generate.generate \
  --output dataset/1024_h.h5 \
  --output-dir dataset \
  --scale-h \
  --dataset-name 1024_h.h5 \
  --resolutions 1024
```

## 3. Train V1 Model

256 baseline example with full SwanLab arguments:

```bash
python -m model.train \
  --model-type mlp \
  --dataset-output dataset/256_h.h5 \
  --output-model out/baseline_256v2.pt \
  --normalization-csv out/baseline_256v2.csv \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project PINN \
  --swanlab-experiment-name baseline_256v2 \
  --swanlab-tags baseline,v2,rho256 
  
```
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
  --output test_data/rho276.h5
```

## 6. Generate Arbitrary-Resolution V1 Flower Data

`--scenario-config` is still supported, but only for V1 phi9 flower generation.

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

