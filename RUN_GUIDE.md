# Run Guide

```bash
# 1. Generate training data
python -m train_generate --output dataset/256_h.h5 --resolutions 1024 --scale-h
# V2 (27D features)
python -m train_generate --output dataset/256_h_v2.h5 --resolutions 1024 --scale-h --augment-gradient

python -m train_generate --output dataset/1024.h5 --resolutions 1024 
python -m train_generate --output dataset/1024_h.h5 --resolutions 1024 --scale-h
python -m train_generate --output dataset/1024_hgradient.h5 --resolutions 1024 --scale-h --augment-gradient

# 2. Train
python -m model.train \
  --dataset-output dataset/256_h.h5 \
  --output-model out/model_256.pt \
  --normalization-csv out/model_256.csv \
  --use-swanlab --swanlab-mode cloud --swanlab-project PINN \
  --swanlab-experiment-name model_256 --swanlab-tags baseline,rho256

# 3. Evaluate split
python -m evaluate.split \
  --data dataset/256_h.h5 --split test \
  --model-path out/model_256.pt --normalization-csv out/model_256.csv

# 4. Generate flower test data
python -m testdata_generate --rho-model 128 \
  --output test_data/rho128.h5

# Generate rho276 test data
python -m testdata_generate --rho-model 276 --scale-h --output test_data/rho276_h.h5

# 5. Evaluate flower
python -m evaluate.flower \
  --data test_data/rho256_h.h5 \
  --model-path out/2430/baseline_256_h.pt --normalization-csv out/2430/baseline_256_h.csv \
  --use-swanlab --swanlab-mode cloud --swanlab-project geometry \
  --swanlab-experiment-name flower_256_phi9h --swanlab-tags version_phi/h,model256-256
```

## Custom Scenario Generation (V1 phi9)
`--scenario-config` is supported for V1 phi9 flower generation:

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

```bash
python -m testdata_generate.generate \
  --scenario-config test_data/flower_custom_v1.json \
  --test-iters 1,5,10,20 \
  --output test_data/flower_custom_v1.h5
```

tmux new -s 1024 
control+b d