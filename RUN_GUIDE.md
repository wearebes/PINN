# Run Guide

```bash
# 1. Generate training data
python -m train_generate --output dataset/256_h.h5 --resolutions 256 --scale-h
# V2 (27D features)
python -m train_generate --output dataset/256_h_v2.h5 --resolutions 256 --scale-h --augment-gradient

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
python -m testdata_generate --rho-model 256 \
  --test-iters 1,2,3,4,5,10,20 --output test_data/rho256.h5

# 5. Evaluate flower
python -m evaluate.flower \
  --data test_data/rho256.h5 \
  --model-path out/model_256.pt --normalization-csv out/model_256.csv \
  --use-swanlab --swanlab-mode cloud --swanlab-project PINN \
  --swanlab-experiment-name flower_256 --swanlab-tags baseline,rho256
```
