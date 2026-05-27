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

# 3. Generate flower test data
# --rho-model 接受任意整数 >= 4；legacy 精调值：256, 266, 276
# 其他分辨率自动推导 L 和 N（h = 1/(rho_model-1)，接口外留 2 格 margin）

python -m testdata_generate --rho-model 512 --scale-h --output dataset/test_data/512_h.h5
python -m testdata_generate --rho-model 512 --augment-gradient --output dataset/test_data/512_hgradient.h5
python -m testdata_generate --rho-model 256 --test-iters 1,5,10,20 --output test_data/rho256_iters.h5

# 4. Evaluate flower
# 每次运行产出一张 2行×3列 overview 图（smooth + acute，最大 iter）
# 输出: out/curvature_viz/flower/flower_curvature_overview_rho<N>.png
python -m evaluate.flower \
  --data dataset/test_data/256_hgradient.h5 \
  --model-path out/256/baseline_256_hgradient.pt \
  --output-dir out/curvature_viz/flower

# 加 SwanLab 日志
python -m evaluate.flower \
  --data dataset/test_data/256_hgradient.h5 \
  --model-path out/256/baseline_256_hgradient.pt \
  --output-dir out/curvature_viz/flower \
  --use-swanlab --swanlab-mode cloud --swanlab-project geometry \
  --swanlab-experiment-name flower_256_hgradient --swanlab-tags v2,rho256
```

tmux new -s 1024 
control+b d


```bash
python -m evaluate.training_curvature --data dataset/266/266.h5 --output-dir dataset/266
```

```bash
# 5. Evaluate ellipse
# Cross-resolution (256 model on 276 grid)
python -m evaluate.ellipse \
  --rho-model 256 \
  --model-path out/128/baseline_128_h-l2e-4.pt \
  --normalization-csv out/128/baseline_128_h-l2e-4.csv \
  --dataset-path dataset/256/256_h.h5 \
  --output-dir out/128
```
