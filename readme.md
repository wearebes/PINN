# Run Guide
V1    (baseline)      : (phi9,                              hκ) , D = 9  , Param = 50,945
V2    (phi/h)         : (phi9/h,                            hκ) , D = 9  , Param = 50,945
V2.1  (phi/h,n)       : ([phi9/h, nx9, ny9],                hκ) , D = 27 , Param = 53,249
V2.2  (phi/ah)         : (phi9/(α·h),                        α·hκ) , D = 9  , Param = 50,945  # alpha-aug α∈{0.5,1.0,2.0}，部署时α=1退化为V2
V2.3  (phi/ah,n)       : ([phi9/(α·h), nx9, ny9],            α·hκ) , D = 27 , Param = 53,249  # 同上+梯度，部署时退化为V2.1
V3    (PCA-18)        : (PCA_18([phi9/h, nx9, ny9]),         hκ) , D = 18 , Param = 52,097

```bash
# 1. Generate training data
python -m train_generate --output dataset/256_h.h5 --resolutions 1024 --scale-h
# V2 (27D features)
python -m train_generate --output dataset/256_h_v2.h5 --resolutions 1024 --scale-h --augment-gradient

python -m train_generate --output dataset/1024.h5 --resolutions 1024 
python -m train_generate --output dataset/1024_h.h5 --resolutions 1024 --scale-h
python -m train_generate --output dataset/1024_hgradient.h5 --resolutions 1024 --scale-h --augment-gradient

# Alpha 增强：每个样本生成 alpha 份副本 (phi/(α·h), α·h·κ)，必须含 1.0
# 样本量 × len(alpha)，推理契约不变（仍消费 phi/h）

python -m train_generate --output dataset/128_ah.h5 --resolutions 128 --augment-alpha 0.25,0.5,1.0
python -m train_generate --output dataset/128_a_gradient.h5 --resolutions 128 --augment-alpha 0.25,0.5,1.0 --augment-gradient

python -m train_generate --output dataset/256_ah.h5 --resolutions 256 --augment-alpha 0.25,0.5,1.0
python -m train_generate --output dataset/256_a_gradient.h5 --resolutions 256 --augment-alpha 0.25,0.5,1.0 --augment-gradient


# 2. Train
python -m model.train \
  --dataset-output dataset/256_h.h5 \
  --output-model out/model_256.pt \
  --normalization-csv out/model_256.csv \
  --use-swanlab --swanlab-mode cloud --swanlab-project PINN \
  --swanlab-experiment-name model_256 --swanlab-tags baseline,rho256

# 2b. Train with PCA-18 (V3, MLP only)
# 源必须是 V2 27D 数据集（--augment-gradient 生成的，如 *_hgradient.h5）
# 首次运行离线把 27D 压成 18D，自动生成同目录 sidecar（之后 freshness 命中则复用）：
#   dataset/256_hgradient_pca18.h5            18D 已投影特征，训练直接读
#   dataset/256_hgradient_pca18_transform.npz 完整 PCA transform（含 components 18×27）
# transform 同时嵌入 checkpoint，评估默认从 checkpoint 取，无需额外传 sidecar
python -m model.train \
  --dataset-output dataset/256_hgradient.h5 \
  --output-model out/model_256_pca18.pt \
  --use-pca \
  --use-swanlab --swanlab-mode cloud --swanlab-project PINN \
  --swanlab-experiment-name model_256_pca18 --swanlab-tags pca18,rho256

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
  --data dataset/test_data/512_hgradient.h5 \
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
```bash
python -m evaluate.training_curvature --data dataset/266/266.h5 --output-dir dataset/266
```

```bash
# 5. Evaluate ellipse
# Cross-resolution (256 model on 276 grid)
python -m evaluate.ellipse \
  --rho-model 64 \
  --model-path out/128/baseline_128_h.pt \
  --normalization-csv out/128/baseline_128_h.csv \
  --dataset-path dataset/128_h.h5 \
  --output-dir out/64/ellipse

# PCA-18 (V3) 模型：去掉 --normalization-csv，transform 自动从 checkpoint 取出
# 评估内部先构造 27D raw features 再投影到 18D，flower/ellipse 命令其余写法不变
python -m evaluate.ellipse \
  --rho-model 64 \
  --model-path out/model_256_pca18.pt \
  --dataset-path dataset/128_hgradient.h5 \
  --output-dir out/64/ellipse_pca18
```
tmux ls
tmux attach -t 1024
tmux new -s 1024 
control+b d

