# Run Guide

## Variants

```
V1    (baseline)      : (phi9,                              hκ) , D = 9  , Param = 50,945
V2    (phi/h)         : (phi9/h,                            hκ) , D = 9  , Param = 50,945
V2.1  (phi/h,n)       : ([phi9/h, nx9, ny9],                hκ) , D = 27 , Param = 53,249
V2.2  (phi/ah)        : (phi9/(α·h),                        α·hκ) , D = 9  , Param = 50,945  # alpha-aug α∈{0.5,1.0,2.0}，部署时α=1退化为V2
V2.3  (phi/ah,n)      : ([phi9/(α·h), nx9, ny9],            α·hκ) , D = 27 , Param = 53,249  # 同上+梯度，部署时退化为V2.1
V3    (PCA-18)        : (PCA_18([phi9/h, nx9, ny9]),         hκ) , D = 18 , Param = 52,097
```

On-disk checkpoints (ρ ∈ {128, 256}):

```
V2    (phi/h, 9D)        out/{ρ}/baseline_{ρ}_h.pt
V2.1  (phi/h+∇φ, 27D)    out/{ρ}/baseline_{ρ}_hgradient.pt      # "27D-grad"
V2.2  (phi/αh, 9D)       out/5527/v22_{ρ}_ah.pt
V2.3  (phi/αh+∇φ, 27D)   out/5527/v23_{ρ}_hgradient_ah.pt
V3    (PCA-18)           out/5527/v3_{ρ}_hgradient.pt           # "PCA"
```


## Environment

```bash
# 1. Generate training data    (实际布局: dataset/{ρ}/{ρ}_*.h5)
python -m train_generate --output dataset/256/256_h.h5         --resolutions 256 --scale-h                       # V2  9D
python -m train_generate --output dataset/16/16_hgradient.h5 --resolutions 16 --scale-h --augment-gradient    # V2.1 27D

python -m train_generate --output dataset/256/256_ah.h5           --resolutions 256 --augment-alpha 0.5,1.0,2.0                     # V2.2 9D
python -m train_generate --output dataset/256/256_hgradient_ah.h5 --resolutions 256 --augment-alpha 0.5,1.0,2.0 --augment-gradient   # V2.3 27D
# (128 同理，把 256 换成 128)

# 11D: keep the same generation arguments; replace --augment-gradient with:
#   --feature-mode phi9_center_normal
# Input: phi9 (same --scale-h / --augment-alpha scaling) + center nx, ny.
# To match the formal cell-centre datasets, also pass:
#   --grid-convention cell_count_cell_centres  (h=1/N, x=(i+0.5)/N)
# Default endpoint_nodes retains the legacy grid.
# Sampling, labels, splits and augmentations are unchanged. Explicit mode overrides
# --augment-gradient. Other modes: phi9 (9D), phi9_full_normal (27D).
python -m train_generate --output dataset/256/256_hcenter_normal.h5 --resolutions 256 --scale-h --feature-mode phi9_center_normal

# 19D: keep all phi9 values; retain normals only at up, left, center, right, down.
# V5 layout: [phi9 | nx_cross5 | ny_cross5]. With --scale-h, phi9 means phi9/h.
# Cross normals preserve 27D stencil indices [1,3,4,5,7] (zero-based).
# This is a feature-only change: keep the other arguments from the 27D run.
# Formal cell-centre example (N64: change resolutions and the output path):
python -m train_generate --output dataset/32/32_cellcenter_hcross_normal.h5 --resolutions 32 --grid-convention cell_count_cell_centres --scale-h --feature-mode phi9_cross_normal

# 2. Train (baseline / V2.x)
python -m model.train \
  --dataset-output dataset/32/32_hgradient.h5 \
  --output-model out/32/baseline_32_hgradient.pt \
  --normalization-csv out/32/baseline_32_hgradient.csv

# 11D uses the same training settings; input_dim is inferred from the dataset.
python -m model.train \
  --dataset-output dataset/256/256_hcenter_normal.h5 \
  --output-model out/256/baseline_256_hcenter_normal.pt \
  --normalization-csv out/256/baseline_256_hcenter_normal.csv

# 19D also infers input_dim from its dataset; reuse the 27D training arguments.
python -m model.train \
  --dataset-output dataset/32/32_cellcenter_hcross_normal.h5 \
  --output-model out/32/cell_32_hcross_normal.pt \
  --normalization-csv out/32/cell_32_hcross_normal_phi9.csv

# 2b. Train with PCA-18 (V3, MLP only)
# 源必须是 V2.1 的 27D 数据集（--augment-gradient 生成的 *_hgradient.h5）。
# 首次运行离线把 27D 压成 18D，自动生成同目录 sidecar（freshness 命中则复用）：
#   dataset/256/256_hgradient_pca18.h5            18D 已投影特征，训练直接读
#   dataset/256/256_hgradient_pca18_transform.npz 完整 PCA transform（含 components 18×27）
# transform 同时嵌入 checkpoint，评估默认从 checkpoint 取，无需额外 sidecar / normalization-csv。
python -m model.train \
  --dataset-output dataset/256/256_hgradient.h5 \
  --output-model out/5527/v3_256_hgradient.pt \
  --use-pca

# 3. Generate flower test data   (实际布局: test_data/flower_rho{ρ}_{h,hgradient}.h5)
# --rho-model 接受任意整数 >= 4；h = 1/(rho_model-1)，接口外留 2 格 margin。
python -m testdata_generate --rho-model 256 --scale-h          --output test_data/flower_rho256_h.h5          # 9D
python -m testdata_generate --rho-model 256 --augment-gradient --output test_data/flower_rho256_hgradient.h5  # 27D (V2.1/V3 共用)
python -m testdata_generate --rho-model 256 --scale-h --feature-mode phi9_center_normal --output test_data/flower_rho256_hcenter_normal.h5  # 11D
python -m testdata_generate --rho-model 256 --scale-h --feature-mode phi9_cross_normal --output test_data/flower_rho256_hcross_normal.h5  # 19D
# These CLI examples retain the generic Flower grid defaults described above.
# Paper comparisons must reuse the paper's exact h=1/256 cell-centre grid and
# frozen-phi0 reinitialization configuration, changing only feature_mode/model.
```

```bash
# 4. Evaluate flower —— 每次产出 2行×3列 overview（smooth+acute 的最大 iter；右列即 |error| vs θ 角度误差）
#    打印的 Flower summary RMSE 是全 20 步聚合，图显示 iter_20 最脏场。

# 27D-grad (V2.1)
python evaluate/flower.py --data dataset/test_data/flower_rho512_hgradient.h5 \
  --model-path out/256/baseline_256_hgradient.pt --device cpu \
  --output-dir out/27d_vs_pca/flower --name 27Dgrad_model256-512

# PCA-18 (V3)：同一份 27D 数据，PCA 自动降到 18D；不要传 --normalization-csv
python evaluate/flower.py --data dataset/test_data/flower_rho512_hgradient.h5 \
  --model-path out/5527/v3_256_hgradient.pt --device cpu \
  --output-dir out/27d_vs_pca/flower --name PCA_model256-512
```

```bash
# 5. Evaluate ellipse —— 4 个 b/a (1.0/0.8/0.5/0.3) 现场构造；中列 h·κ(θ)、底行 |error| vs θ。
# 坑①: --dataset-path 必须用 scale_h=True 的元数据集 dataset/{ρ}/{ρ}_hgradient.h5；
#      别用默认 dataset/256/256.h5(scale_h=False)，dataset/128/128.h5 也不存在。
# 坑②: PCA 模型去掉 --normalization-csv，transform 自动从 checkpoint 取。

# 27D-grad (V2.1)
python evaluate/ellipse.py --rho-model 512 \
  --model-path out/256/baseline_256_hgradient.pt \
  --dataset-path dataset/512/512_hgradient.h5 --device cpu \
  --output-dir out/27d_vs_pca/ellipse --name 27Dgrad_model256-512

# PCA-18 (V3)
python evaluate/ellipse.py --rho-model 512 \
  --model-path out/5527/v3_256_hgradient.pt \
  --dataset-path dataset/512/512_hgradient.h5 --device cpu \
  --output-dir out/27d_vs_pca/ellipse --name PCA_model256-512
```

# 其他评估
$PY -m evaluate.training_curvature --data dataset/266/266.h5 --output-dir dataset/266
```

```bash
# tmux
tmux ls
tmux new -s 1024
tmux attach -t 1024
# detach: Ctrl+b d
```

## Strict 3×3 local normals (15D, V6)

```bash
python -m train_generate.rebuild_local --source dataset/32/32_cellcenter_hcross_normal.h5 --output dataset/32/32_cellcenter_hlocal_normal.h5
python -m model.train --dataset-output dataset/32/32_cellcenter_hlocal_normal.h5 --output-model out/32/cell_32_hlocal_normal.pt --normalization-csv out/32/cell_32_hlocal_normal_phi9.csv
# N64: replace both occurrences of 32 in each path with 64.
python tem/local_normal_comparison/compare.py
```

T