# PINN MLP-Field 项目架构文档

> 最后更新：2026-05-26

---

## 一、项目定位

这是一个 **stencil-to-scalar regression** 任务：训练神经网络，从 level-set 场 φ 在界面节点处的 3×3 局部 stencil 直接预测离散化曲率指标 `h·κ`，以替代传统的中心差分（central-difference）曲率算子。

**核心动机**：中心差分公式在 reinitialization 之后的"脏"φ 场上误差明显增大（截断误差 + 场不精确的双重叠加）。NN 直接学习解析真值，理论上可以在同样的 3×3 stencil 信息下做到比 FD 更高阶的精度，且对场质量退化更鲁棒。

**任务公式**：

```
输入  x ∈ ℝ⁹  (phi9，即 3×3 patch 的 9 个 φ 值，按特定扫描顺序展开)
       或
      x ∈ ℝ²⁷  (phi9 + nx9 + ny9，追加每点的归一化梯度方向，V2 feature)

输出  ŷ ∈ ℝ   (预测的 h·κ，无量纲化曲率)

监督  y = κ_analytic × h  (解析真值，无离散误差)
```

---

## 二、整体 Pipeline

三个通过 HDF5 + checkpoint 文件串联的子系统：

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: TRAIN DATA GENERATION  (train_generate/)                            │
│                                                                                │
│   Circle / Ellipse geometry blueprints                                        │
│   → build φ0 grid (SDF / nonSDF)                                              │
│   → find interface nodes                                                       │
│   → extract phi9 stencil (+ optional grad9)                                   │
│   → compute analytic h·κ target                                               │
│   → split by blueprint (70/15/15)                                             │
│   → save  dataset/*.h5                                                        │
└─────────────────────────────────────┬──────────────────────────────────────────┘
                                      │  load_training_arrays_from_hdf5
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MODEL TRAINING  (model/)                                            │
│                                                                                │
│   fit μ,σ on train split  →  standardize all splits                          │
│   MLP (5 layers) or CNN (2×Conv)  →  MSE loss  →  AdamW                      │
│   early-stop on val MSE  →  save  out/*.pt + *.csv                            │
└─────────────────────────────────────┬──────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┘
         │                   ┌──────────────────────────────────────┐
         │                   │  STAGE 2b: TEST DATA GENERATION      │
         │                   │  (testdata_generate/)                │
         │                   │  flower geometry + HJ reinit steps   │
         │                   │  → test_data/*.h5                    │
         │                   └──────────────────┬───────────────────┘
         ▼                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: EVALUATION  (evaluate/)                                             │
│                                                                                │
│   A. evaluate.split   — in-distribution (test split of training HDF5)        │
│   B. evaluate.ellipse — controlled ellipse OOD cases                         │
│   C. evaluate.flower  — true OOD: flower geometry + reinit stress test        │
│                                                                                │
│   Comparisons: numeric_vs_analytic (FD baseline) / model_vs_analytic (NN)    │
│   Metrics: MSE / MAE / MaxAE                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、训练数据生产（`train_generate/`）

### 3.1 几何蓝图（blueprints）

数据是**分层、确定性、形状均衡**生成的——网格化枚举而非随机抽样。

#### Circle（`CircleGeometryGenerator`）

| 参数 | 取值 |
|---|---|
| 半径范围 | `r ∈ [1.6h, 0.5 − 2h]`，线性等距 |
| 半径数量 | `num_radii = floor((ρ − 8.2) / 2) + 1`，ρ=256 时 ≈ 124 |
| 每半径变体 | `variations = 12`（中心在 `[0.5 ± h/2]` 随机偏移） |
| 默认蓝图数 | ≈ 1488 |

#### Ellipse（`EllipseGeometryGenerator`）

| 参数 | 默认值 |
|---|---|
| 长半轴离散数 | `ellipse_num_a = 48`，`a ∈ [8h, 0.5 − 2h]` 等距 |
| 每 a 变体数 | `ellipse_variations_per_a = 124` |
| 轴比 b/a | `∈ [0.50, 0.90]` 均匀随机 |
| 旋转角 ψ | `∈ [0, π]` 均匀随机 |
| 默认蓝图数 | 5952 |

每个蓝图用确定性子种子（FNV-style bit-hash）保证复现性：
- `CircleGeometryGenerator._subseed(r_idx, v_idx)`
- `EllipseGeometryGenerator._subseed(a_idx, v_idx)`

#### 蓝图分割

`split_blueprint_indices()` 按形状类型分别做 70/15/15 的 blueprint-level split（训练比例可配置），用 `geometry_seed + 7919 * shape_offset` 洗牌，保证两种形状在各 split 中比例一致。

### 3.2 φ0 场构造

对每个蓝图 × initial_field_type 生成初始场：

| 形状 | SDF | nonSDF |
|---|---|---|
| Circle | `φ = ‖(x,y)−c‖ − r` | `φ = ‖(x,y)−c‖² − r²` |
| Ellipse | 需要 Newton 投影（双重精化，见下） | `φ = (u/a)² + (v/b)² − 1` |

**椭圆 SDF 高精度流程**（`build_ellipse_sdf`）：
1. 先用 float64 Newton 迭代投影到最近点角 θ（容差 `1e-12`，最多 30 次）
2. 再用 `mpmath` 80 位精度 Newton 精化（`ellipse_hp_dps=80`，最多 100 次）
3. 最终 SDF = 带符号距离到椭圆曲线

> **设计意图**：训练 target 是解析真值，因此 SDF 本身也必须足够精确，不能引入数值误差污染 label。

### 3.3 界面节点检测（`interface_indices`）

- 找 4-邻接边上 φ 变号的节点（即 `φ[i,j] × φ[i+1,j] ≤ 0` 或 `× φ[i,j+1] ≤ 0`）
- **排除距域边界 ≤2 格的节点**，给 `extract_grad9` 的 ±2 格访问留余量

### 3.4 特征提取

#### V1：phi9（9 维）

`extract_phi9(phi, indices)` 取每个界面节点的 3×3 patch，按 `encode_patch_training_order` 编码（列优先、行倒序）：

```
patch layout (row↑, col→):
  (i−1,j+1) (i, j+1) (i+1,j+1)        phi9 index mapping:
  (i−1,j  ) (i, j  ) (i+1,j  )    →   [0..2] = top col (j+1)
  (i−1,j−1) (i, j−1) (i+1,j−1)        [3..5] = mid col (j)
                                        [6..8] = bot col (j−1)
```

可选 `scale_h=True`：`features = phi9 / h`，使特征与网格尺度无关。

#### V2：phi9 + nx9 + ny9（27 维）

`extract_grad9(phi, indices)` 对 3×3 patch 的每个位置 k 用中心差分计算归一化梯度方向 (nxₖ, nyₖ)：

```
dx_k = φ[rk+1, ck] − φ[rk−1, ck]   # ∝ ∂φ/∂x at stencil pos k
dy_k = φ[rk, ck+1] − φ[rk, ck−1]   # ∝ ∂φ/∂y at stencil pos k
(nxₖ, nyₖ) = (dx_k, dy_k) / ‖(dx_k, dy_k)‖
```

最终 27D 布局：`[phi9_features (9) | nx9 (9) | ny9 (9)]`

> **V2 的物理意义**：显式地把局部法向信息喂给网络，减轻网络需要隐式推断方向的负担。在 `scale_h=True` 下梯度方向已归一化，与 h 无关。

### 3.5 数据增广

#### sign-flip augmentation（`augment_sign_flip=True`，默认开）

利用 `κ(−φ) = −κ(φ)` 的反对称性：

```python
samples.append({"features":  features, "hkappa_target":  hkappa})
samples.append({"features": -features, "hkappa_target": -hkappa})
```

效果：样本量翻倍 + 鼓励网络学到 `f(−x) = −f(x)` 的奇函数归纳偏置。

#### gradient augmentation（`augment_gradient=True`，默认关）

生成 27D V2 features；与 sign-flip 组合时梯度方向同样取反（`−phi9` 对应 `−grad9`）。

### 3.6 监督信号（analytic h·κ）

`compute_hkappa_targets()` 计算解析真值：

- **Circle**：直接 `h·κ = h / r`（常数，无需逐点计算）
- **Ellipse**：
  1. 将界面节点 (x,y) 变换到椭圆局部坐标 (u,v)
  2. float64 Newton 投影 → 高精度 mpmath 精化 → 角参数 θ
  3. 解析公式：`κ = ab / (a²sin²θ + b²cos²θ)^{3/2}`，乘以 h

**关键点**：training target 没有任何离散化误差，是真正的连续曲率。

### 3.7 输出 HDF5 格式

`save_training_dataset_hdf5()` 写入三个 split（train/val/test），每个 split 包含：

| 数组 | 形状 | dtype | 说明 |
|---|---|---|---|
| `phi9` | (N, 9) | float32 | 原始 3×3 stencil |
| `features` | (N, 9 or 27) | float32 | 可能经 scale_h 处理的特征 |
| `hkappa_target` | (N, 1) | float64 | 解析 h·κ |

文件同路径生成 `_manifest.json` 记录完整的 `DataConfig` 快照。

### 3.8 训练数据的统计画像

| 维度 | 特征 |
|---|---|
| 几何类型 | 纯光滑解析形状（圆 + 椭圆），无 reinitialization 后的"脏"场 |
| h·κ 值域 | 圆：`[h/(0.5−2h), h/1.6h] ≈ [2h, 0.625]`；椭圆更宽（与 a/b 有关） |
| 样本量分布 | 被周长/h 加权：大半径形状节点更多 → **小曲率在样本量上占优势** |
| 分割粒度 | Blueprint-level（几何图形级），同一形状的所有节点不跨越 split |
| 分布偏态 | 由于半径等距 → `1/r` 分布右偏，即小 h·κ 多、大 h·κ 少 |

> `evaluate/training_curvature.py` 可以生成 train split 的 hkappa 分布统计（mean/std/分位数/histogram）。

---

## 四、模型架构与训练（`model/`）

### 4.1 模型结构

#### MLP — `HKappaStencilNet`（主力模型）

```
Linear(input_dim, H) → ReLU
Linear(H, H)         → ReLU
Linear(H, H)         → ReLU
Linear(H, H)         → ReLU
Linear(H, 1)
```

- `input_dim`：9（V1）或 27（V2），由 feature transform 决定
- `H = hidden_units`，默认 128
- 参数量：hidden=128, D=9 约 50k；D=27 约 52k（极轻量）

#### CNN — `HKappaCNN`（旁支，仅支持 V1 9D）

```
Conv2d(1, 32, 3×3, padding=1) → ReLU
Conv2d(32, 64, 3×3, padding=1) → ReLU
AdaptiveAvgPool2d(1, 1) → Flatten
Linear(64, 1)
```

> 注：对 3×3 patch 而言，2 层 Conv + AvgPool ≈ 加权线性组合，表达能力与浅 MLP 接近。CNN 主要用于完整性对比，不太可能优于 MLP。

#### 工厂函数

```python
model = create_model(config)  # config 是 MLP_TrainConfig 或 CNN_TrainConfig
```

### 4.2 超参数配置

| 参数 | MLP 默认 | CNN 默认 | 说明 |
|---|---|---|---|
| `lr` | 1e-4 | 1e-4 | AdamW 学习率 |
| `l2_reg` | 0 | 0 | weight_decay |
| `max_epochs` | 1000 | 1000 | 最大轮数 |
| `patience` | 30 | 30 | 早停等待轮数 |
| `batch_size` | 256 | 204800 | mini-batch 大小 |
| `hidden_units` | 128 | — | MLP 隐层宽度 |
| `compile_mode` | `reduce-overhead` | — | torch.compile 模式 |

### 4.3 特征归一化

在训练前由 `fit_feature_transform()` 对 **train split** 拟合 per-feature 均值和标准差：

```python
mean = features_train.mean(axis=0)   # shape (D,)
std  = features_train.std(axis=0)    # shape (D,), clipped to > 0
```

推理时：`x_norm = (x − mean) / std`

**保存位置**：
1. 嵌入 `.pt` checkpoint 的 `feature_transform` 字段（结构化 dict）
2. 同路径的 `.csv` sidecar（`phi_index, mean, std, variance, source_split, feature_count, dataset_path`）

支持 V1（9行）和 V2（27行）两种规格，不可混用。

### 4.4 训练循环（`train_model`）

```
for epoch in range(max_epochs):
    train_loss = run_epoch(train_split, shuffle=True,  optimizer=AdamW)
    val_loss   = run_epoch(val_split,   shuffle=False, optimizer=None)
    
    if val_loss < best_val:
        save_checkpoint_bundle(...)    # 每次改善都保存
        wait = 0
    else:
        wait += 1
        if wait >= patience: break     # 早停
```

**硬件加速**（CUDA 下自动启用）：
- `torch.compile(mode="reduce-overhead")` — 图编译
- `torch.autocast("cuda", float16)` — AMP 混合精度
- `torch.backends.cuda.matmul.allow_tf32 = True` — TF32
- `cudnn.benchmark = True`
- 数据预先 pin_memory + non_blocking 搬到 GPU

### 4.5 Checkpoint 格式

`save_checkpoint_bundle()` 写入：

```python
{
    "checkpoint_format_version": 1,
    "model_config_version": 1,
    "model_type": "mlp" | "cnn",
    "model_config": {<TrainConfig 的 dataclass 字段>},
    "state_dict": {<模型权重>},
    "feature_transform": {
        "transform_kind": "standardize",
        "feature_version": 1 | 2,
        "raw_feature_dim": 9 | 27,
        "output_dim": 9 | 27,
        "feature_order": "phi9" | "phi9+nx9+ny9",
        "source_split": "train",
        "mean": np.ndarray,
        "std":  np.ndarray,
    }
}
```

`load_checkpoint_bundle()` 时会自动处理 legacy V1 baseline 的 key 映射（`remap_checkpoint_state_dict`）。

---

## 五、测试数据生成（`testdata_generate/`）

### 5.1 Flower 几何（`build_flower_phi0`）

```
φ0(x, y) = r − a·cos(p·θ) − b
```

其中 `r = √(x²+y²)`，`θ = arctan2(y,x)`，p=3 为默认花瓣数。

两种预置形态（`legacy_flower_scenarios`）：

| 类型 | a | b | 特征 |
|---|---|---|---|
| `smooth_*` | 0.05 | 0.15 | 曲率变化平缓 |
| `acute_*` | 0.075 | 0.15 | 尖瓣，局部高曲率，接近奇异点 |

各提供 rho_model ∈ {256, 266, 276} 三种分辨率共 6 个 scenario。

### 5.2 Reinitialization（`LevelSetReinitializer`）

Hamilton-Jacobi reinitialization 方程求解：

```
∂φ/∂τ + S(φ0)(|∇φ| − 1) = 0
```

数值格式：
- **空间离散**：WENO5（5阶加权 ENO，`_hj_weno5_1d_eps`）
- **时间积分**：TVD-RK3（3阶 Runge-Kutta）
- **CFL**：默认 0.5
- **符号函数**：平滑版 `S(φ) = φ / √(φ² + (ε·h)²)`，`ε=2.5`
- **符号模式**：`dynamic_phi`（用当前 φ 的符号，而非固定 φ0）

测试数据在 0..20 个 reinit 步上均匀采样，iter=0 是干净的解析 φ0，iter↑ 是逐步"SDF 化"后的场。

### 5.3 解析 target（`hkappa_analytic`）

```python
r   = b + a·cos(p·θ_proj)
r'  = −a·p·sin(p·θ_proj)
r'' = −a·p²·cos(p·θ_proj)
κ   = (r² + 2r'² − r·r'') / (r² + r'²)^{3/2}
h·κ = h × κ
```

`θ_proj` 由 Newton 迭代求投影最近点（`find_projection_theta`），收敛失败时回退 `scipy.optimize.minimize`。

### 5.4 输出 HDF5（`test_data/*.h5`）

| 数组 | 形状 | 说明 |
|---|---|---|
| `phi9` | (N, 9) | 3×3 stencil |
| `features` | (N, 9 or 27) | 可能含梯度 |
| `xy` | (N, 2) | 界面节点物理坐标 |
| `phi0_center` | (N,) | 解析 φ0 在节点处的值（一致性校验用） |
| `hkappa_target` | (N,) | 解析 h·κ |
| `case_id` | (N,) int16 | 对应哪个 scenario |
| `iter` | (N,) int16 | reinit 步数 |
| `rho_model` | (N,) int16 | 分辨率 |
| `h` | (N,) float32 | 网格步长 |

---

## 六、评估系统（`evaluate/`）

### 6.1 公共比较逻辑（`evaluate/shared.py`）

#### 核心指标函数 `compute_metrics(prediction, target)`

```python
{"mse": mean(diff²), "mae": mean(|diff|), "maxae": max(|diff|)}
```

#### Central-difference baseline（`central_difference_hkappa_from_phi9`）

从 phi9 patch 重建标准 FD 曲率公式：

```
κ_FD = (φxx·φy² − 2φx·φy·φxy + φyy·φx²) / (φx² + φy²)^{3/2}
```

其中 φx, φy 用中心差分，φxx, φyy, φxy 用二阶有限差分。这是所有评估中的**基准（baseline）**。

#### 两个核心比较

| 比较名 | 含义 |
|---|---|
| `numeric_vs_analytic` | FD 公式 vs 解析真值 → **FD 的精度上限**（不可突破的截断误差） |
| `model_vs_analytic` | NN 预测 vs 解析真值 → **NN 的绝对精度** |

若 `model_vs_analytic` < `numeric_vs_analytic`，则 NN 已超越 FD。

#### 分组汇总（`group_metric_rows`）

按 label 数组（iter / case_id / rho_model）分组，对每组分别计算 `numeric_vs_analytic` 和 `model_vs_analytic`，便于按维度切片分析。

### 6.2 In-Distribution 评估（`evaluate/split.py`）

**数据来源**：训练 HDF5 的 `test` split（同分布，仅 blueprint 未参与训练）

**用法**：

```bash
python -m evaluate.split \
  --data dataset/256_h.h5 --split test \
  --model-path out/model_256.pt --normalization-csv out/model_256.csv
```

**输出**：`numeric_vs_analytic` 和 `model_vs_analytic` 的 MSE/MAE/MaxAE，打印到 stdout；可选上传 SwanLab。

### 6.3 受控椭圆评估（`evaluate/ellipse.py`）

**数据来源**：即时生成，不依赖 HDF5（从命令行指定的 rho_model 和代码中定义的 `ELLIPSE_CASES` 列表现场计算）

**ELLIPSE_CASES 定义区**（文件中唯一需要编辑的区域）：

```python
ELLIPSE_CASES = (
    EllipseCase(0.10, 0.09),   # a, b
    EllipseCase(0.22, 0.12),
    EllipseCase(0.18, 0.16),
    EllipseCase(0.34, 0.18),
    EllipseCase(0.24, 0.12),
)
```

所有 case 共享命令行传入的 `--rho-model`；center=(0.5, 0.5)，psi=0.0 写死。

**用法**：

```bash
python -m evaluate.ellipse --rho-model 256 \
  --model-path out/model_256.pt \
  --output-dir out/ellipse_eval
```

**泛化维度**：rho_model 可以是 266/276（训练集只有 256），测试轻度跨分辨率外推。

### 6.4 真 OOD 评估 — Flower（`evaluate/flower.py`）

**三重 OOD**：

| OOD 维度 | 训练分布 | 测试分布 |
|---|---|---|
| 几何形状 | 圆 + 椭圆 | 玫瑰曲线 `r = a·cos(pθ) + b` |
| 分辨率 | 仅 ρ=256 | ρ ∈ {256, 266, 276} |
| 场质量 | 干净 SDF / nonSDF | reinit 后的近似 SDF（iter 0..20） |

**用法**：

```bash
python -m evaluate.flower \
  --data test_data/rho256_h.h5 \
  --model-path out/model_256.pt \
  --normalization-csv out/model_256.csv
```

**输出结构**（`evaluate_flower` 返回 dict）：

```python
{
    "summary":            # 全局 model_vs_analytic 汇总
    "numeric_summary":    # 全局 numeric_vs_analytic 汇总
    "numeric_vs_analytic": {"mse", "mae", "maxae"},
    "model_vs_analytic":   {"mse", "mae", "maxae"},
    "cases":              # per-(case_id, iter) 行，含 θ 排序后的曲率曲线
    "by_iter":            # 按 reinit 步分组的 metrics
    "by_case_id":         # 按 scenario 分组的 metrics
    "by_rho_model":       # 按分辨率分组的 metrics
    "angle_bin_rows":     # 按角度 bin 分组的 metrics
    "representative_case": # 自动选取的代表性 case 曲线
    "failed_case_slices": # 失败 case 列表
}
```

**可视化**：`_render_representative_curve()` 生成 θ vs h·κ 对比图（pred/true + 误差曲线）。

### 6.5 Flower 诊断（`evaluate/flower_diagnostics.py`）

独立诊断工具，输出：

| 文件 | 内容 |
|---|---|
| `per_iter_metrics.csv` | per-(rho, case, iter) 行，模型和 FD 的 MSE/MAE/MaxAE |
| `correlations.csv` | Spearman + Pearson 相关性（iter vs mae, grad_norm_error vs mae 等） |
| `diagnostic_plots.png` | 三面板：误差随 iter 变化、梯度归一化误差、MaxAE 分布 |
| `summary.md` | 自动生成的趋势描述和注意事项 |

**用法**：

```bash
python -m evaluate.flower_diagnostics \
  --output-dir out/diagnostics [--rho-models 256,266,276]
```

### 6.6 Step 趋势图（`evaluate/flower_step_plots.py`）

单分辨率的 reinit step 对比图（numeric vs model 的 MSE/MAE/MaxAE 随步数变化），输出：
- `flower_rho{rho}_step_metrics.png`
- `flower_rho{rho}_step_metrics.csv`

### 6.7 训练集曲率统计（`evaluate/training_curvature.py`）

对 train split 的 hkappa 分布做完整统计分析（mean/std/min/p01/p05/p25/p50/p75/p95/p99/max），按 split × shape_type 分组，可输出 histogram 图和分组 CSV。

---

## 七、Feature Version 系统

整个 pipeline 有 V1 / V2 两条严格隔离的特征路径：

| | V1 | V2 |
|---|---|---|
| 特征维度 | 9D | 27D |
| 特征内容 | phi9 | phi9 + nx9 + ny9 |
| 生成开关 | 默认 | `--augment-gradient` |
| CNN 兼容 | ✓ | ✗（CNN 强制 V1） |
| `feature_version` | 1 | 2 |
| `feature_order` | `"phi9"` | `"phi9+nx9+ny9"` |

**不变量约束**：
- 训练数据的 `raw_feature_dim` 必须与 checkpoint 嵌入的 `feature_transform.raw_feature_dim` 一致
- CSV sidecar 的行数（9 或 27）必须与 checkpoint 一致
- 混用会在 `validate_feature_transform()` 或 `apply_feature_transform()` 处抛异常

---

## 八、文件结构索引

```
PINN/
├── train_generate/
│   ├── config.py          DataConfig, GenerationConfig（训练数据所有超参）
│   ├── generate.py        圆/椭圆 generator, phi9 提取, 解析 h·κ, split, 入口
│   └── io.py              HDF5 读写, manifest 生成
│
├── model/
│   ├── config.py          MLP_TrainConfig, CNN_TrainConfig, create_train_config()
│   ├── model.py           HKappaStencilNet, HKappaCNN, create_model()
│   └── train.py           训练循环, early stopping, checkpoint, 入口
│
├── testdata_generate/
│   ├── config.py          TestDataConfig, FlowerScenario, legacy_flower_scenarios()
│   ├── generate.py        flower φ0, 解析 h·κ, reinit 采样, HDF5 写入, 入口
│   └── reinit.py          LevelSetReinitializer（WENO5 + RK3）
│
├── evaluate/
│   ├── shared.py          compute_metrics, fit/apply/save/load feature_transform,
│   │                      central_difference_hkappa, load_model, predict_full_batch,
│   │                      group_metric_rows, checkpoint bundle I/O
│   ├── split.py           in-distribution test split 评估
│   ├── ellipse.py         受控椭圆 OOD 评估
│   ├── flower.py          真 OOD flower 评估主入口
│   ├── flower_diagnostics.py  per-iter 诊断, 相关性分析
│   ├── flower_step_plots.py   reinit 步趋势图
│   ├── flower_overview.py     多分辨率总览图
│   ├── flower_step_plots.py   单分辨率步骤对比图
│   ├── training_curvature.py  训练集 hkappa 分布统计
│   └── curvature_plotting.py  公共绘图工具（angle bins, case summary, overview）
│
├── dataset/               训练 HDF5（*.h5）和 manifest（*_manifest.json）
├── test_data/             花朵测试 HDF5
├── out/                   模型 checkpoint（*.pt）和归一化 CSV（*.csv）
├── logs/                  训练日志
├── swanlog/               SwanLab 本地日志
└── RUN_GUIDE.md           快速运行命令参考
```

---

## 九、关键设计决策与已知风险

### 9.1 训练集无"脏"场

所有训练数据的 φ 都是解析 SDF 或 nonSDF，**没有 reinitialization 后的场**。模型在 flower iter>0 上的泛化依赖于：它学到的是真正的 local differential operator 的行为，而不仅仅是"圆/椭圆 SDF 上的查找表"。`flower` 评估的 per-iter 曲线是验证这一点的核心诊断。

### 9.2 节点级样本量不均衡

Blueprint-level 70/15/15 split 保证了形状分布均衡，但每个 blueprint 贡献的节点数 ∝ 周长 / h。大半径圆的节点数可比小半径圆多 10×。

**后果**：MSE 损失在优化时会被小曲率（大半径）样本主导，可能导致高曲率区域（大 κ，对 interface 动力学通常更重要）的精度被低估。

**诊断**：`evaluate/training_curvature.py` 可以查看 hkappa 在节点级的分布。

### 9.3 h·κ 值域右偏

由于半径等距，`h·κ = h/r` 的分布是倒数分布，支撑在 `[2h, 0.625]` 但密度集中在小值端。如果需要高曲率精度，考虑：

- 对半径取对数等距（`r_i = r_min × (r_max/r_min)^(i/n)`）
- 或对 hkappa 取 log 后再回归

### 9.4 跨分辨率外推的理论前提

训练默认只用 `ρ=256`（`h=1/255`）。ellipse/flower 评估包含 ρ=266/276。

- 若 `scale_h=False`（features = phi9，量纲为 φ 的数值）：不同 ρ 下的 φ 值域不同，外推缺乏理论保证
- 若 `scale_h=True`（features = phi9/h，无量纲化）：不同 ρ 下特征分布接近，外推有理论基础

### 9.5 V1 vs V2 的 trade-off

| | V1 (9D phi9) | V2 (27D phi9+grad) |
|---|---|---|
| 信息量 | 仅局部 φ 值 | 额外包含法向方向 |
| 对非 SDF 的鲁棒性 | 依赖网络隐式推断方向 | 显式提供方向信息，理论上更鲁棒 |
| 参数量 | ≈50k | ≈52k（差异极小） |
| 计算成本 | 低 | 中（grad9 需要 ±2 格访问） |
| scale_h 必要性 | 建议开 | 建议开（grad 已归一化，φ 需要） |

### 9.6 CNN 是对照组

`HKappaCNN` 对 3×3 patch 的 2 层卷积 + adaptive avg pool 在感受野和参数量上与浅 MLP 几乎等价。其存在主要用于验证"卷积结构"是否带来额外收益（通常不会）。

---

## 十、常用命令速查

```bash
# 生成 V1 训练数据（9D phi9/h）
python -m train_generate --output dataset/256_h.h5 --resolutions 256 --scale-h

# 生成 V2 训练数据（27D phi9+grad/h）
python -m train_generate --output dataset/256_h_v2.h5 --resolutions 256 --scale-h --augment-gradient

# 训练
python -m model.train \
  --dataset-output dataset/256_h.h5 \
  --output-model out/model_256.pt \
  --normalization-csv out/model_256.csv

# In-distribution 评估（test split）
python -m evaluate.split \
  --data dataset/256_h.h5 --split test \
  --model-path out/model_256.pt --normalization-csv out/model_256.csv

# 生成 flower 测试数据（rho=256）
python -m testdata_generate --rho-model 256 --scale-h --output test_data/rho256_h.h5

# Flower OOD 评估
python -m evaluate.flower \
  --data test_data/rho256_h.h5 \
  --model-path out/model_256.pt --normalization-csv out/model_256.csv

# 受控椭圆评估
python -m evaluate.ellipse --rho-model 256 --model-path out/model_256.pt

# 训练集 hkappa 分布统计
python -m evaluate.training_curvature --data dataset/256_h.h5 --output-dir out/curvature_stats

# Flower per-iter 诊断
python -m evaluate.flower_diagnostics --output-dir out/diagnostics

# Flower step 趋势图
python -m evaluate.flower_step_plots --rho-model 256
```
