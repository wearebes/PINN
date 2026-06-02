# PINN MLP-Field Research — Agent Context

## 研究背景

**stencil-to-scalar regression**：用神经网络从 level-set 场 φ 在界面节点的 3×3 stencil 直接预测离散曲率 `h·κ`，替代中心差分（FD）算子。FD 在 reinitialization 后的"脏"φ 场上误差明显增大，NN 直接监督在解析真值上，目标是在同样的 stencil 信息下超越 FD 精度并对场质量退化更鲁棒。

特征有三个主要版本：**V1** 9D（phi9，3×3 φ 值）、**V2** 27D（phi9 + nx9 + ny9，追加归一化梯度方向）、**V3** 18D（PCA-18 压缩自 V2 27D）。另有 alpha 增广变体 V2.2 / V2.3（见"近期实验方向"）。

## 工作方式

倾向于先跑通最小 pipeline 验证想法是否有效，再做大范围适配。每次实现后展示实验结果（loss/MSE/MAE/MaxAE 对比、代表性预测图），方便一起判断方向。

## Pipeline

`train_generate/` → `dataset/*.h5` → `model/train.py` → `out/*.pt + *.csv` → `evaluate/`

评估三层：**split**（in-dist test split）、**ellipse**（受控 OOD）、**flower**（真 OOD，花朵几何 + reinit 0..20 步）。所有评估都对比 `numeric_vs_analytic`（FD）和 `model_vs_analytic`（NN）。

## 关键设计决策与风险

- **scale_h**：phi9/h 使特征与网格尺度无关，跨分辨率外推的必要条件；`scale_h=False` 时不同 ρ 的 φ 值域不同，跨分辨率外推无理论保证
- **sign-flip 增广**：`κ(−φ) = −κ(φ)` 反对称性翻倍样本，鼓励奇函数归纳偏置
- **训练集无"脏"场**：训练 φ 全是解析 SDF/nonSDF，flower per-iter 曲线是泛化到 reinit 场的核心诊断
- **Blueprint-level split**：防止同形状节点跨 split 泄露；但节点数 ∝ 周长/h，小曲率样本在 MSE 上占优，高曲率精度可能被低估
- **V2 梯度特征**：显式法向信息对非 SDF 场理论上更鲁棒，参数量从 ~50k 增至 ~53k（极小代价）
- **alpha 增广（V2.2/V2.3）**：`augment_scale_alpha=(0.5,1.0,2.0)` 使 phi9/(α·h) 和 α·h·κ 成对出现，等效于尺度不变性训练；α=1 时退化为标准 phi/h→h·κ，无需部署时做任何特殊处理。ahk 是 target 而非输入特征——混淆这一点会导致错误的 D=10/28 维度和 label-leak 设计。
- **V3 PCA-18**：`ensure_pca_dataset()` 从 V2 27D 数据 offline 拟合 PCA，transform 嵌入 checkpoint，评估时 `apply_feature_transform()` 透明处理；累计 EVR ≥ 0.95 为生成前置检查

## 近期实验方向

### 当前进行中：3 种 feature variant × 2 分辨率（6 训练任务）

目标是对比三种 input 设计在 ρ=128、256 下的表现，全部 MLP（hidden=128，patience=30，L2=0），SwanLab offline 记录。

| Variant | 输入特征 | 目标 | D | Param | 数据集格式 |
|---------|---------|------|---|-------|-----------|
| **V2.2** | phi9/(α·h) | α·h·κ | 9 | 50,945 | `{rho}_ah.h5` |
| **V2.3** | phi9/(α·h) + nx9 + ny9 | α·h·κ | 27 | 53,249 | `{rho}_hgradient_ah.h5` |
| **V3**   | PCA-18(phi9/h + nx9+ny9) | h·κ | 18 | 52,097 | `{rho}_hgradient.h5` + `--use-pca` |

**V2.2 / V2.3 说明**：alpha 增广（α ∈ {0.5, 1.0, 2.0}），输入按 phi9/(α·h) 缩放，目标为 α·h·κ。部署时 α=1 退化为标准 phi9/h → h·κ。ahk（α·h·κ）是**目标**，不作为输入特征（无 label leak）。

**V3 说明**：从现有 `_hgradient.h5`（V2 27D）做 offline PCA-18 压缩，`--use-pca` 标志触发 `ensure_pca_dataset()` 自动生成 `_pca18.h5` sidecar，transform 存入 `.npz` + checkpoint。

### 数据集状态（ρ=128/256）

| 文件 | 用途 | 状态 |
|------|------|------|
| `dataset/128_ah.h5` | V2.2 @ 128 | ✓ 已存在 |
| `dataset/256_ah.h5` | V2.2 @ 256 | ✗ 待生成 |
| `dataset/128_hgradient_ah.h5` | V2.3 @ 128 | ✗ 待生成 |
| `dataset/256_hgradient_ah.h5` | V2.3 @ 256 | ✗ 待生成 |
| `dataset/128_hgradient.h5` | V3 源 @ 128 | ✓ 已存在 |
| `dataset/256_hgradient.h5` | V3 源 @ 256 | ✓ 已存在 |

生成命令（CPU，无需 GPU）：
```bash
# V2.2 @ 256
python -m train_generate --resolutions 256 --dataset-name 256_ah.h5 \
    --augment-alpha 0.5,1.0,2.0 --num-workers 14

# V2.3 @ 128 和 256
python -m train_generate --resolutions 128 --dataset-name 128_hgradient_ah.h5 \
    --augment-gradient --augment-alpha 0.5,1.0,2.0 --num-workers 14
python -m train_generate --resolutions 256 --dataset-name 256_hgradient_ah.h5 \
    --augment-gradient --augment-alpha 0.5,1.0,2.0 --num-workers 14
```

### 训练命令（无需改代码，现有 pipeline 直接支持）

```bash
# V2.2
python -m model.train --dataset-output dataset/128_ah.h5 --output-model out/.../v22_128.pt --l2-reg 0
# V2.3
python -m model.train --dataset-output dataset/128_hgradient_ah.h5 --output-model out/.../v23_128.pt --l2-reg 0
# V3
python -m model.train --dataset-output dataset/128_hgradient.h5 --use-pca --output-model out/.../v3_128.pt --l2-reg 0
```

### 历史方向（已完成）

- scale_h 对比：实验 1 已跑 64/128/256 隐层 × 有无 L2 weight decay
- V2 27D 梯度特征：测试法向信息是否改善 OOD 泛化
- HPC Slurm 3× 并行（ρ=256/266/276），SwanLab offline 模式记录

## 代码结构

```
train_generate/   数据生成（Circle/Ellipse, phi9/grad9, 解析 h·κ, HDF5 I/O）
model/            网络结构（MLP/CNN）、训练循环、create_train_config()
testdata_generate/ 花朵测试数据（WENO5+RK3 reinit, 解析 h·κ）
evaluate/         split / ellipse / flower 评估，FD baseline，诊断图
dataset/          训练 HDF5 + manifest
test_data/        花朵测试 HDF5
out/              模型权重 (*.pt) + 归一化 CSV (*.csv)
```

完整命令见 `RUN_GUIDE.md`，详细架构见 `.claude/docs.md`。

## 环境

WSL + conda (`pinn` env, Python 3.10)，bash syntax，GPU 推荐，`patience=30` early stopping。
