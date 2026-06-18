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

### Experiment 2（已完成）：4 variant × 2 分辨率，MSE resolution sweep

目标：对比四种 input 设计在 ρ=128、256 下的 split 精度和跨分辨率泛化，全部 MLP（hidden=128，patience=30，L2=0）。

| Variant | 输入特征 | 目标 | D | 数据集 |
|---------|---------|------|---|--------|
| **V2.2** | phi9/(α·h) | α·h·κ | 9 | `{rho}_ah.h5` |
| **V2.3** | phi9/(α·h) + nx9 + ny9 | α·h·κ | 27 | `{rho}_hgradient_ah.h5` |
| **V3**   | PCA-18(phi9/h + nx9+ny9) | h·κ | 18 | `{rho}_hgradient.h5` + `--use-pca` |
| **v2_hgrad** | phi9/h + nx9 + ny9 | h·κ | 27 | `{rho}_hgradient.h5` |

**结果摘要（split MSE，`out/mse_sweep/mse_split.csv`）**：V3 和 v2_hgrad 在所有测试分辨率上明显优于 V2.2/V2.3；V3@256 在同分辨率测试下 split MSE ≈ 1.9e-7，接近 FD 精度量级。Alpha 增广（V2.2/V2.3）未带来 split 精度提升，但对 OOD flower 曲线的影响待进一步分析。

**数据集状态**：所有 6 个数据集均已生成。

### 评估脚本

| 脚本 | 用途 |
|------|------|
| `evaluate/flower.py` | 主 OOD flower 评估（reinit 0..20 步）|
| `evaluate/mse_resolution_sweep.py` | 单 variant 跨分辨率 MSE sweep |
| `evaluate/mse_resolution_sweep_split.py` | 所有 variant split MSE 汇总 → `out/mse_sweep/mse_split.csv` |
| `evaluate/mse_resolution_sweep_all.py` | 所有 variant 全量 sweep → `out/mse_sweep/mse_all.csv` |
| `evaluate/flower_step_curves_all.py` | 所有 variant flower step MSE 曲线对比 |
| `evaluate/flower_step_feature_input_compare.py` | feature/input 维度视角的 flower 曲线 |
| `evaluate/test_split_mse.py` | 快速 split MSE 单次检查 |

### 历史实验

- **Experiment 1**：scale_h 对比，64/128/256 隐层 × 有/无 L2 weight decay
- V2 27D 梯度特征初探（法向信息对 OOD 泛化的影响）
- HPC Slurm 3× 并行（ρ=256/266/276），SwanLab offline 模式

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
