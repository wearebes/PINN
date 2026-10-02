# PINN 研究上下文

## 目标与证据

从 level-set 局部 stencil 预测离散曲率 `h·κ`，评估相对数值曲率在解析几何及 reinitialization 场上的误差和泛化能力。网格位置、采样和特征定义以当前生成器与数据元数据为准。

CFD 应用分别验证 Basilisk CLSVOF 和 TwoPhaseFlow/OpenFOAM；比较同一 host 的 NN 与 native baseline。VOF-HF 是对照，不能替代 NN 改善原 solver 的证据；不同 host 的结果不能互相代证。

## 输入输出约定

- 常见输入包括 phi9（9D）、phi9 + nx9 + ny9（27D）以及从 27D 变换得到的 PCA-18；具体名称、归一化和维度由数据及 checkpoint 核实。
- `phi/h` 归一化与 `h·κ` 标签必须配套核实；sign-flip 的曲率符号同步变化。
- alpha 增广中 `α·h·κ` 是目标而不是新增输入；部署及评估按对应训练约定处理。
- PCA 变换与 checkpoint 保持一致；不能将 27D 原始输入直接当成已变换特征。
- 样本统计优先读取 HDF5；曲率幅值用 `mean(|hκ|)`，区分有符号均值。circle/ellipse 计数应由元数据重建并与总数核对。

## 工作入口

`train_generate/` → HDF5 → `model/` → checkpoint → `evaluate/`。split、ellipse、flower 分别反映分布内与不同泛化条件，结果不能混用。

先读仓库 `AGENTS.md`，具体命令查 `readme.md` 并核对源码。`.claude/CLAUDE.md` 保留历史 quick-start，仅作参考，不代表当前命令有效。

复用现有评估脚本，先呈现指标与图。探索代码放 `tem/<topic>/`；原始数据、模型和正式结果保留单一权威来源。实验完成情况、性能数值、可用环境及路径每次按任务现场核验，不在本文件重复维护历史结果表。
