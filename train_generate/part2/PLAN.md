# Part 2 数据生成计划（我的理解版 + 问题清单）

> ⚠️ **已被取代**：本文是对 GPT v1 合约的差距分析（历史记录）。
> 当前采用的方向是 v2/DCTS，实施计划见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。

> 来源：`curvature_data_generation_stratification_contract.md`（GPT 初稿 v1.0）
> 对照：当前实现 [generate.py](generate.py)、[config.py](config.py)、[io.py](io.py)、[testset.py](testset.py)、[../geometry_core.py](../geometry_core.py)
> 作者：Claude 消化整理，2026-06-23
> 状态：**Draft / 待你拍板**。本文不是把 GPT 那份 2100 行重抄一遍，而是
> （1）用我的话讲清楚那份合约到底要做什么，
> （2）把"现在的实现 vs 合约想要的"差距列成一张表，
> （3）把合约里**没讲清楚、需要你决定**的地方挑出来。

---

## 0. 一句话

> 不要用 R、a、b、q 当主分层变量，而是用**无量纲局部曲率** η = |hκ|；
> 圆按 R = h/η 生成，椭圆按 η_max = h/(a·q²) 生成、再用每个采样点的**局部** η_i 重新分 bin；
> **先分 train/val/test，再做 augmentation**；train 集按 shape × η_bin 做平衡采样。

这是一个**数据生成合约（contract）**，目标是把数据这一层冻结成稳定、可复现、可分层的形态，
**之后**才去比较 SDF/non-SDF、augmentation、网络宽深、ResMLP 这些变量。合约第 22 节定义了 Stage 0→5 的顺序，
本文件只覆盖 **Stage 0**：geometry 生成 + 曲率分层 + split 合约。

---

## 1. 核心设计：为什么是 η = |hκ|

模型看到的是**局部 stencil**，预测的目标是 **y = hκ**（不是 κ）。所以：

- 同一个物理曲率 κ，在不同分辨率 h 下，对模型的难度不同 → 要用 hκ 而不是 κ。
- 同一个半径 R，在不同 h 下对应不同的 local stencil 难度 → 要用 h/R。
- 椭圆上曲率沿界面**变化**，光靠 a、b、q 描述不了一个采样点的难度 → 要用每个点的局部 η_i。

所以统一用：

```
y_i  = h · κ_i          # 训练目标（带符号）
η_i  = |y_i| = |h·κ_i|  # 分层 / 采样平衡 / 报告用（不带符号）
```

**曲率范围（固定，不随分辨率变）**：η ∈ [0.008, 0.5]
**分 bin**：8 个 log-uniform bin，edge = 0.008·(0.5/0.008)^(k/8)，每个 bin 宽约 ×1.677。

| bin | 区间 | 圆等效 R/h | 难度 |
|---|---|---|---|
| B0 | [0.0080, 0.0134) | ~125 cells | 几乎平 |
| B1 | [0.0134, 0.0225) | ~50 | 低曲率 |
| B2 | [0.0225, 0.0377) | | |
| B3 | [0.0377, 0.0632) | ~20 | 中等 |
| B4 | [0.0632, 0.1061) | ~10 | 中高 |
| B5 | [0.1061, 0.1778) | | |
| B6 | [0.1778, 0.2982) | ~4 | 高 / under-resolved |
| B7 | [0.2982, 0.5000] | ~2 | 极度 under-resolved |

低于 0.008 → 拒绝或单独统计；高于 0.5 → 拒绝或进 stress-test 池，**不进主训练分布**。

---

## 2. 完整数据生成流程（合约要求的顺序）

**顺序不能乱，关键是"先 split 再 augment"。**

```
1. 建 geometry blueprints（圆 + 椭圆）
2. 在 geometry 级别分 train/val/test split   ← 必须在这一步
3. 对每个 geometry 生成 level-set 场 φ
4. 提取 interface-adjacent base samples（3×3 stencil 有变号）
5. 算每个 base sample 的 exact κ → y=hκ → η → η_bin   ← 椭圆按局部 η_i 重新 bin
6. 建 base packs（一个物理节点 = 一个 pack）
7. 在 train 内按 shape × η_bin 做平衡采样（quota）
8. 在选中的 pack 内展开 augmentation（D4 × sign，可选 ×SDF/non-SDF）
9. 写数据文件（train.h5 / val.h5 / test.h5）
10. 写审计报告（含 leakage 检查，违反就硬失败）
```

### 2.1 圆 blueprint
```
给定目标 η  →  R = h/η                 # 精确可控
中心 c = 0.5 + δ,  δ ~ U(-0.25h, 0.25h) # 仅打破网格相位，不是实验变量
接受条件：R ≤ 0.5 - 2h
```
圆是**常曲率**，整条界面落在同一个 η_bin。

### 2.2 椭圆 blueprint
```
对每个 (q, ψ, η_max_bin)：
  在 bin 内 log-uniform 采 η_max
  a = h / (η_max · q²)        # 由"最大曲率"反推长半轴
  b = q · a
  中心 c = 0.5 + δ
接受条件：b ≥ 2h（短轴别太欠分辨）且旋转后包围盒留 2h 边距
```
- q 固定离散集：`{0.50, 0.65, 0.80, 0.90}`
- ψ 固定离散集：`{0, π/12, π/6, π/4}`（D4 已经在 stencil 级处理旋转/反射，所以不必随机采很多角度）
- 椭圆局部曲率范围：η(t) ∈ [η_max·q³, η_max] → **每个采样点必须按局部 η_i 重新 bin**，不能只用 geometry 级 η_max。
- 椭圆曲率：κ(t) = ab / (a²sin²t + b²cos²t)^{3/2}，需要把网格点投影到椭圆求参数 t（确定性 Newton，max_iter/tol 固定，记录收敛性，失败就拒绝）。

### 2.3 Split（最关键的防泄漏规则）
- **geometry 级**先分，70/15/15；一个圆/椭圆的所有样本只能进一个 split。
- 圆 split key = `(circle, η_bin)`；椭圆 split key = `(ellipse, η_max_bin, q_index, ψ_index)`。
- **绝不允许**：base 在 train、D4 旋转版在 test；SDF 在 train、non-SDF 在 test；正号在 train、负号在 val。

### 2.4 平衡采样（只对 train）
```
N_train(shape, k) = N_train · P(shape) · (1/8)
P(circle)=0.3, P(ellipse)=0.7, k=0..7
```
- 默认 **no replacement**；配额不够就**报告 deficiency**，不默认 oversample。
- val/test 走自然分布（或只做分层报告）。

### 2.5 Augmentation pack
- D4（8）× sign-flip（2）= **16** 个 variant/pack；若含 SDF+non-SDF 则 ×2 = 32。
- sign-flip：φ→-φ，n→-n，hκ→-hκ，但 η=|hκ| 不变。
- **non-SDF 只改输入特征，target 永远来自解析几何**（target/η_bin/geometry_id/pack_id 跟 SDF 版完全一致）。

### 2.6 特征与目标
合约规定 **27D**：`features27 = [φ9/h, nx9, ny9]`，target = `h·κ_exact`。
normal = ∇φ/|∇φ|（finite-difference 或解析，要在 metadata 记录是哪种）。

### 2.7 输出与验收
- 文件：`blueprints/`、`splits/`、`raw/base_samples.parquet`、`processed/{train,val,test}.h5`、`reports/`。
- 7 个 acceptance gate：bin 构造、圆半径一致性(<1e-12)、椭圆 η_max 一致性、椭圆局部 η 范围、split 完整性（无泄漏）、train 平衡(±10%)、投影鲁棒性(失败率<1e-6)。

---

## 3. 现状 vs 合约：差距对照表

> 这就是你说的"现在的 part 2 根本不是我想象的那样"。当前实现其实是**另一套设计**，
> 它只在"圆"上用了 η，椭圆完全是随机采样、没有任何 bin/平衡。

| 维度 | 现在的实现（generate.py / config.py） | 合约想要的 | 差距大小 |
|---|---|---|---|
| **分层变量** | 只有圆用 η；椭圆按 a,q,ψ 随机采，**无 η、无 bin** | 圆+椭圆统一 η=|hκ|，椭圆按局部 η_i 重新 bin | 🔴 根本性 |
| **η 间距** | 圆 **linear** 20 levels（`np.linspace`） | **log-uniform** 8 bins | 🔴 |
| **η 范围** | **随分辨率变**：rho64 时 ≈[0.034, 0.625]（`h/r_max`~`h/r_min`，r_min=1.6h） | **固定** [0.008, 0.5] | 🔴 圆永远不够"平"，高端又超过 0.5 |
| **椭圆生成** | `a~U(a_min,a_max)`、`q~U(0.5,0.9)`、`ψ~U(0,π)` 全随机 | 按 η_max 反推 a；q、ψ 固定离散集 | 🔴 |
| **形状比例** | 200圆:800椭圆 = 20:80（**geometry 计数**） | 30:70（**train 采样权重** + 平衡） | 🟠 概念不同 |
| **平衡采样** | **没有**。所有 interface 节点全保留全 augment，无 quota | train 按 shape×η_bin 配额、no-replacement、报缺额 | 🔴 |
| **split 分层** | 只按 shape_type 分层后 shuffle 70/15/15 | 圆按(η_bin)、椭圆按(η_max_bin,q,ψ) | 🟠 |
| **特征维度** | **28D** = [φ9/h, nx9, ny9, **hk_central**]（多了第28列做残差路线） | **27D** 纯 [φ9/h, nx9, ny9] | 🔴 需你决定（见问题①） |
| **non-SDF** | 圆用代数场 d²-R²、椭圆用隐式场（**无参数**型 non-SDF），且总是同时出 sdf+nonsdf | 给了 φ(1+αφ/h) 这类，但 α/形式未锁定；Stage0 建议先 SDF-only | 🟠 不一致 + 未锁定 |
| **多分辨率** | 单 rho（默认 64），一次一个 | 一个数据集混 `rhos=[256,266,276]` | 🟠 |
| **输出结构** | 单个合并 h5（含 split 列）+ CSV 报告；test 由 [testset.py](testset.py) 另出不重叠 SDF-only | train/val/test 分文件 + parquet + 结构化 reports/ | 🟠 |
| **center jitter** | ±0.5h（`center_min/max = 0.5 ± h/2`） | ±0.25h | 🟢 小 |
| **投影质量** | mpmath 高精度 + 硬 gate（残差≤1e-12，距离≤coarse_min），失败直接 raise | Newton max_iter=30/tol=1e-12，失败则拒绝+计数 | 🟢 现状更严 |

**结论**：当前代码已经把"几何精度 / D4 / sign-flip / 投影 gate / 防泄漏 split"这些**底座**做得很扎实，
但**分层逻辑（log-bin η、椭圆按 η_max 生成、按局部 η 重 bin、shape×bin 平衡采样）几乎是从零开始**，
而且 27D vs 28D 是一个方向性分叉。所以这不是"改几行"，是"换设计"。

---

## 4. 这个计划没讲清楚的问题（核心交付）

> 标注：🔵 = **需要你拍板的决定**；⚪ = 合约自身的 spec gap（实现前必须补定义）。

### ① 🔵 27D 纯特征 vs 28D 残差路线 —— 方向性分叉
合约通篇是 **27D 纯特征 + 直接预测 hκ**，并且把"architecture replacement / new loss"明确划到 out-of-scope。
但当前代码是 **28D**（第 28 列 = `hk_central` 中心差分曲率），明显是为**残差学习**（预测 `hk_exact - hk_central`）准备的。
合约**完全没提**这个残差路线 —— 它既不在 in-scope 也不在 out-of-scope。
- **影响**：决定了 features 维度、HDF5 schema、target 定义、以及 Stage 0 是不是要把残差路线一起冻结。
- **要你定**：Stage 0 数据层是 (a) 回到 27D 纯 hκ，残差路线以后再说；还是 (b) 保留 28D，把 `hk_central` 当成合约的合法扩展。我倾向 **27D 为主、28D 作为可选列**（生成时多写一列不贵，下游想用就用），但这是你的路线决定。

### ② ⚪ 整个数据集的**绝对规模 N_train 没定义**
平衡公式 `N_train(s,k) = N_train·P(s)·(1/8)` 需要一个 N_train，但合约从头到尾没给总样本数目标。
config 里只给了"圆 40/bin/rho、椭圆 4/design-cell"——我算过这只决定 **geometry 数**（圆 960、椭圆 1536），
而每个 geometry 出多少 interface 节点是**变量**（大圆出 ~2πR/h 个节点，小圆出几个）。
- **影响**：没有 N_train 就没法定 quota，Gate 6（±10% 平衡）也无从验。
- **要补**：要么直接给 N_train 目标，要么给"每个 (shape,bin) 取多少 pack"的绝对数。

### ③ ⚪ η_min=0.008 与分辨率**强耦合**，但没写出来
我验证过：圆要达到 η=0.008 需要 R=125h，而接受条件 R≤0.5-2h，**只有 ρ≳255 才放得下**：
ρ=256 刚好（R=0.490 vs 上限 0.492），ρ=128 直接超界（R=0.984）。
合约选的 `rhos=[256,266,276]` 恰好是能塞下 η_min=0.008 的最小区间——**这很可能是故意的，但合约只字未提**。
- **影响**：任何人把 rho 调小、或想复用到 ρ=128 部署网格，B0 bin 会大面积被拒甚至为空，且不同 rho 的 bin 可行性不一致。
- **要补**：要么写明"η_min 随 rho 走"（像现状那样 `η_min=h/r_max`），要么写死"本数据集只对 ρ≥256 有效"。

### ④ ⚪ 椭圆"按 η_max 生成、按全局 η_min 拒绝" → **低曲率弧被系统性丢弃**
合约说椭圆局部 η ∈ [η_max·q³, η_max] 且按局部 η_i 重 bin。但全局 η_min=0.008 的地板会把
`η_max·q³ < 0.008` 的那些点（短轴附近的**低曲率平坦弧**）整段拒掉。我算过：
- q=0.5 时，所有 η_max < 0.064 的椭圆（≈ B0–B4！）都会丢掉短轴段；
- q=0.65 → 阈值 0.029（B0–B2）；q=0.8 → 0.016（B0–B1）；q=0.9 → 0.011（B0）。
- **影响**：低 η_bin（B0–B2）几乎拿不到椭圆贡献，只能靠"大圆"和"高 η_max·高 q 椭圆的短轴"来填——
  与"shape 30:70 + 每个 bin 均衡"的目标**直接冲突**，而且短轴弧本来是界面上弧长占比最大的部分。
- 另外 Gate 4 允许局部 η 到 η_max+1e-10，但 §4.3 又说 >η_max 拒绝——边界处两条规则**互相打架**。
- **要补**：明确低曲率弧被丢是否可接受；或对椭圆放开 η_min 下限（只对圆用 0.008 地板）。

### ⑤ ⚪ 平衡采样的**单位、算法、几何多样性、随机种子**全未定义 → 不可复现 + 可能塌缩
合约把 balance 描述为"选 pack"，但一个 pack（一个物理节点）展开后是 16 或 32 行，
quota 到底数 **pack 还是展开后的 sample**？没说。更要命的是**选哪些 pack 没定义**：
- 同一个大圆能提供上百个**几乎完全相同**（target 一样、stencil 只差 D4）的节点。若某 bin 的 quota 主要从一两个大圆里填，train 多样性会塌缩到极少数几何。
- 合约只约束 shape×η_bin，**没有"bin 内跨 geometry 均匀"或"每个 geometry 最多取 N 个节点"的约束**。
- 选择算法（随机子采样？前 N 个？最远点？）和**采样用的随机种子**都没固定 → Gate 不过的复现性（§18.3 要求精确可复现）自相矛盾。
- **要补**：定 quota 单位、选择算法、per-geometry 节点上限、独立的 sampling seed。

### ⑥ ⚪ val/test 走自然分布 → **高曲率评估被低曲率淹没**，且缺 per-bin 指标
§12.4/Task7 说只平衡 train，val/test 自然分布。但自然分布被"大形状的低曲率节点"主导——
这正是合约 Failure Mode 3 警告的情形，只不过发生在**评估端**：test MSE 会很好看，但高曲率（B6/B7，正是 CFD 欠分辨界面最关心的）样本极少。
- **影响**：用 test MSE 当指标会系统性高估模型在高曲率上的能力。
- **要补**：要么 test 也按 bin 平衡，要么强制**输出 per-η_bin 的分桶指标**（合约现在只要求 η 直方图，不要求分桶误差）。

### ⑦ ⚪ 椭圆 split 分层在小样本下**取整 / 空层未定义**
椭圆 split key = (η_max_bin, q, ψ) = 8×4×4 = **128 个 strata**。config 给每 cell 4 个几何 ×3 rho = 12 个，
70/15/15 → val=1.8、test=1.8，**四舍五入后某些 cell 可能 0 个 val 或 0 个 test**。
- **影响**：分层 split 退化、某些 (q,ψ,bin) 组合在 val/test 完全缺席。
- **要补**：小 strata 的取整规则 / 最小计数保证（现状代码靠"只按 shape 分层"绕过了这个问题）。

### ⑧ 🔵⚪ non-SDF 规则没锁定，且与现有实现**不一致**，是否进 Stage 0 也不清
§9.2 给了 `φ(1+αφ/h)` 这类畸变但"α 小、可控"——**α 和函数形式都没定**，还说"如果项目已有 non-SDF 规则就用现成的"。
而现有代码的 non-SDF 是**代数场**（圆 d²-R²、椭圆隐式 u²/a²+v²/b²-1），是另一种东西（无 α 参数）。
同时 §9.1 和 config `sdf_only_first:true` 暗示 Stage 0 可能**先只做 SDF**，但现状代码总是同时出 sdf+nonsdf。
- **要你定/要补**：(a) Stage 0 是否包含 non-SDF；(b) 若包含，用现有代数场还是合约的 φ(1+αφ/h)，α 取多少。

### ⑨ 🔵 多分辨率 `[256, 266, 276]` 的选择很可疑
三个 rho 只差 ±8%、且 266/276 非 2 的幂、也不是常见部署网格。要么是想测 h-不变性（那为何区间这么窄、不取 128/256/512），
要么是**笔误**。结合问题③，256 是能塞下 η_min=0.008 的下限，266/276 只是多一点余量。
- **要你定**：到底要哪几个 rho？是不是 `[128, 256, 512]` 之类更有意义。

### ⑩ ⚪ 法向 stencil footprint 与"界面邻接"定义不一致；边界层排除没写
§7.1 用 3×3（±1）定义 interface-adjacent，但算 9 个点的 normal 要对每个 stencil 点再 ±1 → 实际要 **±2** 的数据。
合约只说"所有 stencil 值在域内"，没定义 normal 的 footprint，也没说要排除几层边界。
现状代码是对的（[geometry_core.py:67](../geometry_core.py) 排了 2 层边界），但合约这一处欠定义，照抄会越界。

---

## 5. 需要你拍板的几件事（决定后我就能动手）

1. **27D 还是 28D**（问题①）——残差路线进不进 Stage 0 数据层？
2. **Stage 0 要不要 non-SDF**（问题⑧）——只做 SDF 先跑通，还是 SDF+non-SDF 一起；non-SDF 用现有代数场还是合约公式？
3. **rho 取哪几个**（问题⑨）——`[256,266,276]` 照抄，还是改成你真正想要的部署/收敛网格？
4. **数据集规模 N_train**（问题②）——给个目标总量，我才能定 quota。
5. **椭圆低曲率弧丢弃**（问题④）是否可接受 / 椭圆是否单独放开 η_min。

剩下的（③⑤⑥⑦⑩）属于 spec gap，我可以按"安全默认值"先实现并在报告里写明，你 review 时再调；
但上面这 5 个是方向性的，建议你先定。

---

## 6. 我的建议落地顺序（待你确认方向后）

```
Step A: 重写 config —— 固定 [0.008,0.5] log-8-bin、q/ψ 离散集、rhos、N_train、shape 权重
Step B: 重写 blueprint 生成 —— 圆 R=h/η(log)，椭圆 a=h/(η_max q²)，q/ψ 网格
Step C: 椭圆局部 η_i 重 bin + 全局 η 地板策略（问题④的决定）
Step D: geometry-level 分层 split（含小 strata 取整规则）
Step E: shape×η_bin 平衡采样（定 quota 单位/选择算法/per-geom 上限/seed）
Step F: augmentation 展开（D4×sign，可选 SDF/non-SDF）——复用现有已验证代码
Step G: 输出 train/val/test 分文件 + per-bin 报告 + leakage 硬 gate
Step H: 跑 7 个 acceptance gate + dry-run 审计
```

底座（D4、sign-flip、椭圆投影、interface 提取、leakage 检查）现有代码能直接复用，
真正要重写的是 **A–E**（分层与采样这一层）。
