# Part 2 · DCTS v2.1 实施计划（已拍板锁定）

> 方法：**DCTS — Dimensionless-Curvature Targeted Sampling**
> 状态：**v2.1 FULLY LOCKED**（用户 2026-06-23 全部拍板，含 §7 法向=analytic）。本文是写代码的唯一依据。
> 历史：v1 差距分析 [PLAN.md](PLAN.md)（已弃）；v2 审查见 git 历史。
> **所有决定已锁死，无未决项。可直接开工。**

---

## 0. 已锁定的决定（source of truth）

| 项 | v2.1 锁定值 | 来源 |
|---|---|---|
| **分辨率** | **不分 rho**，只生成 **1 份 canonical 无量纲 local-patch 数据集**；rho 仅作 metadata label | 用户①（实测 4 份相同，差 3.9e-15）|
| **坐标系** | 全程 h 归一化：x̃=x/h，φ̃=φ/h，**令 h=1 ⇒ η=\|κ\|，R=1/η，a=1/(η_max q²)** | D1 |
| **曲率变量** | η = \|h·κ_exact\| | DCTS |
| **范围** | **η_min=0.006，η_max=2/3**（≈0.6666667）；不摇摆成 0.004 | 用户② |
| **目标密度** | log-uniform，用 **100 个 log 等距 fine-bin** 近似 | DCTS |
| **采样单位** | base pack | DCTS |
| **预算** | train 100k / val 20k / test 20k packs（每 fine-bin 1000/200/200）| v2 §4.5 |
| **形状权重** | circle:ellipse = **30:70**（每 fine-bin：train 300/700）| v2 §4.6 |
| **圆生成** | **direct-by-fine-bin-quota，无候选池、无 per-geometry cap** | 用户 D2 |
| **椭圆生成** | **local-η-conditioned**（不是纯 arc-length）：η_max log 扫描 + 按 local η(t) 选 t-段 + 段内 arc-length 采样 | 用户 D3 |
| **椭圆 q / ψ** | q∈{0.50,0.65,0.80,0.90}，ψ∈{0,π/12,π/6,π/4} | v2 §9.1 |
| **椭圆 per-geometry cap** | train 24 / val 12 / test 12，**按 (split,shape,fine_bin,geometry_id) 计**（见 §3.4）| v2 §12 |
| **特征** | 27D = [φ9/h, nx9, ny9]；**hk_central + residual 作 auxiliary 存盘** | v2 §6 |
| **目标** | y = h·κ_exact | v2 §6 |
| **法向** | **analytic（locked, A）**：圆 n=(x−c)/‖x−c‖、椭圆投影点法向；Stage0 不混入 FD 退化。`normal_source` 留作后续 ablation 轴 | 用户③（§7）|
| **augmentation** | D4×sign，**做成 config flag**，Stage0 默认 enabled | 用户 D5 |
| **SDF** | Stage0 **SDF-only**；non-SDF（φ+0.5·φ\|φ\|/h）留给 Stage1 | v2 §15 |
| **椭圆投影** | **float64 deterministic Newton**（tol 1e-12, max_iter 30），**不用 mpmath**；target κ 由已知 t 解析 | 用户 D4/⑥ |
| **split** | split-specific 生成 seed（合成几何天然不重叠）+ leakage 硬 gate | v2 §13.2 |

---

## 1. DCTS 最终规格（无量纲，h=1）

每个 base pack = 一个界面邻接节点的局部 5×5 `φ/h` 补丁。

- **fine-bin 边**：`e_j = 0.006·(2/3 / 0.006)^(j/100)`, j=0..100（101 个边，最后一个 bin 含右端点）。
- **coarse regime**：100 fine-bin 合成 8 段，仅用于出图，不参与训练配额。
- **patch 几何**（锁定定义）：节点 (i,j)∈{-2,-1,0,1,2}²（5×5），中心节点 (0,0)；
  特征用内 3×3 {-1,0,1}²；外圈仅作法向的 ±1 差分 footprint。
- **target 密度实现**：每个 (split, shape, fine_bin) 填到固定 quota → 整体 log-uniform。

---

## 2. 圆生成（D2：direct-by-fine-bin-quota）

圆是常曲率、零拒绝、η 可精确控制 ⇒ 不走候选池、不需要 cap、不需要 post-balance。

```
for split in (train, val, test):
  for j in range(100):                       # 每个 fine-bin
    for _ in range(circle_quota[split]):     # train=300, val/test=60
      eta   ~ LogUniform(e_j, e_{j+1})
      theta ~ U(0, 2π)
      d0    ~ U(-0.5, 0.5)                    # 单位 h
      R = 1/eta
      n = (cosθ, sinθ);  c = -(d0+R)·n
      phi[i,j] = ||(i,j) - c|| - R   for (i,j) in 5×5
      target_hk = eta                        # = h·κ, 解析精确
      pack_id = f"{split}:circle:{j}:{k}"
```
- §20.2 gate：每个圆 |η - 1/R| < 1e-12（解析必过）。
- 中心节点 φ=d0，|d0|<0.5 ⇒ 3×3 必有变号（无需拒绝）。
- ⚠️ 高 η（≳0.5）时圆心奇点进 patch → 见 §7 法向问题。

---

## 3. 椭圆生成（D3：local-η-conditioned）

最终 balancing 变量是 **local η_i**，所以候选生成也围绕 local η(t) 来填，而不是寄望 arc-length 自然填满
（实测纯 arc-length 下尖端只占 9.5%–18.5%，高 bin 填不满）。

### 3.1 单个椭圆的 t→η 表
```
给定 (q, ψ, η_max):
  a = 1/(η_max·q²);  b = q·a                  # 单位 h
  t_table = linspace(0, 2π, dense)            # e.g. 4096
  eta(t)  = a·b / (a²sin²t + b²cos²t)^1.5     # = h·κ(t), 复用 ellipse_hkappa_from_theta
  arclen(t) 累积表（用于段内 arc-length 采样）
```
local η 范围 = [η_max·q³, η_max]。该椭圆**只能**贡献落在此范围内的 fine-bin。

### 3.2 填某个目标 fine-bin j
```
要让 η(t) 能达到 bin j（中心≈η*）:  需 η_max·q³ ≤ η* ≤ η_max
  ⇒ 对每个 q，在 η_max ∈ [η*, η*/q³] ∩ [0.006, 2/3] 内 log 扫描若干 η_max
for 每个候选 (q, ψ, η_max):
  选出 eta(t) ∈ [e_j, e_{j+1}) 的 t-段
  段内按 arc-length 均匀采 ≤ cap 个点
  每个点: 选界面点 x⊥(t) → 法向 n(t) → d0~U(-0.5,0.5) → x0 = x⊥ + d0·n
          对 5×5 各节点用 float64 投影求最近点 → signed dist → φ/h
          target_hk = h·κ(t)   （已知 t，解析）
round-robin 跨不同 (q,ψ,η_max) 椭圆取点，直到 bin j 的椭圆 quota 满
```

### 3.3 投影（D4：不用 mpmath）
- target 曲率**不投影**（t 已知）。
- 仅 5×5 的其余 24 节点求 φ：用 `project_theta_to_axis_aligned_ellipse`（float64, tol 1e-12, max_iter 30）+ **coarse-global 兜底**（高 η 凹侧越过 evolute 时保证取全局最近点）。
- §20.3 gate：|η_max - 1/(a q²)| < 1e-12。

### 3.4 per-geometry cap 语义（要点）
cap 按 **(split, shape, fine_bin, geometry_id)** 计（依 v2 §20.5 测试）：一个椭圆横跨多个 bin，
对**每个** bin 最多贡献 cap 个点。候选池规模由**最高的 fine-bin 决定**（η_max 窗口窄、尖端稀），
所以池子要按最难填的高 bin 放大。geometry_id = 一个 (q,ψ,η_max) 实例。

---

## 4. 特征 / 目标 / 法向

```
features27 = [φ9/h, nx9, ny9]      # 内 3×3；法向 = ANALYTIC（见 §7，已锁）
target_hk  = h·κ_exact             # 解析（圆 1/R；椭圆已知 t 的 κ(t)）
# auxiliary（存盘，非默认训练目标）:
hk_central          = 中心差分曲率(φ9/h)      # 复用 central_difference_hkappa_from_phi9_float64
target_residual_hk  = hk_exact - hk_central
```
- **法向 = analytic（locked）**：圆 `n=(node−c)/‖node−c‖`；椭圆 = 各节点投影到椭圆后的单位外法向（投影已为求 φ 算过，法向顺带得到）。`normal_source="analytic"` 写进盘。
- hk_central 是 auxiliary，其在高 η 的噪声不影响主线（残差路线是 Stage4）。

---

## 5. augmentation / split / 输出 / gate（锁定，从简）

- **augmentation**（D5）：`d4_sign.enabled: true`（默认），但**可配置关闭**供 Stage2 做 no-aug vs D4×sign。
  - 复用 [generate.py](generate.py) 的 `transform_d4_features28` / `transform_sign_flip_features28` / `_D4_SIGMAS`。
  - 1 train pack → 16 行；val/test 默认只存 canonical（aug 副本可选、打标、不进主指标）。
  - 文档措辞要改：**D4 现在是 equivariance/对称正则，不再是"补取向覆盖"**（连续 θ 已覆盖取向）。
- **split**：三套独立 seed 生成 train/val/test 合成几何；仍跑 leakage 硬 gate（geometry_id / pack_id 不跨 split）。
- **输出**（无量纲，去掉 SDF-only 下多余字段）：
  ```
  processed/{train,val,test}.h5 :
    /features27[N,27] /target_hk[N,1] /hk_central[N,1] /target_residual_hk[N,1]
    /eta[N,1] /fine_bin[N] /coarse_regime[N]
    /pack_id /geometry_id /shape /h(=1) /d4_id /sign_id /sdf_mode(="sdf") /normal_source
  raw/{circle,ellipse}_candidates.parquet, blueprints/, reports/
  ```
- **gate**：v2 §19 的 Gate1–10 全实现，先对 smoke 跑全绿再放大。
  追加 **Gate11（mandatory 诊断）**：即使用 analytic 法向，也从同一 phi5×5 算 FD ‖∇φ‖，每 fine-bin 报告 <0.5/<0.2/<0.1 占比 + medial-axis 邻近度统计 → `reports/gate11_normal_degeneration.csv`，见 §7。

---

## 6. 构建顺序 + 复用映射

**复用（别重写）**：D4/sign transforms、`central_difference_hkappa_from_phi9_float64`（[generate.py](generate.py)）；
`ellipse_hkappa_from_theta`、`project_theta_to_axis_aligned_ellipse`、`project_theta_scalar`（coarse 兜底）、
`build_ellipse_nonsdf`（[geometry_core.py](../geometry_core.py)）；io 的 HDF5/CSV helper（[io.py](io.py)，改 schema）。

**新写**（建议 `train_generate/part2/dcts/`）：`config.py / bins.py / patch.py / circle.py / ellipse.py / select.py / augment.py / write.py / reports.py / gates.py / __main__.py`。

**顺序**（每步独立验证）：
```
A config+bins      → fine-edge 单测(len=101,端点,单增)
B patch+circle     → 圆一致性 1e-12；§7 法向退化率报告
C ellipse          → η_max 一致性；验证 local-η-conditioned 能覆盖全部 100 bin（含最高 bin）
D select           → leakage + cap 单测；round-robin 命中 quota
E augment(复用)     → D4 重算 hk_central <1e-10、sign-flip 精确
F write+reports     → 出 smoke(train 10k packs)+全报告
G gates            → smoke 跑 Gate1–11 全绿 → 放大 main(100k)
```

---

## 7. 法向 = analytic（已锁，选项 A）+ Gate11 强制诊断

**决定（用户拍板）**：Stage0 用 **analytic 法向**。Stage0 的目的是隔离「规定的无量纲曲率分布」的效果，
**不**把 SDF medial-axis 附近的有限差分法向退化混进来。η_max=2/3 不下调。

- 圆：`n = (node − c)/‖node − c‖`。
- 椭圆：节点投影到椭圆后的单位外法向。
- 曲率 target 也解析（圆 1/R；椭圆已知 t 的 κ(t)）。
- ⇒ **Stage0 = analytic SDF + analytic normal + analytic curvature**，全解析、零 FD 退化。

**为什么仍要诊断**（实测，v1/v2 都没提）：高 η 时解析 SDF 的奇点（圆心 / 椭圆 evolute，距离≈1.5h）进入 5×5 patch，
**如果**用 FD 法向会被打坏。随机 (θ,d₀) 下 min‖∇φ‖_FD（干净≈2.0）：η=0.5→2.5%、0.6→10.3%、**2/3→15.1%** 的 patch 退化到 <0.5（min 见 5.9e-3）。
Stage0 选 A 已规避它进入特征，但**必须把它量化记录**，以支撑后续法向 ablation 的解读。

### Gate11（mandatory diagnostics，即使用 analytic 法向也要算）
1. **从同一份 phi5×5 计算 FD 梯度模退化**——即便特征用的是 analytic 法向，也照样算一遍 FD `‖∇φ‖`，作为「若改用 FD 会有多坏」的诊断基线。
2. **每个 fine-bin 报告** min‖∇φ‖_FD 落在 **<0.5 / <0.2 / <0.1** 的样本占比（三档）。
3. **报告 medial-axis / 曲率中心邻近度统计**——每个节点到圆心 / 椭圆 evolute 的最近距离分布，**高 η bin 重点**。
- 输出 `reports/gate11_normal_degeneration.csv`（按 fine_bin × shape）。非硬失败 gate，是必出诊断表。

### normal_source = 后续 ablation 轴（不在 Stage0 实现，但 schema 预留）
```
normal_source ∈ { analytic | finite_difference_from_phi | deployment_operator_mycs }
```
Stage0 固定 `analytic`，写进每行 schema 的 `normal_source` 字段。FD / mycs 一致性作为**独立 ablation**，
与 SDF/non-SDF（Stage1）、合成-phase-vs-部署-phase（疑点④）并列——同属「训练/部署一致性」问题，连着 [[plic-sdf-needs-mycs]]（部署需 mycs 法向）。

---

## 8. 其余我会按默认处理的小项（不阻塞，smoke 后你可调）

1. **高-bin 椭圆多样性天然窄**：η→2/3 的椭圆样本都是"尖端≈R=1.5h 的圆 + 曲率梯度"，30:70 在顶 bin 主要买到"类圆"patch。→ 默认接受，但 reports 里**每 bin 报告椭圆 distinct (q,ψ,η_max) 数**让它可见。
2. **极端 bin 缺额策略**：高 bin 若填不满 700 椭圆，默认 `replacement=false` + 记录 deficiency + `allow_deficiency` flag（不静默 oversample）。
3. **"生成到填满"循环的可复现性**：seed = f(split_seed, fine_bin, round)，确定性。
4. **coarse regime** 100→8 的 12/13 不均匀分段：仅出图，无所谓。

---

## 9. 一句话

> 生成 1 份无量纲 DCTS 数据集：圆按 fine-bin 配额直接生成、椭圆按 local-η 条件采样，
> 100 个 log fine-bin post-balance 到 log-uniform，**analytic SDF + analytic 法向 + analytic 曲率**，
> D4×sign（可关）在 split 后展开；Gate1–11 全绿（Gate11 必出法向退化诊断）→ 先 smoke 后 main。
> **所有决定已锁，可直接开工。**
