# Experiment Report Figure Style Guide

本文件是 `Experiment/report` 图表的正式绘图规范。后续替换报告图时，优先遵守本文件；若具体脚本和本文件冲突，以本文件为准并同步修改脚本。

## 1. 适用范围

当前报告图集位于 `Experiment/report/`，包含：

| 文件 | 图表类型 | 角色 |
|---|---|---|
| `cross_resolution_mse.png` | quantitative line plot | 跨训练/测试分辨率 MSE 对比 |
| `cross_resolution_mae.png` | quantitative line plot | 跨训练/测试分辨率 MAE 对比 |
| `hk_mse.png` | quantitative line plot | 按 true curvature magnitude 分箱的 MSE |
| `hk_mae.png` | quantitative line plot | 按 true curvature magnitude 分箱的 MAE |
| `flower256.png` | quantitative line plot | flower rho=256 MSE step 曲线 |
| `flower256sdf.png` | quantitative line plot | flower rho=256 SDF/gradient consistency 曲线 |
| `flower_hot.png` | image plate + colorbar | flower 界面局部绝对误差空间分布 |
| `stationarybubble_64_process.png` | scatter + line, 2-panel | stationary bubble (L6/grid64) 曲率重建误差与 Ca 收敛过程 |

如果正文只选其中四张，仍从这组正式图中选择；不要另开一套颜色或手工改图。

## 2. 后端与导出

- 只使用 Python/matplotlib 路线生成论文图；不混用 R/ggplot 或 notebook 截图。
- `Experiment/report/` 只保留报告入口 PNG/CSV/MD；SVG/PDF/TIFF 等可编辑或印刷级导出保留在可复现源目录，不同步到本目录。
- SVG 必须保留可编辑文字：`svg.fonttype = "none"`。
- PDF 字体应保留 TrueType：`pdf.fonttype = 42`。
- 报告图必须从可复现输出整理到 `Experiment/report/`，不要手工截图、裁剪或重新另存。

## 3. 全局版式

- 背景：白色。
- 字体：sans-serif，优先 Arial/Helvetica，允许 DejaVu Sans fallback。
- 坐标轴：保留 left/bottom spine，关闭 top/right spine。
- legend：无边框，尽量短标签；线型含义优先由 legend 表示，颜色含义由 caption 或同一图内一致映射解释。
- 网格：默认关闭。只有读数困难时允许非常浅的 y-grid，但报告主图默认不用网格。
- marker：默认不用。线图依靠颜色和线型区分，避免 marker 造成顶刊图过重。
- 线宽：主线约 1.2-1.6；辅助/参考线不应比主线更重。

## 4. 主色板

| 语义 | 颜色 | Hex | 用途 |
|---|---|---:|---|
| acute / high-curvature family | deep blue | `#0F4D92` | flower acute；rho=256 主色 |
| smooth / lower-curvature family | muted red | `#B64342` | flower smooth；rho=512 对照色 |
| medium-resolution support | muted teal | `#42949E` | rho=128；辅助对照 |
| neutral support | neutral dark | `#4D4D4D` | rho=64；非方法主角 |
| reference / FD / analytic | near black | `#272727` | FD、reference、analytic outline |
| inactive neutral | neutral mid | `#767676` | 次要标注、低优先级元素 |

绿色只用于明确的 improvement/gain 标注，不用于普通方法类别；红色不默认表示失败，除非 caption 明确说明。

## 5. Flower 图规则

### `flower256.png`

- 颜色表示 flower family：
  - smooth: `#B64342`
  - acute: `#0F4D92`
- 线型表示方法：
  - NN: solid
  - FD: dashed
- legend 只解释线型：`NN`, `FD`。
- y 轴使用 log scale，标签为 `MSE`。

### `flower256sdf.png`

- 颜色同 flower family：
  - smooth: `#B64342`
  - acute: `#0F4D92`
- 线型表示统计量：
  - max: solid
  - mean: dashed
- legend 只解释线型：`max`, `mean`。
- y 轴使用 log scale，标签为 `||grad phi|-1|` 的数学排版版本。

### `flower_hot.png`

- 使用 sequential error colormap，不使用 diverging colormap。
- 当前正式方案为黄-橙-红序列，用于 absolute error。
- 黑色/深灰轮廓只作为几何边界和可读性辅助，不表示方法类别。
- colorbar 必须写清楚是 absolute error，并标明比较对象：`|h kappa_NN - h kappa_analytic|`。

## 6. Cross-resolution 图规则

### 颜色

训练分辨率颜色固定如下：

| rho_train | Hex |
|---:|---:|
| 64 | `#4D4D4D` |
| 128 | `#42949E` |
| 256 | `#0F4D92` |
| 512 | `#B64342` |
| FD | `#272727`, dashed |

### 版式

- x 轴：`Test resolution, rho_test`。
- y 轴：`MSE` 或 `MAE`。
- y 轴：log scale。
- legend 标题：`rho_train`。
- legend 标签：`64`, `128`, `256`, `512`, `FD`。
- 不使用 marker，不使用 grid。

## 7. h-kappa 分箱图规则

### 文件

- `hk_mse.png`: binned MSE versus true curvature magnitude.
- `hk_mae.png`: binned MAE versus true curvature magnitude.

### 颜色

沿用 cross-resolution 的训练分辨率颜色：

| rho_train | Hex |
|---:|---:|
| 64 | `#4D4D4D` |
| 128 | `#42949E` |
| 256 | `#0F4D92` |
| 512 | `#B64342` |
| FD | `#272727`, dashed |

### 版式

- x 轴：`|h kappa_analytic|`。
- y 轴：`MSE` 或 `MAE`。
- y 轴：log scale。
- legend 标题：`rho_train`。
- legend 标签：`64`, `128`, `256`, `512`, `FD`。
- 不使用 marker，不使用 grid。
- 不把 FD 写成 `central difference` 长标签；图注中解释 FD 即 central-difference baseline。

## 8. Stationary bubble 图规则

### 文件

- `stationarybuubble_64.png`：stationary curvature diagnostic（`stationary_curvature_process`，level 6 / grid_n 64）在 `t/T = 0, 1/3, 2/3, 1` 四个时刻的 angle-resolved `h*kappa`，以及同 run 的 `Ca_max` trace。文件名保留历史拼写。
- `stationarybubble_128.png`：同一 `stationary_curvature_process` / `NN27_RAW` / `CLSVOF-LS` host 路径，level 7 / grid_n 128，完整跑到 `t/T = 1`。
- `stationarybubble_256.png`：同一 `stationary_curvature_process` / `NN27_RAW` / `CLSVOF-LS` host 路径，level 8 / grid_n 256，完整跑到 `t/T = 1`。
- `stationarybubble_64_process.png`：stationary bubble case（`stationary_curvature_process`，level 6 / grid_n 64）在 `t/T = 0, 1/3, 2/3, 1` 四个时刻的气泡几何状态，1x4 small multiples。

这组图属于 CFD-injection（cfd_applications_cleanroom）实验族。

`stationary_curvature_process_20260703T052139Z` 曾启动 level 8 / 9（grid_n 256 / 512）补跑，但本地 full-time solver 推进过慢，中断时 level 8 仅到 `t/T≈0.0101`，未生成 manifest summary / final figure。该 partial raw bundle 不进入 `Experiment/report`，也不能作为正式图引用。正式 N256 使用 `stationary_curvature_process_20260703T074754Z/NN27_RAW_L8` 的完整 summary。N512 正在单独以 level 9 full-time run 补跑，完整 summary 生成后再进入 report。

### 颜色

| 语义 | Hex | 用途 |
|---|---:|---|
| 边界 / reference / analytic | `#272727` | 圆边界实线，沿用第 4 节 reference/FD/analytic 语义 |
| 填充 | `#42949E`（alpha 0.22） | 气泡内部浅色填充，从第 4 节 muted teal 扩展而来 |

### 版式

- 每个 panel 画解析圆（R=0.4，由 `hk_native` 反推网格间距核验一致），无 marker、无散点，因为这四个时刻的真实界面轮廓数据（VOF/level-set facet dump）不存在——pipeline 只落盘了界面带探针点（网格单元中心，用于挑出 `|d|/Δ≤1` 那圈，并非真实界面）和标量 Ca trace。画探针点会被误读成"界面形状"，所以改为直接画解析几何。
- 四个 panel 之间的圆理应看起来完全一样：这是诚实的结论（stationary bubble 的几何状态本来就不随时间变化），不要为了"看起来有信息量"而人为夸大或编造形变。
- 若要展示这个 case 真正随时间变化的量（曲率重建误差、Ca_max 收敛过程），用 pipeline 自带的诊断图：`cfd_applications_cleanroom/results/figures/stationary_curvature_process_20260702T121230Z_stationary_curvature_process_plate.png`，不要混进这张状态图里。
- `Ca_max` trace panel 的方法标签必须显式写出最终值，例如 `NN final Ca_max = 8.19e-06`，避免读者需要回查 raw summary 才知道收敛后的量级。

## 9. 可复现来源

`Experiment/report` 里的 PNG 是最终报告入口；同名 SVG/PDF 不进入本目录。需要可编辑或印刷级输出时，从下表的可复现源目录读取。当前同步关系如下：

| Report file | Reproducible source |
|---|---|
| `cross_resolution_mse.png` | `out/baseline_hgradient_cross_resolution/cross_resolution_mse.png` |
| `cross_resolution_mae.png` | `out/baseline_hgradient_cross_resolution/cross_resolution_mae.png` |
| `hk_mse.png` | `out/baseline_hgradient_cross_resolution/error_vs_hk_label_mse.png` |
| `hk_mae.png` | `out/baseline_hgradient_cross_resolution/error_vs_hk_label_mae.png` |
| `flower256.png` | `out/flower_hgradient_cross_resolution/research_figures/nature_style/flower_mse_train256_test256_cfl0p5_nature.png` |
| `flower256sdf.png` | `out/flower_hgradient_cross_resolution/research_figures/nature_style/flower_grad_error_train256_test256_cfl0p5_nature_redblue.png` |
| `flower_hot.png` | `out/flower_hgradient_cross_resolution/research_figures/nature_style/flower_nn_prediction_error_snapshots_train256_test256_cfl0p5_steps0-10-20-30_nature.png` |
| `stationarybuubble_64.png` | `cfd_applications_cleanroom/results/figures/stationary_curvature_process_20260702T121230Z_stationary_curvature_process_plate.png` |
| `stationarybubble_128.png` | `cfd_applications_cleanroom/results/figures/stationary_curvature_process_20260702T130840Z_stationary_curvature_process_plate.png` |
| `stationarybubble_256.png` | `cfd_applications_cleanroom/results/figures/stationary_curvature_process_20260703T074754Z_L8_only_stationary_curvature_process_plate.png` |
| `stationarybubble_64_process.png` | `tem/stationarybubble_report_figure/out/stationarybubble_64_process.png`（由 `tem/stationarybubble_report_figure/plot_stationarybubble_64_process.py` 生成，画解析几何，用 `cfd_applications_cleanroom/results/raw/stationary/stationary_curvature_process_20260702T121230Z/NN27_RAW_L6/curvature_process.csv` 里的 `grid_n`/`hk_native` 核验半径，不重跑 CFD） |

跨分辨率图可用已有 CSV 重绘，重绘后只把 PNG 放入 `Experiment/report/`：

```bash
python -m evaluate.baseline_hgradient_cross_resolution --plot-only
```

stationary bubble 图可用已有 source data 重绘，重绘后只把 PNG 放入 `Experiment/report/`：

```bash
python tem/stationarybubble_report_figure/plot_stationarybubble_64_process.py
```

## 10. 修改规则

- 新增图表时，先判断它是 quantitative line plot、image plate、heatmap、bar chart 还是 schematic/composite，再继承本文件的颜色语义。
- 同一方法或同一 family 不得在不同 panel 中换色。
- 同一颜色不得在同一组图中同时表示两种冲突语义。
- 如果必须增加新颜色，优先从 muted teal、neutral mid、neutral light 中扩展；不要引入高饱和默认色。
- 修改颜色、线型或输出来源时，必须同步更新本文件和相应绘图脚本。
