# Stationary-Ellipse L8 (256×256) — WSL → Mac Transfer Runbook

**TL;DR (给 Codex-on-WSL):** 代码用 git 同步；**结果数据不走 git**，打成一个 `.tar.gz` 离线传回 Mac。你只需在 WSL 上跑 `pack_...` 脚本产出合格包，然后把包挪到 Mac。Mac 端由另一位 agent 跑 `import_...` 完成合并/出图/验收。

---

## 0. 前提：先同步代码（这是 git 唯一的用途）

```bash
cd <repo-root-on-WSL>
git pull                      # 确保 scripts/ 与 Mac 一致（当前分支：MLP-field）
git status --short            # 应无本地改动冲突
```

激活环境（本仓库用 conda `pinn`）：

```bash
conda activate pinn           # 或者 export PYTHON_BIN=<pinn 的 python 绝对路径>
export PYTHON_BIN="${PYTHON_BIN:-python}"
```

---

## 1. 打包（run 已完成 → 只需这一步）

椭圆 L8 已经跑完，raw 输出应在
`cfd_applications_cleanroom/results/raw/stationary_ellipse/stationary_ellipse_curvature_process_E{1,2}_*/{NN_DISABLE,NN_PROBE_ONLY,NN27_RAW,NN27_D4}_L8/`。

```bash
cd <repo-root-on-WSL>
bash cfd_applications_cleanroom/scripts/pack_stationary_ellipse_l8_results.sh
```

该脚本会：`audit` → 列出 8 行完整 L8 路径 → 打 `tar.gz` → **自校验**。

> 若想从零重跑再打包，用一条龙：
> `bash cfd_applications_cleanroom/scripts/run_and_pack_stationary_ellipse_l8_wsl.sh`
> （= preflight → run → audit → pack；可用 `MAX_JOBS=<n>` 控并发，默认 4）

### 合格判据（必须看到）

打包结尾会调用 `validate_stationary_ellipse_l8_package.py`，**必须打印**：

```
PASS package=.../stationary_ellipse_l8_results_<UTC>.tar.gz
PASS stationary_ellipse_l8_package_complete=8/8
```

校验器硬性检查：正好 **8** 行 = E1/E2 × {`NN_DISABLE`,`NN_PROBE_ONLY`,`NN27_RAW`,`NN27_D4`}，每个目录含 `summary.json` + `surface_tension_trace.csv`，且每个 `summary.json` 满足
`method` 匹配 & `level==8` & `reached_final_time==true` & `final_tau>=0.999`。
**看到 `FAIL ...` 就是有行没跑到终态 —— 不要传，先补跑。**

---

## 2. 产物形式

在 `cfd_applications_cleanroom/results/transfer/` 下生成（`<UTC>` = `date -u +%Y%m%dT%H%M%SZ`）：

| 文件 | 内容 |
| --- | --- |
| `stationary_ellipse_l8_results_<UTC>.tar.gz` | 8 个完整 L8 结果目录 + WSL 后台日志 + manifest + paths |
| `stationary_ellipse_l8_results_<UTC>.manifest.txt` | 包内路径清单 + 打包时间/repo 根 |
| `stationary_ellipse_l8_results_<UTC>.paths.txt` | 8 行完整结果目录相对路径 |

`results/transfer/` **已被 gitignore** —— 不要 `git add`，不要 commit 这个包。

---

## 3. 传回 Mac（唯一人工步骤，离线，不走 git）

把 `.tar.gz` 挪到 **Mac 上同一相对路径**：

```bash
# 示例（scp / rsync / 共享盘 / U 盘均可）
scp cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_<UTC>.tar.gz \
    <mac-host>:~/research/PINN/cfd_applications_cleanroom/results/transfer/
```

**可选但推荐**：两端各算一次 sha256 核对跨线完整性（脚本本身只做结构校验，不校验 checksum）：

```bash
shasum -a 256 cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_<UTC>.tar.gz
```

---

## 4. Mac 端会做什么（契约，供你了解，不必你执行）

```bash
bash cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh \
  cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_<UTC>.tar.gz
```

= 验包 → 解包进 `results/raw` → `merge_stationary_ellipse_summary.py` 从 raw 重建 summary（L6/L7 raw 已在 Mac，L8 从包里来）→ finalize 渲染 6 张 process plate → `verify_stationary_ellipse_delivery.py` 交付门：

```
PASS stationary_ellipse_process_ready=24/24
PASS stationary_ellipse_diagnostic_ready=12/12
PASS stationary_ellipse_l8_complete=8/8
PASS stationary_ellipse_process_png_plates=6/6
```

随后走 report source-archive（`tem/report_source_archive/`）：生成椭圆 `source.csv.gz`+`_summary.md`(sha256) → 过 redraw 字节级 gate → **只把 `_summary.md` 提交 git**（`source.csv.gz` 超 100MB，gitignore，外部备份）。

---

## 5. 只提交这些（你在 WSL commit 的东西）

- ✅ 代码/脚本改动、本 runbook
- ❌ **不提交** `results/transfer/*.tar.gz`、`results/raw/**`、`results/source_data/**`、`summary.csv`（全部 gitignore / 走离线包）

---

### Checklist（贴回给 Mac 端 agent）

- [ ] `git pull` 完成，分支 `MLP-field`
- [ ] `pack_...sh` 输出 `PASS ...8/8`
- [ ] 包名 `stationary_ellipse_l8_results_<UTC>.tar.gz`，sha256 = `<填>`
- [ ] 已 scp 到 Mac 的 `results/transfer/`
