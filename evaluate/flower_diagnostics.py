"""
Standalone diagnostic tool for the flower petal evaluation.

Produces four output files in --output-dir:
  per_iter_metrics.csv  — per (rho/case/iter) row, all metrics
  correlations.csv      — Spearman + Pearson for key variable pairs
  diagnostic_plots.png  — three-panel plots per rho
  summary.md            — auto-generated trends + caveats

Usage:
    python -m evaluate.flower_diagnostics \\
        --output-dir <dir> [--device cpu|cuda] [--rho-models 256,266,276]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    central_difference_hkappa_from_phi9,
    compute_metrics,
    decode_phi9_to_patch,
    load_model_from_checkpoint,
    predict_hkappa_full_batch,
    resolve_feature_transform,
)
from evaluate.flower import load_flower_dataset
from testdata_generate.generate import (
    build_flower_phi0,
    build_grid,
    build_raw_features,
    find_projection_theta,
    hkappa_analytic,
    interface_indices,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA_DIR = _PROJECT_ROOT / "test_data"
_MODEL_DIR = _PROJECT_ROOT / "out"


# ---------------------------------------------------------------------------
# iter=0 generation (in-memory, no HDF5 write)
# ---------------------------------------------------------------------------

def _build_iter0_arrays(
    scenario_map: dict[int, dict[str, Any]],
    *,
    scale_h: bool,
) -> dict[str, np.ndarray]:
    """Generate iter=0 data (raw phi0, before any reinit) for each scenario.

    Uses find_projection_theta + hkappa_analytic for labeling — the same path as
    _append_case_iter in testdata_generate/generate.py — not the training-side
    compute_hkappa_targets() which only works for circles/ellipses.
    """
    phi9_list: list[np.ndarray] = []
    features_list: list[np.ndarray] = []
    xy_list: list[np.ndarray] = []
    hkappa_list: list[np.ndarray] = []
    case_id_list: list[np.ndarray] = []
    iter_list: list[np.ndarray] = []
    rho_model_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []

    for case_id in sorted(scenario_map.keys()):
        sc = scenario_map[case_id]
        a = float(sc["a"])
        b = float(sc["b"])
        p = int(sc["p"])
        L = float(sc["L"])
        N = int(sc["N"])
        rho_model = int(sc["rho_model"])

        X, Y, h = build_grid(L, N)
        phi0 = build_flower_phi0(X, Y, a, b, p)
        indices = interface_indices(phi0)
        if indices.size == 0:
            exp_id = sc.get("exp_id", case_id)
            print(f"    [iter=0] case={case_id} {exp_id}: no interface nodes, skipping")
            continue

        rows_idx = indices[:, 0]
        cols_idx = indices[:, 1]
        xy = np.column_stack((X[rows_idx, cols_idx], Y[rows_idx, cols_idx])).astype(np.float32)
        theta_proj = find_projection_theta(xy, a, b, p)
        phi9, features = build_raw_features(phi0, indices, scale_h=scale_h, h=h)
        hkappa = hkappa_analytic(theta_proj, h, a, b, p)
        count = indices.shape[0]

        phi9_list.append(phi9)
        features_list.append(features)
        xy_list.append(xy)
        hkappa_list.append(hkappa)
        case_id_list.append(np.full((count,), case_id, dtype=np.int16))
        iter_list.append(np.full((count,), 0, dtype=np.int16))
        rho_model_list.append(np.full((count,), rho_model, dtype=np.int16))
        h_list.append(np.full((count,), h, dtype=np.float32))
        exp_id = sc.get("exp_id", case_id)
        print(f"    [iter=0] case={case_id} {exp_id} samples={count}")

    def _cat(lst: list[np.ndarray], dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        if not lst:
            return np.zeros(shape, dtype=dtype)
        return np.concatenate(lst, axis=0).astype(dtype, copy=False)

    return {
        "phi9": _cat(phi9_list, np.float32, (0, 9)),
        "features": _cat(features_list, np.float32, (0, 9)),
        "xy": _cat(xy_list, np.float32, (0, 2)),
        "hkappa_target": _cat(hkappa_list, np.float32, (0,)),
        "case_id": _cat(case_id_list, np.int16, (0,)),
        "iter": _cat(iter_list, np.int16, (0,)),
        "rho_model": _cat(rho_model_list, np.int16, (0,)),
        "h": _cat(h_list, np.float32, (0,)),
    }


# ---------------------------------------------------------------------------
# SDF gradient-norm diagnostics (raw phi9, not features)
# ---------------------------------------------------------------------------

def _grad_norm_from_phi9(phi9: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-sample ||grad phi|| / h via central differences on the 3x3 stencil.

    phi9 must be the raw (unscaled) stencil values in distance units.
    h is per-sample grid spacing.
    Result should be ~1 for a proper SDF.
    """
    patch = decode_phi9_to_patch(phi9).astype(np.float64, copy=False)
    # patch[n, row, col] = phi[r + row-1, c + col-1], indices in {0,1,2}
    phi_row_diff = 0.5 * (patch[:, 2, 1] - patch[:, 0, 1])  # central diff across rows
    phi_col_diff = 0.5 * (patch[:, 1, 2] - patch[:, 1, 0])  # central diff across cols
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    return np.sqrt(phi_row_diff**2 + phi_col_diff**2) / h_arr


def _sdf_metrics(phi9: np.ndarray, h: np.ndarray) -> dict[str, float]:
    grad_norm = _grad_norm_from_phi9(phi9, h)
    err = np.abs(grad_norm - 1.0)
    return {
        "mean_abs_grad_norm_error": float(np.mean(err)),
        "p95_abs_grad_norm_error": float(np.percentile(err, 95)),
    }


def _stencil_metrics(phi9: np.ndarray, h: np.ndarray) -> dict[str, float]:
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    phi9_f = np.asarray(phi9, dtype=np.float64)
    center = phi9_f[:, 4]
    return {
        "mean_abs_phi_center_over_h": float(np.mean(np.abs(center) / h_arr)),
        "mean_abs_phi9_over_h": float(np.mean(np.abs(phi9_f) / h_arr[:, None])),
        "mean_phi9_std_over_h": float(np.mean(np.std(phi9_f, axis=1) / h_arr)),
    }


# ---------------------------------------------------------------------------
# Per-iter metric rows
# ---------------------------------------------------------------------------

def _compute_per_iter_metrics(
    arrays: dict[str, np.ndarray],
    prediction: np.ndarray,
    *,
    normalization_source: str,
    case_label_map: dict[int, str],
    rho_model: int,
) -> list[dict[str, Any]]:
    phi9 = np.asarray(arrays["phi9"], dtype=np.float32)
    hkappa_target = np.asarray(arrays["hkappa_target"], dtype=np.float64).reshape(-1)
    case_ids = np.asarray(arrays["case_id"]).reshape(-1)
    iterations = np.asarray(arrays["iter"]).reshape(-1)
    h_arr = np.asarray(arrays["h"], dtype=np.float32).reshape(-1)

    try:
        numeric_all = central_difference_hkappa_from_phi9(phi9)
        numeric_ok = True
    except ValueError as exc:
        print(f"  WARNING: central_difference_hkappa_from_phi9 failed: {exc}")
        numeric_all = None
        numeric_ok = False

    rows: list[dict[str, Any]] = []
    for case_id in sorted(int(v) for v in np.unique(case_ids)):
        case_mask = case_ids == case_id
        case_label = case_label_map.get(case_id, str(case_id))
        for iteration in sorted(int(v) for v in np.unique(iterations[case_mask])):
            mask = case_mask & (iterations == iteration)
            target_m = hkappa_target[mask]
            pred_m = prediction[mask]
            phi9_m = phi9[mask]
            h_m = h_arr[mask]

            row: dict[str, Any] = {
                "rho_model": rho_model,
                "case_label": case_label,
                "iter": iteration,
                "sample_count": int(np.sum(mask)),
                "normalization_source": normalization_source,
            }

            model_m = compute_metrics(pred_m, target_m)
            row["model_mse"] = model_m["mse"]
            row["model_mae"] = model_m["mae"]
            row["model_maxae"] = model_m["maxae"]

            if numeric_ok and numeric_all is not None:
                num_m = numeric_all[mask]
                num_metrics = compute_metrics(num_m, target_m)
                row["numeric_mse"] = num_metrics["mse"]
                row["numeric_mae"] = num_metrics["mae"]
                row["numeric_maxae"] = num_metrics["maxae"]
                mvn = compute_metrics(pred_m, num_m)
                row["model_vs_numeric_mse"] = mvn["mse"]
                row["model_vs_numeric_mae"] = mvn["mae"]
                row["model_vs_numeric_maxae"] = mvn["maxae"]
            else:
                for k in (
                    "numeric_mse", "numeric_mae", "numeric_maxae",
                    "model_vs_numeric_mse", "model_vs_numeric_mae", "model_vs_numeric_maxae",
                ):
                    row[k] = float("nan")

            row.update(_sdf_metrics(phi9_m, h_m))
            row.update(_stencil_metrics(phi9_m, h_m))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _compute_correlations(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scipy.stats import pearsonr, spearmanr

    PAIRS = [
        ("iter", "model_mae"),
        ("iter", "numeric_mae"),
        ("iter", "mean_abs_grad_norm_error"),
        ("mean_abs_grad_norm_error", "model_mae"),
        ("mean_abs_grad_norm_error", "numeric_mae"),
    ]

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = (int(row["rho_model"]), str(row["case_label"]))
        groups.setdefault(key, []).append(row)

    corr_rows: list[dict[str, Any]] = []
    for (rho, case_label), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda r: r["iter"])
        for x_key, y_key in PAIRS:
            xs = [float(r[x_key]) for r in group_rows]
            ys = [float(r[y_key]) for r in group_rows]
            has_nan = any(np.isnan(v) for v in xs + ys)
            if has_nan or len(xs) < 3:
                spear, pear = float("nan"), float("nan")
            else:
                spear = float(spearmanr(xs, ys)[0])
                pear = float(pearsonr(xs, ys)[0])
            corr_rows.append({
                "rho_model": rho,
                "case_label": case_label,
                "x_var": x_key,
                "y_var": y_key,
                "spearman_rho": spear,
                "pearson_r": pear,
                "n_points": len(xs),
            })
    return corr_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(metric_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = sorted({int(r["rho_model"]) for r in metric_rows})
    n_rhos = len(rhos)
    fig, axes = plt.subplots(n_rhos, 3, figsize=(15, 5 * n_rhos), squeeze=False)
    fig.suptitle("Flower Petal Diagnostics", fontsize=14)

    for row_idx, rho in enumerate(rhos):
        rho_rows = [r for r in metric_rows if r["rho_model"] == rho]
        cases = sorted({r["case_label"] for r in rho_rows})
        ax0, ax1, ax2 = axes[row_idx]

        for case in cases:
            case_rows = sorted([r for r in rho_rows if r["case_label"] == case], key=lambda r: r["iter"])
            iters = [r["iter"] for r in case_rows]
            ax0.plot(iters, [r["model_mae"] for r in case_rows], marker="o", label=f"{case} model")
            ax0.plot(iters, [r["numeric_mae"] for r in case_rows], marker="x", linestyle="--", label=f"{case} numeric")
        ax0.set_title(f"rho={rho}: MAE vs iter")
        ax0.set_xlabel("iter")
        ax0.set_ylabel("MAE")
        ax0.legend(fontsize=6)
        ax0.grid(True, alpha=0.3)

        for case in cases:
            case_rows = sorted([r for r in rho_rows if r["case_label"] == case], key=lambda r: r["iter"])
            iters = [r["iter"] for r in case_rows]
            ax1.plot(iters, [r["mean_abs_grad_norm_error"] for r in case_rows], marker="o", label=f"{case} mean")
            ax1.plot(iters, [r["p95_abs_grad_norm_error"] for r in case_rows], marker="x", linestyle="--", label=f"{case} p95")
        ax1.set_title(f"rho={rho}: grad norm error vs iter")
        ax1.set_xlabel("iter")
        ax1.set_ylabel("|grad_norm - 1|")
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)

        for case in cases:
            case_rows = [r for r in rho_rows if r["case_label"] == case]
            gn = [r["mean_abs_grad_norm_error"] for r in case_rows]
            mm = [r["model_mae"] for r in case_rows]
            nm = [r["numeric_mae"] for r in case_rows]
            ax2.scatter(gn, mm, marker="o", s=20, label=f"{case} model", alpha=0.7)
            ax2.scatter(gn, nm, marker="x", s=20, label=f"{case} numeric", alpha=0.7)
        ax2.set_title(f"rho={rho}: grad norm error vs MAE")
        ax2.set_xlabel("|grad_norm - 1| (mean)")
        ax2.set_ylabel("MAE")
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  plots -> {output_path}")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _trend_label(values: list[float], q: int) -> str:
    if len(values) < 2:
        return "insufficient data"
    early = float(np.mean(values[:q]))
    late = float(np.mean(values[-q:]))
    if late < early:
        return "decreasing"
    if late > early:
        return "increasing"
    return "flat"


def _make_summary(
    metric_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Flower Petal Diagnostic Summary\n\n")
    lines.append(
        "_Generated automatically by `evaluate/flower_diagnostics.py`. "
        "These are correlation-based findings — not causal proof._\n\n"
    )
    lines.append("## Caveats\n\n")
    lines.append(
        "- **Axis convention mismatch (known, unfixed)**: `train_generate` uses "
        '`np.meshgrid(..., indexing="ij")` while `testdata_generate` uses `indexing="xy"`. '
        "Both sides use identical `_PHI9_ROW_OFFSETS/_PHI9_COL_OFFSETS`, meaning the physical "
        "coordinate interpretation of the phi9 stencil differs between training and testing. "
        "Curvature is a scalar invariant, so the hkappa result is unaffected, but any "
        "spatially directional analysis should be treated with caution.\n"
    )
    lines.append(
        "- **Correlation, not causation**: lower grad norm error co-occurring with lower model "
        "error is consistent with the hypothesis that reinit moves inputs toward the training "
        "distribution, but this has not been experimentally isolated or controlled.\n"
    )
    lines.append(
        "- **Do not use historical SwanLab runs without auditing**: at least one known run "
        "(run-20260523_140349) has a config/model mismatch (rho256.h5 dataset paired with "
        "baseline_266.pt model). Only freshly generated local runs are reliable.\n"
    )
    lines.append(
        "- **numeric MAE direction varies by case**: numeric MSE often increases with iter, "
        "but numeric MAE is not universally monotone (e.g. acute_276 MAE decreases). "
        "Per-case trends are reported below; no universal direction is assumed.\n\n"
    )

    rhos = sorted({int(r["rho_model"]) for r in metric_rows})
    for rho in rhos:
        rho_rows = [r for r in metric_rows if r["rho_model"] == rho]
        cases = sorted({r["case_label"] for r in rho_rows})
        norm_source = rho_rows[0]["normalization_source"] if rho_rows else "unknown"

        lines.append(f"## rho={rho}\n\n")
        lines.append(f"normalization_source: `{norm_source}`\n\n")

        for case in cases:
            case_rows = sorted(
                [r for r in rho_rows if r["case_label"] == case], key=lambda r: r["iter"]
            )
            if not case_rows:
                continue
            iters = [r["iter"] for r in case_rows]
            model_maes = [r["model_mae"] for r in case_rows]
            numeric_maes = [r["numeric_mae"] for r in case_rows]
            grad_errs = [r["mean_abs_grad_norm_error"] for r in case_rows]
            q = max(1, len(case_rows) // 4)

            lines.append(f"### {case}\n\n")
            lines.append(f"- iter range: {min(iters)}-{max(iters)}, n_iters={len(iters)}\n")
            lines.append(
                f"- model MAE: {model_maes[0]:.4e} -> {model_maes[-1]:.4e} "
                f"(**{_trend_label(model_maes, q)}**)\n"
            )
            lines.append(
                f"- numeric MAE: {numeric_maes[0]:.4e} -> {numeric_maes[-1]:.4e} "
                f"(**{_trend_label(numeric_maes, q)}**)\n"
            )
            lines.append(
                f"- mean grad norm error: {grad_errs[0]:.4e} -> {grad_errs[-1]:.4e} "
                f"(**{_trend_label(grad_errs, q)}**)\n"
            )

            def _get_spear(x_var: str, y_var: str, _rho: int = rho, _case: str = case) -> str:
                v = next(
                    (cr["spearman_rho"] for cr in corr_rows
                     if cr["rho_model"] == _rho and cr["case_label"] == _case
                     and cr["x_var"] == x_var and cr["y_var"] == y_var),
                    float("nan"),
                )
                return f"{v:.3f}" if not np.isnan(v) else "nan"

            lines.append(
                f"- Spearman(iter, model_mae)={_get_spear('iter', 'model_mae')}, "
                f"Spearman(iter, numeric_mae)={_get_spear('iter', 'numeric_mae')}, "
                f"Spearman(grad_err, model_mae)={_get_spear('mean_abs_grad_norm_error', 'model_mae')}\n\n"
            )

    lines.append("---\n\n")
    lines.append(
        "_Model improvement with iter is an empirical observation from these experiments, "
        "not a theoretically guaranteed result for all smooth curves or all MLPs of this type._\n"
    )

    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"  summary -> {output_path}")


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        print(f"  csv (empty) -> {output_path}")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  csv -> {output_path}")


# ---------------------------------------------------------------------------
# Per-rho routine
# ---------------------------------------------------------------------------

def run_diagnostics_for_rho(
    rho_model: int,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    data_path = _TEST_DATA_DIR / f"rho{rho_model}.h5"
    model_path = _MODEL_DIR / f"baseline_{rho_model}.pt"
    print(f"\n[rho={rho_model}] data={data_path.name} model={model_path.name}")

    bundle = load_flower_dataset(data_path)
    arrays_existing = bundle["arrays"]
    attrs = bundle["attrs"]
    case_label_map = bundle["case_label_map"]
    scenario_map = bundle["scenario_map"]
    scale_h = bool(attrs.get("scale_h", False))

    print("  generating iter=0 data (in-memory)...")
    arrays_iter0 = _build_iter0_arrays(scenario_map, scale_h=scale_h)

    def _merge(key: str, dtype: Any) -> np.ndarray:
        a0 = np.asarray(arrays_iter0.get(key, np.zeros(0, dtype=dtype)), dtype=dtype)
        a1 = np.asarray(arrays_existing.get(key, np.zeros(0, dtype=dtype)), dtype=dtype)
        parts = [p for p in [a0, a1] if p.size > 0]
        return np.concatenate(parts, axis=0) if parts else np.zeros(a0.shape, dtype=dtype)

    merged: dict[str, np.ndarray] = {
        "phi9": _merge("phi9", np.float32),
        "features": _merge("features", np.float32),
        "hkappa_target": _merge("hkappa_target", np.float32),
        "case_id": _merge("case_id", np.int16),
        "iter": _merge("iter", np.int16),
        "rho_model": _merge("rho_model", np.int16),
        "h": _merge("h", np.float32),
    }

    print("  loading model and predicting...")
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    feature_transform, normalization_source = resolve_feature_transform(
        model_path=model_path,
        explicit_path=None,
        checkpoint_meta=checkpoint_meta,
    )
    features_merged = np.asarray(merged["features"], dtype=np.float32)
    prediction = predict_hkappa_full_batch(
        model, features_merged, transform=feature_transform, device=device
    )

    print("  computing per-iter metrics...")
    return _compute_per_iter_metrics(
        merged,
        prediction,
        normalization_source=normalization_source,
        case_label_map=case_label_map,
        rho_model=rho_model,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_rho_models(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flower petal diagnostic tool: per-iter SDF and error metrics."
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write diagnostic outputs.")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda).")
    parser.add_argument(
        "--rho-models",
        type=str,
        default="256,266,276",
        help="Comma-separated rho_model values (default: 256,266,276).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rho_models = _parse_rho_models(args.rho_models)

    all_metric_rows: list[dict[str, Any]] = []
    for rho in rho_models:
        rows = run_diagnostics_for_rho(rho, device=device)
        all_metric_rows.extend(rows)

    print("\n[diagnostics] computing correlations...")
    corr_rows = _compute_correlations(all_metric_rows)

    print("[diagnostics] writing outputs...")
    _write_csv(all_metric_rows, output_dir / "per_iter_metrics.csv")
    _write_csv(corr_rows, output_dir / "correlations.csv")
    _make_plots(all_metric_rows, output_dir / "diagnostic_plots.png")
    _make_summary(all_metric_rows, corr_rows, output_dir / "summary.md")

    print(f"\n[diagnostics] done. outputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
