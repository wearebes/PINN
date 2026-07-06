from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.flower import evaluate_flower
from testdata_generate.config import TestDataConfig, make_scenarios_for_rho_model
from testdata_generate.generate import build_flower_phi0, build_grid, generate_test_data
from testdata_generate.reinit import LevelSetReinitializer


DEFAULT_RESOLUTIONS = (64, 128, 256, 512)
DEFAULT_TEST_ITERS = tuple(range(1, 31))
DEFAULT_PROCESS_STEPS = (0, 15, 30)
FAMILIES = ("smooth", "acute")


@dataclass(frozen=True)
class PairSpec:
    train_rho: int
    test_rho: int
    model_path: Path
    data_path: Path


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _parse_positive_int_tuple(raw: str) -> tuple[int, ...]:
    values = _parse_int_tuple(raw)
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"Expected positive integers, got {values}.")
    return values


def _model_path(train_rho: int) -> Path:
    return Path("out") / str(train_rho) / f"baseline_{train_rho}_hgradient.pt"


def _data_path(output_dir: Path, test_rho: int) -> Path:
    return output_dir / "test_data" / f"flower_rho{test_rho}_hgradient_iters1-30.h5"


def _family_from_case_label(case_label: str) -> str:
    text = str(case_label)
    if "_" not in text:
        return text
    return text.rsplit("_", 1)[0]


def _dataset_matches(path: Path, *, rho: int, test_iters: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            required = {"features", "phi9", "xy", "hkappa_target", "case_id", "iter", "rho_model", "h"}
            if missing := required - set(handle.keys()):
                print(f"[data-check] {path} missing fields: {sorted(missing)}")
                return False
            if int(handle.attrs.get("feature_version", -1)) != 2:
                return False
            if int(handle.attrs.get("feature_dim_raw", -1)) != 27:
                return False
            if str(handle.attrs.get("feature_order", "")) != "phi9+nx9+ny9":
                return False
            if not bool(handle.attrs.get("scale_h", False)):
                return False
            if tuple(sorted({int(item) for item in handle["iter"][:].reshape(-1)})) != tuple(test_iters):
                return False
            if tuple(sorted({int(item) for item in handle["rho_model"][:].reshape(-1)})) != (int(rho),):
                return False
            if int(handle["features"].shape[1]) != 27:
                return False
    except OSError:
        return False
    return True


def _ensure_flower_dataset(
    *,
    path: Path,
    rho: int,
    test_iters: tuple[int, ...],
    overwrite: bool,
) -> Path:
    if not overwrite and _dataset_matches(path, rho=rho, test_iters=test_iters):
        print(f"[data] reuse rho={rho}: {path}")
        return path
    print(f"[data] generate rho={rho}: {path}")
    cfg = TestDataConfig(
        rho_model=int(rho),
        requested_rho_model=int(rho),
        test_iters=tuple(test_iters),
        scale_h=True,
        augment_gradient=True,
    )
    return generate_test_data(cfg, output=path)


def _setup_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save_figure(fig: Any, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "tiff": stem.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    return {key: str(value.resolve()) for key, value in outputs.items()}


def _flower_reinitializer() -> LevelSetReinitializer:
    return LevelSetReinitializer(
        indexing="xy",
        cfl=0.5,
        eps_weno=1.0e-6,
        eps_sign_factor=2.5,
        sign_mode="dynamic_phi",
        time_order=3,
        space_order=5,
    )


def _evaluate_pair(spec: PairSpec, *, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = evaluate_flower(
        dataset_path=spec.data_path,
        model_path=spec.model_path,
        normalization_csv_path=None,
        device=device,
        angle_bin_deg=30.0,
    )
    rows: list[dict[str, Any]] = []
    for case in result["cases"]:
        family = _family_from_case_label(str(case["case_label"]))
        numeric = case["numeric_vs_analytic"]
        model = case["model_vs_analytic"]
        rows.append({
            "train_rho": int(spec.train_rho),
            "test_rho": int(spec.test_rho),
            "family": family,
            "case_label": str(case["case_label"]),
            "iter": int(case["iter"]),
            "sample_count": int(case["sample_count"]),
            "numeric_mse": float(numeric["mse"]),
            "numeric_mae": float(numeric["mae"]),
            "numeric_maxae": float(numeric["maxae"]),
            "model_mse": float(model["mse"]),
            "model_mae": float(model["mae"]),
            "model_maxae": float(model["maxae"]),
            "model_path": str(spec.model_path.resolve()),
            "data_path": str(spec.data_path.resolve()),
        })
    summary = {
        "train_rho": int(spec.train_rho),
        "test_rho": int(spec.test_rho),
        "sample_count": int(result["sample_count"]),
        "failed_case_count": int(result["failed_case_count"]),
        "feature_version": int(result["feature_version"]),
        "raw_feature_dim": int(result["raw_feature_dim"]),
        "model_input_dim": int(result["model_input_dim"]),
        "normalization_source": str(result["normalization_source"]),
        "model_mse": float(result["model_vs_analytic"]["mse"]),
        "model_mae": float(result["model_vs_analytic"]["mae"]),
        "model_maxae": float(result["model_vs_analytic"]["maxae"]),
        "numeric_mse": float(result["numeric_vs_analytic"]["mse"]),
        "numeric_mae": float(result["numeric_vs_analytic"]["mae"]),
        "numeric_maxae": float(result["numeric_vs_analytic"]["maxae"]),
        "model_path": str(spec.model_path.resolve()),
        "data_path": str(spec.data_path.resolve()),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return path


def _summarize_family_step_curves(
    rows: list[dict[str, Any]],
    family: str,
    *,
    train_rho: int | None = None,
    test_rho: int | None = None,
) -> dict[str, np.ndarray]:
    summary_rows = [
        row for row in rows
        if row["family"] == family
        and (train_rho is None or int(row["train_rho"]) == int(train_rho))
        and (test_rho is None or int(row["test_rho"]) == int(test_rho))
    ]
    if not summary_rows:
        raise ValueError(
            f"No flower-step rows available for family={family!r}, "
            f"train_rho={train_rho}, test_rho={test_rho}."
        )

    iters = np.asarray(sorted({int(row["iter"]) for row in summary_rows}), dtype=np.int64)
    numeric_median: list[float] = []
    numeric_q25: list[float] = []
    numeric_q75: list[float] = []
    model_median: list[float] = []
    model_q25: list[float] = []
    model_q75: list[float] = []
    pair_count: list[int] = []
    for step in iters:
        step_rows = [row for row in summary_rows if int(row["iter"]) == int(step)]
        numeric_values = np.asarray([float(row["numeric_mse"]) for row in step_rows], dtype=np.float64)
        model_values = np.asarray([float(row["model_mse"]) for row in step_rows], dtype=np.float64)
        numeric_q25.append(float(np.percentile(numeric_values, 25.0)))
        numeric_median.append(float(np.percentile(numeric_values, 50.0)))
        numeric_q75.append(float(np.percentile(numeric_values, 75.0)))
        model_q25.append(float(np.percentile(model_values, 25.0)))
        model_median.append(float(np.percentile(model_values, 50.0)))
        model_q75.append(float(np.percentile(model_values, 75.0)))
        pair_count.append(len({(int(row["train_rho"]), int(row["test_rho"])) for row in step_rows}))

    return {
        "iter": iters,
        "pair_count": np.asarray(pair_count, dtype=np.int64),
        "numeric_median": np.asarray(numeric_median, dtype=np.float64),
        "numeric_q25": np.asarray(numeric_q25, dtype=np.float64),
        "numeric_q75": np.asarray(numeric_q75, dtype=np.float64),
        "model_median": np.asarray(model_median, dtype=np.float64),
        "model_q25": np.asarray(model_q25, dtype=np.float64),
        "model_q75": np.asarray(model_q75, dtype=np.float64),
    }


def _plot_family_summary(
    *,
    family: str,
    rows: list[dict[str, Any]],
    output_stem: Path,
    train_rho: int | None = None,
    test_rho: int | None = None,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    summary = _summarize_family_step_curves(rows, family, train_rho=train_rho, test_rho=test_rho)
    family_rows = [
        row for row in rows
        if row["family"] == family
        and (train_rho is None or int(row["train_rho"]) == int(train_rho))
        and (test_rho is None or int(row["test_rho"]) == int(test_rho))
    ]
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in family_rows:
        by_pair.setdefault((int(row["train_rho"]), int(row["test_rho"])), []).append(row)

    fig, ax = plt.subplots(figsize=(3.55, 2.55), constrained_layout=True)
    for pair_rows in by_pair.values():
        series = sorted(pair_rows, key=lambda item: int(item["iter"]))
        x = [int(item["iter"]) for item in series]
        ax.plot(x, [float(item["numeric_mse"]) for item in series], color="0.58", linestyle="--", linewidth=0.55, alpha=0.22)
        ax.plot(x, [float(item["model_mse"]) for item in series], color="#4C78A8", linewidth=0.55, alpha=0.18)

    x_summary = summary["iter"]
    ax.fill_between(
        x_summary,
        summary["numeric_q25"],
        summary["numeric_q75"],
        color="0.65",
        alpha=0.18,
        linewidth=0.0,
    )
    ax.fill_between(
        x_summary,
        summary["model_q25"],
        summary["model_q75"],
        color="#4C78A8",
        alpha=0.18,
        linewidth=0.0,
    )
    ax.plot(x_summary, summary["numeric_median"], color="0.35", linestyle="--", linewidth=1.35)
    ax.plot(x_summary, summary["model_median"], color="#2F5F98", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlim(float(x_summary.min()) - 0.5, float(x_summary.max()) + 4.5)
    ax.set_xlabel("reinit step")
    ax.set_ylabel("MSE")
    ax.grid(True, which="major", axis="y", color="0.88", linewidth=0.55)
    ax.grid(True, which="minor", axis="y", color="0.93", linewidth=0.35)
    label_x = float(x_summary.max()) + 0.7
    pair_total = int(np.max(summary["pair_count"]))
    numeric_label = "FD" if pair_total == 1 else "FD median"
    model_label = "NN" if pair_total == 1 else "NN median"
    ax.text(label_x, float(summary["numeric_median"][-1]), numeric_label, color="0.25", ha="left", va="center")
    ax.text(label_x, float(summary["model_median"][-1]), model_label, color="#2F5F98", ha="left", va="center")
    if train_rho is not None and test_rho is not None:
        ax.set_title(f"{family.capitalize()} flower: train rho={train_rho}, test rho={test_rho}")
    else:
        ax.set_title(f"{family.capitalize()} flower: summary across {pair_total} resolution pairs")
    return _save_figure(fig, output_stem)


def _plot_family_grid(
    *,
    family: str,
    rows: list[dict[str, Any]],
    train_resolutions: tuple[int, ...],
    test_resolutions: tuple[int, ...],
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["family"] == family:
            by_pair.setdefault((int(row["train_rho"]), int(row["test_rho"])), []).append(row)

    fig, axes = plt.subplots(
        len(train_resolutions),
        len(test_resolutions),
        figsize=(7.1, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(train_resolutions), len(test_resolutions))
    for i, train_rho in enumerate(train_resolutions):
        for j, test_rho in enumerate(test_resolutions):
            ax = axes[i, j]
            series = sorted(by_pair.get((train_rho, test_rho), []), key=lambda item: int(item["iter"]))
            if not series:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            x = [int(item["iter"]) for item in series]
            y_model = [float(item["model_mse"]) for item in series]
            y_numeric = [float(item["numeric_mse"]) for item in series]
            ax.plot(x, y_numeric, color="0.45", linestyle="--", linewidth=1.1, label="FD")
            ax.plot(x, y_model, color="#4C78A8", linewidth=1.35, label="NN")
            ax.set_yscale("log")
            ax.grid(True, which="major", axis="y", color="0.88", linewidth=0.55)
            ax.grid(True, which="minor", axis="y", color="0.93", linewidth=0.35)
            ax.set_title(fr"train $\rho={train_rho}$, test $\rho={test_rho}$")
            if i == len(train_resolutions) - 1:
                ax.set_xlabel("reinit step")
            if j == 0:
                ax.set_ylabel("MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(f"{family.capitalize()} flower: baseline_hgradient across resolution pairs", y=1.045)
    return _save_figure(fig, output_stem)


def _plot_reinit_process(
    *,
    family: str,
    rho: int,
    steps: tuple[int, ...],
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    scenarios = make_scenarios_for_rho_model(int(rho))
    scenario = next(item for item in scenarios if item.experiment_type == family)
    X, Y, h = build_grid(scenario.L, scenario.N)
    phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
    phi = phi0.astype(np.float64, copy=True)
    reinitializer = _flower_reinitializer()
    snapshots: dict[int, np.ndarray] = {0: phi0.copy()}
    for iteration in range(1, max(steps) + 1):
        phi = reinitializer.reinitialize(phi, h, 1)
        if iteration in steps:
            snapshots[int(iteration)] = phi.copy()

    fig, axes = plt.subplots(1, len(steps), figsize=(7.1, 2.25), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    contour_levels = h * np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float64)
    for ax, step in zip(axes, steps, strict=True):
        current = snapshots[int(step)]
        ax.contour(X, Y, current, levels=contour_levels, colors="red", linewidths=0.55, linestyles="solid")
        ax.contour(X, Y, current, levels=[0.0], colors="red", linewidths=1.0, linestyles="solid")
        ax.contour(X, Y, phi0, levels=[0.0], colors="black", linewidths=1.35, linestyles="solid")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {step}")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("0.2")
    fig.suptitle(f"{family.capitalize()} flower level-set isolines under reinitialization", y=1.14)
    fig.text(
        0.5,
        1.045,
        "increasing amount of reinitialization steps",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.33, 1.005),
            (0.67, 1.005),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="black",
            clip_on=False,
        )
    )
    return _save_figure(fig, output_stem)


def _plot_levelset_isolines(
    *,
    family: str,
    rho: int,
    steps: tuple[int, ...],
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scenarios = make_scenarios_for_rho_model(int(rho))
    scenario = next(item for item in scenarios if item.experiment_type == family)
    X, Y, h = build_grid(scenario.L, scenario.N)
    phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
    phi = phi0.astype(np.float64, copy=True)
    reinitializer = _flower_reinitializer()
    snapshots: dict[int, np.ndarray] = {0: phi0.copy()}
    for iteration in range(1, max(steps) + 1):
        phi = reinitializer.reinitialize(phi, h, 1)
        if iteration in steps:
            snapshots[int(iteration)] = phi.copy()

    levels = h * np.arange(-5, 6, dtype=np.float64)
    fig, axes = plt.subplots(1, len(steps), figsize=(7.1, 2.65), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.045, top=0.755, wspace=0.06)
    axes = np.asarray(axes).reshape(-1)
    for ax, step in zip(axes, steps, strict=True):
        current = snapshots[int(step)]
        ax.contour(
            X,
            Y,
            current,
            levels=levels,
            colors="#d62728",
            linewidths=0.72,
            linestyles="solid",
            antialiased=True,
        )
        ax.contour(X, Y, phi0, levels=[0.0], colors="black", linewidths=1.3, linestyles="solid")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {step}", pad=5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("0.2")

    handles = [
        Line2D([0], [0], color="#d62728", lw=1.0, label=r"$\phi = k\Delta x,\ k=-5,\ldots,5$"),
        Line2D([0], [0], color="black", lw=1.3, label="analytic interface"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.875), frameon=False)
    fig.suptitle(f"{family.capitalize()} flower level-set isolines", y=0.985)
    return _save_figure(fig, output_stem)


def _gradient_norm_centered(phi: np.ndarray, h: float) -> np.ndarray:
    gx = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2.0 * float(h))
    gy = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2.0 * float(h))
    return np.sqrt(gx * gx + gy * gy)


def _plot_sdf_error_heatmap(
    *,
    family: str,
    rho: int,
    steps: tuple[int, ...],
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    scenarios = make_scenarios_for_rho_model(int(rho))
    scenario = next(item for item in scenarios if item.experiment_type == family)
    X, Y, h = build_grid(scenario.L, scenario.N)
    phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
    phi = phi0.astype(np.float64, copy=True)
    reinitializer = _flower_reinitializer()
    snapshots: dict[int, np.ndarray] = {0: phi0.copy()}
    for iteration in range(1, max(steps) + 1):
        phi = reinitializer.reinitialize(phi, h, 1)
        if iteration in steps:
            snapshots[int(iteration)] = phi.copy()

    errors: dict[int, np.ma.MaskedArray] = {}
    finite_values: list[np.ndarray] = []
    for step in steps:
        current = snapshots[int(step)]
        sdf_error = _gradient_norm_centered(current, h) - 1.0
        band = np.abs(current[1:-1, 1:-1]) <= 6.0 * h
        masked = np.ma.masked_where(~band, sdf_error)
        errors[int(step)] = masked
        if masked.count() > 0:
            finite_values.append(np.asarray(masked.compressed(), dtype=np.float64))
    if not finite_values:
        raise ValueError("No near-interface values are available for SDF-error heatmap.")
    robust = float(np.percentile(np.abs(np.concatenate(finite_values)), 98.0))
    vmax = max(0.02, min(robust, 1.0))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, len(steps), figsize=(7.1, 2.45), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    extent = [
        float(X[1:-1, 1:-1].min()),
        float(X[1:-1, 1:-1].max()),
        float(Y[1:-1, 1:-1].min()),
        float(Y[1:-1, 1:-1].max()),
    ]
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white", alpha=1.0)
    image = None
    for ax, step in zip(axes, steps, strict=True):
        current = snapshots[int(step)]
        image = ax.imshow(
            errors[int(step)],
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.contour(X, Y, current, levels=[0.0], colors="black", linewidths=1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {step}")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("0.2")
    if image is None:
        raise ValueError("SDF-error heatmap image was not created.")
    cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.86, pad=0.02)
    cbar.set_label(r"$|\nabla\phi| - 1$")
    fig.suptitle(f"{family.capitalize()} flower near-interface SDF error", y=1.04)
    return _save_figure(fig, output_stem)


def _compute_sdf_error_curve(
    *,
    family: str,
    rho: int,
    max_step: int,
) -> list[dict[str, Any]]:
    scenarios = make_scenarios_for_rho_model(int(rho))
    scenario = next(item for item in scenarios if item.experiment_type == family)
    X, Y, h = build_grid(scenario.L, scenario.N)
    phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
    phi = phi0.astype(np.float64, copy=True)
    reinitializer = _flower_reinitializer()
    rows: list[dict[str, Any]] = []
    for step in range(0, int(max_step) + 1):
        current = phi0 if step == 0 else phi
        abs_error = np.abs(_gradient_norm_centered(current, h) - 1.0)
        band = np.abs(current[1:-1, 1:-1]) <= 6.0 * h
        values = np.asarray(abs_error[band], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No near-interface SDF-error values for {family} step {step}.")
        rows.append({
            "family": family,
            "rho": int(rho),
            "step": int(step),
            "sample_count": int(values.size),
            "mean_abs_sdf_error": float(np.mean(values)),
            "max_abs_sdf_error": float(np.max(values)),
        })
        if step < int(max_step):
            phi = reinitializer.reinitialize(phi, h, 1)
    return rows


def _write_sdf_error_curve_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_csv(
        path,
        rows,
        ["family", "rho", "step", "sample_count", "mean_abs_sdf_error", "max_abs_sdf_error"],
    )


def _plot_sdf_error_curve(
    *,
    family: str,
    rows: list[dict[str, Any]],
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No SDF-error curve rows are available.")
    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    mean_values = np.asarray([float(row["mean_abs_sdf_error"]) for row in rows], dtype=np.float64)
    max_values = np.asarray([float(row["max_abs_sdf_error"]) for row in rows], dtype=np.float64)
    if np.any(mean_values <= 0.0) or np.any(max_values <= 0.0):
        raise ValueError("SDF-error curve values must be positive for log-scale plotting.")

    fig, ax = plt.subplots(figsize=(3.05, 2.55), constrained_layout=True)
    ax.plot(steps, max_values, color="#1f4aff", linewidth=1.25, label="max")
    ax.plot(steps, mean_values, color="#1f4aff", linewidth=1.15, linestyle=(0, (4.5, 4.5)), label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("steps")
    ax.set_ylabel(r"$\left||\nabla\phi|-1\right|$")
    ax.set_title(f"{family.capitalize()} flower SDF-error decay", fontsize=8.5, pad=5)
    ax.grid(True, which="major", axis="y", color="0.88", linewidth=0.55)
    ax.grid(True, which="minor", axis="y", color="0.93", linewidth=0.35)
    ax.legend(loc="lower left", handlelength=1.6)
    return _save_figure(fig, output_stem)


def _validate_outputs(
    *,
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    train_resolutions: tuple[int, ...],
    test_resolutions: tuple[int, ...],
    test_iters: tuple[int, ...],
) -> dict[str, Any]:
    expected_pairs = len(train_resolutions) * len(test_resolutions)
    if len(summary_rows) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} summary rows, got {len(summary_rows)}.")
    for train_rho in train_resolutions:
        for test_rho in test_resolutions:
            for family in FAMILIES:
                subset = [
                    row for row in detail_rows
                    if int(row["train_rho"]) == train_rho
                    and int(row["test_rho"]) == test_rho
                    and row["family"] == family
                ]
                got_iters = tuple(sorted(int(row["iter"]) for row in subset))
                if got_iters != test_iters:
                    raise ValueError(
                        f"Missing detail rows for train={train_rho} test={test_rho} family={family}: "
                        f"expected iters {test_iters}, got {got_iters}."
                    )
                for row in subset:
                    values = [
                        row["numeric_mse"],
                        row["numeric_mae"],
                        row["numeric_maxae"],
                        row["model_mse"],
                        row["model_mae"],
                        row["model_maxae"],
                    ]
                    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                        raise ValueError(f"Non-finite metric row: {row}")
    return {
        "expected_pairs": expected_pairs,
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "families": list(FAMILIES),
        "test_iters": list(test_iters),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline_hgradient flower OOD tests across train/test resolutions and plot reinit-step curves."
    )
    parser.add_argument("--train-resolutions", type=_parse_positive_int_tuple, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--test-resolutions", type=_parse_positive_int_tuple, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--test-iters", type=_parse_int_tuple, default=DEFAULT_TEST_ITERS)
    parser.add_argument("--process-rho", type=int, default=256)
    parser.add_argument("--process-steps", type=_parse_int_tuple, default=DEFAULT_PROCESS_STEPS)
    parser.add_argument("--sdf-curve-max-step", type=int, default=50)
    parser.add_argument("--figure-train-rho", type=int, default=256)
    parser.add_argument("--figure-test-rho", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(Path("out") / "flower_hgradient_cross_resolution"))
    parser.add_argument("--overwrite-data", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    test_iters = tuple(sorted({int(item) for item in args.test_iters}))
    if not test_iters or min(test_iters) < 0:
        raise ValueError("--test-iters must contain non-negative integers.")
    process_steps = tuple(sorted({int(item) for item in args.process_steps}))
    if not process_steps or min(process_steps) < 0:
        raise ValueError("--process-steps must contain non-negative integers.")
    if int(args.sdf_curve_max_step) < 1:
        raise ValueError("--sdf-curve-max-step must be at least 1.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_resolutions = tuple(int(item) for item in args.train_resolutions)
    test_resolutions = tuple(int(item) for item in args.test_resolutions)
    figure_train_rho = int(args.figure_train_rho)
    figure_test_rho = int(args.figure_test_rho)

    data_paths = {
        rho: _ensure_flower_dataset(
            path=_data_path(output_dir, rho),
            rho=rho,
            test_iters=test_iters,
            overwrite=bool(args.overwrite_data),
        )
        for rho in test_resolutions
    }

    specs: list[PairSpec] = []
    for train_rho in train_resolutions:
        model_path = _model_path(train_rho)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing baseline_hgradient checkpoint: {model_path.resolve()}")
        for test_rho in test_resolutions:
            specs.append(PairSpec(train_rho, test_rho, model_path, data_paths[test_rho]))

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[eval {index:02d}/{len(specs):02d}] train rho={spec.train_rho} -> test rho={spec.test_rho}", flush=True)
        pair_rows, pair_summary = _evaluate_pair(spec, device=device)
        detail_rows.extend(pair_rows)
        summary_rows.append(pair_summary)
        print(
            f"    model MSE={pair_summary['model_mse']:.6e} "
            f"numeric MSE={pair_summary['numeric_mse']:.6e} "
            f"failed={pair_summary['failed_case_count']}",
            flush=True,
        )

    validation = _validate_outputs(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        train_resolutions=train_resolutions,
        test_resolutions=test_resolutions,
        test_iters=test_iters,
    )

    detail_csv = _write_csv(
        output_dir / "flower_step_metrics.csv",
        detail_rows,
        [
            "train_rho", "test_rho", "family", "case_label", "iter", "sample_count",
            "numeric_mse", "numeric_mae", "numeric_maxae",
            "model_mse", "model_mae", "model_maxae",
            "model_path", "data_path",
        ],
    )
    summary_csv = _write_csv(
        output_dir / "flower_pair_summary.csv",
        summary_rows,
        [
            "train_rho", "test_rho", "sample_count", "failed_case_count",
            "feature_version", "raw_feature_dim", "model_input_dim", "normalization_source",
            "numeric_mse", "numeric_mae", "numeric_maxae",
            "model_mse", "model_mae", "model_maxae",
            "model_path", "data_path",
        ],
    )

    figure_outputs = {
        family: _plot_family_summary(
            family=family,
            rows=detail_rows,
            train_rho=figure_train_rho,
            test_rho=figure_test_rho,
            output_stem=output_dir / f"flower_{family}_step_mse_train{figure_train_rho}_test{figure_test_rho}",
        )
        for family in FAMILIES
    }
    reinit_outputs = {
        family: _plot_reinit_process(
            family=family,
            rho=int(args.process_rho),
            steps=process_steps,
            output_stem=output_dir / f"reinitialization_process_{family}",
        )
        for family in FAMILIES
    }
    levelset_isoline_outputs = {
        family: _plot_levelset_isolines(
            family=family,
            rho=int(args.process_rho),
            steps=process_steps,
            output_stem=output_dir / f"levelset_isolines_{family}",
        )
        for family in FAMILIES
    }
    sdf_error_outputs = {
        family: _plot_sdf_error_heatmap(
            family=family,
            rho=int(args.process_rho),
            steps=process_steps,
            output_stem=output_dir / f"sdf_error_heatmap_{family}",
        )
        for family in FAMILIES
    }
    sdf_error_curve_rows_by_family = {
        family: _compute_sdf_error_curve(
            family=family,
            rho=int(args.process_rho),
            max_step=int(args.sdf_curve_max_step),
        )
        for family in FAMILIES
    }
    sdf_error_curve_csvs = {
        family: str(
            _write_sdf_error_curve_csv(
                output_dir / f"sdf_error_curve_{family}.csv",
                rows,
            ).resolve()
        )
        for family, rows in sdf_error_curve_rows_by_family.items()
    }
    sdf_error_curve_outputs = {
        family: _plot_sdf_error_curve(
            family=family,
            rows=rows,
            output_stem=output_dir / f"sdf_error_curve_{family}",
        )
        for family, rows in sdf_error_curve_rows_by_family.items()
    }

    manifest = {
        "task": "baseline_hgradient flower cross-resolution OOD evaluation",
        "train_resolutions": list(train_resolutions),
        "test_resolutions": list(test_resolutions),
        "figure_train_rho": figure_train_rho,
        "figure_test_rho": figure_test_rho,
        "test_iters": list(test_iters),
        "device": str(device),
        "detail_csv": str(detail_csv.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "data_paths": {str(rho): str(path.resolve()) for rho, path in data_paths.items()},
        "figure_outputs": figure_outputs,
        "reinitialization_outputs": reinit_outputs,
        "levelset_isoline_outputs": levelset_isoline_outputs,
        "sdf_error_heatmap_outputs": sdf_error_outputs,
        "sdf_error_curve_csvs": sdf_error_curve_csvs,
        "sdf_error_curve_outputs": sdf_error_curve_outputs,
        "validation": validation,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved detail metrics: {detail_csv}")
    print(f"Saved pair summary: {summary_csv}")
    print(f"Saved manifest: {manifest_path}")
    for family, outputs in figure_outputs.items():
        print(f"Figure {family}: {outputs['png']}")
    for family, outputs in reinit_outputs.items():
        print(f"Reinit {family}: {outputs['png']}")
    for family, outputs in levelset_isoline_outputs.items():
        print(f"Level-set isolines {family}: {outputs['png']}")
    for family, outputs in sdf_error_outputs.items():
        print(f"SDF error heatmap {family}: {outputs['png']}")
    for family, outputs in sdf_error_curve_outputs.items():
        print(f"SDF error curve {family}: {outputs['png']}")


if __name__ == "__main__":
    main()
