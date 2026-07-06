from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_RESOLUTIONS = (64, 128, 256, 512)
FLOWER_STYLE_TRAIN_RHO_PALETTE = {
    64: "#4D4D4D",
    128: "#42949E",
    256: "#0F4D92",
    512: "#B64342",
}
FLOWER_STYLE_FALLBACK_COLORS = ["#4D4D4D", "#42949E", "#0F4D92", "#B64342", "#767676", "#A8A8A8"]
FLOWER_STYLE_REFERENCE = "#272727"


@dataclass(frozen=True)
class EvalPaths:
    train_rho: int
    test_rho: int
    model_path: Path
    data_path: Path


def _parse_resolutions(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one resolution.")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"Resolutions must be positive, got {values}.")
    return values


def _default_model_path(train_rho: int) -> Path:
    return Path("out") / str(train_rho) / f"baseline_{train_rho}_hgradient.pt"


def _default_data_path(test_rho: int) -> Path:
    return Path("dataset") / str(test_rho) / f"{test_rho}_hgradient.h5"


def _iter_slices(n: int, chunk_size: int):
    for start in range(0, int(n), int(chunk_size)):
        stop = min(start + int(chunk_size), int(n))
        yield start, stop


def _require_split(handle: Any, split: str) -> Any:
    if split not in handle:
        raise ValueError(f"Dataset {handle.filename} does not contain split {split!r}.")
    group = handle[split]
    for key in ("features", "hkappa_target"):
        if key not in group:
            raise ValueError(f"Dataset {handle.filename} split {split!r} is missing {key!r}.")
    return group


def _scan_targets(
    data_paths: list[Path],
    split: str,
    chunk_size: int,
    *,
    collect_values: bool,
) -> dict[str, Any]:
    import h5py

    global_min = np.inf
    global_max = -np.inf
    global_positive_min = np.inf
    per_dataset: list[dict[str, Any]] = []
    collected: list[np.ndarray] = []
    for data_path in data_paths:
        with h5py.File(data_path, "r") as handle:
            group = _require_split(handle, split)
            targets = group["hkappa_target"]
            n = int(targets.shape[0])
            if n <= 0:
                raise ValueError(f"Dataset {data_path} split {split!r} is empty.")
            local_min = np.inf
            local_max = -np.inf
            local_positive_min = np.inf
            for start, stop in _iter_slices(n, chunk_size):
                values = np.asarray(targets[start:stop], dtype=np.float64).reshape(-1)
                if values.size == 0:
                    continue
                if not np.isfinite(values).all():
                    raise ValueError(f"Dataset {data_path} split {split!r} has non-finite hkappa_target values.")
                local_min = min(local_min, float(np.min(values)))
                local_max = max(local_max, float(np.max(values)))
                positive_values = values[values > 0.0]
                if positive_values.size:
                    local_positive_min = min(local_positive_min, float(np.min(positive_values)))
                if collect_values:
                    collected.append(values)
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
            global_positive_min = min(global_positive_min, local_positive_min)
            per_dataset.append({
                "dataset_path": str(data_path.resolve()),
                "n": n,
                "hk_min": local_min,
                "hk_max": local_max,
                "hk_positive_min": local_positive_min if np.isfinite(local_positive_min) else None,
            })
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Could not determine finite h*kappa target range.")
    if np.isclose(global_min, global_max):
        pad = max(1.0e-6, abs(global_min) * 1.0e-6)
        global_min -= pad
        global_max += pad
    result = {
        "hk_min": float(global_min),
        "hk_max": float(global_max),
        "hk_positive_min": float(global_positive_min) if np.isfinite(global_positive_min) else None,
        "per_dataset": per_dataset,
    }
    if collect_values:
        result["target_values"] = np.concatenate(collected).astype(np.float64, copy=False)
    return result


def _make_hk_edges(
    *,
    hk_min: float,
    hk_max: float,
    hk_positive_min: float | None,
    n_bins: int,
    bin_mode: str,
    target_values: np.ndarray | None,
) -> np.ndarray:
    if bin_mode == "linear":
        edges = np.linspace(float(hk_min), float(hk_max), int(n_bins) + 1, dtype=np.float64)
    elif bin_mode == "quantile":
        if target_values is None:
            raise ValueError("target_values are required for quantile h*kappa bins.")
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
        edges = np.quantile(np.asarray(target_values, dtype=np.float64), quantiles)
        edges = np.unique(edges.astype(np.float64, copy=False))
    elif bin_mode == "positive-quantile":
        if target_values is None:
            raise ValueError("target_values are required for positive-quantile h*kappa bins.")
        positive = np.asarray(target_values, dtype=np.float64)
        positive = positive[positive > 0.0]
        if positive.size == 0:
            raise ValueError("positive-quantile h*kappa bins require at least one positive target value.")
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
        edges = np.quantile(positive, quantiles)
        edges = np.unique(edges.astype(np.float64, copy=False))
    elif bin_mode == "positive-log":
        if hk_positive_min is None or not np.isfinite(hk_positive_min) or hk_positive_min <= 0.0:
            raise ValueError("positive-log h*kappa bins require at least one positive target value.")
        if hk_max <= hk_positive_min:
            raise ValueError(
                f"positive-log h*kappa bins need hk_max > hk_positive_min, got {hk_max} <= {hk_positive_min}."
            )
        edges = np.geomspace(float(hk_positive_min), float(hk_max), int(n_bins) + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported h*kappa bin mode {bin_mode!r}.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("h*kappa bin edges are not strictly increasing.")
    if edges.shape[0] < 9:
        raise ValueError(f"h*kappa binning produced only {edges.shape[0] - 1} bins; need at least 8.")
    return edges


def _update_bins(
    *,
    values: np.ndarray,
    squared_error: np.ndarray,
    abs_error: np.ndarray,
    edges: np.ndarray,
    sum_sq: np.ndarray,
    sum_abs: np.ndarray,
    sum_hk: np.ndarray,
    counts: np.ndarray,
) -> None:
    in_range = (values >= edges[0]) & (values <= edges[-1])
    if not np.any(in_range):
        return
    values = values[in_range]
    squared_error = squared_error[in_range]
    abs_error = abs_error[in_range]
    bin_index = np.searchsorted(edges, values, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    counts += np.bincount(bin_index, minlength=counts.shape[0]).astype(np.int64)
    sum_sq += np.bincount(bin_index, weights=squared_error, minlength=sum_sq.shape[0]).astype(np.float64)
    sum_abs += np.bincount(bin_index, weights=abs_error, minlength=sum_abs.shape[0]).astype(np.float64)
    sum_hk += np.bincount(bin_index, weights=values, minlength=sum_hk.shape[0]).astype(np.float64)


def _evaluate_pair(
    *,
    paths: EvalPaths,
    split: str,
    device: Any,
    chunk_size: int,
    hk_edges: np.ndarray,
    aggregate_bins: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    import h5py
    import torch

    from evaluate.shared import apply_feature_transform, load_model_from_checkpoint, resolve_feature_transform

    model, checkpoint_meta = load_model_from_checkpoint(paths.model_path, device=device)
    transform, transform_source = resolve_feature_transform(
        model_path=paths.model_path,
        explicit_path=None,
        checkpoint_meta=checkpoint_meta,
    )
    raw_feature_dim = int(transform["raw_feature_dim"])
    output_dim = int(transform["output_dim"])
    pair_counts = np.zeros(len(hk_edges) - 1, dtype=np.int64)
    pair_sum_sq = np.zeros(len(hk_edges) - 1, dtype=np.float64)
    pair_sum_abs = np.zeros(len(hk_edges) - 1, dtype=np.float64)
    pair_sum_hk = np.zeros(len(hk_edges) - 1, dtype=np.float64)

    total_count = 0
    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0

    agg = aggregate_bins[paths.train_rho]

    with h5py.File(paths.data_path, "r") as handle:
        group = _require_split(handle, split)
        features_ds = group["features"]
        target_ds = group["hkappa_target"]
        if int(features_ds.ndim) != 2:
            raise ValueError(f"{paths.data_path} split {split!r} features must be 2D, got {features_ds.shape}.")
        if int(features_ds.shape[1]) != raw_feature_dim:
            raise ValueError(
                f"Feature dimension mismatch for train rho={paths.train_rho}, test rho={paths.test_rho}: "
                f"dataset has {features_ds.shape[1]}D features but model transform expects {raw_feature_dim}D."
            )
        n = int(features_ds.shape[0])
        if int(target_ds.shape[0]) != n:
            raise ValueError(f"{paths.data_path} split {split!r} target length does not match features.")
        if n <= 0:
            raise ValueError(f"{paths.data_path} split {split!r} is empty.")

        for start, stop in _iter_slices(n, chunk_size):
            features = np.asarray(features_ds[start:stop], dtype=np.float32)
            target = np.asarray(target_ds[start:stop], dtype=np.float64).reshape(-1)
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite features in {paths.data_path} rows {start}:{stop}.")
            if not np.isfinite(target).all():
                raise ValueError(f"Non-finite targets in {paths.data_path} rows {start}:{stop}.")
            transformed = apply_feature_transform(features, transform)
            batch = torch.from_numpy(transformed).to(device)
            with torch.inference_mode():
                pred = model(batch).detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            diff = pred - target
            abs_err = np.abs(diff)
            sq_err = diff * diff
            total_count += int(target.shape[0])
            sum_sq += float(np.sum(sq_err, dtype=np.float64))
            sum_abs += float(np.sum(abs_err, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(abs_err)))

            _update_bins(
                values=target,
                squared_error=sq_err,
                abs_error=abs_err,
                edges=hk_edges,
                sum_sq=pair_sum_sq,
                sum_abs=pair_sum_abs,
                sum_hk=pair_sum_hk,
                counts=pair_counts,
            )
            _update_bins(
                values=target,
                squared_error=sq_err,
                abs_error=abs_err,
                edges=hk_edges,
                sum_sq=agg["sum_sq"],
                sum_abs=agg["sum_abs"],
                sum_hk=agg["sum_hk"],
                counts=agg["count"],
            )

    return {
        "train_rho": int(paths.train_rho),
        "test_rho": int(paths.test_rho),
        "split": split,
        "n": int(total_count),
        "mse": float(sum_sq / total_count),
        "mae": float(sum_abs / total_count),
        "maxae": float(max_abs),
        "model_path": str(paths.model_path.resolve()),
        "data_path": str(paths.data_path.resolve()),
        "model_type": str(checkpoint_meta["model_type"]),
        "transform_kind": str(transform["transform_kind"]),
        "transform_source": str(transform_source),
        "raw_feature_dim": raw_feature_dim,
        "model_input_dim": output_dim,
        "pair_bin_count": pair_counts,
        "pair_bin_sum_sq": pair_sum_sq,
        "pair_bin_sum_abs": pair_sum_abs,
        "pair_bin_sum_hk": pair_sum_hk,
    }


def _evaluate_central_difference(
    *,
    data_path: Path,
    test_rho: int,
    split: str,
    chunk_size: int,
    hk_edges: np.ndarray,
    aggregate_bins: dict[str, np.ndarray],
) -> dict[str, Any]:
    import h5py

    from evaluate.shared import central_difference_hkappa_from_phi9

    total_count = 0
    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0

    with h5py.File(data_path, "r") as handle:
        group = _require_split(handle, split)
        if "phi9" not in group:
            raise ValueError(f"Dataset {data_path} split {split!r} is missing 'phi9' for central difference.")
        phi9_ds = group["phi9"]
        target_ds = group["hkappa_target"]
        if int(phi9_ds.ndim) != 2 or int(phi9_ds.shape[1]) != 9:
            raise ValueError(f"{data_path} split {split!r} phi9 must have shape (N, 9), got {phi9_ds.shape}.")
        n = int(phi9_ds.shape[0])
        if int(target_ds.shape[0]) != n:
            raise ValueError(f"{data_path} split {split!r} target length does not match phi9.")
        if n <= 0:
            raise ValueError(f"{data_path} split {split!r} is empty.")

        for start, stop in _iter_slices(n, chunk_size):
            phi9 = np.asarray(phi9_ds[start:stop], dtype=np.float32)
            target = np.asarray(target_ds[start:stop], dtype=np.float64).reshape(-1)
            if not np.isfinite(phi9).all():
                raise ValueError(f"Non-finite phi9 in {data_path} rows {start}:{stop}.")
            if not np.isfinite(target).all():
                raise ValueError(f"Non-finite targets in {data_path} rows {start}:{stop}.")
            pred = central_difference_hkappa_from_phi9(phi9)
            diff = pred - target
            abs_err = np.abs(diff)
            sq_err = diff * diff
            total_count += int(target.shape[0])
            sum_sq += float(np.sum(sq_err, dtype=np.float64))
            sum_abs += float(np.sum(abs_err, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(abs_err)))
            _update_bins(
                values=target,
                squared_error=sq_err,
                abs_error=abs_err,
                edges=hk_edges,
                sum_sq=aggregate_bins["sum_sq"],
                sum_abs=aggregate_bins["sum_abs"],
                sum_hk=aggregate_bins["sum_hk"],
                counts=aggregate_bins["count"],
            )

    return {
        "method": "central_difference",
        "test_rho": int(test_rho),
        "split": split,
        "n": int(total_count),
        "mse": float(sum_sq / total_count),
        "mae": float(sum_abs / total_count),
        "maxae": float(max_abs),
        "data_path": str(data_path.resolve()),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "train_rho", "test_rho", "split", "n", "mse", "mae", "maxae",
        "model_type", "transform_kind", "transform_source", "raw_feature_dim",
        "model_input_dim", "model_path", "data_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_central_difference_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["method", "test_rho", "split", "n", "mse", "mae", "maxae", "data_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_hk_curve_csv(
    path: Path,
    *,
    hk_edges: np.ndarray,
    aggregate_bins: dict[int, dict[str, np.ndarray]],
) -> None:
    fieldnames = [
        "train_rho", "hk_left", "hk_right", "hk_center", "hk_mean", "count", "mse", "mae",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        centers = 0.5 * (hk_edges[:-1] + hk_edges[1:])
        for train_rho in sorted(aggregate_bins):
            agg = aggregate_bins[train_rho]
            count = agg["count"]
            for idx in range(count.shape[0]):
                n = int(count[idx])
                writer.writerow({
                    "train_rho": int(train_rho),
                    "hk_left": float(hk_edges[idx]),
                    "hk_right": float(hk_edges[idx + 1]),
                    "hk_center": float(centers[idx]),
                    "hk_mean": float(agg["sum_hk"][idx] / n) if n else "",
                    "count": n,
                    "mse": float(agg["sum_sq"][idx] / n) if n else "",
                    "mae": float(agg["sum_abs"][idx] / n) if n else "",
                })


def _write_central_difference_hk_curve_csv(
    path: Path,
    *,
    hk_edges: np.ndarray,
    aggregate_bins: dict[str, np.ndarray],
) -> None:
    fieldnames = ["method", "hk_left", "hk_right", "hk_center", "hk_mean", "count", "mse", "mae"]
    centers = 0.5 * (hk_edges[:-1] + hk_edges[1:])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        count = aggregate_bins["count"]
        for idx in range(count.shape[0]):
            n = int(count[idx])
            writer.writerow({
                "method": "central_difference",
                "hk_left": float(hk_edges[idx]),
                "hk_right": float(hk_edges[idx + 1]),
                "hk_center": float(centers[idx]),
                "hk_mean": float(aggregate_bins["sum_hk"][idx] / n) if n else "",
                "count": n,
                "mse": float(aggregate_bins["sum_sq"][idx] / n) if n else "",
                "mae": float(aggregate_bins["sum_abs"][idx] / n) if n else "",
            })


def _setup_matplotlib():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def _save_publication_figure(fig, stem: Path) -> dict[str, str]:
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


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for plot-only mode: {path.resolve()}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_cross_resolution_metric(
    *,
    rows: list[dict[str, Any]],
    central_difference_rows: list[dict[str, Any]],
    train_resolutions: tuple[int, ...],
    test_resolutions: tuple[int, ...],
    metric_key: str,
    y_label: str,
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    palette = FLOWER_STYLE_TRAIN_RHO_PALETTE
    fallback = FLOWER_STYLE_FALLBACK_COLORS
    metric_by_pair = {
        (int(row["train_rho"]), int(row["test_rho"])): float(row[metric_key])
        for row in rows
    }
    x_positions = np.arange(len(test_resolutions), dtype=float)

    fig, ax = plt.subplots(figsize=(3.54, 2.28), constrained_layout=True)
    for idx, train_rho in enumerate(train_resolutions):
        y = [metric_by_pair[(train_rho, test_rho)] for test_rho in test_resolutions]
        color = palette.get(train_rho, fallback[idx % len(fallback)])
        ax.plot(
            x_positions,
            y,
            linewidth=1.25,
            color=color,
            solid_capstyle="round",
            label=str(train_rho),
        )

    cd_by_test = {int(row["test_rho"]): float(row[metric_key]) for row in central_difference_rows}
    cd_y = [cd_by_test[test_rho] for test_rho in test_resolutions]
    ax.plot(
        x_positions,
        cd_y,
        color=FLOWER_STYLE_REFERENCE,
        linestyle=(0, (3.8, 2.0)),
        linewidth=1.15,
        label="FD",
    )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(rho) for rho in test_resolutions])
    ax.set_xlim(-0.12, x_positions[-1] + 0.12)
    ax.set_xlabel(r"Test resolution, $\rho_{\mathrm{test}}$")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(False)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", which="major", length=2.8, width=0.62)
    ax.tick_params(axis="y", which="minor", length=1.4, width=0.32)
    ax.legend(
        title=r"$\rho_{\mathrm{train}}$",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=5,
        handlelength=1.45,
        columnspacing=0.75,
        handletextpad=0.35,
        labelspacing=0.15,
        borderaxespad=0.0,
        fontsize=6.1,
        title_fontsize=6.1,
    )
    return _save_publication_figure(fig, output_stem)


def _plot_hk_curve(
    *,
    hk_curve_csv: Path,
    central_difference_hk_curve_csv: Path,
    xscale: str,
    metric_key: str,
    y_label: str,
    eta_c: float | None = None,
    output_stem: Path,
) -> dict[str, str]:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    data = np.genfromtxt(hk_curve_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No rows found in {hk_curve_csv}.")
    data = np.atleast_1d(data)
    x_min = float(np.nanmin(np.asarray(data["hk_left"], dtype=np.float64)))
    x_max = float(np.nanmax(np.asarray(data["hk_right"], dtype=np.float64)))
    palette = FLOWER_STYLE_TRAIN_RHO_PALETTE
    fallback = FLOWER_STYLE_FALLBACK_COLORS
    train_values = sorted({int(value) for value in data["train_rho"]})

    def display_x(values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64)
        if xscale != "low-curvature":
            return raw
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        normalized = (np.log10(raw) - log_min) / (log_max - log_min)
        return np.sqrt(np.clip(normalized, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(3.54, 2.48), constrained_layout=True)
    for idx, train_rho in enumerate(train_values):
        mask = data["train_rho"] == train_rho
        sub = data[mask]
        counts = np.asarray(sub["count"], dtype=np.int64)
        values = np.asarray(sub[metric_key], dtype=np.float64)
        x_field = "hk_mean" if "hk_mean" in sub.dtype.names else "hk_center"
        x = np.asarray(sub[x_field], dtype=np.float64)
        finite = (counts > 0) & np.isfinite(values) & (values > 0.0)
        if not np.any(finite):
            continue
        color = palette.get(train_rho, fallback[idx % len(fallback)])
        ax.plot(
            display_x(x[finite]),
            values[finite],
            linewidth=1.25,
            color=color,
            solid_capstyle="round",
            label=str(train_rho),
        )

    cd = np.genfromtxt(central_difference_hk_curve_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    cd = np.atleast_1d(cd)
    x_min = min(x_min, float(np.nanmin(np.asarray(cd["hk_left"], dtype=np.float64))))
    x_max = max(x_max, float(np.nanmax(np.asarray(cd["hk_right"], dtype=np.float64))))
    counts = np.asarray(cd["count"], dtype=np.int64)
    values = np.asarray(cd[metric_key], dtype=np.float64)
    x_field = "hk_mean" if "hk_mean" in cd.dtype.names else "hk_center"
    x = np.asarray(cd[x_field], dtype=np.float64)
    finite = (counts > 0) & np.isfinite(values) & (values > 0.0)
    if np.any(finite):
        ax.plot(
            display_x(x[finite]),
            values[finite],
            color=FLOWER_STYLE_REFERENCE,
            linestyle=(0, (3.8, 2.0)),
            linewidth=1.15,
            label="FD",
        )

    ax.set_yscale("log")
    if xscale in ("linear", "log"):
        ax.set_xscale(xscale)
        ax.set_xlim(x_min, x_max)
    if xscale == "log":
        tick_candidates = np.array([0.002028374, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.625], dtype=np.float64)
        ticks = tick_candidates[(tick_candidates >= x_min) & (tick_candidates <= x_max)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    elif xscale == "low-curvature":
        tick_candidates = np.array([0.002028374, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.625], dtype=np.float64)
        ticks = tick_candidates[(tick_candidates >= x_min) & (tick_candidates <= x_max)]
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(display_x(ticks))
        ax.set_xticklabels([f"{value:g}" for value in ticks])
    if eta_c is not None and np.isfinite(float(eta_c)) and x_min < float(eta_c) < x_max:
        x_eta = float(eta_c) if xscale != "low-curvature" else float(display_x(np.asarray([eta_c]))[0])
        ax.axvline(x_eta, color="0.35", linestyle=(0, (1.2, 1.8)), linewidth=0.65, zorder=0)
        ax.text(
            x_eta,
            0.96,
            r"$\eta_c$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.0,
            color="0.25",
        )
    ax.set_xlabel(r"$|h\kappa_{\mathrm{analytic}}|$")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=36))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(False)
    ax.tick_params(axis="both", which="major", length=2.8, width=0.62)
    ax.tick_params(axis="both", which="minor", length=1.25, width=0.3)
    ax.legend(
        title=r"$\rho_{\mathrm{train}}$",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=5,
        handlelength=1.45,
        columnspacing=0.72,
        handletextpad=0.35,
        labelspacing=0.15,
        borderaxespad=0.0,
        fontsize=6.1,
        title_fontsize=6.1,
    )
    return _save_publication_figure(fig, output_stem)


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    target_scan: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    central_difference_rows: list[dict[str, Any]],
    figure_outputs: dict[str, Any],
) -> None:
    manifest = {
        "task": "baseline_hgradient cross-resolution test-split evaluation",
        "split": args.split,
        "train_resolutions": list(args.train_resolutions),
        "test_resolutions": list(args.test_resolutions),
        "chunk_size": int(args.chunk_size),
        "hk_bins": int(args.hk_bins),
        "hk_bin_mode": str(args.hk_bin_mode),
        "hk_curve_filter": "h*kappa > 0" if args.hk_bin_mode.startswith("positive-") else "all targets within bin edges",
        "hk_xscale": str(args.hk_xscale_resolved),
        "eta_c": None if args.eta_c is None else float(args.eta_c),
        "target_scan": target_scan,
        "metrics_csv": str((Path(args.output_dir) / "cross_resolution_metrics.csv").resolve()),
        "central_difference_metrics_csv": str((Path(args.output_dir) / "central_difference_metrics.csv").resolve()),
        "hk_curve_csv": str((Path(args.output_dir) / "hk_label_binned_mse.csv").resolve()),
        "central_difference_hk_curve_csv": str((Path(args.output_dir) / "central_difference_hk_label_binned_mse.csv").resolve()),
        "figure_outputs": figure_outputs,
        "runtime_notes": {
            "device": str(args.device_resolved),
            "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK", ""),
            "mplconfigdir": os.environ.get("MPLCONFIGDIR", ""),
        },
        "metrics_rows": [
            {
                key: row[key]
                for key in ("train_rho", "test_rho", "n", "mse", "mae", "maxae", "transform_kind", "transform_source")
            }
            for row in metrics_rows
        ],
        "central_difference_rows": [
            {
                key: row[key]
                for key in ("method", "test_rho", "n", "mse", "mae", "maxae")
            }
            for row in central_difference_rows
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline_hgradient checkpoints on cross-resolution training-dataset test splits."
    )
    parser.add_argument("--train-resolutions", type=_parse_resolutions, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--test-resolutions", type=_parse_resolutions, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--hk-bins", type=int, default=160)
    parser.add_argument(
        "--hk-bin-mode",
        choices=("positive-quantile", "positive-log", "quantile", "linear"),
        default="positive-quantile",
    )
    parser.add_argument("--hk-xscale", choices=("auto", "linear", "log", "low-curvature"), default="auto")
    parser.add_argument(
        "--eta-c",
        type=float,
        default=None,
        help="Optional hybrid threshold to draw on the h*kappa curve. Omit when no threshold is defined.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(Path("out") / "baseline_hgradient_cross_resolution"))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing CSV outputs without rerunning model or FD evaluation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.hk_bins < 8:
        raise ValueError("--hk-bins must be at least 8.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    train_resolutions = tuple(int(item) for item in args.train_resolutions)
    test_resolutions = tuple(int(item) for item in args.test_resolutions)
    args.hk_xscale_resolved = "log" if (args.hk_xscale == "auto" and args.hk_bin_mode == "positive-log") else (
        "linear" if args.hk_xscale == "auto" else args.hk_xscale
    )

    if args.plot_only:
        metrics_csv = output_dir / "cross_resolution_metrics.csv"
        central_difference_metrics_csv = output_dir / "central_difference_metrics.csv"
        hk_curve_csv = output_dir / "hk_label_binned_mse.csv"
        central_difference_hk_curve_csv = output_dir / "central_difference_hk_label_binned_mse.csv"
        figure_outputs = {
            "cross_resolution_mse": _plot_cross_resolution_metric(
                rows=_read_csv_dicts(metrics_csv),
                central_difference_rows=_read_csv_dicts(central_difference_metrics_csv),
                train_resolutions=train_resolutions,
                test_resolutions=test_resolutions,
                metric_key="mse",
                y_label="MSE",
                output_stem=output_dir / "cross_resolution_mse",
            ),
            "cross_resolution_mae": _plot_cross_resolution_metric(
                rows=_read_csv_dicts(metrics_csv),
                central_difference_rows=_read_csv_dicts(central_difference_metrics_csv),
                train_resolutions=train_resolutions,
                test_resolutions=test_resolutions,
                metric_key="mae",
                y_label="MAE",
                output_stem=output_dir / "cross_resolution_mae",
            ),
            "error_vs_hk_label_mse": _plot_hk_curve(
                hk_curve_csv=hk_curve_csv,
                central_difference_hk_curve_csv=central_difference_hk_curve_csv,
                xscale=args.hk_xscale_resolved,
                metric_key="mse",
                y_label="MSE",
                eta_c=args.eta_c,
                output_stem=output_dir / "error_vs_hk_label_mse",
            ),
            "error_vs_hk_label_mae": _plot_hk_curve(
                hk_curve_csv=hk_curve_csv,
                central_difference_hk_curve_csv=central_difference_hk_curve_csv,
                xscale=args.hk_xscale_resolved,
                metric_key="mae",
                y_label="MAE",
                eta_c=args.eta_c,
                output_stem=output_dir / "error_vs_hk_label_mae",
            ),
        }
        for figure_name, outputs in figure_outputs.items():
            print(f"Figure {figure_name}: {outputs['png']}")
        return

    import torch

    args.device_resolved = torch.device(args.device)

    paths: list[EvalPaths] = []
    for train_rho in train_resolutions:
        model_path = _default_model_path(train_rho)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path.resolve()}")
        for test_rho in test_resolutions:
            data_path = _default_data_path(test_rho)
            if not data_path.exists():
                raise FileNotFoundError(f"Missing dataset: {data_path.resolve()}")
            paths.append(EvalPaths(train_rho, test_rho, model_path, data_path))

    unique_data_paths = [_default_data_path(test_rho) for test_rho in test_resolutions]
    target_scan = _scan_targets(
        unique_data_paths,
        args.split,
        args.chunk_size,
        collect_values=args.hk_bin_mode in {"quantile", "positive-quantile"},
    )
    hk_edges = _make_hk_edges(
        hk_min=target_scan["hk_min"],
        hk_max=target_scan["hk_max"],
        hk_positive_min=target_scan["hk_positive_min"],
        n_bins=args.hk_bins,
        bin_mode=args.hk_bin_mode,
        target_values=target_scan.get("target_values"),
    )
    target_scan.pop("target_values", None)
    aggregate_bins = {
        train_rho: {
            "count": np.zeros(len(hk_edges) - 1, dtype=np.int64),
            "sum_sq": np.zeros(len(hk_edges) - 1, dtype=np.float64),
            "sum_abs": np.zeros(len(hk_edges) - 1, dtype=np.float64),
            "sum_hk": np.zeros(len(hk_edges) - 1, dtype=np.float64),
        }
        for train_rho in train_resolutions
    }
    central_difference_bins = {
        "count": np.zeros(len(hk_edges) - 1, dtype=np.int64),
        "sum_sq": np.zeros(len(hk_edges) - 1, dtype=np.float64),
        "sum_abs": np.zeros(len(hk_edges) - 1, dtype=np.float64),
        "sum_hk": np.zeros(len(hk_edges) - 1, dtype=np.float64),
    }

    central_difference_rows: list[dict[str, Any]] = []
    for index, test_rho in enumerate(test_resolutions, start=1):
        data_path = _default_data_path(test_rho)
        print(f"[CD {index:02d}/{len(test_resolutions):02d}] test rho={test_rho}", flush=True)
        cd_row = _evaluate_central_difference(
            data_path=data_path,
            test_rho=test_rho,
            split=args.split,
            chunk_size=args.chunk_size,
            hk_edges=hk_edges,
            aggregate_bins=central_difference_bins,
        )
        central_difference_rows.append(cd_row)
        print(
            f"    N={cd_row['n']:,} MSE={cd_row['mse']:.6e} "
            f"MAE={cd_row['mae']:.6e} MaxAE={cd_row['maxae']:.6e}",
            flush=True,
        )

    metrics_rows: list[dict[str, Any]] = []
    total = len(paths)
    for index, pair_paths in enumerate(paths, start=1):
        print(
            f"[{index:02d}/{total:02d}] "
            f"train rho={pair_paths.train_rho} -> test rho={pair_paths.test_rho}",
            flush=True,
        )
        row = _evaluate_pair(
            paths=pair_paths,
            split=args.split,
            device=args.device_resolved,
            chunk_size=args.chunk_size,
            hk_edges=hk_edges,
            aggregate_bins=aggregate_bins,
        )
        metrics_rows.append(row)
        print(
            f"    N={row['n']:,} MSE={row['mse']:.6e} "
            f"MAE={row['mae']:.6e} MaxAE={row['maxae']:.6e}",
            flush=True,
        )

    metrics_csv = output_dir / "cross_resolution_metrics.csv"
    central_difference_metrics_csv = output_dir / "central_difference_metrics.csv"
    hk_curve_csv = output_dir / "hk_label_binned_mse.csv"
    central_difference_hk_curve_csv = output_dir / "central_difference_hk_label_binned_mse.csv"
    _write_metrics_csv(metrics_csv, metrics_rows)
    _write_central_difference_metrics_csv(central_difference_metrics_csv, central_difference_rows)
    _write_hk_curve_csv(hk_curve_csv, hk_edges=hk_edges, aggregate_bins=aggregate_bins)
    _write_central_difference_hk_curve_csv(
        central_difference_hk_curve_csv,
        hk_edges=hk_edges,
        aggregate_bins=central_difference_bins,
    )

    figure_outputs = {
        "cross_resolution_mse": _plot_cross_resolution_metric(
            rows=metrics_rows,
            central_difference_rows=central_difference_rows,
            train_resolutions=train_resolutions,
            test_resolutions=test_resolutions,
            metric_key="mse",
            y_label="MSE",
            output_stem=output_dir / "cross_resolution_mse",
        ),
        "cross_resolution_mae": _plot_cross_resolution_metric(
            rows=metrics_rows,
            central_difference_rows=central_difference_rows,
            train_resolutions=train_resolutions,
            test_resolutions=test_resolutions,
            metric_key="mae",
            y_label="MAE",
            output_stem=output_dir / "cross_resolution_mae",
        ),
        "error_vs_hk_label_mse": _plot_hk_curve(
            hk_curve_csv=hk_curve_csv,
            central_difference_hk_curve_csv=central_difference_hk_curve_csv,
            xscale=args.hk_xscale_resolved,
            metric_key="mse",
            y_label="MSE",
            eta_c=args.eta_c,
            output_stem=output_dir / "error_vs_hk_label_mse",
        ),
        "error_vs_hk_label_mae": _plot_hk_curve(
            hk_curve_csv=hk_curve_csv,
            central_difference_hk_curve_csv=central_difference_hk_curve_csv,
            xscale=args.hk_xscale_resolved,
            metric_key="mae",
            y_label="MAE",
            eta_c=args.eta_c,
            output_stem=output_dir / "error_vs_hk_label_mae",
        ),
    }
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        target_scan=target_scan,
        metrics_rows=metrics_rows,
        central_difference_rows=central_difference_rows,
        figure_outputs=figure_outputs,
    )
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved central difference metrics: {central_difference_metrics_csv}")
    print(f"Saved h*kappa curve data: {hk_curve_csv}")
    print(f"Saved central difference h*kappa curve data: {central_difference_hk_curve_csv}")
    print(f"Saved manifest: {output_dir / 'manifest.json'}")
    for figure_name, outputs in figure_outputs.items():
        print(f"Figure {figure_name}: {outputs['png']}")


if __name__ == "__main__":
    main()
