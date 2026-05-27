from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    add_swanlab_args,
    build_swanlab_image,
    build_swanlab_table_payload,
    csv_to_list,
    ensure_output_dir,
    init_swanlab_run,
    write_csv_rows,
)
from train_generate.generate import build_grid, build_phi0_grid, compute_hkappa_targets, interface_indices
from train_generate.io import load_training_metadata_from_hdf5


SUMMARY_FIELDNAMES = (
    "stat_mode",
    "split",
    "shape_type",
    "blueprint_count",
    "node_count",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "max",
)
DEFAULT_CHUNK_SIZE = 1_000_000


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.total += float(np.sum(arr, dtype=np.float64))
        self.total_sq += float(np.sum(arr * arr, dtype=np.float64))
        self.min_value = min(self.min_value, float(np.min(arr)))
        self.max_value = max(self.max_value, float(np.max(arr)))

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = self.total / float(self.count)
        variance = max(self.total_sq / float(self.count) - mean * mean, 0.0)
        return {
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": float(self.min_value),
            "max": float(self.max_value),
        }


@dataclass
class HistogramAccumulator:
    edges: np.ndarray

    def __post_init__(self) -> None:
        self.counts = np.zeros((self.edges.shape[0] - 1,), dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        hist, _ = np.histogram(arr, bins=self.edges)
        self.counts += hist.astype(np.int64, copy=False)

    def quantile(self, q: float) -> float:
        total = int(np.sum(self.counts))
        if total == 0:
            return float("nan")
        target = float(q) * float(total - 1)
        cdf = np.cumsum(self.counts, dtype=np.int64)
        idx = int(np.searchsorted(cdf, target, side="right"))
        idx = min(max(idx, 0), self.counts.shape[0] - 1)
        prev = int(cdf[idx - 1]) if idx > 0 else 0
        count = int(self.counts[idx])
        if count <= 0:
            return float(self.edges[idx])
        frac = (target - float(prev)) / float(count)
        frac = min(max(frac, 0.0), 1.0)
        return float(self.edges[idx] + frac * (self.edges[idx + 1] - self.edges[idx]))


def _iter_dataset_chunks(dataset: h5py.Dataset, *, chunk_size: int = DEFAULT_CHUNK_SIZE):
    total = int(dataset.shape[0])
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        yield np.asarray(dataset[start:stop], dtype=np.float64).reshape(-1)


def _build_hist_edges(min_value: float, max_value: float, *, bins: int = 256) -> np.ndarray:
    lo = float(min_value)
    hi = float(max_value)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-6)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)


def _summary_row(
    *,
    stat_mode: str,
    split: str,
    shape_type: str,
    blueprint_count: int,
    node_count: int,
    summary: dict[str, float],
    percentiles: dict[str, float],
) -> dict[str, Any]:
    return {
        "stat_mode": stat_mode,
        "split": split,
        "shape_type": shape_type,
        "blueprint_count": int(blueprint_count),
        "node_count": int(node_count),
        "mean": float(summary["mean"]),
        "std": float(summary["std"]),
        "min": float(summary["min"]),
        "p01": float(percentiles["p01"]),
        "p05": float(percentiles["p05"]),
        "p25": float(percentiles["p25"]),
        "p50": float(percentiles["p50"]),
        "p75": float(percentiles["p75"]),
        "p95": float(percentiles["p95"]),
        "p99": float(percentiles["p99"]),
        "max": float(summary["max"]),
    }


def _percentiles_from_hist(hist: HistogramAccumulator) -> dict[str, float]:
    return {
        "p01": hist.quantile(0.01),
        "p05": hist.quantile(0.05),
        "p25": hist.quantile(0.25),
        "p50": hist.quantile(0.50),
        "p75": hist.quantile(0.75),
        "p95": hist.quantile(0.95),
        "p99": hist.quantile(0.99),
    }


def _percentiles_from_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {key: float("nan") for key in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    return {
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _hist_density(counts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    widths = np.diff(edges)
    total = float(np.sum(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if total <= 0.0:
        return centers, np.zeros_like(centers)
    density = counts.astype(np.float64) / (total * widths)
    return centers, density


def _representative_ellipse_case(data_config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blueprint = {
        "meta": {
            "shape_type": "ellipse",
            "resolution": 266,
            "blueprint_id": "ellipse_rep_rho266_a0.18_b0.16",
        },
        "params": {
            "center": [0.5, 0.5],
            "a": 0.18,
            "b": 0.16,
            "psi": 0.0,
            "h": 1.0 / 265.0,
        },
    }
    X, Y = build_grid(266)
    phi0 = build_phi0_grid(blueprint, "sdf", data_config=data_config, X=X, Y=Y)
    indices = interface_indices(phi0)
    hkappa = compute_hkappa_targets(blueprint, indices, data_config=data_config, X=X, Y=Y).reshape(-1)
    rows = indices[:, 0]
    cols = indices[:, 1]
    xy = np.column_stack((X[rows, cols], Y[rows, cols])).astype(np.float64)
    order = np.argsort(np.arctan2(xy[:, 1] - 0.5, xy[:, 0] - 0.5), kind="mergesort")
    return xy[order], hkappa[order], np.arctan2(xy[order, 1] - 0.5, xy[order, 0] - 0.5)


def _geometry_worker(task: tuple[str, dict[str, Any], Any]) -> tuple[str, str, np.ndarray]:
    split_name, blueprint, data_config = task
    rho = int(blueprint["meta"]["resolution"])
    shape_type = str(blueprint["meta"]["shape_type"])
    X, Y = build_grid(rho)
    # Geometry-deduplicated stats only need boundary nodes on the zero level set.
    # Using the matching nonsdf initializer avoids the expensive full-grid ellipse SDF solve.
    phi0 = build_phi0_grid(blueprint, "nonsdf", data_config=data_config, X=X, Y=Y)
    indices = interface_indices(phi0)
    hkappa = compute_hkappa_targets(blueprint, indices, data_config=data_config, X=X, Y=Y).reshape(-1)
    return split_name, shape_type, hkappa


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize training-set curvature distributions and geometry-deduplicated boundary curvature.")
    parser.add_argument("--data", type=str, default=str(Path("dataset") / "256.h5"))
    parser.add_argument("--output-dir", type=str, default=str(Path("out") / "curvature_viz" / "training"))
    add_swanlab_args(parser)
    return parser


def run_training_curvature(*, dataset_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    output_dir_path = ensure_output_dir(output_dir)
    metadata = load_training_metadata_from_hdf5(dataset_path)
    data_config = metadata["config"]

    sample_stats = {name: RunningStats() for name in ("train", "val", "test", "all")}
    with h5py.File(dataset_path, "r") as handle:
        for split_name in ("train", "val", "test"):
            if split_name not in handle:
                continue
            dataset = handle[split_name]["hkappa_target"]
            for chunk in _iter_dataset_chunks(dataset):
                sample_stats[split_name].update(chunk)
                sample_stats["all"].update(chunk)

        histograms = {
            name: HistogramAccumulator(
                _build_hist_edges(sample_stats[name].min_value, sample_stats[name].max_value)
            )
            for name in sample_stats
        }
        for split_name in ("train", "val", "test"):
            if split_name not in handle:
                continue
            dataset = handle[split_name]["hkappa_target"]
            for chunk in _iter_dataset_chunks(dataset):
                histograms[split_name].update(chunk)
                histograms["all"].update(chunk)

    geometry_values: dict[str, dict[str, list[np.ndarray]]] = {
        split_name: {"all": [], "circle": [], "ellipse": []}
        for split_name in ("train", "val", "test", "all")
    }
    geometry_blueprint_counts: dict[str, dict[str, int]] = {
        split_name: {"all": 0, "circle": 0, "ellipse": 0}
        for split_name in ("train", "val", "test", "all")
    }
    geometry_node_counts: dict[str, dict[str, int]] = {
        split_name: {"all": 0, "circle": 0, "ellipse": 0}
        for split_name in ("train", "val", "test", "all")
    }
    geometry_tasks = [
        (split_name, metadata["blueprints"][int(blueprint_idx)], data_config)
        for split_name in ("train", "val", "test")
        for blueprint_idx in metadata["split_blueprint_indices"].get(split_name, [])
    ]
    max_workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for split_name, shape_type, hkappa in executor.map(_geometry_worker, geometry_tasks, chunksize=8):
            geometry_values[split_name]["all"].append(hkappa)
            geometry_values[split_name][shape_type].append(hkappa)
            geometry_values["all"]["all"].append(hkappa)
            geometry_values["all"][shape_type].append(hkappa)
            for bucket in (split_name, "all"):
                geometry_blueprint_counts[bucket]["all"] += 1
                geometry_blueprint_counts[bucket][shape_type] += 1
                geometry_node_counts[bucket]["all"] += int(hkappa.size)
                geometry_node_counts[bucket][shape_type] += int(hkappa.size)

    summary_rows: list[dict[str, Any]] = []
    total_blueprints = int(sum(metadata["split_blueprint_counts"].values()))
    for split_name in ("train", "val", "test", "all"):
        split_blueprints = (
            int(metadata["split_blueprint_counts"].get(split_name, 0))
            if split_name != "all"
            else total_blueprints
        )
        summary_rows.append(
            _summary_row(
                stat_mode="sample",
                split=split_name,
                shape_type="all",
                blueprint_count=split_blueprints,
                node_count=int(sample_stats[split_name].count),
                summary=sample_stats[split_name].summary(),
                percentiles=_percentiles_from_hist(histograms[split_name]),
            )
        )

    geometry_arrays: dict[str, dict[str, np.ndarray]] = {
        split_name: {
            shape_type: (
                np.concatenate(values, axis=0).astype(np.float64, copy=False)
                if values
                else np.zeros((0,), dtype=np.float64)
            )
            for shape_type, values in geometry_values[split_name].items()
        }
        for split_name in geometry_values
    }
    for split_name in ("train", "val", "test", "all"):
        for shape_type in ("all", "circle", "ellipse"):
            arr = geometry_arrays[split_name][shape_type]
            summary_rows.append(
                _summary_row(
                    stat_mode="geometry",
                    split=split_name,
                    shape_type=shape_type,
                    blueprint_count=int(geometry_blueprint_counts[split_name][shape_type]),
                    node_count=int(geometry_node_counts[split_name][shape_type]),
                    summary=RunningStats(
                        count=int(arr.size),
                        total=float(np.sum(arr, dtype=np.float64)),
                        total_sq=float(np.sum(arr * arr, dtype=np.float64)),
                        min_value=float(np.min(arr)) if arr.size else float("nan"),
                        max_value=float(np.max(arr)) if arr.size else float("nan"),
                    ).summary(),
                    percentiles=_percentiles_from_values(arr),
                )
            )

    summary_path = write_csv_rows(
        output_dir_path / "training_curvature_summary.csv",
        summary_rows,
        fieldnames=SUMMARY_FIELDNAMES,
    )

    rep_xy, rep_hkappa, _ = _representative_ellipse_case(data_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_min = min(sample_stats["all"].min_value, float(np.min(geometry_arrays["all"]["all"])))
    plot_max = max(sample_stats["all"].max_value, float(np.max(geometry_arrays["all"]["all"])))
    plot_edges = _build_hist_edges(plot_min, plot_max)
    plot_sample_counts = {
        split_name: np.zeros((plot_edges.shape[0] - 1,), dtype=np.int64)
        for split_name in ("train", "val", "test", "all")
    }
    with h5py.File(dataset_path, "r") as handle:
        for split_name in ("train", "val", "test"):
            if split_name not in handle:
                continue
            for chunk in _iter_dataset_chunks(handle[split_name]["hkappa_target"]):
                hist = np.histogram(chunk, bins=plot_edges)[0].astype(np.int64, copy=False)
                plot_sample_counts[split_name] += hist
                plot_sample_counts["all"] += hist

    # Clip x-axis to p0.5–p99.5 so the bulk of the distribution is visible
    xlim_lo = histograms["all"].quantile(0.005)
    xlim_hi = histograms["all"].quantile(0.995)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    ax_sample = axes[0, 0]
    for split_name in ("train", "val", "test"):
        centers, density = _hist_density(plot_sample_counts[split_name], plot_edges)
        ax_sample.plot(centers, density, linewidth=1.4, label=split_name)
    ax_sample.set_title("Sample curvature distribution")
    ax_sample.set_xlabel("h*kappa")
    ax_sample.set_ylabel("density")
    ax_sample.set_xlim(xlim_lo, xlim_hi)
    ax_sample.grid(True, alpha=0.25)
    ax_sample.legend(loc="best")

    ax_compare = axes[0, 1]
    sample_centers, sample_density = _hist_density(plot_sample_counts["all"], plot_edges)
    geometry_all_hist, geometry_all_edges = np.histogram(geometry_arrays["all"]["all"], bins=plot_edges)
    geometry_centers, geometry_density = _hist_density(geometry_all_hist, geometry_all_edges)
    ax_compare.plot(sample_centers, sample_density, linewidth=1.4, label="sample_all")
    ax_compare.plot(geometry_centers, geometry_density, linewidth=1.4, label="geometry_all")
    ax_compare.set_title("Sample vs geometry-deduplicated")
    ax_compare.set_xlabel("h*kappa")
    ax_compare.set_ylabel("density")
    ax_compare.set_xlim(xlim_lo, xlim_hi)
    ax_compare.grid(True, alpha=0.25)
    ax_compare.legend(loc="best")

    ax_shapes = axes[1, 0]
    circle_hist, circle_edges = np.histogram(geometry_arrays["all"]["circle"], bins=plot_edges)
    ellipse_hist, ellipse_edges = np.histogram(geometry_arrays["all"]["ellipse"], bins=plot_edges)
    circle_centers, circle_density = _hist_density(circle_hist, circle_edges)
    ellipse_centers, ellipse_density = _hist_density(ellipse_hist, ellipse_edges)
    ax_shapes.plot(circle_centers, circle_density, linewidth=1.4, label="circle")
    ax_shapes.plot(ellipse_centers, ellipse_density, linewidth=1.4, label="ellipse")
    ax_shapes.set_title("Geometry distribution by shape")
    ax_shapes.set_xlabel("h*kappa")
    ax_shapes.set_ylabel("density")
    ax_shapes.set_xlim(xlim_lo, xlim_hi)
    ax_shapes.grid(True, alpha=0.25)
    ax_shapes.legend(loc="best")

    ax_rep = axes[1, 1]
    closed_xy = np.vstack([rep_xy, rep_xy[:1]])
    ax_rep.plot(closed_xy[:, 0], closed_xy[:, 1], color="0.80", linewidth=1.0)
    scatter = ax_rep.scatter(rep_xy[:, 0], rep_xy[:, 1], c=rep_hkappa, cmap="coolwarm", s=10, linewidths=0.0)
    fig.colorbar(scatter, ax=ax_rep, fraction=0.046, pad=0.04)
    ax_rep.set_aspect("equal")
    ax_rep.set_xticks([])
    ax_rep.set_yticks([])
    ax_rep.set_title("Representative ellipse boundary curvature\nrho=266, a=0.18, b=0.16")

    stats_path = (output_dir_path / "training_curvature_stats.png").resolve()
    fig.savefig(stats_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_rows": summary_rows,
        "summary_path": summary_path,
        "stats_path": stats_path,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_training_curvature(dataset_path=args.data, output_dir=args.output_dir)
    print("Task: training curvature summary")
    print(f"Summary CSV: {result['summary_path']}")
    print(f"Overview image: {result['stats_path']}")
    if args.use_swanlab:
        import swanlab

        run = init_swanlab_run(
            project=args.swanlab_project or None,
            experiment_name=args.swanlab_experiment_name or None,
            description=args.swanlab_description or None,
            tags=csv_to_list(args.swanlab_tags),
            group=args.swanlab_group or None,
            workspace=args.swanlab_workspace or None,
            logdir=args.swanlab_logdir or None,
            mode=args.swanlab_mode or None,
            config={
                "task": "training curvature",
                "dataset_path": str(Path(args.data).resolve()),
            },
        )
        run.log({
            "train_curvature/overview": build_swanlab_image(swanlab, result["stats_path"]),
            "train_curvature/summary_table": build_swanlab_table_payload(
                swanlab,
                result["summary_rows"],
                fieldnames=SUMMARY_FIELDNAMES,
            ),
        })
        run.finish()


if __name__ == "__main__":
    main()
