from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_RESOLUTIONS = (64, 128, 256, 512)
DEFAULT_DATASET_PATTERN = "dataset/{resolution}/{resolution}_hgradient.h5"
DEFAULT_OUTPUT_DIR = Path("results") / "hk_distribution_by_resolution"
DEFAULT_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class ResolutionData:
    resolution: int
    path: Path
    values: np.ndarray
    attrs: dict[str, str]


def _iter_chunks(dataset: h5py.Dataset, chunk_size: int) -> Iterable[np.ndarray]:
    total = int(dataset.shape[0])
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        yield np.asarray(dataset[start:stop], dtype=np.float64).reshape(-1)


def _read_values(path: Path, split: str, chunk_size: int) -> tuple[np.ndarray, dict[str, str]]:
    chunks: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        if split not in handle:
            raise KeyError(f"{path} does not contain split {split!r}")
        group = handle[split]
        if "hkappa_target" not in group:
            raise KeyError(f"{path}:{split} does not contain hkappa_target")
        for chunk in _iter_chunks(group["hkappa_target"], chunk_size):
            chunks.append(chunk)
        attrs = {
            "feature_order": str(handle.attrs.get("feature_order", "")),
            "feature_transform": str(handle.attrs.get("feature_transform", "")),
            "augment_sign_flip": str(handle.attrs.get("augment_sign_flip", "")),
            "shape_types_json": str(handle.attrs.get("shape_types_json", "")),
            "initial_field_types_json": str(handle.attrs.get("initial_field_types_json", "")),
        }
    if not chunks:
        values = np.zeros((0,), dtype=np.float64)
    else:
        values = np.concatenate(chunks).astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}:{split}/hkappa_target contains non-finite values")
    return values, attrs


def _resolve_dataset_path(pattern: str, resolution: int) -> Path:
    return Path(pattern.format(resolution=resolution, res=resolution)).resolve()


def load_resolution_data(args: argparse.Namespace) -> list[ResolutionData]:
    data: list[ResolutionData] = []
    for resolution in args.resolutions:
        path = _resolve_dataset_path(args.dataset_pattern, resolution)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset for N={resolution}: {path}")
        values, attrs = _read_values(path, args.split, args.chunk_size)
        if args.quantity == "abs":
            values = np.abs(values)
        data.append(
            ResolutionData(
                resolution=int(resolution),
                path=path,
                values=values,
                attrs=attrs,
            )
        )
    return data


def _summary_row(item: ResolutionData, quantity: str, split: str) -> dict[str, object]:
    values = item.values
    if values.size == 0:
        stats = {name: float("nan") for name in ("mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")}
    else:
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }
    return {
        "resolution": item.resolution,
        "split": split,
        "quantity": quantity,
        "dataset": str(item.path),
        "n_samples": int(values.size),
        **stats,
        **item.attrs,
    }


def _write_summary(data: list[ResolutionData], output_dir: Path, quantity: str, split: str) -> Path:
    rows = [_summary_row(item, quantity, split) for item in data]
    fieldnames = list(rows[0].keys()) if rows else []
    path = output_dir / f"{quantity}_hk_distribution_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_edges(data: list[ResolutionData], bins: int, quantity: str) -> np.ndarray:
    all_values = np.concatenate([item.values for item in data if item.values.size])
    if all_values.size == 0:
        return np.linspace(-1.0, 1.0, bins + 1)
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if quantity == "signed":
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-6)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, bins + 1)


def _display_xlim(data: list[ResolutionData], quantity: str, quantile: float | None) -> tuple[float, float]:
    all_values = np.concatenate([item.values for item in data if item.values.size])
    if all_values.size == 0:
        return (-1.0, 1.0)
    if quantile is None:
        lo = float(np.min(all_values))
        hi = float(np.max(all_values))
    elif quantity == "signed":
        bound = float(np.quantile(np.abs(all_values), quantile))
        lo, hi = -bound, bound
    else:
        lo = 0.0
        hi = float(np.quantile(all_values, quantile))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-6)
        lo -= pad
        hi += pad
    return lo, hi


def _write_bin_counts(data: list[ResolutionData], edges: np.ndarray, output_dir: Path, quantity: str, split: str) -> Path:
    path = output_dir / f"{quantity}_hk_distribution_bins.csv"
    with path.open("w", newline="") as handle:
        fieldnames = ["resolution", "split", "quantity", "bin_left", "bin_right", "bin_center", "count", "fraction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        centers = 0.5 * (edges[:-1] + edges[1:])
        for item in data:
            counts, _ = np.histogram(item.values, bins=edges)
            total = float(np.sum(counts))
            for left, right, center, count in zip(edges[:-1], edges[1:], centers, counts, strict=True):
                writer.writerow(
                    {
                        "resolution": item.resolution,
                        "split": split,
                        "quantity": quantity,
                        "bin_left": float(left),
                        "bin_right": float(right),
                        "bin_center": float(center),
                        "count": int(count),
                        "fraction": float(count / total) if total else 0.0,
                    }
                )
    return path


def _plot(data: list[ResolutionData], edges: np.ndarray, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pinn_matplotlib_cache"))

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
        }
    )

    n_panels = len(data)
    fig_width = max(6.2, 1.75 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 2.15), sharex=True, sharey=False)
    if n_panels == 1:
        axes = [axes]
    xlim = _display_xlim(data, args.quantity, args.xlim_quantile)
    bar_color = "#2B7BB9"
    for ax, item in zip(axes, data, strict=True):
        counts, _ = np.histogram(item.values, bins=edges)
        width = np.diff(edges)
        ax.bar(edges[:-1], counts, width=width, align="edge", color=bar_color, edgecolor=bar_color, linewidth=0.1)
        ax.set_title(f"N={item.resolution}", fontsize=12, pad=8)
        ax.set_xlabel(r"$|h\kappa|$" if args.quantity == "abs" else r"$h\kappa$", fontsize=10)
        ax.set_xlim(*xlim)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=9)
        ax.margins(x=0)
    axes[0].set_ylabel("number of samples", fontsize=10)
    if args.show_title:
        title = f"{args.split} split hk distribution by resolution"
        if args.xlim_quantile is not None:
            percent = 100.0 * args.xlim_quantile
            title += f" ({percent:.1f}% central range shown)"
        fig.suptitle(title, fontsize=10, y=1.05)
    fig.tight_layout(w_pad=1.1)

    stem = output_dir / f"{args.quantity}_hk_distribution_by_resolution_{args.split}"
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot hkappa_target histograms across dataset resolutions.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--dataset-pattern", default=DEFAULT_DATASET_PATTERN)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--quantity", default="signed", choices=("signed", "abs"))
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--xlim-quantile", type=float, default=0.995, help="Use None via --full-xlim to show full min/max range.")
    parser.add_argument("--full-xlim", action="store_true")
    parser.add_argument("--show-title", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.full_xlim:
        args.xlim_quantile = None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_resolution_data(args)
    edges = _build_edges(data, args.bins, args.quantity)
    summary_path = _write_summary(data, output_dir, args.quantity, args.split)
    bin_path = _write_bin_counts(data, edges, output_dir, args.quantity, args.split)
    figure_paths = _plot(data, edges, output_dir, args)

    print("Task: hk distribution by resolution")
    print(f"Split: {args.split}")
    print(f"Quantity: {args.quantity}")
    print("Datasets:")
    for item in data:
        print(f"  N={item.resolution}: {item.path} ({item.values.size} samples)")
    print(f"Summary CSV: {summary_path}")
    print(f"Bin-count CSV: {bin_path}")
    for path in figure_paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
