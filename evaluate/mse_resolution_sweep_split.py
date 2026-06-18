"""
Test-split MSE vs resolution for three feature variants (V2.2 / V2.3 / V3),
each trained at rho=128 and rho=256, evaluated at rho in {64, 128, 256, 512}.

Layout: two plots (trained@128 / trained@256).
  x = test resolution, y = MSE vs analytic h*kappa.
  3 model curves + FD central-difference baseline.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.shared import central_difference_hkappa_from_phi9, compute_metrics
from evaluate.test_split_mse import evaluate_split
from train_generate.io import load_training_arrays_from_hdf5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "out" / "5527"
OUTPUT_DIR = PROJECT_ROOT / "out" / "mse_sweep"

TRAIN_RESOLUTIONS = [128, 256]
TEST_RESOLUTIONS = [64, 128, 256, 512]

VARIANTS = [
    {
        "key": "v22",
        "label": "V2.2  (φ/αh, 9D)",
        "color": "#1f77b4",
        "marker": "o",
        "data_suffix": "_h",
        "ckpt": {
            128: MODEL_DIR / "v22_128_ah.pt",
            256: MODEL_DIR / "v22_256_ah.pt",
        },
    },
    {
        "key": "v23",
        "label": "V2.3  (φ/αh + ∇φ, 27D)",
        "color": "#ff7f0e",
        "marker": "s",
        "data_suffix": "_hgradient",
        "ckpt": {
            128: MODEL_DIR / "v23_128_hgradient_ah.pt",
            256: MODEL_DIR / "v23_256_hgradient_ah.pt",
        },
    },
    {
        "key": "v3",
        "label": "V3  (PCA-18)",
        "color": "#2ca02c",
        "marker": "^",
        "data_suffix": "_hgradient",
        "ckpt": {
            128: MODEL_DIR / "v3_128_hgradient.pt",
            256: MODEL_DIR / "v3_256_hgradient.pt",
        },
    },
    {
        "key": "v2_hgrad",
        "label": "V2  (φ/h + ∇φ, 27D)",
        "color": "#9467bd",
        "marker": "D",
        "data_suffix": "_hgradient",
        "ckpt": {
            128: PROJECT_ROOT / "out" / "128" / "baseline_128_hgradient.pt",
            256: PROJECT_ROOT / "out" / "256" / "baseline_256_hgradient.pt",
        },
    },
]

NUMERIC_COLOR = "#7f7f7f"


def _data_path(rho: int, suffix: str) -> Path:
    return PROJECT_ROOT / "dataset" / str(rho) / f"{rho}{suffix}.h5"


def _fd_mse(data_path: Path, device: torch.device) -> float:
    data = load_training_arrays_from_hdf5(data_path)
    phi9   = data["splits"]["test"]["phi9"]
    target = data["splits"]["test"]["hkappa_target"].reshape(-1)
    numeric = central_difference_hkappa_from_phi9(phi9)
    return float(compute_metrics(numeric, target)["mse"])


def run_all(device: torch.device) -> tuple[dict, dict]:
    results: dict = {tr: {v["key"]: {} for v in VARIANTS} for tr in TRAIN_RESOLUTIONS}
    numeric_baseline: dict = {}

    for train_res in TRAIN_RESOLUTIONS:
        print(f"\n=== Models trained @ ρ={train_res} ===")
        for variant in VARIANTS:
            model_path = variant["ckpt"][train_res]
            for test_res in TEST_RESOLUTIONS:
                data_path = _data_path(test_res, variant["data_suffix"])
                label = f"  {variant['key']:4s} train={train_res} | test={test_res}"

                # FD baseline (compute once per test_res, shared across variants)
                if test_res not in numeric_baseline:
                    # use _h suffix (phi9 is the same across _h / _hgradient)
                    numeric_baseline[test_res] = _fd_mse(
                        _data_path(test_res, "_h"), device
                    )

                print(f"{label} ... ", end="", flush=True)
                result = evaluate_split(
                    model_path=model_path,
                    data_path=data_path,
                    split="test",
                    device=device,
                )
                mse = float(result["metrics"]["mse"])
                results[train_res][variant["key"]][test_res] = mse
                print(f"model_MSE={mse:.4e}  FD_MSE={numeric_baseline[test_res]:.4e}")

    return results, numeric_baseline


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(TEST_RESOLUTIONS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Test resolution (ρ)", fontsize=12)
    ax.set_ylabel("MSE  (vs analytic h·κ)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)


def plot_train_res(results: dict, numeric_baseline: dict, train_res: int, ax: plt.Axes) -> None:
    ax.plot(
        TEST_RESOLUTIONS,
        [numeric_baseline[r] for r in TEST_RESOLUTIONS],
        color=NUMERIC_COLOR, linestyle="--", marker="x", linewidth=1.5,
        markersize=7, label="Central diff (FD)", zorder=1,
    )
    for variant in VARIANTS:
        mses = [results[train_res][variant["key"]][r] for r in TEST_RESOLUTIONS]
        ax.plot(
            TEST_RESOLUTIONS, mses,
            color=variant["color"], linestyle="-", marker=variant["marker"],
            linewidth=2, markersize=7, label=variant["label"], zorder=2,
        )
    _style_axis(ax)
    ax.set_title(f"Models trained @ ρ={train_res}", fontsize=13)
    ax.legend(fontsize=10)


def make_plots(results: dict, numeric_baseline: dict) -> None:
    for train_res in TRAIN_RESOLUTIONS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plot_train_res(results, numeric_baseline, train_res, ax)
        fig.suptitle("Test-split MSE vs resolution — three variants", fontsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"mse_split_trained{train_res}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Test-split MSE vs resolution — three variants", fontsize=14, y=1.02)
    for ax, train_res in zip(axes, TRAIN_RESOLUTIONS):
        plot_train_res(results, numeric_baseline, train_res, ax)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "mse_split_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined: {out_path}")
    plt.close(fig)


def write_csv(results: dict, numeric_baseline: dict) -> None:
    out_path = OUTPUT_DIR / "mse_split.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["train_res", "variant", "test_res", "model_mse", "fd_mse"])
        for train_res in TRAIN_RESOLUTIONS:
            for variant in VARIANTS:
                for test_res in TEST_RESOLUTIONS:
                    writer.writerow([
                        train_res, variant["key"], test_res,
                        f"{results[train_res][variant['key']][test_res]:.6e}",
                        f"{numeric_baseline[test_res]:.6e}",
                    ])
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, numeric_baseline = run_all(device)
    make_plots(results, numeric_baseline)
    write_csv(results, numeric_baseline)
    print("\nDone.")
