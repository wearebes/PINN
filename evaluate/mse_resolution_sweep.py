"""
Run flower test at resolutions 128, 256, 512, 1024 for non-regularized models
trained at 128 and 256. Produce two comparison plots (h-feature and hgradient-feature).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.flower import evaluate_flower

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "h": {
        128: PROJECT_ROOT / "out" / "128" / "baseline_128_h.pt",
        256: PROJECT_ROOT / "out" / "256" / "baseline_256_h.pt",
    },
    "hgradient": {
        128: PROJECT_ROOT / "out" / "128" / "baseline_128_hgradient.pt",
        256: PROJECT_ROOT / "out" / "256" / "baseline_256_hgradient.pt",
    },
}

TEST_DATA = {
    "h": {
        128: PROJECT_ROOT / "test_data" / "flower_rho128_h.h5",
        256: PROJECT_ROOT / "test_data" / "flower_rho256_h.h5",
        512: PROJECT_ROOT / "test_data" / "flower_rho512_h.h5",
        1024: PROJECT_ROOT / "test_data" / "flower_rho1024_h.h5",
    },
    "hgradient": {
        128: PROJECT_ROOT / "test_data" / "flower_rho128_hgradient.h5",
        256: PROJECT_ROOT / "test_data" / "flower_rho256_hgradient.h5",
        512: PROJECT_ROOT / "test_data" / "flower_rho512_hgradient.h5",
        1024: PROJECT_ROOT / "test_data" / "flower_rho1024_hgradient.h5",
    },
}

TEST_RESOLUTIONS = [128, 256, 512, 1024]

COLORS = {128: "#1f77b4", 256: "#ff7f0e"}
NUMERIC_COLOR = "#7f7f7f"

OUTPUT_DIR = PROJECT_ROOT / "out" / "mse_sweep"


def run_all() -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {"h": {}, "hgradient": {}}

    for feature_type in ("h", "hgradient"):
        print(f"\n=== Feature type: {feature_type} ===")
        for train_res, model_path in MODELS[feature_type].items():
            results[feature_type][train_res] = {}
            for test_res in TEST_RESOLUTIONS:
                data_path = TEST_DATA[feature_type][test_res]
                print(f"  model_train={train_res} | test_rho={test_res} ... ", end="", flush=True)
                result = evaluate_flower(
                    dataset_path=data_path,
                    model_path=model_path,
                    normalization_csv_path=None,
                    device=device,
                )
                mse_model = float(result["model_vs_analytic"]["mse"])
                mse_numeric = float(result["numeric_vs_analytic"]["mse"])
                results[feature_type][train_res][test_res] = {
                    "model_mse": mse_model,
                    "numeric_mse": mse_numeric,
                }
                print(f"model_MSE={mse_model:.4e}  numeric_MSE={mse_numeric:.4e}")

    return results


def plot_feature_type(
    results: dict[str, dict],
    feature_type: str,
    ax: plt.Axes,
) -> None:
    resolutions = TEST_RESOLUTIONS
    numeric_mse = [
        results[feature_type][128][r]["numeric_mse"] for r in resolutions
    ]
    ax.plot(
        resolutions,
        numeric_mse,
        color=NUMERIC_COLOR,
        linestyle="--",
        marker="s",
        linewidth=1.5,
        markersize=6,
        label="Central diff (numeric)",
        zorder=1,
    )

    for train_res, color in COLORS.items():
        mses = [results[feature_type][train_res][r]["model_mse"] for r in resolutions]
        ax.plot(
            resolutions,
            mses,
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"Model trained @ ρ={train_res}",
            zorder=2,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(resolutions)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Test resolution (ρ)", fontsize=12)
    ax.set_ylabel("MSE  (model vs analytic h·κ)", fontsize=12)
    feat_label = "φ/h" if feature_type == "h" else "φ/h + ∇φ"
    ax.set_title(f"Feature type: {feat_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def make_plots(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Flower test MSE vs resolution — non-regularized models", fontsize=14, y=1.01)

    for ax, feature_type in zip(axes, ("h", "hgradient")):
        plot_feature_type(results, feature_type, ax)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "mse_resolution_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved combined plot: {out_path}")
    plt.close(fig)

    for feature_type in ("h", "hgradient"):
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_feature_type(results, feature_type, ax)
        feat_label = "phi_h" if feature_type == "h" else "phi_hgradient"
        fig.suptitle("Flower test MSE vs resolution — non-regularized models", fontsize=13)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"mse_sweep_{feat_label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    import matplotlib.ticker
    results = run_all()
    make_plots(results)
    print("\nDone.")
