"""
Flower-test MSE vs resolution for FIVE models, split into two plots by TRAINING resolution.

Models (each trained @ rho=128 and @ rho=256, all non-regularized L2=0):
  baseline φ/h        (V2,   9D,  no aug)  -> out/{res}/baseline_{res}_h.pt
  V2.2  φ/(α·h)       (9D,  α-aug)         -> out/5527/v22_{res}_ah.pt
  baseline φ/h+∇φ     (V2.1, 27D, no aug)  -> out/{res}/baseline_{res}_hgradient.pt
  V2.3  φ/(α·h)+∇φ    (27D, α-aug)         -> out/5527/v23_{res}_hgradient_ah.pt
  V3    PCA-18        (27D->18)            -> out/5527/v3_{res}_hgradient.pt
plus the central-difference (FD) numeric baseline.

Each model uses its feature-matched flower test data (9D raw -> _h, 27D raw -> _hgradient);
normalization / PCA is taken from each checkpoint (normalization_csv_path=None).
x = test resolution (rho), y = MSE vs analytic h*kappa (log-log).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.flower import evaluate_flower

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "out"
OUTPUT_DIR = OUT / "mse_sweep"

TRAIN_RESOLUTIONS = [128, 256]
TEST_RESOLUTIONS = [128, 256, 512, 1024]

# Ordered so the legend groups 9D (baseline vs α-aug), then 27D (baseline vs α-aug), then V3.
MODELS = [
    {
        "key": "base_h", "label": "baseline φ/h (9D)",
        "color": "#1f77b4", "marker": "o", "data_kind": "h",
        "ckpt": {128: OUT / "128" / "baseline_128_h.pt", 256: OUT / "256" / "baseline_256_h.pt"},
    },
    {
        "key": "v22", "label": "V2.2  φ/(α·h) (9D, α-aug)",
        "color": "#17becf", "marker": "v", "data_kind": "h",
        "ckpt": {128: OUT / "5527" / "v22_128_ah.pt", 256: OUT / "5527" / "v22_256_ah.pt"},
    },
    {
        "key": "base_hg", "label": "baseline φ/h+∇φ (27D)",
        "color": "#9467bd", "marker": "D", "data_kind": "hgradient",
        "ckpt": {128: OUT / "128" / "baseline_128_hgradient.pt",
                 256: OUT / "256" / "baseline_256_hgradient.pt"},
    },
    {
        "key": "v23", "label": "V2.3  φ/(α·h)+∇φ (27D, α-aug)",
        "color": "#ff7f0e", "marker": "s", "data_kind": "hgradient",
        "ckpt": {128: OUT / "5527" / "v23_128_hgradient_ah.pt",
                 256: OUT / "5527" / "v23_256_hgradient_ah.pt"},
    },
    {
        "key": "v3", "label": "V3  PCA-18",
        "color": "#2ca02c", "marker": "^", "data_kind": "hgradient",
        "ckpt": {128: OUT / "5527" / "v3_128_hgradient.pt",
                 256: OUT / "5527" / "v3_256_hgradient.pt"},
    },
]

TEST_DATA = {
    "h": {r: PROJECT_ROOT / "test_data" / f"flower_rho{r}_h.h5" for r in TEST_RESOLUTIONS},
    "hgradient": {r: PROJECT_ROOT / "test_data" / f"flower_rho{r}_hgradient.h5" for r in TEST_RESOLUTIONS},
}

NUMERIC_COLOR = "#7f7f7f"


def run_all() -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {tr: {m["key"]: {} for m in MODELS} for tr in TRAIN_RESOLUTIONS}
    numeric_baseline: dict = {}

    for train_res in TRAIN_RESOLUTIONS:
        print(f"\n=== Models trained @ ρ={train_res} ===")
        for model in MODELS:
            model_path = model["ckpt"][train_res]
            for test_res in TEST_RESOLUTIONS:
                data_path = TEST_DATA[model["data_kind"]][test_res]
                print(f"  {model['key']:7s} train={train_res} | test_rho={test_res} ... ",
                      end="", flush=True)
                result = evaluate_flower(
                    dataset_path=data_path,
                    model_path=model_path,
                    normalization_csv_path=None,
                    device=device,
                )
                mse_model = float(result["model_vs_analytic"]["mse"])
                mse_numeric = float(result["numeric_vs_analytic"]["mse"])
                results[train_res][model["key"]][test_res] = {
                    "model_mse": mse_model, "numeric_mse": mse_numeric,
                }
                numeric_baseline.setdefault(test_res, mse_numeric)
                print(f"model_MSE={mse_model:.4e}  numeric_MSE={mse_numeric:.4e}")

    return results, numeric_baseline


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(TEST_RESOLUTIONS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Test resolution (ρ)", fontsize=12)
    ax.set_ylabel("MSE  (model vs analytic h·κ)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)


def plot_train_res(results: dict, numeric_baseline: dict, train_res: int, ax: plt.Axes) -> None:
    ax.plot(
        TEST_RESOLUTIONS, [numeric_baseline[r] for r in TEST_RESOLUTIONS],
        color=NUMERIC_COLOR, linestyle="--", marker="x", linewidth=1.5,
        markersize=7, label="Central diff (numeric)", zorder=1,
    )
    for model in MODELS:
        mses = [results[train_res][model["key"]][r]["model_mse"] for r in TEST_RESOLUTIONS]
        ax.plot(
            TEST_RESOLUTIONS, mses,
            color=model["color"], linestyle="-", marker=model["marker"],
            linewidth=2, markersize=7, label=model["label"], zorder=2,
        )
    _style_axis(ax)
    ax.set_title(f"Models trained @ ρ={train_res}", fontsize=13)
    ax.legend(fontsize=9)


def make_plots(results: dict, numeric_baseline: dict) -> None:
    for train_res in TRAIN_RESOLUTIONS:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_train_res(results, numeric_baseline, train_res, ax)
        fig.suptitle("Flower-test MSE vs resolution — baselines + α-aug + PCA (non-regularized)",
                     fontsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"mse_all_trained{train_res}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle("Flower-test MSE vs resolution — baselines + α-aug + PCA (non-regularized)",
                 fontsize=14, y=1.02)
    for ax, train_res in zip(axes, TRAIN_RESOLUTIONS):
        plot_train_res(results, numeric_baseline, train_res, ax)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "mse_all_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot: {out_path}")
    plt.close(fig)


def write_csv(results: dict) -> None:
    out_path = OUTPUT_DIR / "mse_all.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["train_res", "model", "test_res", "model_mse", "numeric_mse"])
        for train_res in TRAIN_RESOLUTIONS:
            for model in MODELS:
                for test_res in TEST_RESOLUTIONS:
                    cell = results[train_res][model["key"]][test_res]
                    writer.writerow([train_res, model["key"], test_res,
                                     f"{cell['model_mse']:.6e}", f"{cell['numeric_mse']:.6e}"])
    print(f"Saved table: {out_path}")


if __name__ == "__main__":
    results, numeric_baseline = run_all()
    make_plots(results, numeric_baseline)
    write_csv(results)
    print("\nDone.")
