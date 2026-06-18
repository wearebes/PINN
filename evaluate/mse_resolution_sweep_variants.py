"""
Flower-test MSE vs resolution for the THREE trained feature variants (V2.2 / V2.3 / V3),
each trained at rho=128 and rho=256, evaluated at rho in {128, 256, 512, 1024}.

Layout: two plots, split by TRAINING resolution.
  - Plot 1 (trained @128): three model curves (V2.2, V2.3, V3) + FD numeric baseline.
  - Plot 2 (trained @256): same three variants.
x = test resolution (rho), y = MSE vs analytic h*kappa.

Each variant uses its feature-matched flower test data:
  V2.2 (raw 9D, phi/h)        -> flower_rho{N}_h.h5
  V2.3 (raw 27D, phi/h+grad)  -> flower_rho{N}_hgradient.h5
  V3   (raw 27D -> PCA18)      -> flower_rho{N}_hgradient.h5  (checkpoint PCA reduces 27->18)
Feature normalization / PCA is taken from each checkpoint's embedded transform
(normalization_csv_path=None). All three variants are non-regularized (L2=0).
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
MODEL_DIR = PROJECT_ROOT / "out" / "5527"
OUTPUT_DIR = PROJECT_ROOT / "out" / "mse_sweep"

TRAIN_RESOLUTIONS = [128, 256]
TEST_RESOLUTIONS = [128, 256, 512, 1024]

# Three trained variants. data_kind picks the feature-matched flower test file.
VARIANTS = [
    {
        "key": "v22",
        "label": "V2.2  (φ/h, 9D)",
        "color": "#1f77b4",
        "marker": "o",
        "data_kind": "h",
        "ckpt": {
            128: MODEL_DIR / "v22_128_ah.pt",
            256: MODEL_DIR / "v22_256_ah.pt",
        },
    },
    {
        "key": "v23",
        "label": "V2.3  (φ/h + ∇φ, 27D)",
        "color": "#ff7f0e",
        "marker": "s",
        "data_kind": "hgradient",
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
        "data_kind": "hgradient",
        "ckpt": {
            128: MODEL_DIR / "v3_128_hgradient.pt",
            256: MODEL_DIR / "v3_256_hgradient.pt",
        },
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

    # results[train_res][variant_key][test_res] = {"model_mse":..., "numeric_mse":...}
    results: dict = {tr: {v["key"]: {} for v in VARIANTS} for tr in TRAIN_RESOLUTIONS}
    # FD baseline depends only on test_res (same phi9 across _h / _hgradient files).
    numeric_baseline: dict = {}

    for train_res in TRAIN_RESOLUTIONS:
        print(f"\n=== Models trained @ ρ={train_res} ===")
        for variant in VARIANTS:
            model_path = variant["ckpt"][train_res]
            for test_res in TEST_RESOLUTIONS:
                data_path = TEST_DATA[variant["data_kind"]][test_res]
                print(f"  {variant['key']:4s} train={train_res} | test_rho={test_res} ... ",
                      end="", flush=True)
                result = evaluate_flower(
                    dataset_path=data_path,
                    model_path=model_path,
                    normalization_csv_path=None,
                    device=device,
                )
                mse_model = float(result["model_vs_analytic"]["mse"])
                mse_numeric = float(result["numeric_vs_analytic"]["mse"])
                results[train_res][variant["key"]][test_res] = {
                    "model_mse": mse_model,
                    "numeric_mse": mse_numeric,
                }
                # FD baseline is model-independent; capture once per test_res.
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
        TEST_RESOLUTIONS,
        [numeric_baseline[r] for r in TEST_RESOLUTIONS],
        color=NUMERIC_COLOR, linestyle="--", marker="x", linewidth=1.5,
        markersize=7, label="Central diff (numeric)", zorder=1,
    )
    for variant in VARIANTS:
        mses = [results[train_res][variant["key"]][r]["model_mse"] for r in TEST_RESOLUTIONS]
        ax.plot(
            TEST_RESOLUTIONS, mses,
            color=variant["color"], linestyle="-", marker=variant["marker"],
            linewidth=2, markersize=7, label=variant["label"], zorder=2,
        )
    _style_axis(ax)
    ax.set_title(f"Models trained @ ρ={train_res}", fontsize=13)
    ax.legend(fontsize=10)


def make_plots(results: dict, numeric_baseline: dict) -> None:
    # Two standalone figures (the deliverable).
    for train_res in TRAIN_RESOLUTIONS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plot_train_res(results, numeric_baseline, train_res, ax)
        fig.suptitle("Flower-test MSE vs resolution — three variants (non-regularized)",
                     fontsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"mse_variants_trained{train_res}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)

    # Combined side-by-side (bonus).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Flower-test MSE vs resolution — three variants (non-regularized)",
                 fontsize=14, y=1.02)
    for ax, train_res in zip(axes, TRAIN_RESOLUTIONS):
        plot_train_res(results, numeric_baseline, train_res, ax)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "mse_variants_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot: {out_path}")
    plt.close(fig)


def write_csv(results: dict, numeric_baseline: dict) -> None:
    out_path = OUTPUT_DIR / "mse_variants.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["train_res", "variant", "test_res", "model_mse", "numeric_mse"])
        for train_res in TRAIN_RESOLUTIONS:
            for variant in VARIANTS:
                for test_res in TEST_RESOLUTIONS:
                    cell = results[train_res][variant["key"]][test_res]
                    writer.writerow([
                        train_res, variant["key"], test_res,
                        f"{cell['model_mse']:.6e}",
                        f"{cell['numeric_mse']:.6e}",
                    ])
    print(f"Saved table: {out_path}")


if __name__ == "__main__":
    results, numeric_baseline = run_all()
    make_plots(results, numeric_baseline)
    write_csv(results, numeric_baseline)
    print("\nDone.")
