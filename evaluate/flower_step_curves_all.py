"""
Flower MSE vs REINIT STEP (1..20), not averaged over steps.

Two figures, one per TRAINING resolution. Each figure is a 2x4 grid:
  rows  = flower shape (smooth, acute)
  cols  = test resolution (128, 256, 512, 1024)
Each panel: 5 models + FD numeric baseline, x = reinit step, y = MSE (log).

Per-step MSE comes from evaluate_flower(...)["cases"], where every (case_label, iter)
row carries its own model_vs_analytic / numeric_vs_analytic metrics
(same data source used by evaluate/flower_step_plots.py).

Models / test-data mapping / colors are reused from mse_resolution_sweep_all.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.flower import evaluate_flower
from evaluate.mse_resolution_sweep_all import (
    MODELS, TEST_DATA, TRAIN_RESOLUTIONS, TEST_RESOLUTIONS, OUTPUT_DIR, NUMERIC_COLOR,
)

SHAPES = ["smooth", "acute"]


def _family(case_label: str) -> str:
    return str(case_label).split("_")[0]


def run_all() -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # results[train_res][model_key][test_res][shape] = {step: mse}
    results: dict = {
        tr: {m["key"]: {ts: {} for ts in TEST_RESOLUTIONS} for m in MODELS}
        for tr in TRAIN_RESOLUTIONS
    }
    # numeric[test_res][shape] = {step: mse}  (model-independent FD baseline)
    numeric: dict = {ts: {} for ts in TEST_RESOLUTIONS}

    for train_res in TRAIN_RESOLUTIONS:
        print(f"\n=== trained @ρ={train_res} ===")
        for model in MODELS:
            model_path = model["ckpt"][train_res]
            for test_res in TEST_RESOLUTIONS:
                data_path = TEST_DATA[model["data_kind"]][test_res]
                print(f"  {model['key']:7s} train={train_res} | test={test_res} ...", flush=True)
                res = evaluate_flower(
                    dataset_path=data_path, model_path=model_path,
                    normalization_csv_path=None, device=device,
                )
                for row in res["cases"]:
                    fam = _family(row["case_label"])
                    step = int(row["iter"])
                    results[train_res][model["key"]][test_res].setdefault(fam, {})[step] = \
                        float(row["model_vs_analytic"]["mse"])
                    numeric[test_res].setdefault(fam, {}).setdefault(
                        step, float(row["numeric_vs_analytic"]["mse"]))

    return results, numeric


def make_figure(results: dict, numeric: dict, train_res: int) -> None:
    nrows, ncols = len(SHAPES), len(TEST_RESOLUTIONS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows), sharex=True)
    handles, labels = [], []

    for i, shape in enumerate(SHAPES):
        for j, test_res in enumerate(TEST_RESOLUTIONS):
            ax = axes[i][j]
            nd = numeric[test_res].get(shape, {})
            n_steps = sorted(nd)
            if n_steps:
                ax.plot(n_steps, [nd[s] for s in n_steps], color=NUMERIC_COLOR,
                        linestyle="--", marker="x", markersize=3, linewidth=1.4,
                        label="Central diff (numeric)")
            for model in MODELS:
                md = results[train_res][model["key"]][test_res].get(shape, {})
                steps = sorted(md)
                if steps:
                    ax.plot(steps, [md[s] for s in steps], color=model["color"],
                            linestyle="-", marker=model["marker"], markersize=3,
                            linewidth=1.5, label=model["label"])
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{shape},  test ρ={test_res}", fontsize=11)
            if i == nrows - 1:
                ax.set_xlabel("reinit step")
            if j == 0:
                ax.set_ylabel("MSE  (model vs analytic h·κ)", fontsize=10)
            if not handles:
                handles, labels = ax.get_legend_handles_labels()

    fig.suptitle(f"Flower MSE vs reinit step — models trained @ ρ={train_res}",
                 fontsize=15, y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    out = OUTPUT_DIR / f"mse_step_trained{train_res}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def write_csv(results: dict, numeric: dict) -> None:
    out = OUTPUT_DIR / "mse_step.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["train_res", "model", "test_res", "shape", "step", "model_mse", "numeric_mse"])
        for tr in TRAIN_RESOLUTIONS:
            for model in MODELS:
                for ts in TEST_RESOLUTIONS:
                    for shape, steps in results[tr][model["key"]][ts].items():
                        for step in sorted(steps):
                            nm = numeric[ts].get(shape, {}).get(step, float("nan"))
                            w.writerow([tr, model["key"], ts, shape, step,
                                        f"{steps[step]:.6e}", f"{nm:.6e}"])
    print(f"Saved table: {out}")


if __name__ == "__main__":
    results, numeric = run_all()
    for train_res in TRAIN_RESOLUTIONS:
        make_figure(results, numeric, train_res)
    write_csv(results, numeric)
    print("\nDone.")
