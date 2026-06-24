"""Symmetry/consistency check on the Test 2 pooled cache, now multi-model.

Panel 1: MSE vs eta, central-diff baseline + one model+/model- pair per
checkpoint in MODELS (positive branch vs exact sign-flipped-twin branch).

Panel 2: for each checkpoint, the FULL 16-element group the part2/DCTS
training augmentation actually used (8 D4 stencil re-orientations x 2 sign
flips -- see train_generate/part2/dcts/augment.py) applied to that
checkpoint's own input. D4 elements leave the target unchanged; sign-flip
negates it -- so after undoing the known sign, all 16 transformed predictions
per sample are independent estimates of the SAME target_hk. Reports per eta
bin: MSE(consensus, target), mean |single - consensus|, max |single -
consensus|.

Caveat (matters for interpreting MODELS that aren't dcts_main_wd0): part2's
DCTS pipeline trains WITH this exact 16-element D4 x sign augmentation, so
its consistency under all 16 views is a "did it learn what it was taught"
check. The part1 baselines under dataset/256_hgradient.h5 etc. only used
augment_sign_flip (Z2), NEVER a D4 stencil-rotation augmentation -- so for
those checkpoints the D4 dimension of this same panel is a genuine OOD
generalization probe (does it generalize to relabeled-stencil orientations it
never saw), not a "did it learn its own training signal" check. Both
checkpoints use the identical 27D phi9/h + nx9 + ny9 (FD-normal) convention
(verified: feature_transform/feature_order match, and a direct sanity check
on this dataset gives sane low-eta predictions -- see verify_methodology.py
patterns), so feeding the same Test 2 cache into both is valid.

X-axis: eta bins ordered LARGE -> SMALL left to right (descending), on an
evenly-spaced categorical axis with the actual eta range as tick labels on a
subset of bins.

Usage:
    python tem/eta_bin_breakdown/plot_symmetry_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate.shared import load_model_from_checkpoint, predict_hkappa_full_batch, resolve_feature_transform
from train_generate.part2.config import D4_NAMES
from train_generate.part2.generate import (
    central_difference_hkappa_from_phi9_float64,
    transform_d4_features28,
    transform_sign_flip_features28,
)

RESOLUTION_SWEEP_H5 = Path("dataset/part2_dcts/main/processed/resolution_sweep.h5")
OUTPUT_PNG = Path("tem/eta_bin_breakdown/symmetry_comparison.png")
ETA_MIN, ETA_MAX = 0.006, 2.0 / 3.0
N_BINS = 25
TICK_EVERY = 3

# (label, checkpoint path, trained_with_d4_aug)
MODELS = [
    ("dcts_main_wd0 (part2/DCTS)", Path("out/7367/dcts_main_wd0.pt"), True),
    ("baseline_256_hgradient (part1)", Path("out/256/baseline_256_hgradient.pt"), False),
]

# distinct color pairs per model: (model_pos, model_neg, consensus, violation)
COLOR_SETS = [
    ("#d62728", "#1f77b4", "#ff7f0e", "#9467bd"),
    ("#2ca02c", "#8c564b", "#17becf", "#bcbd22"),
]


def evaluate_model(model_path: Path, features27: np.ndarray, hk_central: np.ndarray, target_hk: np.ndarray, device):
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    transform, _ = resolve_feature_transform(model_path=model_path, explicit_path=None, checkpoint_meta=checkpoint_meta)

    pred_pos = predict_hkappa_full_batch(model, features27, transform=transform, device=device)
    features27_neg = (-features27).astype(np.float32)
    pred_neg = predict_hkappa_full_batch(model, features27_neg, transform=transform, device=device)

    features28_raw = np.concatenate([features27.astype(np.float64), hk_central.reshape(-1, 1)], axis=1)
    aligned = np.empty((16, features27.shape[0]), dtype=np.float64)
    combo = 0
    for d4_name in D4_NAMES:
        d4_feat28 = transform_d4_features28(features28_raw, d4_name=d4_name)
        for sign_id in (0, 1):
            if sign_id:
                feat27_g, _ = transform_sign_flip_features28(d4_feat28[:, :27], target_hk)
                parity = -1.0
            else:
                feat27_g = d4_feat28[:, :27]
                parity = 1.0
            pred_g = predict_hkappa_full_batch(model, feat27_g.astype(np.float32), transform=transform, device=device)
            aligned[combo] = parity * pred_g
            combo += 1
    consensus = aligned.mean(axis=0)
    mean_violation = np.mean(np.abs(aligned - consensus[None, :]), axis=0)
    max_violation = np.max(np.abs(aligned - consensus[None, :]), axis=0)
    return {
        "pred_pos": pred_pos, "pred_neg": pred_neg,
        "consensus": consensus, "mean_violation": mean_violation, "max_violation": max_violation,
    }


def bin_stat(values_mask, fn):
    return float(fn(values_mask)) if values_mask.size else float("nan")


def main() -> None:
    with h5py.File(RESOLUTION_SWEEP_H5, "r") as handle:
        target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
        hk_central = np.asarray(handle["hk_central"][:], dtype=np.float64).reshape(-1)
        features27 = np.asarray(handle["features27"][:], dtype=np.float32)
    target_hk_neg = -target_hk
    hk_central_neg = -hk_central

    phi9_neg = -features27[:, :9].astype(np.float64)
    fd_oddness_residual = float(np.max(np.abs(
        central_difference_hkappa_from_phi9_float64(phi9_neg).reshape(-1) - (-hk_central)
    )))
    print(f"FD central-diff exact-oddness check: max|FD(-phi9) - (-FD(phi9))| = {fd_oddness_residual:.3e} (should be ~0)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for label, path, _ in MODELS:
        print(f"Evaluating {label} ({path}) ...")
        res = evaluate_model(path, features27, hk_central, target_hk, device)
        consensus_mse = float(np.mean((res["consensus"] - target_hk) ** 2))
        plain_mse = float(np.mean((res["pred_pos"] - target_hk) ** 2))
        print(f"  16-fold consensus MSE = {consensus_mse:.3e}  vs plain single-orientation MSE = {plain_mse:.3e}")
        results[label] = res

    eta = np.abs(target_hk)
    edges = ETA_MIN * np.power(ETA_MAX / ETA_MIN, np.arange(N_BINS + 1) / N_BINS)
    bin_idx = np.clip(np.searchsorted(edges, eta, side="right") - 1, 0, N_BINS - 1)
    order = np.arange(N_BINS)[::-1]
    labels = [f"[{edges[b]:.3f},{edges[b+1]:.3f})" for b in range(N_BINS)]
    labels_ord = [labels[b] for b in order]
    xs = np.arange(N_BINS)

    def reorder(arr: np.ndarray) -> np.ndarray:
        return arr[order]

    central_mse_pos = np.full(N_BINS, np.nan)
    central_mse_neg = np.full(N_BINS, np.nan)
    counts = np.zeros(N_BINS, dtype=np.int64)
    per_model_bins = {
        label: {k: np.full(N_BINS, np.nan) for k in ("mse_pos", "mse_neg", "cons_mse", "mean_viol", "max_viol")}
        for label, _, _ in MODELS
    }

    for b in range(N_BINS):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            continue
        central_mse_pos[b] = float(np.mean((hk_central[mask] - target_hk[mask]) ** 2))
        central_mse_neg[b] = float(np.mean((hk_central_neg[mask] - target_hk_neg[mask]) ** 2))
        for label, _, _ in MODELS:
            res = results[label]
            bins = per_model_bins[label]
            bins["mse_pos"][b] = float(np.mean((res["pred_pos"][mask] - target_hk[mask]) ** 2))
            bins["mse_neg"][b] = float(np.mean((res["pred_neg"][mask] - target_hk_neg[mask]) ** 2))
            bins["cons_mse"][b] = float(np.mean((res["consensus"][mask] - target_hk[mask]) ** 2))
            bins["mean_viol"][b] = float(np.mean(res["mean_violation"][mask]))
            bins["max_viol"][b] = float(np.max(res["max_violation"][mask]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.3]})

    ax1.plot(xs, reorder(central_mse_pos), color="#7f7f7f", linestyle="--", marker="x", linewidth=1.4, markersize=5,
              label="Central diff, eta>0", zorder=1)
    ax1.plot(xs, reorder(central_mse_neg), color="#1a1a1a", linestyle=":", marker="+", linewidth=1.4, markersize=7,
              label="Central diff, eta<0 (twin)", zorder=1)
    for (label, _, trained_d4), (c_pos, c_neg, _, _) in zip(MODELS, COLOR_SETS):
        bins = per_model_bins[label]
        ax1.plot(xs, reorder(bins["mse_pos"]), color=c_pos, marker="o", linewidth=1.7, markersize=4.5,
                  label=f"{label}, eta>0", zorder=3)
        ax1.plot(xs, reorder(bins["mse_neg"]), color=c_neg, marker="s", linewidth=1.7, markersize=4.5, linestyle="-.",
                  label=f"{label}, eta<0 (twin)", zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("MSE(h*kappa)", fontsize=11)
    ax1.set_title("Test 2 (pooled rho=32..1024): MSE vs eta bin, descending |h*kappa|", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right", ncol=1)

    for (label, _, trained_d4), (_, _, c_cons, c_viol) in zip(MODELS, COLOR_SETS):
        bins = per_model_bins[label]
        d4_tag = "D4-trained" if trained_d4 else "D4 NEVER trained (OOD probe)"
        ax2.plot(xs, reorder(bins["cons_mse"]), color=c_cons, marker="^", linewidth=1.7, markersize=4.5,
                  label=f"{label}: MSE(16-fold consensus, target)")
        ax2.plot(xs, reorder(bins["mean_viol"]), color=c_viol, marker="o", linewidth=1.7, markersize=4.5,
                  label=f"{label}: mean violation [{d4_tag}]")
        ax2.plot(xs, reorder(bins["max_viol"]), color=c_viol, linestyle="--", linewidth=1.2, alpha=0.7,
                  label=f"{label}: max violation")
    ax2.set_yscale("log")
    ax2.set_xlabel("eta bin (descending |h*kappa| -->)", fontsize=11)
    ax2.set_ylabel("16-fold D4xsign\nconsistency", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7, loc="upper right", ncol=1)

    tick_pos = xs[::TICK_EVERY]
    tick_lab = [labels_ord[i] for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=8)

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PNG}")

    header = f"{'eta range (desc)':<20s}{'n':>7s}{'FD+':>10s}{'FD-':>10s}"
    for label, _, _ in MODELS:
        short = label.split(" ")[0]
        header += f"{short+'_mse+':>16s}{short+'_mse-':>16s}{short+'_cons':>16s}{short+'_mnV':>14s}{short+'_mxV':>14s}"
    print("\n" + header)
    for b in order:
        row = f"{labels[b]:<20s}n={counts[b]:5d}  {central_mse_pos[b]:.2e} {central_mse_neg[b]:.2e}"
        for label, _, _ in MODELS:
            bins = per_model_bins[label]
            row += (
                f"  {bins['mse_pos'][b]:.2e} {bins['mse_neg'][b]:.2e} "
                f"{bins['cons_mse'][b]:.2e} {bins['mean_viol'][b]:.2e} {bins['max_viol'][b]:.2e}"
            )
        print(row)


if __name__ == "__main__":
    main()
