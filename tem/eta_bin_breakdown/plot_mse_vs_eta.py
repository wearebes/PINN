"""Ad hoc plot: MSE(h*kappa) vs eta=|h*kappa|, fine log-bins, central-diff vs model.

Companion to check_eta_bins.py's coarse 4-bin table -- this renders the actual
curve the user asked for (log-log MSE vs eta) using ~25 log-uniform bins over
the locked [eta_min, eta_max] = [0.006, 2/3] range, pooled across all 6 grid
resolutions in the Test 2 cache.

Usage:
    python tem/eta_bin_breakdown/plot_mse_vs_eta.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate.shared import load_model_from_checkpoint, predict_hkappa_full_batch, resolve_feature_transform

RESOLUTION_SWEEP_H5 = Path("dataset/part2_dcts/main/processed/resolution_sweep.h5")
MODEL_PATH = Path("out/7367/dcts_main_wd0.pt")
OUTPUT_PNG = Path("tem/eta_bin_breakdown/mse_vs_eta.png")
ETA_MIN, ETA_MAX = 0.006, 2.0 / 3.0
N_BINS = 25


def main() -> None:
    with h5py.File(RESOLUTION_SWEEP_H5, "r") as handle:
        target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
        hk_central = np.asarray(handle["hk_central"][:], dtype=np.float64).reshape(-1)
        features27 = np.asarray(handle["features27"][:], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_meta = load_model_from_checkpoint(MODEL_PATH, device=device)
    transform, _ = resolve_feature_transform(model_path=MODEL_PATH, explicit_path=None, checkpoint_meta=checkpoint_meta)
    pred = predict_hkappa_full_batch(model, features27, transform=transform, device=device)

    eta = np.abs(target_hk)
    edges = ETA_MIN * np.power(ETA_MAX / ETA_MIN, np.arange(N_BINS + 1) / N_BINS)
    centers = np.sqrt(edges[:-1] * edges[1:])
    bin_idx = np.clip(np.searchsorted(edges, eta, side="right") - 1, 0, N_BINS - 1)

    central_mse = np.full(N_BINS, np.nan)
    model_mse = np.full(N_BINS, np.nan)
    counts = np.zeros(N_BINS, dtype=np.int64)
    for b in range(N_BINS):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            continue
        central_mse[b] = float(np.mean((hk_central[mask] - target_hk[mask]) ** 2))
        model_mse[b] = float(np.mean((pred[mask] - target_hk[mask]) ** 2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(centers, central_mse, color="#7f7f7f", linestyle="--", marker="x", linewidth=1.5, markersize=6,
            label="Central diff (no model)")
    ax.plot(centers, model_mse, color="#d62728", marker="o", linewidth=1.8, markersize=5,
            label=f"model ({MODEL_PATH.stem})")
    for x in (0.02, 0.1, 0.3):
        ax.axvline(x, color="black", alpha=0.15, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eta = |h*kappa|", fontsize=12)
    ax.set_ylabel("MSE(h*kappa)", fontsize=12)
    ax.set_title("Test 2 (resolution sweep, pooled rho=32..1024): MSE vs eta", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUTPUT_PNG}")
    print(f"{'eta range':<22s}{'n':>8s}{'central MSE':>14s}{'model MSE':>14s}")
    for b in range(N_BINS):
        print(f"[{edges[b]:.4f},{edges[b+1]:.4f})  n={counts[b]:6d}  central={central_mse[b]:.3e}  model={model_mse[b]:.3e}")


if __name__ == "__main__":
    main()
