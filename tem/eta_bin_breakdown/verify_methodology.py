"""Ad hoc verification of two methodology worries raised by the user:

1. Does model evaluation use the TRAIN-derived feature mean/std (no leakage from
   val/test/eval-only data), and does it actually match the real training
   population (not just trust the 'source_split=train' label)?
2. Is the reported "overall" MSE the correctly sample-count-weighted pooled
   MSE, or could it secretly be an unweighted average of the 8 coarse_regime
   bin MSEs (which would over-weight the rare high-curvature bins)? Also:
   the canonical val/test splits never include negative h*kappa (no sign-flip
   augmentation outside train) -- does the model actually generalize
   symmetrically to negative curvature, or has nothing ever tested that?

Usage:
    python tem/eta_bin_breakdown/verify_methodology.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate.shared import compute_metrics, load_model_from_checkpoint, predict_hkappa_full_batch, resolve_feature_transform

MODEL_PATH = Path("out/7367/dcts_main_wd0.pt")
TRAIN_H5 = Path("dataset/part2_dcts/main/processed/train.h5")
RESOLUTION_SWEEP_H5 = Path("dataset/part2_dcts/main/processed/resolution_sweep.h5")
TEST_FD_H5 = Path("dataset/part2_dcts/main/processed/test_fd.h5")


def check_normalization_provenance() -> dict:
    print("=== 1. Normalization (mean/std) provenance ===")
    device = torch.device("cpu")
    model, checkpoint_meta = load_model_from_checkpoint(MODEL_PATH, device=device)
    transform, transform_source = resolve_feature_transform(model_path=MODEL_PATH, explicit_path=None, checkpoint_meta=checkpoint_meta)
    print(f"  transform_source actually used by evaluate/* and our scripts: {transform_source!r}")
    print(f"  embedded transform: source_split={transform['source_split']!r}  dataset_path={transform['dataset_path']!r}")

    with h5py.File(TRAIN_H5, "r") as handle:
        feats = np.asarray(handle["features27"][:], dtype=np.float64)
    real_mean = feats.mean(axis=0)
    real_std = feats.std(axis=0)
    ckpt_mean = np.asarray(transform["mean"], dtype=np.float64)
    ckpt_std = np.asarray(transform["std"], dtype=np.float64)
    rel_err_std = np.max(np.abs(real_std - ckpt_std) / np.abs(real_std))
    max_abs_err_mean = np.max(np.abs(real_mean - ckpt_mean))
    print(f"  recomputed directly from local train.h5 (N={feats.shape[0]}):")
    print(f"    max|mean_train - mean_checkpoint|       = {max_abs_err_mean:.3e}")
    print(f"    max relative |std_train - std_checkpoint| = {rel_err_std:.3e}")
    print("  (small relative gap expected: local train.h5 vs the cluster's converted v7.h5 copy;")
    print("   a val/test-leak bug would show as a LARGE, qualitatively different gap, not this.)")
    return {"model": model, "transform": transform, "device": device}


def naive_vs_weighted_bin_average() -> None:
    print("\n=== 2a. Is 'overall' MSE an unweighted average of 8 coarse-regime MSEs? ===")
    import csv

    rows = []
    with open("out/7367/test2_resolution_sweep_metrics.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["resolution"] == "32" and row["model"] == "dcts_main_wd0" and row["group_kind"] == "coarse_regime":
                rows.append((int(row["sample_count"]), float(row["model_mse"])))
    assert len(rows) == 8, f"expected 8 coarse_regime rows for rho=32, got {len(rows)}"
    counts = np.array([r[0] for r in rows], dtype=np.float64)
    mses = np.array([r[1] for r in rows], dtype=np.float64)
    naive_unweighted = float(np.mean(mses))
    weighted = float(np.sum(counts * mses) / np.sum(counts))
    print(f"  per-regime sample counts: {counts.astype(int).tolist()}  (sum={int(counts.sum())})")
    print(f"  naive unweighted mean-of-8-MSEs : {naive_unweighted:.4e}")
    print(f"  sample-count-weighted mean      : {weighted:.4e}")
    print(f"  ratio naive/weighted            : {naive_unweighted/weighted:.2f}x")

    with h5py.File(RESOLUTION_SWEEP_H5, "r") as handle:
        rho_col = np.asarray(handle["rho"][:], dtype=np.int32)
        target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
        features27 = np.asarray(handle["features27"][:], dtype=np.float32)
    mask = rho_col == 32
    device = torch.device("cpu")
    model, checkpoint_meta = load_model_from_checkpoint(MODEL_PATH, device=device)
    transform, _ = resolve_feature_transform(model_path=MODEL_PATH, explicit_path=None, checkpoint_meta=checkpoint_meta)
    pred = predict_hkappa_full_batch(model, features27[mask], transform=transform, device=device)
    true_pooled = compute_metrics(pred, target_hk[mask])["mse"]
    print(f"  TRUE pooled MSE recomputed from raw rho=32 samples (N={int(mask.sum())}): {true_pooled:.4e}")
    print(f"  -> matches weighted mean? {np.isclose(true_pooled, weighted, rtol=1e-3)}  "
          f"matches naive unweighted? {np.isclose(true_pooled, naive_unweighted, rtol=1e-3)}")


def sign_symmetry_check(model, transform, device) -> None:
    print("\n=== 2b. Negative-curvature generalization (val/test/eval are all eta>0; train is sign-balanced) ===")
    for label, path in (("Test 2 (resolution_sweep, pooled)", RESOLUTION_SWEEP_H5), ("Test 1 (test_fd)", TEST_FD_H5)):
        with h5py.File(path, "r") as handle:
            target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
            features27 = np.asarray(handle["features27"][:], dtype=np.float32)
        frac_neg = float(np.mean(target_hk < 0))
        pred_pos = predict_hkappa_full_batch(model, features27, transform=transform, device=device)
        mse_pos = compute_metrics(pred_pos, target_hk)["mse"]

        features27_flipped = (-features27).astype(np.float32)
        target_hk_flipped = -target_hk
        pred_neg = predict_hkappa_full_batch(model, features27_flipped, transform=transform, device=device)
        mse_neg = compute_metrics(pred_neg, target_hk_flipped)["mse"]

        exact_antisymmetric = float(np.max(np.abs(pred_neg - (-pred_pos))))
        print(f"  [{label}] N={target_hk.size}  frac(target_hk<0) in this eval set = {frac_neg:.3f}")
        print(f"    MSE on as-stored (all eta>0) inputs      : {mse_pos:.4e}")
        print(f"    MSE on sign-flipped (eta<0 twin) inputs   : {mse_neg:.4e}  (ratio neg/pos = {mse_neg/mse_pos:.3f}x)")
        print(f"    max|model(-x) - (-model(x))| (exact-antisymmetry residual): {exact_antisymmetric:.4e}")


def main() -> None:
    ctx = check_normalization_provenance()
    naive_vs_weighted_bin_average()
    sign_symmetry_check(ctx["model"], ctx["transform"], ctx["device"])


if __name__ == "__main__":
    main()
