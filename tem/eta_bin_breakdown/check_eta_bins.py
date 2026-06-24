"""Ad hoc check: bin Test 2 (resolution-sweep) error by eta=|h*kappa|.

Question: does the central-diff (no-model) baseline's ~1e-3-2e-3 overall MSE
(the gray dashed line in out/7367/test2_resolution_sweep.png) come mostly
from the high-curvature bins, or is it spread evenly? Decomposes
mean((hk_central - target_hk)^2) by eta bin and reports each bin's share of
the total sum-of-squared-error (SSE), pooled and per resolution.

Usage:
    python tem/eta_bin_breakdown/check_eta_bins.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

RESOLUTION_SWEEP_H5 = Path("dataset/part2_dcts/main/processed/resolution_sweep.h5")
TEST_FD_H5 = Path("dataset/part2_dcts/main/processed/test_fd.h5")
EDGES = (0.006, 0.02, 0.1, 0.3, 2.0 / 3.0)


def load(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        out = {
            "target_hk": np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1),
            "hk_central": np.asarray(handle["hk_central"][:], dtype=np.float64).reshape(-1),
        }
        out["rho"] = np.asarray(handle["rho"][:], dtype=np.int32) if "rho" in handle else None
    return out


def report(eta: np.ndarray, hk_central: np.ndarray, target_hk: np.ndarray, *, label: str) -> None:
    sq_err = (hk_central - target_hk) ** 2
    bin_idx = np.clip(np.searchsorted(EDGES, eta, side="right") - 1, 0, len(EDGES) - 2)
    total_sse = float(sq_err.sum())
    total_mse = float(sq_err.mean())
    print(f"  [{label}] N={eta.size}  overall central-diff MSE={total_mse:.4e}")
    for b in range(len(EDGES) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            print(f"    eta in [{EDGES[b]:.3f},{EDGES[b+1]:.3f}): n=0")
            continue
        bin_sse = float(sq_err[mask].sum())
        bin_mse = bin_sse / n
        frac = 100.0 * bin_sse / total_sse if total_sse > 0 else float("nan")
        print(
            f"    eta in [{EDGES[b]:.3f},{EDGES[b+1]:.3f}): n={n:6d} ({100.0*n/eta.size:4.1f}%)  "
            f"MSE={bin_mse:.4e}  share of total SSE={frac:5.1f}%"
        )


def main() -> None:
    print("=== Test 2 (resolution sweep): central-diff baseline error by eta bin ===")
    data = load(RESOLUTION_SWEEP_H5)
    eta = np.abs(data["target_hk"])
    report(eta, data["hk_central"], data["target_hk"], label="pooled (all resolutions)")
    if data["rho"] is not None:
        for rho in sorted(set(data["rho"].tolist())):
            mask = data["rho"] == rho
            report(eta[mask], data["hk_central"][mask], data["target_hk"][mask], label=f"rho={rho}")

    if TEST_FD_H5.exists():
        print("\n=== Test 1 (in-distribution geometries, central-diff normals) ===")
        data1 = load(TEST_FD_H5)
        eta1 = np.abs(data1["target_hk"])
        report(eta1, data1["hk_central"], data1["target_hk"], label="test_fd.h5")


if __name__ == "__main__":
    main()
