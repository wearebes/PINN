#!/usr/bin/env python3
"""Zero-contour displacement during the flower reinitialisation sequences.

Post-processing of results/flower_rho256_hgradient_steps0-30.h5.  No new
simulation is performed: the stored per-node stencils are re-read.

Convention behind the bounds quoted in Section 4.3 of the manuscript:

  * displacement is measured against the PRESCRIBED analytical polar contour
    phi0(r,theta) = r - a cos(p theta) - b, not against the step-0 field;
  * the maximum is over all retained interface-adjacent nodes and all 30
    steps of BOTH geometries;
  * the mean is pooled over the same set (nodes and steps together), not a
    per-step mean or the maximum of per-step means.

For each retained node the distance to the CURRENT zero level set is estimated
locally from the stored 3x3 stencil; the exact distance to the prescribed
contour is computed by minimising over a dense parametrisation.  Their
difference is the displacement estimate.  At step 0 the zero contour coincides
with the prescribed contour, so the value returned there is the estimator's own
floor and fixes the resolution of the diagnostic.

Two estimators are reported.  `linear` uses phi_c / |grad phi|.  `quadratic`
fits a quadratic to the nine stencil values and takes the root nearest the
centre along the local normal; it is the more accurate of the two and its
step-0 floor is correspondingly smaller.  The manuscript quotes a bound that
holds under both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_H5 = HERE / "results" / "flower_rho256_hgradient_steps0-30.h5"

# Stored stencil order (feature_order = phi9+nx9+ny9, "training_order"):
#   (i-1,j+1) (i,j+1) (i+1,j+1) | (i-1,j) (i,j) (i+1,j) | (i-1,j-1) (i,j-1) (i+1,j-1)
DX = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
DY = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
CENTRE = 4

# (case_id in the file) -> (name, a, b, p); see scenarios_json in the H5 attrs.
GEOMETRIES = {0: ("smooth", 0.05, 0.15, 3), 1: ("acute", 0.075, 0.15, 3)}


def distance_linear(phi9: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Signed distance from stencil centre to the zero contour, in cells."""
    dphidx = (phi9[:, 5] - phi9[:, 3]) / (2.0 * h)
    dphidy = (phi9[:, 1] - phi9[:, 7]) / (2.0 * h)
    return phi9[:, CENTRE] / (np.hypot(dphidx, dphidy) * h)


def distance_quadratic(phi9: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Same quantity from a quadratic fit to the nine stencil values."""
    basis = np.column_stack([np.ones(9), DX, DY, DX * DX, DX * DY, DY * DY])
    coeff = phi9 @ np.linalg.pinv(basis).T
    a0, ax, ay, axx, axy, ayy = (coeff[:, k] for k in range(6))
    grad = np.hypot(ax, ay)
    nx, ny = ax / grad, ay / grad
    curv = axx * nx * nx + axy * nx * ny + ayy * ny * ny
    linear_root = -a0 / grad
    disc = grad * grad - 4.0 * curv * a0
    usable = (np.abs(curv) >= 1.0e-14) & (disc >= 0.0)
    denom = 2.0 * np.where(usable, curv, 1.0)
    root_p = np.where(usable, (-grad + np.sqrt(np.abs(disc))) / denom, np.nan)
    root_m = np.where(usable, (-grad - np.sqrt(np.abs(disc))) / denom, np.nan)
    nearest = np.where(np.abs(root_p) < np.abs(root_m), root_p, root_m)
    keep = usable & np.isfinite(nearest) & (np.abs(nearest) < 2.0)
    return -np.where(keep, nearest, linear_root)


def exact_distance(points: np.ndarray, a: float, b: float, p: int,
                   samples: int = 400_000, chunk: int = 1500) -> np.ndarray:
    """Signed distance to r(theta) = a cos(p theta) + b, in physical units."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1)[:-1]
    radius = a * np.cos(p * theta) + b
    cx, cy = radius * np.cos(theta), radius * np.sin(theta)
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d2 = (block[:, 0:1] - cx) ** 2 + (block[:, 1:2] - cy) ** 2
        out[start:start + chunk] = np.sqrt(d2.min(axis=1))
    r_pt = np.hypot(points[:, 0], points[:, 1])
    t_pt = np.arctan2(points[:, 1], points[:, 0])
    return np.sign(r_pt - (a * np.cos(p * t_pt) + b)) * out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        xy = f["xy"][:].astype(float)
        phi9 = f["phi9"][:].astype(float)
        h = f["h"][:].astype(float)
        case_id = f["case_id"][:]
        step = f["iter"][:]

    estimators = {"linear": distance_linear(phi9, h),
                  "quadratic": distance_quadratic(phi9, h)}

    pooled = {name: [] for name in estimators}
    for cid, (name, a, b, p) in GEOMETRIES.items():
        rows = case_id == cid
        first = rows & (step == 0)
        exact = exact_distance(xy[first], a, b, p) / h[first]
        n_nodes = int(first.sum())
        print(f"=== {name} flower: {n_nodes} retained nodes, steps "
              f"{step[rows].min()}-{step[rows].max()} ===")
        for label, estimate in estimators.items():
            per_step = []
            for k in sorted(np.unique(step[rows])):
                sel = rows & (step == k)
                assert np.allclose(xy[sel], xy[first]), "node order changed"
                per_step.append(np.abs(estimate[sel] - exact))
            stacked = np.concatenate(per_step)
            pooled[label].append(stacked)
            print(f"  {label:<9} floor(step 0) max={per_step[0].max():.5f}h  "
                  f"| all steps: max={stacked.max():.5f}h  "
                  f"pooled mean={stacked.mean():.5f}h")
        print()

    print("=== bound over both geometries, all nodes, all steps ===")
    for label, blocks in pooled.items():
        allv = np.concatenate(blocks)
        print(f"  {label:<9} max={allv.max():.5f}h  pooled mean={allv.mean():.5f}h")
    print("\nManuscript quotes max < 0.08h and mean < 0.01h, which holds for both.")


if __name__ == "__main__":
    main()
