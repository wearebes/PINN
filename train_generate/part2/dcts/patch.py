"""5x5 patch geometry, shared by circle and ellipse generators.

Convention (matches train_generate.geometry_core.STENCIL_OFFSETS and the
central-difference operator in part2/generate.py):
  - A patch node offset is (i, j) = (x-offset, y-offset), i,j in {-2,...,2}.
  - phi5 is stored as a (..., 5, 5) array indexed [ix, iy] with ix=i+2, iy=j+2.
  - The inner 3x3 (i,j in {-1,0,1}) carries the features; the outer ring exists
    only as the +/-1 finite-difference footprint for the Gate 11 diagnostic.
  - phi9 / nx9 / ny9 are ordered exactly as STENCIL_OFFSETS so that
    central_difference_hkappa_from_phi9_float64 reproduces the curvature.
"""
from __future__ import annotations

import numpy as np

from train_generate.geometry_core import STENCIL_OFFSETS

# All 25 offsets in (ix outer, iy inner) order so result.reshape(5, 5) is [ix, iy].
_PATCH_OFFSETS_5x5 = np.asarray(
    [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)],
    dtype=np.float64,
)  # (25, 2) as (x-offset, y-offset)

# Index of each STENCIL_OFFSETS entry inside the flattened 5x5 array.
_INNER_FLAT_INDEX = np.asarray(
    [int((di + 2) * 5 + (dj + 2)) for (di, dj) in STENCIL_OFFSETS],
    dtype=np.int64,
)


def patch_offsets_5x5() -> np.ndarray:
    """(25, 2) float64 array of (x-offset, y-offset), reshape(5,5) -> [ix, iy]."""
    return _PATCH_OFFSETS_5x5.copy()


def inner_phi9(phi5: np.ndarray) -> np.ndarray:
    """Extract the inner 9 values in STENCIL order. phi5: (N,5,5) -> (N,9)."""
    phi5 = np.asarray(phi5, dtype=np.float64)
    flat = phi5.reshape(phi5.shape[0], 25)
    return flat[:, _INNER_FLAT_INDEX]


def fd_grad_norm_inner(phi5: np.ndarray) -> np.ndarray:
    """Finite-difference ||grad phi|| at the inner 9 nodes (h=1). phi5:(N,5,5)->(N,9).

    Diagnostic only (Gate 11). Uses central +/-1 differences, hence the outer ring.
    """
    phi5 = np.asarray(phi5, dtype=np.float64)
    out = np.empty((phi5.shape[0], 9), dtype=np.float64)
    for k, (di, dj) in enumerate(STENCIL_OFFSETS):
        ix = int(di) + 2
        iy = int(dj) + 2
        gx = 0.5 * (phi5[:, ix + 1, iy] - phi5[:, ix - 1, iy])
        gy = 0.5 * (phi5[:, ix, iy + 1] - phi5[:, ix, iy - 1])
        out[:, k] = np.sqrt(gx * gx + gy * gy)
    return out


def inner_offsets_xy() -> np.ndarray:
    """(9, 2) STENCIL-ordered inner offsets as float64 (x-offset, y-offset)."""
    return STENCIL_OFFSETS.astype(np.float64, copy=False)
