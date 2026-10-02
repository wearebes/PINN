"""Shared contracts for generated, unstandardized stencil features."""
from __future__ import annotations

FEATURE_MODES = ("phi9", "phi9_full_normal", "phi9_center_normal", "phi9_cross_normal", "phi9_local_normal")
FEATURE_CONTRACTS = {
    "phi9_local_normal": (6, 15, "phi9+nx_center+ny_center+ny_left+ny_right+nx_top+nx_bottom"),
    "phi9": (1, 9, "phi9"),
    "phi9_full_normal": (2, 27, "phi9+nx9+ny9"),
    "phi9_center_normal": (4, 11, "phi9+nx_center+ny_center"),
    "phi9_cross_normal": (5, 19, "phi9+nx_cross5+ny_cross5"),
}


def resolve_feature_mode(mode: str | None, augment_gradient: bool = False) -> str:
    if mode is None:
        return "phi9_full_normal" if augment_gradient else "phi9"
    if mode not in FEATURE_CONTRACTS:
        raise ValueError(f"Unknown feature_mode {mode!r}; expected one of {FEATURE_MODES}.")
    return mode


def feature_contract(mode: str | None, augment_gradient: bool = False) -> tuple[int, int, str]:
    return FEATURE_CONTRACTS[resolve_feature_mode(mode, augment_gradient)]


def local_normal_features(phi9):
    """Six normals from phi9 ONLY, in canonical TL,T,TR,L,C,R,BL,B,BR order.

    Return nx_C, ny_C, ny_L, ny_R, nx_T, nx_B. Tangential derivatives
    are centered; boundary-normal derivatives are first-order inward
    differences. The common positive 1/h cancels during normalization.
    Accept raw phi9 or phi9/h; zero gradients produce zero components.
    """
    import numpy as np
    p = np.asarray(phi9)
    if p.ndim != 2 or p.shape[1] != 9:
        raise ValueError(f"Expected (N,9) phi stencil, got {p.shape}")
    p = p.astype(np.float64, copy=False)
    gx = np.column_stack((.5*(p[:,5]-p[:,3]), p[:,4]-p[:,3],
                          p[:,5]-p[:,4], .5*(p[:,2]-p[:,0]), .5*(p[:,8]-p[:,6])))
    gy = np.column_stack((.5*(p[:,1]-p[:,7]), .5*(p[:,0]-p[:,6]),
                          .5*(p[:,2]-p[:,8]), p[:,1]-p[:,4], p[:,4]-p[:,7]))
    mag = np.hypot(gx, gy)
    mag = np.where(mag > 0, mag, 1.)
    nx, ny = gx/mag, gy/mag
    return np.column_stack((nx[:,0],ny[:,0],ny[:,1],ny[:,2],nx[:,3],nx[:,4])).astype(np.float32)
