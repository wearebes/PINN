"""Fine-bin / coarse-regime machinery for DCTS.

100 log-uniform fine-bins over [eta_min, eta_max]; edges e_j = eta_min*(ratio)^(j/N).
The last bin is closed on the right (includes eta_max). 100 fine-bins are folded
into 8 coarse regimes for plotting only (not used for training quotas).
"""
from __future__ import annotations

import numpy as np


def fine_bin_edges(eta_min: float, eta_max: float, n: int) -> np.ndarray:
    """Return the n+1 log-uniform bin edges, strictly increasing, float64."""
    j = np.arange(n + 1, dtype=np.float64)
    ratio = float(eta_max) / float(eta_min)
    edges = float(eta_min) * np.power(ratio, j / float(n))
    # Pin the endpoints exactly (guard against float drift at the extremes).
    edges[0] = float(eta_min)
    edges[-1] = float(eta_max)
    return edges.astype(np.float64, copy=False)


def fine_bin_centers(edges: np.ndarray) -> np.ndarray:
    """Geometric centers of each fine-bin (the log-midpoint), float64."""
    edges = np.asarray(edges, dtype=np.float64)
    return np.sqrt(edges[:-1] * edges[1:])


def fine_bin_index(eta: np.ndarray | float, edges: np.ndarray) -> np.ndarray | int:
    """Map eta -> fine-bin index in [0, n-1] (last bin closed on the right)."""
    edges = np.asarray(edges, dtype=np.float64)
    n = edges.size - 1
    eta_arr = np.asarray(eta, dtype=np.float64)
    # searchsorted with side='right' gives j+1 for e_j <= eta < e_{j+1}; subtract 1.
    idx = np.searchsorted(edges, eta_arr, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    if np.isscalar(eta) or eta_arr.ndim == 0:
        return int(idx)
    return idx.astype(np.int64, copy=False)


def coarse_regime(fine_bin: np.ndarray | int, n_fine: int, n_coarse: int) -> np.ndarray | int:
    """Fold a fine-bin index into a coarse regime in [0, n_coarse-1]."""
    fb = np.asarray(fine_bin, dtype=np.int64)
    cr = (fb * int(n_coarse)) // int(n_fine)
    cr = np.clip(cr, 0, int(n_coarse) - 1)
    if np.isscalar(fine_bin) or fb.ndim == 0:
        return int(cr)
    return cr.astype(np.int64, copy=False)


def sample_eta_loguniform(edges: np.ndarray, fine_bin: int, rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample `size` etas log-uniformly within fine-bin `fine_bin`."""
    lo = float(edges[fine_bin])
    hi = float(edges[fine_bin + 1])
    log_eta = rng.uniform(np.log(lo), np.log(hi), size=size)
    return np.exp(log_eta).astype(np.float64, copy=False)
