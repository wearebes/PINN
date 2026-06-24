"""Circle pack generation: direct-by-fine-bin-quota (LOCKED D2).

Circles are constant-curvature and exactly controllable, so there is no
candidate pool, no per-geometry cap, and no post-balancing. For each
(split, fine_bin) we draw `circle_quota` packs directly:

    eta   ~ LogUniform(e_j, e_{j+1})        # = h*kappa (h=1)
    theta ~ U(0, 2pi)                        # interface orientation
    d0    ~ U(-0.5, 0.5)                     # signed center-node distance (units h)
    R = 1/eta;  n = (cos t, sin t);  c = -(d0 + R) * n
    phi(node) = ||node - c|| - R
    target_hk = +eta                         # analytic, = h*kappa

Normals are analytic: n(node) = (node - c)/||node - c|| (outward = +grad phi).
"""
from __future__ import annotations

import numpy as np

from train_generate.part2.dcts import bins, patch
from train_generate.part2.dcts.config import DctsConfig


def generate_circle_packs(
    config: DctsConfig,
    *,
    split: str,
    fine_bin: int,
    edges: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> dict:
    """Generate `n` circle packs for one (split, fine_bin). Returns a batch dict."""
    if n <= 0:
        return _empty_batch()

    eta = bins.sample_eta_loguniform(edges, fine_bin, rng, n)  # (n,)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    d0 = rng.uniform(config.d0_min, config.d0_max, size=n)

    radius = 1.0 / eta  # (n,)
    nx_dir = np.cos(theta)
    ny_dir = np.sin(theta)
    cx = -(d0 + radius) * nx_dir  # (n,)
    cy = -(d0 + radius) * ny_dir

    offsets = patch.patch_offsets_5x5()  # (25, 2)
    ox = offsets[:, 0]  # (25,)
    oy = offsets[:, 1]

    # node global pos = offset; distance to center c per pack.
    dx = ox[None, :] - cx[:, None]  # (n, 25)
    dy = oy[None, :] - cy[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    phi25 = dist - radius[:, None]  # (n, 25) -- true SDF
    phi5 = phi25.reshape(n, 5, 5)

    phi9 = patch.inner_phi9(phi5)  # (n, 9)
    fd_grad_norm9 = patch.fd_grad_norm_inner(phi5)

    # Non-SDF field (opt-in): same zero-crossing, quadratic, |grad| = 2*dist (not 1).
    # Matches train_generate.geometry_core.build_circle_nonsdf exactly.
    phi25_nonsdf = dist * dist - (radius * radius)[:, None]
    phi9_nonsdf = patch.inner_phi9(phi25_nonsdf.reshape(n, 5, 5))

    # Analytic outward normals at the inner 9 nodes (STENCIL order).
    inner = patch.inner_offsets_xy()  # (9, 2)
    indx = inner[:, 0][None, :] - cx[:, None]  # (n, 9)
    indy = inner[:, 1][None, :] - cy[:, None]
    nmag = np.sqrt(indx * indx + indy * indy)
    nx9 = indx / nmag
    ny9 = indy / nmag

    hk_exact = eta.copy()  # = +1/R
    fb = np.full(n, int(fine_bin), dtype=np.int64)
    cr = np.full(n, bins.coarse_regime(fine_bin, config.n_fine_bins, config.n_coarse_regimes), dtype=np.int64)

    # Medial axis = circle center; min distance from any patch node to it.
    medial_min_dist = np.min(dist, axis=1)

    geometry_id = [f"circle:{split}:b{fine_bin:02d}:k{k:06d}" for k in range(n)]
    pack_id = list(geometry_id)  # one pack per circle

    return {
        "shape": "circle",
        "phi9": phi9,
        "phi9_nonsdf": phi9_nonsdf,
        "nx9": nx9,
        "ny9": ny9,
        "hk_exact": hk_exact,
        "eta": eta,
        "fine_bin": fb,
        "coarse_regime": cr,
        "phi5": phi5,
        "fd_grad_norm9": fd_grad_norm9,
        "medial_min_dist": medial_min_dist,
        "geometry_id": geometry_id,
        "pack_id": pack_id,
        # circle has no (q, psi, eta_max); kept NaN so the schema is uniform.
        "q": np.full(n, np.nan),
        "psi": np.full(n, np.nan),
        "eta_max": np.full(n, np.nan),
        # circle consistency check payload:
        "radius": radius,
    }


def _empty_batch() -> dict:
    return {
        "shape": "circle",
        "phi9": np.zeros((0, 9)),
        "phi9_nonsdf": np.zeros((0, 9)),
        "nx9": np.zeros((0, 9)),
        "ny9": np.zeros((0, 9)),
        "hk_exact": np.zeros((0,)),
        "eta": np.zeros((0,)),
        "fine_bin": np.zeros((0,), dtype=np.int64),
        "coarse_regime": np.zeros((0,), dtype=np.int64),
        "phi5": np.zeros((0, 5, 5)),
        "fd_grad_norm9": np.zeros((0, 9)),
        "medial_min_dist": np.zeros((0,)),
        "geometry_id": [],
        "pack_id": [],
        "q": np.zeros((0,)),
        "psi": np.zeros((0,)),
        "eta_max": np.zeros((0,)),
        "radius": np.zeros((0,)),
    }
