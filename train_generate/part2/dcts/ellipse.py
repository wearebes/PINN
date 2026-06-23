"""Ellipse pack generation: local-eta-conditioned sampling (LOCKED D3).

Given (q, psi, eta_max):  a = 1/(eta_max*q^2),  b = q*a   (units h, h=1)
  eta(t) = h * kappa(t) = a*b / (a^2 sin^2 t + b^2 cos^2 t)^1.5
  local eta range = [eta_max*q^3, eta_max]   (min at minor-axis end, max at tip)

To fill a target fine-bin we select the t-segment(s) where eta(t) lies in the bin
and arc-length-sample within them. The curvature target uses the KNOWN interface
parameter t analytically (never the projection). Only the surrounding 5x5 nodes are
projected to the ellipse (float64 Newton, tol 1e-12, max_iter 30, coarse-global
fallback) to get signed distance phi and the per-node analytic outward normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_generate.geometry_core import (
    ellipse_hkappa_from_theta,
    project_theta_to_axis_aligned_ellipse,
)
from train_generate.part2.dcts import bins, patch
from train_generate.part2.dcts.config import DctsConfig

TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class EllipseGeom:
    q: float
    psi: float
    eta_max: float
    a: float
    b: float
    q_idx: int
    psi_idx: int
    em_idx: int


def make_geom(q: float, psi: float, eta_max: float, *, q_idx: int, psi_idx: int, em_idx: int,
              tol: float) -> EllipseGeom:
    a = 1.0 / (float(eta_max) * float(q) ** 2)
    b = float(q) * a
    # Locked consistency gate: |eta_max - 1/(a q^2)| < tol.
    err = abs(float(eta_max) - 1.0 / (a * q * q))
    if err >= tol:
        raise AssertionError(f"ellipse eta_max consistency failed: err={err:.3e} (q={q}, eta_max={eta_max})")
    return EllipseGeom(float(q), float(psi), float(eta_max), float(a), float(b), int(q_idx), int(psi_idx), int(em_idx))


def eta_table(a: float, b: float, dense: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense t -> (t, eta(t), local arc-length element ds(t)) over [0, 2pi)."""
    t = np.linspace(0.0, TWO_PI, int(dense), endpoint=False, dtype=np.float64)
    eta = ellipse_hkappa_from_theta(t, h=1.0, a=float(a), b=float(b))
    dt = TWO_PI / float(dense)
    speed = np.sqrt(a * a * np.sin(t) ** 2 + b * b * np.cos(t) ** 2)
    ds = speed * dt
    return t, eta, ds


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Contiguous True runs of indices in `mask`, merging the t=0/2pi wraparound."""
    if not mask.any():
        return []
    idx = np.nonzero(mask)[0]
    breaks = np.nonzero(np.diff(idx) != 1)[0]
    runs = np.split(idx, breaks + 1)
    if bool(mask[0]) and bool(mask[-1]) and len(runs) > 1:
        runs[0] = np.concatenate([runs[-1], runs[0]])
        runs = runs[:-1]
    return runs


def _allocate(n_take: int, weights: list[float]) -> list[int]:
    """Largest-remainder allocation of n_take across segments proportional to weight."""
    total = float(sum(weights))
    if total <= 0.0:
        return [0] * len(weights)
    exact = [n_take * w / total for w in weights]
    base = [int(np.floor(e)) for e in exact]
    rem = int(n_take - sum(base))
    if rem > 0:
        order = sorted(range(len(weights)), key=lambda i: -(exact[i] - base[i]))
        for i in order[:rem]:
            base[i] += 1
    return base


def arclength_sample_band(
    t: np.ndarray, eta: np.ndarray, ds: np.ndarray, *, lo: float, hi: float, cap: int, phase: float = 0.5
) -> np.ndarray:
    """Arc-length-uniform t-samples (<= cap) where lo <= eta(t) < hi.

    Samples WITHIN each contiguous in-band t-segment (plan s3.2), allocating the
    budget across segments proportional to their arc length. Interpolation never
    crosses a segment gap, so every returned t has eta(t) in [lo, hi).
    """
    mask = (eta >= lo) & (eta < hi)
    n_in = int(np.count_nonzero(mask))
    if n_in == 0:
        return np.zeros((0,), dtype=np.float64)
    runs = _contiguous_runs(mask)
    seg_w = [float(ds[run].sum()) for run in runs]
    n_take = int(min(int(cap), n_in))
    alloc = _allocate(n_take, seg_w)

    samples: list[np.ndarray] = []
    for run, k in zip(runs, alloc):
        if k <= 0:
            continue
        tr = np.unwrap(t[run])  # monotone within the (possibly wrapped) segment
        wr = ds[run]
        cum = np.cumsum(wr)
        seg_total = float(cum[-1])
        if seg_total <= 0.0:
            samples.append(np.mod(tr[:k], TWO_PI))
            continue
        targets = (np.arange(k, dtype=np.float64) + float(phase)) / float(k) * seg_total
        ts = np.interp(targets, cum, tr)
        samples.append(np.mod(ts, TWO_PI))

    if not samples:
        return np.zeros((0,), dtype=np.float64)
    out = np.concatenate(samples)
    out = np.unique(out)  # drop any degenerate duplicates from tiny segments
    return out.astype(np.float64, copy=False)


def build_batch(
    config: DctsConfig,
    geom: EllipseGeom,
    *,
    split: str,
    fine_bin: int,
    t_values: np.ndarray,
    d0_values: np.ndarray,
    geom_id: str,
) -> dict:
    """Build a pack batch for chosen interface params t_values (+ d0) on one ellipse."""
    t_values = np.asarray(t_values, dtype=np.float64).reshape(-1)
    d0_values = np.asarray(d0_values, dtype=np.float64).reshape(-1)
    n = t_values.size
    if n == 0:
        return _empty_batch()
    a, b, psi = geom.a, geom.b, geom.psi
    cos_psi, sin_psi = float(np.cos(psi)), float(np.sin(psi))

    # Interface point P(t) and outward unit normal n(t), in ellipse-local frame.
    P_u = a * np.cos(t_values)
    P_v = b * np.sin(t_values)
    nlu = b * np.cos(t_values)
    nlv = a * np.sin(t_values)
    nmag = np.sqrt(nlu * nlu + nlv * nlv)
    nlu /= nmag
    nlv /= nmag
    x0_u = P_u + d0_values * nlu  # patch center node, local frame
    x0_v = P_v + d0_values * nlv

    offsets = patch.patch_offsets_5x5()  # (25, 2) as global (x,y) offsets
    ox = offsets[:, 0]
    oy = offsets[:, 1]
    # Global offset (i,j) -> local offset (rotation by -psi).
    loc_u = ox * cos_psi + oy * sin_psi  # (25,)
    loc_v = -ox * sin_psi + oy * cos_psi

    u_nodes = x0_u[:, None] + loc_u[None, :]  # (n, 25)
    v_nodes = x0_v[:, None] + loc_v[None, :]

    theta = project_theta_to_axis_aligned_ellipse(
        u_nodes.reshape(-1), v_nodes.reshape(-1), a=a, b=b,
        max_iter=int(config.newton_max_iter), tol=float(config.newton_tol),
    ).reshape(n, 25)

    Qu = a * np.cos(theta)
    Qv = b * np.sin(theta)
    dist = np.sqrt((u_nodes - Qu) ** 2 + (v_nodes - Qv) ** 2)
    inside = (u_nodes ** 2) / (a * a) + (v_nodes ** 2) / (b * b) < 1.0
    phi25 = np.where(inside, -dist, dist)
    phi5 = phi25.reshape(n, 5, 5)

    # Per-node analytic outward normal (at the projection point), rotated to global.
    n_u = b * np.cos(theta)
    n_v = a * np.sin(theta)
    nm = np.sqrt(n_u * n_u + n_v * n_v)
    n_u /= nm
    n_v /= nm
    nx25 = n_u * cos_psi - n_v * sin_psi
    ny25 = n_u * sin_psi + n_v * cos_psi

    phi9 = patch.inner_phi9(phi5)
    nx9 = patch.inner_phi9(nx25.reshape(n, 5, 5))
    ny9 = patch.inner_phi9(ny25.reshape(n, 5, 5))
    fd_grad_norm9 = patch.fd_grad_norm_inner(phi5)

    # Analytic curvature target from the KNOWN interface parameter t.
    hk_exact = ellipse_hkappa_from_theta(t_values, h=1.0, a=a, b=b)
    eta = hk_exact.copy()

    # Medial-axis (evolute) proximity: distance from each node to the centre of
    # curvature at its own projection, min over the 25 nodes.
    kappa_node = ellipse_hkappa_from_theta(theta, h=1.0, a=a, b=b)
    Rc = 1.0 / kappa_node
    Eu = Qu - Rc * n_u
    Ev = Qv - Rc * n_v
    dist_evolute = np.sqrt((u_nodes - Eu) ** 2 + (v_nodes - Ev) ** 2)
    medial_min_dist = np.min(dist_evolute, axis=1)

    fb = np.full(n, int(fine_bin), dtype=np.int64)
    cr = np.full(n, bins.coarse_regime(fine_bin, config.n_fine_bins, config.n_coarse_regimes), dtype=np.int64)
    geometry_id = [geom_id] * n
    pack_id = [f"{geom_id}:n{k:04d}" for k in range(n)]

    return {
        "shape": "ellipse",
        "phi9": phi9,
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
        "q": np.full(n, geom.q),
        "psi": np.full(n, geom.psi),
        "eta_max": np.full(n, geom.eta_max),
        "radius": np.full(n, np.nan),  # circles only; NaN for ellipse
    }


def _empty_batch() -> dict:
    z = np.zeros
    return {
        "shape": "ellipse",
        "phi9": z((0, 9)), "nx9": z((0, 9)), "ny9": z((0, 9)),
        "hk_exact": z((0,)), "eta": z((0,)),
        "fine_bin": z((0,), dtype=np.int64), "coarse_regime": z((0,), dtype=np.int64),
        "phi5": z((0, 5, 5)), "fd_grad_norm9": z((0, 9)), "medial_min_dist": z((0,)),
        "geometry_id": [], "pack_id": [],
        "q": z((0,)), "psi": z((0,)), "eta_max": z((0,)), "radius": z((0,)),
    }
