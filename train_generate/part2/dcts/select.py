"""Selection / generation of canonical packs for one split (LOCKED D2/D3/D4).

Circles: direct-by-fine-bin-quota. Ellipses: local-eta-conditioned candidate
build (eta_max log-scan x q x psi), arc-length sampling per geometry, then
round-robin across geometries with a per-(split,shape,fine_bin,geometry_id) cap
until the bin quota is met. Shortfalls are recorded as deficiencies, never
silently oversampled. eta_max log-grid is split-phased so train/val/test
ellipse shapes are disjoint (leakage hard gate).
"""
from __future__ import annotations

import numpy as np

from train_generate.part2.dcts import circle, ellipse
from train_generate.part2.dcts.config import DctsConfig, seed_from_parts

# Canonical keys merged across batches.
_ARRAY_KEYS = (
    "phi9", "phi9_nonsdf", "nx9", "ny9", "hk_exact", "eta", "fine_bin", "coarse_regime",
    "phi5", "fd_grad_norm9", "medial_min_dist", "q", "psi", "eta_max", "radius",
)
_LIST_KEYS = ("geometry_id", "pack_id", "shape")


def candidate_eta_max(
    config: DctsConfig, fine_bin: int, q: float, split: str, centers: np.ndarray, *, q_idx: int
) -> np.ndarray:
    """Per-split log-uniform draws of eta_max in [max(eta*,emin), min(eta*/q^3, emax)].

    Drawn from a split-seeded RNG so train/val/test ellipse shapes are disjoint
    with probability 1 (continuous), while every bin center eta* stays reachable.
    """
    star = float(centers[fine_bin])
    lo = max(star, config.eta_min)
    hi = min(star / (q ** 3), config.eta_max)
    if lo > hi:
        return np.zeros((0,), dtype=np.float64)
    if hi <= lo:
        return np.asarray([lo], dtype=np.float64)
    n = int(config.ellipse_eta_max_scan)
    rng = np.random.default_rng(seed_from_parts(config.split_seed(split), fine_bin, 3, q_idx))
    vals = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    return np.clip(vals, lo, hi)


def build_ellipse_candidates(
    config: DctsConfig, *, split: str, fine_bin: int, edges: np.ndarray, centers: np.ndarray
) -> list[tuple["ellipse.EllipseGeom", str, list[float], list[float]]]:
    """Candidate (geom, geom_id, t_list, d0_list) per geometry for one (split, fine_bin).

    Each geometry's t_list is arc-length sampled within the bin and capped at
    config.cap(split) (no replacement). This is the exact pool the round-robin
    selection draws from, so capacity preflight and selection cannot drift.
    """
    lo_e = float(edges[fine_bin])
    hi_e = float(edges[fine_bin + 1])
    cap = config.cap(split)
    phase = config.split_phase(split)
    geom_payloads: list[tuple[ellipse.EllipseGeom, str, list[float], list[float]]] = []
    geom_counter = 0
    for q_idx, q in enumerate(config.q_values):
        em_values = candidate_eta_max(config, fine_bin, q, split, centers, q_idx=q_idx)
        for em_idx, em in enumerate(em_values):
            for psi_idx, psi in enumerate(config.psi_values):
                geom = ellipse.make_geom(
                    q, psi, float(em), q_idx=q_idx, psi_idx=psi_idx, em_idx=em_idx, tol=config.eta_consistency_tol
                )
                t, eta_t, ds = ellipse.eta_table(geom.a, geom.b, config.ellipse_dense_t)
                t_cand = ellipse.arclength_sample_band(t, eta_t, ds, lo=lo_e, hi=hi_e, cap=cap, phase=phase)
                if t_cand.size == 0:
                    geom_counter += 1
                    continue
                rng = np.random.default_rng(seed_from_parts(config.split_seed(split), fine_bin, 2, geom_counter))
                d0 = rng.uniform(config.d0_min, config.d0_max, size=t_cand.size)
                geom_id = f"ellipse:{split}:b{fine_bin:02d}:q{q_idx}:p{psi_idx}:e{em_idx}"
                geom_payloads.append((geom, geom_id, list(t_cand), list(d0)))
                geom_counter += 1
    return geom_payloads


def select_ellipse_for_bin(
    config: DctsConfig, *, split: str, fine_bin: int, edges: np.ndarray, centers: np.ndarray, quota: int
) -> tuple[list[dict], dict]:
    """Round-robin ellipse selection for one (split, fine_bin). Returns (batches, record)."""
    geom_payloads = build_ellipse_candidates(config, split=split, fine_bin=fine_bin, edges=edges, centers=centers)
    cap = config.cap(split)
    total_available = sum(len(p[2]) for p in geom_payloads)

    # Round-robin: one point per geometry per pass, until quota or exhaustion.
    selected: dict[int, tuple[ellipse.EllipseGeom, str, list[float], list[float]]] = {}
    taken = 0
    cursors = [0] * len(geom_payloads)
    progressed = True
    while taken < quota and progressed:
        progressed = False
        for gi, (geom, geom_id, t_list, d0_list) in enumerate(geom_payloads):
            if taken >= quota:
                break
            c = cursors[gi]
            if c >= len(t_list):
                continue
            entry = selected.setdefault(gi, (geom, geom_id, [], []))
            entry[2].append(t_list[c])
            entry[3].append(d0_list[c])
            cursors[gi] = c + 1
            taken += 1
            progressed = True

    batches: list[dict] = []
    for gi in sorted(selected):
        geom, geom_id, t_sel, d0_sel = selected[gi]
        batches.append(
            ellipse.build_batch(
                config, geom, split=split, fine_bin=fine_bin,
                t_values=np.asarray(t_sel), d0_values=np.asarray(d0_sel), geom_id=geom_id,
            )
        )

    record = {
        "split": split,
        "shape": "ellipse",
        "fine_bin": int(fine_bin),
        "quota": int(quota),
        "selected": int(taken),
        "deficiency": int(max(0, quota - taken)),
        "candidate_geometries": int(len(geom_payloads)),
        "geometries_used": int(len(selected)),
        "total_available_candidates": int(total_available),
        "cap": int(cap),
    }
    return batches, record


def generate_split(config: DctsConfig, *, split: str, edges: np.ndarray, centers: np.ndarray) -> tuple[dict, list[dict]]:
    """Generate all canonical packs for one split. Returns (merged_batch, occupancy_records)."""
    circle_quota = config.circle_quota(split)
    ellipse_quota = config.ellipse_quota(split)
    split_seed = config.split_seed(split)

    all_batches: list[dict] = []
    records: list[dict] = []
    for j in range(config.n_fine_bins):
        crng = np.random.default_rng(seed_from_parts(split_seed, j, 1))
        cbatch = circle.generate_circle_packs(config, split=split, fine_bin=j, edges=edges, rng=crng, n=circle_quota)
        all_batches.append(cbatch)
        records.append({
            "split": split, "shape": "circle", "fine_bin": int(j),
            "quota": int(circle_quota), "selected": int(circle_quota), "deficiency": 0,
            "candidate_geometries": "", "geometries_used": int(circle_quota),
            "total_available_candidates": "", "cap": "",
        })
        ebatches, erec = select_ellipse_for_bin(
            config, split=split, fine_bin=j, edges=edges, centers=centers, quota=ellipse_quota
        )
        all_batches.extend(ebatches)
        records.append(erec)

    merged = concat_batches(all_batches)
    return merged, records


def concat_batches(batches: list[dict]) -> dict:
    """Concatenate per-geometry batch dicts into one split-level batch."""
    nonempty = [b for b in batches if int(np.asarray(b["phi9"]).shape[0]) > 0]
    out: dict = {}
    for key in _ARRAY_KEYS:
        if not nonempty:
            out[key] = np.zeros((0,))
            continue
        out[key] = np.concatenate([np.asarray(b[key]) for b in nonempty], axis=0)
    for key in ("geometry_id", "pack_id"):
        out[key] = [s for b in nonempty for s in b[key]]
    out["shape"] = [b["shape"] for b in nonempty for _ in range(int(np.asarray(b["phi9"]).shape[0]))]
    return out
