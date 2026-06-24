"""D4 x sign x field-mode augmentation, reusing part2/generate.py transforms.

Augmentation is expanded AFTER the split, and only for `config.augment_splits`
(default: train). 1 canonical pack -> 16 rows (8 D4 elements x 2 sign flips),
or 32 rows when config.nonsdf_enabled (x2 field modes). val/test keep only
the canonical (d4='e', sign=0) row(s) -- 1 per field mode in play.

Under continuous theta sampling, orientation is already covered; D4 x sign now
acts as an equivariance / symmetry regulariser, not orientation coverage.
field_mode is an orthogonal axis: circle.py/ellipse.py already compute BOTH
phi9 (true SDF) and phi9_nonsdf (quadratic implicit form, same zero-crossing,
non-unit gradient) from the SAME geometry parameters; this module just picks
which one feeds the D4/sign pipeline for a given combo. nx9/ny9/target_hk/eta
are field-mode-invariant (normal direction and curvature are geometry
properties, not field-representation properties), so they are reused as-is.

Reused (not reimplemented):
  transform_d4_features28, transform_sign_flip_features28,
  central_difference_hkappa_from_phi9_float64   (train_generate/part2/generate.py)
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from train_generate.part2.config import D4_NAMES
from train_generate.part2.generate import (
    central_difference_hkappa_from_phi9_float64,
    transform_d4_features28,
    transform_sign_flip_features28,
)
from train_generate.part2.dcts.config import DctsConfig

FIELD_MODES = ("sdf", "nonsdf")
_PHI9_KEY = {"sdf": "phi9", "nonsdf": "phi9_nonsdf"}


def _features28(merged: dict, *, field_mode: str = "sdf") -> np.ndarray:
    phi9 = np.asarray(merged[_PHI9_KEY[field_mode]], dtype=np.float64)
    nx9 = np.asarray(merged["nx9"], dtype=np.float64)
    ny9 = np.asarray(merged["ny9"], dtype=np.float64)
    hk_central = central_difference_hkappa_from_phi9_float64(phi9)  # (N,1)
    return np.concatenate([phi9, nx9, ny9, hk_central], axis=1)  # (N,28)


def _combos(config: DctsConfig, split: str) -> list[tuple[int, str, int, str]]:
    """(d4_id, d4_name, sign_id, field_mode) combos for this split."""
    fields = FIELD_MODES if config.nonsdf_enabled else FIELD_MODES[:1]
    if config.d4_sign_enabled and split in config.augment_splits:
        return [
            (d4_id, name, sign, field)
            for d4_id, name in enumerate(D4_NAMES)
            for sign in (0, 1)
            for field in fields
        ]
    return [(0, "e", 0, field) for field in fields]


def iter_rows(config: DctsConfig, merged: dict, *, split: str) -> Iterator[dict]:
    """Yield one row-dict (all N canonical packs transformed) per d4 x sign x field combo."""
    n = int(np.asarray(merged["phi9"]).shape[0])
    if n == 0:
        return
    features28_by_field = {field: _features28(merged, field_mode=field) for field in FIELD_MODES}
    hk_exact = np.asarray(merged["hk_exact"], dtype=np.float64).reshape(-1, 1)
    eta = np.asarray(merged["eta"], dtype=np.float64).reshape(-1)
    fine_bin = np.asarray(merged["fine_bin"], dtype=np.int64)
    coarse = np.asarray(merged["coarse_regime"], dtype=np.int64)
    geometry_id = list(merged["geometry_id"])
    pack_id = list(merged["pack_id"])
    shape = list(merged["shape"])

    for d4_id, d4_name, sign_id, field_mode in _combos(config, split):
        d4_feat = transform_d4_features28(features28_by_field[field_mode], d4_name=d4_name)
        if sign_id:
            feat, hk_t = transform_sign_flip_features28(d4_feat, hk_exact)
        else:
            feat, hk_t = d4_feat, hk_exact
        features27 = feat[:, :27]
        hk_central_out = feat[:, 27:28]
        target_residual = hk_t - hk_central_out
        pack_suffix = pack_id if field_mode == "sdf" else [f"{p}:nonsdf" for p in pack_id]
        yield {
            "features27": features27.astype(np.float32),
            "target_hk": hk_t.astype(np.float32),
            "hk_central": hk_central_out.astype(np.float32),
            "target_residual_hk": target_residual.astype(np.float32),
            "eta": eta.reshape(-1, 1).astype(np.float32),
            "fine_bin": fine_bin.astype(np.int32),
            "coarse_regime": coarse.astype(np.int32),
            "pack_id": list(pack_suffix),
            "geometry_id": list(geometry_id),
            "shape": list(shape),
            "h": np.full(n, float(config.h), dtype=np.float64),
            "d4_id": np.full(n, int(d4_id), dtype=np.int8),
            "sign_id": np.full(n, int(sign_id), dtype=np.int8),
            "sdf_mode": [field_mode] * n,
            "normal_source": [config.normal_source] * n,
        }


def verify_augmentation(config: DctsConfig, merged: dict) -> dict:
    """Gate 7/8: D4 hk_central recompute < 1e-10; sign-flip exact. Checked per field_mode
    actually in play (sdf always; nonsdf too when config.nonsdf_enabled). Returns max errors."""
    n = int(np.asarray(merged["phi9"]).shape[0])
    if n == 0:
        return {"max_d4_hk_central_error": 0.0, "max_sign_feature_error": 0.0, "max_sign_target_error": 0.0, "passed": True}
    hk_exact = np.asarray(merged["hk_exact"], dtype=np.float64).reshape(-1, 1)
    fields = FIELD_MODES if config.nonsdf_enabled else FIELD_MODES[:1]
    max_d4 = 0.0
    max_sf = 0.0
    max_st = 0.0
    for field_mode in fields:
        features28 = _features28(merged, field_mode=field_mode)
        hk_central = features28[:, 27:28]
        for d4_name in D4_NAMES:
            d4_feat = transform_d4_features28(features28, d4_name=d4_name)
            recomputed = central_difference_hkappa_from_phi9_float64(d4_feat[:, :9])
            max_d4 = max(max_d4, float(np.max(np.abs(recomputed - hk_central))))
            flipped, flipped_hk = transform_sign_flip_features28(d4_feat, hk_exact)
            max_sf = max(max_sf, float(np.max(np.abs(flipped + d4_feat))))
            max_st = max(max_st, float(np.max(np.abs(flipped_hk + hk_exact))))
    return {
        "max_d4_hk_central_error": max_d4,
        "max_sign_feature_error": max_sf,
        "max_sign_target_error": max_st,
        "passed": bool(max_d4 <= 1.0e-10 and max_sf == 0.0 and max_st == 0.0),
    }


def verify_nonsdf_consistency(merged: dict) -> dict:
    """New check: sdf and nonsdf must share the exact same interface (sign pattern),
    since both are built from the same geometry parameters by construction."""
    phi9_sdf = np.asarray(merged["phi9"], dtype=np.float64)
    phi9_nonsdf = np.asarray(merged["phi9_nonsdf"], dtype=np.float64)
    sign_mismatch = int(np.count_nonzero(np.sign(phi9_sdf) != np.sign(phi9_nonsdf)))
    return {"sign_mismatch_count": sign_mismatch, "passed": bool(sign_mismatch == 0)}
