import numpy as np

from levelset_static_bubble.contracts import RADIUS
from levelset_static_bubble.d4 import D4_MATRICES, transform_raw27
from levelset_static_bubble.features import PHI9_OFFSETS
from levelset_static_bubble.geometry import h_from_N, phi_at_index
from levelset_static_bubble.model_adapter import ModelBundle


def curvature_exact(config: str, N: int, i: int, j: int) -> float:
    return 1.0 / RADIUS


def curvature_lsf_fd(config: str, N: int, i: int, j: int) -> float:
    h = h_from_N(N)
    p = {offset: phi_at_index(config, N, i + offset[0], j + offset[1]) for offset in PHI9_OFFSETS}
    px = 0.5 * (p[(1, 0)] - p[(-1, 0)])
    py = 0.5 * (p[(0, 1)] - p[(0, -1)])
    pxx = p[(1, 0)] - 2.0 * p[(0, 0)] + p[(-1, 0)]
    pyy = p[(0, 1)] - 2.0 * p[(0, 0)] + p[(0, -1)]
    pxy = 0.25 * (p[(1, 1)] - p[(1, -1)] - p[(-1, 1)] + p[(-1, -1)])
    denom = (px * px + py * py) ** 1.5
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(f"non-finite curvature denominator at {(config, N, i, j)}")
    hkappa = (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / denom
    return float(hkappa / h)


def curvature_nn27_raw(model_bundle: ModelBundle, raw: np.ndarray, h: float) -> float:
    y = model_bundle.predict_hkappa(np.asarray(raw, dtype=np.float32).reshape(1, 27))[0]
    return float(y / h)


def curvature_nn27_d4(model_bundle: ModelBundle, raw: np.ndarray, h: float) -> float:
    transformed = np.stack([transform_raw27(raw, matrix) for matrix in D4_MATRICES], axis=0)
    y = model_bundle.predict_hkappa(transformed)
    return float(np.mean(y) / h)
