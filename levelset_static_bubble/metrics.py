import numpy as np

from levelset_static_bubble.contracts import RADIUS


def curvature_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("curvature_stats requires at least one selected interface node")
    mean = float(np.mean(arr))
    exact = 1.0 / RADIUS
    return {
        "kappa_exact": exact,
        "kappa_mean": mean,
        "kappa_std": float(np.sqrt(np.mean((arr - mean) ** 2))),
        "kappa_linf_error": float(np.max(np.abs(arr - exact))),
    }
