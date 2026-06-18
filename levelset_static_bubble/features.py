import numpy as np

from levelset_static_bubble.contracts import ETA_NORMAL
from levelset_static_bubble.geometry import h_from_N, phi_at_index

PHI9_OFFSETS = (
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)


def phi9(config: str, N: int, i: int, j: int) -> np.ndarray:
    return np.asarray(
        [phi_at_index(config, N, i + di, j + dj) for di, dj in PHI9_OFFSETS],
        dtype=np.float32,
    )


def normal_at(config: str, N: int, i: int, j: int) -> tuple[float, float]:
    h = h_from_N(N)
    dx = (phi_at_index(config, N, i + 1, j) - phi_at_index(config, N, i - 1, j)) / (2.0 * h)
    dy = (phi_at_index(config, N, i, j + 1) - phi_at_index(config, N, i, j - 1)) / (2.0 * h)
    denom = (dx * dx + dy * dy + ETA_NORMAL) ** 0.5
    return float(dx / denom), float(dy / denom)


def normal9(config: str, N: int, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    vals = [normal_at(config, N, i + di, j + dj) for di, dj in PHI9_OFFSETS]
    nx = np.asarray([v[0] for v in vals], dtype=np.float32)
    ny = np.asarray([v[1] for v in vals], dtype=np.float32)
    return nx, ny


def raw27(config: str, N: int, i: int, j: int) -> np.ndarray:
    h = h_from_N(N)
    p9 = phi9(config, N, i, j) / np.float32(h)
    nx, ny = normal9(config, N, i, j)
    return np.concatenate([p9, nx, ny]).astype(np.float32, copy=False)
