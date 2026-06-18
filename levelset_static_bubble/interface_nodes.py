import numpy as np

from levelset_static_bubble.contracts import BAND_FACTOR
from levelset_static_bubble.geometry import grid_size, h_from_N, phi_at_index

FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def is_cross_node(config: str, N: int, i: int, j: int) -> bool:
    center = phi_at_index(config, N, i, j)
    for di, dj in FOUR_NEIGHBORS:
        try:
            neighbor = phi_at_index(config, N, i + di, j + dj)
        except IndexError:
            continue
        if center * neighbor <= 0.0:
            return True
    return False


def has_valid_stencil(config: str, N: int, i: int, j: int) -> bool:
    if config == "Q":
        return True
    n = grid_size("F", N)
    return 2 <= i <= n - 3 and 2 <= j <= n - 3


def interface_indices(config: str, N: int) -> np.ndarray:
    h = h_from_N(N)
    n = grid_size(config, N)
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if abs(phi_at_index(config, N, i, j)) <= BAND_FACTOR * h:
                if is_cross_node(config, N, i, j) and has_valid_stencil(config, N, i, j):
                    out.append((i, j))
    return np.asarray(out, dtype=np.int64)
