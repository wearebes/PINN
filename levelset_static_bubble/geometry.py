import numpy as np

from levelset_static_bubble.contracts import RADIUS, RESOLUTIONS


def h_from_N(N: int) -> float:
    if N not in RESOLUTIONS:
        raise ValueError(f"N must be one of {RESOLUTIONS}, got {N}.")
    return 1.0 / float(N - 1)


def grid_size(config: str, N: int) -> int:
    h_from_N(N)
    if config == "Q":
        return N
    if config == "F":
        return 2 * N - 1
    raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")


def coordinates(config: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    h = h_from_N(N)
    n = grid_size(config, N)
    axis = np.arange(n, dtype=np.float64) * h
    return np.meshgrid(axis, axis, indexing="ij")


def phi(config: str, N: int) -> np.ndarray:
    x, y = coordinates(config, N)
    if config == "Q":
        return np.sqrt(x * x + y * y) - RADIUS
    if config == "F":
        return np.sqrt((x - 1.0) ** 2 + (y - 1.0) ** 2) - RADIUS
    raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")


def phi_value(config: str, x: float, y: float) -> float:
    if config == "Q":
        return float((x * x + y * y) ** 0.5 - RADIUS)
    if config == "F":
        return float(((x - 1.0) ** 2 + (y - 1.0) ** 2) ** 0.5 - RADIUS)
    raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")


def phi_at_index(config: str, N: int, i: int, j: int) -> float:
    h = h_from_N(N)
    if config == "Q":
        return phi_value("Q", abs(i * h), abs(j * h))
    n = grid_size("F", N)
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError(f"F index out of bounds for grid {n}: {(i, j)}")
    return phi_value("F", i * h, j * h)
