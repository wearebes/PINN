from levelset_static_bubble.contracts import MU, T_SIGMA
from levelset_static_bubble.geometry import grid_size, h_from_N


def test_grid_and_physical_constants_match_plan() -> None:
    assert h_from_N(64) == 1.0 / 63.0
    assert grid_size("Q", 64) == 64
    assert grid_size("F", 64) == 127
    assert abs(MU - 8.16496580927726e-3) < 1e-15
    assert abs(T_SIGMA - 0.7155417527999327) < 1e-15
