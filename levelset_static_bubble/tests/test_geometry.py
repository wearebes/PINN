from levelset_static_bubble.contracts import RADIUS
from levelset_static_bubble.geometry import phi


def test_quadrant_phi_grid_matches_contract() -> None:
    q = phi("Q", 64)
    assert q.shape == (64, 64)
    assert abs(q[0, 0] + RADIUS) < 1e-15
    assert abs(q[25, 0]) < 1.0 / 63.0


def test_full_circle_phi_grid_matches_contract() -> None:
    f = phi("F", 64)
    assert f.shape == (127, 127)
    assert abs(f[63, 63] + RADIUS) < 1e-15
    assert abs(f[63 + 25, 63]) < 1.0 / 63.0
