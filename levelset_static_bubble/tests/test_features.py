import numpy as np

from levelset_static_bubble.features import raw27
from levelset_static_bubble.interface_nodes import interface_indices


def test_raw27_feature_is_finite_and_uses_unit_normals() -> None:
    idx = interface_indices("F", 64)
    i, j = map(int, idx[len(idx) // 2])
    r = raw27("F", 64, i, j)
    assert r.shape == (27,)
    assert np.isfinite(r).all()
    nx = r[9:18]
    ny = r[18:27]
    assert np.max(np.abs(np.sqrt(nx * nx + ny * ny) - 1.0)) < 1e-6
