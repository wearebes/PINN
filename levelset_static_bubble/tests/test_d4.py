import numpy as np

from levelset_static_bubble.d4 import D4_MATRICES, transform_raw27
from levelset_static_bubble.features import raw27
from levelset_static_bubble.interface_nodes import interface_indices


def test_d4_transforms_raw27_and_preserves_normals() -> None:
    idx = interface_indices("F", 64)
    i, j = map(int, idx[len(idx) // 2])
    r = raw27("F", 64, i, j)
    transformed = [transform_raw27(r, m) for m in D4_MATRICES]
    assert len(transformed) == 8
    assert all(x.shape == (27,) for x in transformed)
    assert all(np.isfinite(x).all() for x in transformed)

    identity = transform_raw27(r, D4_MATRICES[0])
    np.testing.assert_allclose(identity, r, rtol=0.0, atol=0.0)

    for matrix in D4_MATRICES:
        recovered = transform_raw27(transform_raw27(r, matrix), matrix.T)
        np.testing.assert_allclose(recovered, r, rtol=0.0, atol=1e-6)

    for value in transformed:
        nx = value[9:18]
        ny = value[18:27]
        np.testing.assert_allclose(np.sqrt(nx * nx + ny * ny), 1.0, rtol=0.0, atol=1e-6)
