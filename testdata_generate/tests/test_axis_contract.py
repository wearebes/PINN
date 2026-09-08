from __future__ import annotations

import numpy as np

from testdata_generate.generate import build_grid
from train_generate.generate import extract_grad9


def test_flower_grid_matches_training_axis_contract() -> None:
    X, Y, _ = build_grid(L=1.0, N=9)

    assert X[1, 4] > X[0, 4]
    assert X[4, 1] == X[4, 0]
    assert Y[4, 1] > Y[4, 0]
    assert Y[1, 4] == Y[0, 4]

    phi = 2.0 * X + 3.0 * Y
    center = np.asarray([[4, 4]], dtype=np.int64)
    normal = extract_grad9(phi, center)[0, 4]
    expected = np.asarray([2.0, 3.0]) / np.sqrt(13.0)

    np.testing.assert_allclose(normal, expected, rtol=0.0, atol=1.0e-7)
