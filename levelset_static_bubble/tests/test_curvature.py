import numpy as np

from levelset_static_bubble.curvature import curvature_exact, curvature_lsf_fd
from levelset_static_bubble.interface_nodes import interface_indices
from levelset_static_bubble.metrics import curvature_stats


def test_exact_curvature_is_constant_on_selected_nodes() -> None:
    idx = interface_indices("F", 64)
    exact = np.asarray([curvature_exact("F", 64, int(i), int(j)) for i, j in idx], dtype=np.float64)
    np.testing.assert_allclose(exact, 2.5, rtol=0.0, atol=0.0)


def test_lsf_fd_curvature_is_finite_and_reports_stats() -> None:
    idx = interface_indices("F", 64)
    values = np.asarray([curvature_lsf_fd("F", 64, int(i), int(j)) for i, j in idx], dtype=np.float64)
    assert np.isfinite(values).all()
    stats = curvature_stats(values)
    assert stats["kappa_exact"] == 2.5
    assert stats["kappa_std"] >= 0.0
    assert stats["kappa_linf_error"] >= 0.0


def test_lsf_fd_mean_approaches_exact_under_refinement() -> None:
    idx64 = interface_indices("F", 64)
    idx128 = interface_indices("F", 128)
    mean64 = np.mean([curvature_lsf_fd("F", 64, int(i), int(j)) for i, j in idx64])
    mean128 = np.mean([curvature_lsf_fd("F", 128, int(i), int(j)) for i, j in idx128])
    assert abs(mean128 - 2.5) < abs(mean64 - 2.5)
