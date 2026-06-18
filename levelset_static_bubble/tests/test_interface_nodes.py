from levelset_static_bubble.geometry import h_from_N, phi_at_index
from levelset_static_bubble.interface_nodes import (
    has_valid_stencil,
    interface_indices,
    is_cross_node,
)


def test_quadrant_phi_index_uses_analytic_symmetry_ghosts() -> None:
    assert phi_at_index("Q", 64, -25, 0) == phi_at_index("Q", 64, 25, 0)
    assert phi_at_index("Q", 64, 0, -25) == phi_at_index("Q", 64, 0, 25)


def test_full_circle_phi_index_rejects_out_of_bounds() -> None:
    try:
        phi_at_index("F", 64, -1, 0)
    except IndexError:
        pass
    else:
        raise AssertionError("F index out of bounds should raise IndexError")


def test_interface_indices_match_band_cross_and_stencil_contracts() -> None:
    for config in ("Q", "F"):
        idx = interface_indices(config, 64)
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert len(idx) > 0
        h = h_from_N(64)
        for i, j in idx:
            assert abs(phi_at_index(config, 64, int(i), int(j))) <= 1.5 * h + 1e-15
            assert is_cross_node(config, 64, int(i), int(j))
            assert has_valid_stencil(config, 64, int(i), int(j))
