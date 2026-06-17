from __future__ import annotations

import numpy as np

from cfd_static_bubble.scripts.nn27_contract import (
    build_circle_raw27,
    checkpoint_paths,
    d4_average_inputs,
    load_v2_checkpoint_bundle,
)


def test_checkpoint_paths_find_repo_v2_assets():
    ckpt, csv = checkpoint_paths(64)
    assert ckpt.name == "baseline_64_hgradient.pt"
    assert csv.name == "baseline_64_hgradient.csv"
    assert ckpt.exists()
    assert csv.exists()


def test_load_v2_checkpoint_bundle_rejects_wrong_contract():
    bundle = load_v2_checkpoint_bundle(64)
    assert bundle["feature_transform"]["feature_order"] == "phi9+nx9+ny9"
    assert int(bundle["model_config"]["input_dim"]) == 27


def test_build_circle_raw27_matches_expected_shape():
    raw27, meta = build_circle_raw27(64, radius=0.4, center=(0.5, 0.5))
    assert raw27.ndim == 2
    assert raw27.shape[1] == 27
    assert meta["interface_count"] > 0


def test_d4_average_identity_on_symmetric_center_sample():
    phi9 = np.asarray(
        [
            np.sqrt(2.0), 1.0, np.sqrt(2.0),
            1.0, 0.0, 1.0,
            np.sqrt(2.0), 1.0, np.sqrt(2.0),
        ],
        dtype=np.float32,
    )
    nx9 = np.asarray(
        [
            -1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0),
            -1.0, 0.0, 1.0,
            -1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0),
        ],
        dtype=np.float32,
    )
    ny9 = np.asarray(
        [
            1.0 / np.sqrt(2.0), 1.0, 1.0 / np.sqrt(2.0),
            0.0, 0.0, 0.0,
            -1.0 / np.sqrt(2.0), -1.0, -1.0 / np.sqrt(2.0),
        ],
        dtype=np.float32,
    )
    center_like = np.concatenate([phi9, nx9, ny9], axis=0).reshape(1, 27)
    transformed = d4_average_inputs(center_like)
    for sample in transformed:
        assert np.allclose(sample, center_like, atol=1e-6)
