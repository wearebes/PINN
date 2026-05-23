from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from train_generate.config import DataConfig, GenerationConfig
from train_generate.generate import generate_training_splits
from train_generate.io import load_training_arrays_from_hdf5, save_training_dataset_hdf5
from testdata_generate.config import FlowerScenario, TestDataConfig, compute_grid_spacing
from testdata_generate.generate import generate_test_data


SMALL_DATA_KW = dict(
    resolutions=(16,),
    geometry_seed=42,
    variations=1,
    initial_field_types=("sdf",),
    augment_sign_flip=False,
    shape_types=("circle",),
    ellipse_num_a=2,
    ellipse_variations_per_a=2,
)


def _make_data_config(scale_h: bool) -> DataConfig:
    return DataConfig(scale_h=scale_h, **SMALL_DATA_KW)


def _make_gen_config(tmp_path: Path, name: str) -> GenerationConfig:
    return GenerationConfig(num_workers=1, output_dir=tmp_path, dataset_name=name)


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _attr_bool(value) -> bool:
    return bool(np.asarray(value).item())


def test_train_off_features_equal_phi9(tmp_path):
    cfg = _make_data_config(scale_h=False)
    gen = _make_gen_config(tmp_path, "off.h5")
    bundle = generate_training_splits(data_config=cfg, generation_config=gen)
    out = save_training_dataset_hdf5(bundle)

    with h5py.File(out, "r") as f:
        assert _attr_bool(f.attrs["scale_h"]) is False
        assert _attr_str(f.attrs["feature_transform"]) == "phi9"
        assert int(f.attrs["dataset_format_version"]) == 5
        assert int(f.attrs["feature_version"]) == 1
        assert int(f.attrs["feature_dim_raw"]) == 9
        assert _attr_str(f.attrs["feature_order"]) == "phi9"
        for split in ("train", "val", "test"):
            phi9 = f[split]["phi9"][:]
            feats = f[split]["features"][:]
            assert phi9.dtype == np.float32 and feats.dtype == np.float32
            assert phi9.shape[1] == 9 and feats.shape[1] == 9
            assert np.array_equal(phi9, feats), f"split={split}: features must equal phi9 when scale_h=False"


def test_train_on_features_equal_phi9_over_h(tmp_path):
    cfg_off = _make_data_config(scale_h=False)
    cfg_on = _make_data_config(scale_h=True)
    gen_off = _make_gen_config(tmp_path, "off2.h5")
    gen_on = _make_gen_config(tmp_path, "on.h5")
    out_off = save_training_dataset_hdf5(
        generate_training_splits(data_config=cfg_off, generation_config=gen_off)
    )
    out_on = save_training_dataset_hdf5(
        generate_training_splits(data_config=cfg_on, generation_config=gen_on)
    )

    with h5py.File(out_on, "r") as fon, h5py.File(out_off, "r") as foff:
        assert _attr_bool(fon.attrs["scale_h"]) is True
        assert _attr_str(fon.attrs["feature_transform"]) == "phi9_over_h"
        assert int(fon.attrs["dataset_format_version"]) == 5

        blueprints = json.loads(fon["blueprints_json"][()].decode())
        h_values = {float(bp["params"]["h"]) for bp in blueprints}
        assert len(h_values) == 1, (
            "Test assumes a single resolution -> single h value; "
            f"got h_values={h_values}"
        )
        h_const = np.float32(next(iter(h_values)))

        for split in ("train", "val", "test"):
            phi9_on = fon[split]["phi9"][:]
            feats_on = fon[split]["features"][:]
            phi9_off = foff[split]["phi9"][:]
            target_on = fon[split]["hkappa_target"][:]
            target_off = foff[split]["hkappa_target"][:]

            assert np.array_equal(phi9_on, phi9_off), f"phi9 must be invariant to scale_h ({split})"
            assert np.array_equal(target_on, target_off), f"hkappa_target must be invariant to scale_h ({split})"

            expected = (phi9_on / h_const).astype(np.float32)
            assert np.allclose(feats_on, expected, rtol=0, atol=1e-6), (
                f"split={split}: features must equal phi9 / h when scale_h=True"
            )


def test_train_backward_compat_missing_attr(tmp_path):
    cfg = _make_data_config(scale_h=False)
    gen = _make_gen_config(tmp_path, "legacy.h5")
    out = save_training_dataset_hdf5(
        generate_training_splits(data_config=cfg, generation_config=gen)
    )
    with h5py.File(out, "a") as f:
        del f.attrs["scale_h"]
        if "feature_transform" in f.attrs:
            del f.attrs["feature_transform"]

    bundle = load_training_arrays_from_hdf5(out)
    assert bundle["config"].scale_h is False


def test_testdata_off_features_equal_phi9(tmp_path):
    L = 0.2
    N = 41
    h = compute_grid_spacing(L=L, N=N)
    scenario = FlowerScenario(
        exp_id="t1",
        experiment_type="smooth",
        rho_model=64,
        L=L,
        N=N,
        h=h,
        a=0.05,
        b=0.15,
        p=3,
    )
    cfg = TestDataConfig(
        test_iters=(1,),
        rho_model=64,
        output_dir=tmp_path,
        dataset_name="td_off.h5",
        scenarios=(scenario,),
        config_source="unit_test",
        requested_rho_model=64,
        scale_h=False,
    )
    out = generate_test_data(cfg)
    with h5py.File(out, "r") as f:
        assert _attr_bool(f.attrs["scale_h"]) is False
        assert _attr_str(f.attrs["feature_transform"]) == "phi9"
        phi9 = f["phi9"][:]
        feats = f["features"][:]
        assert np.array_equal(phi9, feats)


def test_testdata_on_features_equal_phi9_over_h(tmp_path):
    L = 0.2
    N = 41
    h = compute_grid_spacing(L=L, N=N)
    scenario = FlowerScenario(
        exp_id="t1",
        experiment_type="smooth",
        rho_model=64,
        L=L,
        N=N,
        h=h,
        a=0.05,
        b=0.15,
        p=3,
    )
    cfg_off = TestDataConfig(
        test_iters=(1,),
        rho_model=64,
        output_dir=tmp_path,
        dataset_name="td_off2.h5",
        scenarios=(scenario,),
        config_source="unit_test",
        requested_rho_model=64,
        scale_h=False,
    )
    cfg_on = TestDataConfig(
        test_iters=(1,),
        rho_model=64,
        output_dir=tmp_path,
        dataset_name="td_on.h5",
        scenarios=(scenario,),
        config_source="unit_test",
        requested_rho_model=64,
        scale_h=True,
    )
    out_off = generate_test_data(cfg_off)
    out_on = generate_test_data(cfg_on)

    with h5py.File(out_on, "r") as fon, h5py.File(out_off, "r") as foff:
        assert _attr_bool(fon.attrs["scale_h"]) is True
        assert _attr_str(fon.attrs["feature_transform"]) == "phi9_over_h"
        phi9_on = fon["phi9"][:]
        phi9_off = foff["phi9"][:]
        feats_on = fon["features"][:]
        target_on = fon["hkappa_target"][:]
        target_off = foff["hkappa_target"][:]
        h_arr = fon["h"][:]

        assert np.array_equal(phi9_on, phi9_off)
        assert np.array_equal(target_on, target_off)
        expected = (phi9_on / h_arr[:, None].astype(np.float32)).astype(np.float32)
        assert np.allclose(feats_on, expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
