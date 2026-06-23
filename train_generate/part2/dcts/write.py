"""HDF5 writer for the DCTS Stage-0 dataset (new dimensionless schema).

Schema (per IMPLEMENTATION_PLAN.md s5), SDF-only, analytic normals:
  /features27[N,27]  /target_hk[N,1]  /hk_central[N,1]  /target_residual_hk[N,1]
  /eta[N,1]  /fine_bin[N]  /coarse_regime[N]
  /pack_id  /geometry_id  /shape  /h(=1)  /d4_id  /sign_id  /sdf_mode  /normal_source

Reuses the generic dataset/append helpers from part2/io.py (no schema change there).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from train_generate.part2.io import _numeric_dataset, _string_dataset
from train_generate.part2.dcts import augment
from train_generate.part2.dcts.config import DctsConfig

_NUMERIC = {
    "features27": ((0, 27), np.float32),
    "target_hk": ((0, 1), np.float32),
    "hk_central": ((0, 1), np.float32),
    "target_residual_hk": ((0, 1), np.float32),
    "eta": ((0, 1), np.float32),
    "fine_bin": ((0,), np.int32),
    "coarse_regime": ((0,), np.int32),
    "h": ((0,), np.float64),
    "d4_id": ((0,), np.int8),
    "sign_id": ((0,), np.int8),
}
_STRING = ("pack_id", "geometry_id", "shape", "sdf_mode", "normal_source")


def _create_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    ds: dict[str, h5py.Dataset] = {}
    for name, (shape, dtype) in _NUMERIC.items():
        ds[name] = _numeric_dataset(handle, name, shape, dtype)
    for name in _STRING:
        ds[name] = _string_dataset(handle, name, (0,))
    return ds


def _append(datasets: dict[str, h5py.Dataset], row: dict) -> int:
    n = int(np.asarray(row["features27"]).shape[0])
    if n == 0:
        return 0
    start = int(datasets["features27"].shape[0])
    end = start + n
    for name, dataset in datasets.items():
        dataset.resize((end,) + dataset.shape[1:])
        dataset[start:end] = row[name]
    return n


def write_split_hdf5(config: DctsConfig, merged: dict, *, split: str, path: Path) -> dict:
    """Write one split's augmented rows to HDF5. Returns counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_canonical = int(np.asarray(merged["phi9"]).shape[0])
    n_rows = 0
    n_combos = 0
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_family"] = "part2_dcts_stage0"
        handle.attrs["schema_version"] = 1
        handle.attrs["split"] = split
        handle.attrs["tag"] = config.tag
        handle.attrs["h"] = float(config.h)
        handle.attrs["eta_min"] = float(config.eta_min)
        handle.attrs["eta_max"] = float(config.eta_max)
        handle.attrs["n_fine_bins"] = int(config.n_fine_bins)
        handle.attrs["sdf_mode"] = config.sdf_mode
        handle.attrs["normal_source"] = config.normal_source
        handle.attrs["d4_sign_enabled"] = bool(config.d4_sign_enabled)
        handle.attrs["augmented"] = bool(config.d4_sign_enabled and split in config.augment_splits)
        handle.attrs["canonical_pack_count"] = n_canonical
        handle.attrs["features27_columns"] = "[phi9:0-8, nx9:9-17, ny9:18-26]"
        datasets = _create_datasets(handle)
        for row in augment.iter_rows(config, merged, split=split):
            n_rows += _append(datasets, row)
            n_combos += 1
        handle.attrs["row_count"] = int(n_rows)
    return {"split": split, "canonical_packs": n_canonical, "combos": n_combos, "rows": n_rows, "path": str(path)}
