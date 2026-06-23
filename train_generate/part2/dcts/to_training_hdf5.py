"""Adapter: repackage the DCTS Stage-0 split files into ONE v7-style training HDF5.

Lets model/train.py consume the DCTS dataset with NO change to the trainer or
train_generate/io.py. The trainer's loader (load_training_arrays_from_hdf5)
expects a single file with train/val/test groups, the V2 27D feature contract,
and dataset_format_version in {3..7}. DCTS writes three flat-table files instead,
so we translate:

    features27            -> <split>/features        (N, 27)
    features27[:, :9]     -> <split>/phi9            (N, 9)   (phi9 prefix)
    target_hk             -> <split>/hkappa_target   (N, 1)

and stamp the attrs/datasets the loader checks. The DCTS dataset is dimensionless
(h=1); `resolutions`/`blueprints_json` are written as neutral placeholders the
trainer only uses for display.

Usage:
    python -m train_generate.part2.dcts.to_training_hdf5 --tag main
    python -m train_generate.part2.dcts.to_training_hdf5 --in-dir <dir> --out <file.h5>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")


def convert(in_dir: str | Path, out_path: str | Path) -> dict:
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes: dict[str, int] = {}
    with h5py.File(out_path, "w") as out:
        # --- feature contract the loader hard-checks (V2 27D) ---
        out.attrs["dataset_format_version"] = 7
        out.attrs["feature_version"] = 2
        out.attrs["feature_dim_raw"] = 27
        out.attrs["feature_order"] = "phi9+nx9+ny9"
        # --- descriptive attrs (all have loader-side defaults; set for honesty) ---
        out.attrs["initial_field_types_json"] = json.dumps(["sdf"])
        out.attrs["shape_types_json"] = json.dumps(["circle", "ellipse"])
        out.attrs["scale_h"] = True
        out.attrs["feature_transform"] = "phi9+nx9+ny9"
        out.attrs["augment_sign_flip"] = True
        out.attrs["augment_gradient"] = True
        out.attrs["augment_scale_alpha_json"] = json.dumps([])
        # --- provenance ---
        out.attrs["dcts_source"] = "part2_dcts_stage0"
        out.attrs["dcts_dimensionless_h1"] = True
        # --- required top-level datasets (placeholders: DCTS is dimensionless) ---
        out.create_dataset("resolutions", data=np.asarray([256], dtype=np.int32))
        out.create_dataset("blueprints_json", data=json.dumps([]).encode("utf-8"))

        for split in SPLITS:
            src = in_dir / f"{split}.h5"
            if not src.exists():
                if split == "test":
                    continue
                raise FileNotFoundError(f"DCTS split file not found: {src}")
            with h5py.File(src, "r") as f:
                feats = np.asarray(f["features27"][:], dtype=np.float32)
                tgt = np.asarray(f["target_hk"][:], dtype=np.float32)
            if feats.ndim != 2 or feats.shape[1] != 27:
                raise ValueError(f"{src}: features27 must be (N,27), got {feats.shape}")
            if tgt.ndim == 1:
                tgt = tgt.reshape(-1, 1)
            grp = out.create_group(split)
            grp.create_dataset("phi9", data=feats[:, :9], compression="gzip")
            grp.create_dataset("features", data=feats, compression="gzip")
            grp.create_dataset("hkappa_target", data=tgt, compression="gzip")
            sizes[split] = int(feats.shape[0])
    return {"out": str(out_path.resolve()), "sizes": sizes}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repackage DCTS splits into a v7 training HDF5.")
    p.add_argument("--tag", default="main", help="DCTS tag (main|smoke) under dataset/part2_dcts/<tag>/.")
    p.add_argument("--in-dir", default="", help="Override input dir (DCTS processed/ folder).")
    p.add_argument("--out", default="", help="Override output HDF5 path.")
    a = p.parse_args(argv)
    in_dir = Path(a.in_dir) if a.in_dir else Path(f"dataset/part2_dcts/{a.tag}/processed")
    out = Path(a.out) if a.out else Path(f"dataset/part2_dcts/{a.tag}/training/dcts_{a.tag}_v7.h5")
    result = convert(in_dir, out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
