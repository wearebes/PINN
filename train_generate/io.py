from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import DataConfig, GenerationConfig, default_dataset_name


DATASET_FORMAT_VERSION = 5
SUPPORTED_DATASET_FORMAT_VERSIONS = {3, 4, 5}
FIELD_ORDER = ("phi9", "features", "hkappa_target")
FIELD_DTYPES = {
    "phi9": np.float32,
    "features": np.float32,
    "hkappa_target": np.float32,
}
DEFAULT_PREVIEW_SAMPLES = 5


def normalize_generation_config(generation_config: GenerationConfig | None) -> GenerationConfig:
    cfg = generation_config or GenerationConfig()
    if int(cfg.num_workers) < 1:
        raise ValueError("GenerationConfig.num_workers must be >= 1.")
    if int(cfg.generation_batch_size) < 1:
        raise ValueError("GenerationConfig.generation_batch_size must be >= 1.")
    return GenerationConfig(
        num_workers=int(cfg.num_workers),
        generation_batch_size=int(cfg.generation_batch_size),
        output_dir=Path(cfg.output_dir),
        dataset_name=str(cfg.dataset_name or default_dataset_name()),
    )


def dataset_manifest_path(dataset_path: str | Path) -> Path:
    return Path(dataset_path).with_suffix(".json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _preview_records_from_split(split: dict[str, np.ndarray], *, limit: int = DEFAULT_PREVIEW_SAMPLES) -> list[dict[str, Any]]:
    n_items = min(int(split["phi9"].shape[0]), int(limit))
    preview: list[dict[str, Any]] = []
    for row_idx in range(n_items):
        preview.append(
            {
                field_name: np.asarray(split[field_name][row_idx]).reshape(-1).tolist()
                for field_name in FIELD_ORDER
                if field_name in split
            }
        )
    return preview


def build_dataset_manifest(bundle: dict[str, Any], dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    generation_cfg = normalize_generation_config(bundle["generation_config"])
    splits: dict[str, Any] = {}
    for split_name, split in bundle["splits"].items():
        available_fields = [field_name for field_name in FIELD_ORDER if field_name in split]
        splits[split_name] = {
            "size": int(split["features"].shape[0]),
            "fields": [
                {
                    "name": field_name,
                    "shape": list(split[field_name].shape),
                    "dtype": str(split[field_name].dtype),
                }
                for field_name in available_fields
            ],
            "sample_preview": _preview_records_from_split(split),
        }

    return {
        "dataset_file": {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "manifest_path": str(dataset_manifest_path(dataset_path)),
        },
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "task": "3x3 stencil features -> h*kappa",
        "configs": {
            "data": _json_ready(asdict(bundle["config"])),
            "generation": _json_ready(asdict(generation_cfg)),
        },
        "split_blueprint_counts": _json_ready(bundle.get("split_blueprint_counts", {})),
        "split_shape_blueprint_counts": _json_ready(bundle.get("split_shape_blueprint_counts", {})),
        "splits": splits,
    }


def save_dataset_manifest(bundle: dict[str, Any], dataset_path: str | Path) -> Path:
    manifest_path = dataset_manifest_path(dataset_path)
    manifest = build_dataset_manifest(bundle, dataset_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def read_dataset_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    manifest_path = dataset_manifest_path(dataset_path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_training_dataset_hdf5(
    bundle: dict[str, Any],
    generation_config: GenerationConfig | None = None,
    *,
    path: str | Path | None = None,
) -> Path:
    generation_cfg = normalize_generation_config(generation_config or bundle.get("generation_config"))
    output_path = Path(path) if path is not None else generation_cfg.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        data_cfg = bundle["config"]
        raw_feature_dim = int(bundle["splits"]["train"]["features"].shape[1]) if "train" in bundle["splits"] else 9
        if raw_feature_dim != 9:
            raise ValueError(f"V1 training datasets must store 9D phi9 features, got raw_feature_dim={raw_feature_dim}.")
        handle.attrs["dataset_format_version"] = DATASET_FORMAT_VERSION
        handle.attrs["geometry_seed"] = int(data_cfg.geometry_seed)
        handle.attrs["variations"] = int(data_cfg.variations)
        handle.attrs["initial_field_types_json"] = json.dumps(list(data_cfg.initial_field_types))
        handle.attrs["augment_sign_flip"] = bool(data_cfg.augment_sign_flip)
        handle.attrs["feature_version"] = 1
        handle.attrs["feature_dim_raw"] = 9
        handle.attrs["feature_order"] = "phi9"
        handle.attrs["shape_types_json"] = json.dumps(list(data_cfg.shape_types))
        handle.attrs["train_fraction"] = float(data_cfg.train_fraction)
        handle.attrs["val_fraction"] = float(data_cfg.val_fraction)
        handle.attrs["ellipse_num_a"] = int(data_cfg.ellipse_num_a)
        handle.attrs["ellipse_variations_per_a"] = int(data_cfg.ellipse_variations_per_a)
        handle.attrs["ellipse_axis_ratio_min"] = float(data_cfg.ellipse_axis_ratio_min)
        handle.attrs["ellipse_axis_ratio_max"] = float(data_cfg.ellipse_axis_ratio_max)
        handle.attrs["ellipse_rotation_min"] = float(data_cfg.ellipse_rotation_min)
        handle.attrs["ellipse_rotation_max"] = float(data_cfg.ellipse_rotation_max)
        handle.attrs["ellipse_a_min_factor"] = float(data_cfg.ellipse_a_min_factor)
        handle.attrs["ellipse_sdf_newton_max_iter"] = int(data_cfg.ellipse_sdf_newton_max_iter)
        handle.attrs["ellipse_sdf_newton_tol"] = float(data_cfg.ellipse_sdf_newton_tol)
        handle.attrs["ellipse_hp_dps"] = int(data_cfg.ellipse_hp_dps)
        handle.attrs["ellipse_hp_newton_max_iter"] = int(data_cfg.ellipse_hp_newton_max_iter)
        handle.attrs["generation_batch_size"] = int(generation_cfg.generation_batch_size)
        handle.attrs["num_workers"] = int(generation_cfg.num_workers)
        handle.attrs["output_dir"] = str(Path(generation_cfg.output_dir))
        handle.attrs["dataset_name"] = str(generation_cfg.dataset_name)
        handle.attrs["scale_h"] = bool(data_cfg.scale_h)
        handle.attrs["feature_transform"] = "phi9_over_h" if data_cfg.scale_h else "phi9"
        handle.create_dataset("resolutions", data=np.asarray(data_cfg.resolutions, dtype=np.int32))
        handle.create_dataset("blueprints_json", data=json.dumps(bundle.get("blueprints", [])).encode("utf-8"))
        split_group = handle.create_group("split_blueprint_indices")
        for split_name, indices in bundle.get("split_blueprint_indices", {}).items():
            split_group.create_dataset(split_name, data=np.asarray(indices, dtype=np.int32))
        for split_name, split in bundle["splits"].items():
            group = handle.create_group(split_name)
            for field_name in FIELD_ORDER:
                if field_name not in split:
                    continue
                group.create_dataset(
                    field_name,
                    data=np.asarray(split[field_name], dtype=FIELD_DTYPES[field_name]),
                    compression="gzip",
                )

    save_dataset_manifest(
        {
            **bundle,
            "generation_config": generation_cfg,
        },
        output_path,
    )
    return output_path


def build_dataset_summary_from_hdf5(dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    with h5py.File(dataset_path, "r") as handle:
        splits: dict[str, Any] = {}
        for split_name in ("train", "val", "test"):
            if split_name not in handle:
                continue
            group = handle[split_name]
            available_fields = [field_name for field_name in FIELD_ORDER if field_name in group]
            size_field = "features" if "features" in group else "phi9"
            splits[split_name] = {
                "size": int(group[size_field].shape[0]),
                "fields": [
                    {
                        "name": field_name,
                        "shape": list(group[field_name].shape),
                        "dtype": str(group[field_name].dtype),
                    }
                    for field_name in available_fields
                ],
            }
        return {
            "dataset_file": {
                "name": dataset_path.name,
                "path": str(dataset_path),
                "manifest_path": str(dataset_manifest_path(dataset_path)),
            },
            "dataset_format_version": int(handle.attrs.get("dataset_format_version", 0)),
            "task": "3x3 stencil features -> h*kappa",
            "configs": {
                "data": {
                    "resolutions": [int(item) for item in handle["resolutions"][:]],
                    "geometry_seed": int(handle.attrs.get("geometry_seed", 0)),
                    "variations": int(handle.attrs.get("variations", 0)),
                    "initial_field_types": json.loads(str(handle.attrs.get("initial_field_types_json", "[]"))),
                    "augment_sign_flip": bool(handle.attrs.get("augment_sign_flip", False)),
                    "feature_version": int(handle.attrs.get("feature_version", 1)),
                    "feature_dim_raw": int(handle.attrs.get("feature_dim_raw", 9)),
                    "feature_order": str(handle.attrs.get("feature_order", "phi9")),
                    "shape_types": json.loads(str(handle.attrs.get("shape_types_json", '["circle"]'))),
                    "train_fraction": float(handle.attrs.get("train_fraction", 0.0)),
                    "val_fraction": float(handle.attrs.get("val_fraction", 0.0)),
                    "ellipse_num_a": int(handle.attrs.get("ellipse_num_a", DataConfig().ellipse_num_a)),
                    "ellipse_variations_per_a": int(
                        handle.attrs.get("ellipse_variations_per_a", DataConfig().ellipse_variations_per_a)
                    ),
                    "ellipse_axis_ratio_min": float(
                        handle.attrs.get("ellipse_axis_ratio_min", DataConfig().ellipse_axis_ratio_min)
                    ),
                    "ellipse_axis_ratio_max": float(
                        handle.attrs.get("ellipse_axis_ratio_max", DataConfig().ellipse_axis_ratio_max)
                    ),
                    "ellipse_rotation_min": float(
                        handle.attrs.get("ellipse_rotation_min", DataConfig().ellipse_rotation_min)
                    ),
                    "ellipse_rotation_max": float(
                        handle.attrs.get("ellipse_rotation_max", DataConfig().ellipse_rotation_max)
                    ),
                    "ellipse_a_min_factor": float(
                        handle.attrs.get("ellipse_a_min_factor", DataConfig().ellipse_a_min_factor)
                    ),
                    "ellipse_sdf_newton_max_iter": int(
                        handle.attrs.get("ellipse_sdf_newton_max_iter", DataConfig().ellipse_sdf_newton_max_iter)
                    ),
                    "ellipse_sdf_newton_tol": float(
                        handle.attrs.get("ellipse_sdf_newton_tol", DataConfig().ellipse_sdf_newton_tol)
                    ),
                    "ellipse_hp_dps": int(handle.attrs.get("ellipse_hp_dps", DataConfig().ellipse_hp_dps)),
                    "ellipse_hp_newton_max_iter": int(
                        handle.attrs.get("ellipse_hp_newton_max_iter", DataConfig().ellipse_hp_newton_max_iter)
                    ),
                    "scale_h": bool(handle.attrs.get("scale_h", False)),
                    "feature_transform": str(handle.attrs.get("feature_transform", "phi9")),
                },
                "generation": {
                    "generation_batch_size": int(handle.attrs.get("generation_batch_size", 0)),
                    "num_workers": int(handle.attrs.get("num_workers", 0)),
                    "output_dir": str(handle.attrs.get("output_dir", dataset_path.parent)),
                    "dataset_name": str(handle.attrs.get("dataset_name", dataset_path.name)),
                },
            },
            "splits": splits,
        }


def load_training_arrays_from_hdf5(path: str | Path) -> dict[str, Any]:
    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path.resolve()}")

    with h5py.File(h5_path, "r") as handle:
        dataset_format_version = int(handle.attrs.get("dataset_format_version", 0))
        if dataset_format_version not in SUPPORTED_DATASET_FORMAT_VERSIONS:
            raise ValueError(
                f"Dataset format version {dataset_format_version} is incompatible with the current stencil pipeline. "
                f"Supported versions are {sorted(SUPPORTED_DATASET_FORMAT_VERSIONS)}; "
                f"regenerate the HDF5 dataset to get dataset_format_version={DATASET_FORMAT_VERSION}."
            )
        raw_initial_field_types = handle.attrs.get("initial_field_types_json", '["sdf", "nonsdf"]')
        if isinstance(raw_initial_field_types, bytes):
            raw_initial_field_types = raw_initial_field_types.decode("utf-8")
        raw_shape_types = handle.attrs.get("shape_types_json", '["circle"]')
        if isinstance(raw_shape_types, bytes):
            raw_shape_types = raw_shape_types.decode("utf-8")
        feature_version = int(handle.attrs.get("feature_version", 1))
        raw_feature_dim = int(handle.attrs.get("feature_dim_raw", 9))
        feature_order = str(handle.attrs.get("feature_order", "phi9"))
        if feature_version != 1 or raw_feature_dim != 9 or feature_order != "phi9":
            raise ValueError(
                f"Dataset {h5_path.resolve()} is not a supported V1 phi9 dataset. "
                f"Got feature_version={feature_version}, feature_dim_raw={raw_feature_dim}, feature_order={feature_order!r}. "
                "Regenerate the dataset with the V1-only train_generate pipeline."
            )
        data_config = DataConfig(
            resolutions=tuple(int(item) for item in handle["resolutions"][:]),
            geometry_seed=int(handle.attrs.get("geometry_seed", DataConfig().geometry_seed)),
            variations=int(handle.attrs.get("variations", DataConfig().variations)),
            initial_field_types=tuple(json.loads(str(raw_initial_field_types))),
            augment_sign_flip=bool(handle.attrs.get("augment_sign_flip", False)),
            shape_types=tuple(json.loads(str(raw_shape_types))),
            train_fraction=float(handle.attrs.get("train_fraction", DataConfig().train_fraction)),
            val_fraction=float(handle.attrs.get("val_fraction", DataConfig().val_fraction)),
            ellipse_num_a=int(handle.attrs.get("ellipse_num_a", DataConfig().ellipse_num_a)),
            ellipse_variations_per_a=int(
                handle.attrs.get("ellipse_variations_per_a", DataConfig().ellipse_variations_per_a)
            ),
            ellipse_axis_ratio_min=float(
                handle.attrs.get("ellipse_axis_ratio_min", DataConfig().ellipse_axis_ratio_min)
            ),
            ellipse_axis_ratio_max=float(
                handle.attrs.get("ellipse_axis_ratio_max", DataConfig().ellipse_axis_ratio_max)
            ),
            ellipse_rotation_min=float(handle.attrs.get("ellipse_rotation_min", DataConfig().ellipse_rotation_min)),
            ellipse_rotation_max=float(handle.attrs.get("ellipse_rotation_max", DataConfig().ellipse_rotation_max)),
            ellipse_a_min_factor=float(handle.attrs.get("ellipse_a_min_factor", DataConfig().ellipse_a_min_factor)),
            ellipse_sdf_newton_max_iter=int(
                handle.attrs.get("ellipse_sdf_newton_max_iter", DataConfig().ellipse_sdf_newton_max_iter)
            ),
            ellipse_sdf_newton_tol=float(handle.attrs.get("ellipse_sdf_newton_tol", DataConfig().ellipse_sdf_newton_tol)),
            ellipse_hp_dps=int(handle.attrs.get("ellipse_hp_dps", DataConfig().ellipse_hp_dps)),
            ellipse_hp_newton_max_iter=int(
                handle.attrs.get("ellipse_hp_newton_max_iter", DataConfig().ellipse_hp_newton_max_iter)
            ),
            scale_h=bool(handle.attrs.get("scale_h", False)),
        )
        generation_config = normalize_generation_config(
            GenerationConfig(
                num_workers=int(handle.attrs.get("num_workers", GenerationConfig().num_workers)),
                generation_batch_size=int(handle.attrs.get("generation_batch_size", GenerationConfig().generation_batch_size)),
                output_dir=Path(str(handle.attrs.get("output_dir", h5_path.parent))),
                dataset_name=str(handle.attrs.get("dataset_name", h5_path.name)),
            )
        )
        raw_blueprints = handle["blueprints_json"][()]
        if isinstance(raw_blueprints, bytes):
            raw_blueprints = raw_blueprints.decode("utf-8")
        blueprints = json.loads(str(raw_blueprints))
        split_blueprint_indices = {
            name: [int(item) for item in handle["split_blueprint_indices"][name][:]]
            for name in ("train", "val", "test")
            if "split_blueprint_indices" in handle and name in handle["split_blueprint_indices"]
        }
        splits: dict[str, dict[str, np.ndarray]] = {}
        for split_name in ("train", "val", "test"):
            if split_name not in handle:
                continue
            group = handle[split_name]
            missing_fields = [field_name for field_name in ("phi9", "hkappa_target") if field_name not in group]
            if missing_fields:
                raise ValueError(
                    f"Split {split_name!r} in dataset {h5_path.resolve()} is missing required fields: {missing_fields}."
                )
            phi9 = np.asarray(group["phi9"][:], dtype=np.float32)
            features = np.asarray(group["features"][:], dtype=np.float32) if "features" in group else phi9.copy()
            if phi9.ndim != 2 or phi9.shape[1] != 9:
                raise ValueError(f"Split {split_name!r} phi9 must have shape (N, 9), got {phi9.shape}.")
            if features.ndim != 2 or features.shape[1] != 9:
                raise ValueError(
                    f"Split {split_name!r} features must have shape (N, 9) for the V1 pipeline, got {features.shape}."
                )
            splits[split_name] = {
                "phi9": phi9,
                "features": features,
                "hkappa_target": np.asarray(group["hkappa_target"][:], dtype=np.float32),
            }

    return {
        "config": data_config,
        "generation_config": generation_config,
        "blueprints": blueprints,
        "split_blueprint_indices": split_blueprint_indices,
        "split_blueprint_counts": {name: len(ids) for name, ids in split_blueprint_indices.items()},
        "split_shape_blueprint_counts": {
            name: {
                shape_type: int(
                    sum(1 for idx in ids if str(blueprints[idx]["meta"].get("shape_type", "circle")) == shape_type)
                )
                for shape_type in sorted({str(blueprint["meta"].get("shape_type", "circle")) for blueprint in blueprints})
            }
            for name, ids in split_blueprint_indices.items()
        },
        "splits": splits,
        "sizes": {name: int(split["features"].shape[0]) for name, split in splits.items()},
    }
