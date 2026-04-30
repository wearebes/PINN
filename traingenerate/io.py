from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import DataConfig, GenerationConfig, default_dataset_name


DATASET_FORMAT_VERSION = 3
FIELD_ORDER = ("phi9", "hkappa_target")
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


def _preview_records_from_split(split: dict[str, np.ndarray], *, limit: int = DEFAULT_PREVIEW_SAMPLES) -> list[dict[str, float]]:
    n_items = min(int(split["phi9"].shape[0]), int(limit))
    preview: list[dict[str, float]] = []
    for row_idx in range(n_items):
        preview.append(
            {
                field_name: float(np.asarray(split[field_name][row_idx]).reshape(-1)[0].item())
                for field_name in FIELD_ORDER
            }
        )
    return preview


def build_dataset_manifest(bundle: dict[str, Any], dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    generation_cfg = normalize_generation_config(bundle["generation_config"])
    splits: dict[str, Any] = {}
    for split_name, split in bundle["splits"].items():
        splits[split_name] = {
            "size": int(split["phi9"].shape[0]),
            "fields": [
                {
                    "name": field_name,
                    "shape": list(split[field_name].shape),
                    "dtype": str(split[field_name].dtype),
                }
                for field_name in FIELD_ORDER
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
        "task": "3x3 phi stencil -> h*kappa",
        "configs": {
            "data": _json_ready(asdict(bundle["config"])),
            "generation": _json_ready(asdict(generation_cfg)),
        },
        "split_blueprint_counts": _json_ready(bundle.get("split_blueprint_counts", {})),
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
        handle.attrs["dataset_format_version"] = DATASET_FORMAT_VERSION
        handle.attrs["geometry_seed"] = int(data_cfg.geometry_seed)
        handle.attrs["variations"] = int(data_cfg.variations)
        handle.attrs["initial_field_types_json"] = json.dumps(list(data_cfg.initial_field_types))
        handle.attrs["n_samples_per_circle"] = int(data_cfg.n_samples_per_circle)
        handle.attrs["train_fraction"] = float(data_cfg.train_fraction)
        handle.attrs["val_fraction"] = float(data_cfg.val_fraction)
        handle.attrs["generation_batch_size"] = int(generation_cfg.generation_batch_size)
        handle.attrs["num_workers"] = int(generation_cfg.num_workers)
        handle.attrs["output_dir"] = str(Path(generation_cfg.output_dir))
        handle.attrs["dataset_name"] = str(generation_cfg.dataset_name)
        handle.create_dataset("resolutions", data=np.asarray(data_cfg.resolutions, dtype=np.int32))
        handle.create_dataset("blueprints_json", data=json.dumps(bundle.get("blueprints", [])).encode("utf-8"))
        split_group = handle.create_group("split_blueprint_indices")
        for split_name, indices in bundle.get("split_blueprint_indices", {}).items():
            split_group.create_dataset(split_name, data=np.asarray(indices, dtype=np.int32))
        for split_name, split in bundle["splits"].items():
            group = handle.create_group(split_name)
            for field_name in FIELD_ORDER:
                group.create_dataset(field_name, data=np.asarray(split[field_name], dtype=np.float32), compression="gzip")

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
            splits[split_name] = {
                "size": int(group["phi9"].shape[0]),
                "fields": [
                    {
                        "name": field_name,
                        "shape": list(group[field_name].shape),
                        "dtype": str(group[field_name].dtype),
                    }
                    for field_name in FIELD_ORDER
                ],
            }
        return {
            "dataset_file": {
                "name": dataset_path.name,
                "path": str(dataset_path),
                "manifest_path": str(dataset_manifest_path(dataset_path)),
            },
            "dataset_format_version": int(handle.attrs.get("dataset_format_version", 0)),
            "task": "3x3 phi stencil -> h*kappa",
            "configs": {
                "data": {
                    "resolutions": [int(item) for item in handle["resolutions"][:]],
                    "geometry_seed": int(handle.attrs.get("geometry_seed", 0)),
                    "variations": int(handle.attrs.get("variations", 0)),
                    "initial_field_types": json.loads(str(handle.attrs.get("initial_field_types_json", "[]"))),
                    "n_samples_per_circle": int(handle.attrs.get("n_samples_per_circle", 0)),
                    "train_fraction": float(handle.attrs.get("train_fraction", 0.0)),
                    "val_fraction": float(handle.attrs.get("val_fraction", 0.0)),
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
        if dataset_format_version != DATASET_FORMAT_VERSION:
            raise ValueError(
                f"Dataset format version {dataset_format_version} is incompatible with the current stencil pipeline. "
                f"Regenerate the HDF5 dataset so it has dataset_format_version={DATASET_FORMAT_VERSION}."
            )
        raw_initial_field_types = handle.attrs.get("initial_field_types_json", '["sdf", "nonsdf"]')
        if isinstance(raw_initial_field_types, bytes):
            raw_initial_field_types = raw_initial_field_types.decode("utf-8")
        data_config = DataConfig(
            resolutions=tuple(int(item) for item in handle["resolutions"][:]),
            geometry_seed=int(handle.attrs.get("geometry_seed", DataConfig().geometry_seed)),
            variations=int(handle.attrs.get("variations", DataConfig().variations)),
            initial_field_types=tuple(json.loads(str(raw_initial_field_types))),
            n_samples_per_circle=int(handle.attrs.get("n_samples_per_circle", DataConfig().n_samples_per_circle)),
            train_fraction=float(handle.attrs.get("train_fraction", DataConfig().train_fraction)),
            val_fraction=float(handle.attrs.get("val_fraction", DataConfig().val_fraction)),
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
        splits = {
            split_name: {
                field_name: np.asarray(handle[split_name][field_name][:], dtype=np.float32)
                for field_name in FIELD_ORDER
            }
            for split_name in ("train", "val", "test")
        }

    return {
        "config": data_config,
        "generation_config": generation_config,
        "blueprints": blueprints,
        "split_blueprint_indices": split_blueprint_indices,
        "split_blueprint_counts": {name: len(ids) for name, ids in split_blueprint_indices.items()},
        "splits": splits,
        "sizes": {name: int(split["phi9"].shape[0]) for name, split in splits.items()},
    }
