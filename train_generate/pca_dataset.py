"""Offline PCA-18 (V3) dataset generation and feature-transform persistence.

The training server has weak CPU, so we do not want to compute the PCA
projection inside the training loop. Instead we compress a V2 27D HDF5 dataset
(phi9 + nx9 + ny9) into an 18D PCA dataset once, offline, and let training read
the 18D features directly. Evaluation reconstructs the raw 27D features and
applies the *same* transform (standardize 27D -> project to 18D) so that the
training and evaluation feature spaces stay identical.

Persistence (see PCA-18 plan V3):
    dataset/<stem>_pca18.h5            18D features, ready for training
    dataset/<stem>_pca18_transform.npz full V3 feature_transform + provenance
    checkpoint (.pt)                   embeds the same feature_transform
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import DataConfig, GenerationConfig
from .io import (
    DATASET_FORMAT_VERSION,
    FIELD_DTYPES,
    FIELD_ORDER,
    load_training_arrays_from_hdf5,
    normalize_generation_config,
    save_dataset_manifest,
)


PCA_DIM = 18
PCA_MIN_EVR = 0.95
PCA_TRANSFORM_SCHEMA_VERSION = 1

RAW_FEATURE_DIM_V3 = 27
FEATURE_ORDER_V3 = "pca18(phi9+nx9+ny9)"
TRANSFORM_KIND_V3 = "standardize_pca18"


def pca_dataset_paths(source_path: str | Path) -> tuple[Path, Path]:
    """Return the (h5, npz) sidecar paths next to ``source_path``."""
    source_path = Path(source_path)
    stem = source_path.stem
    parent = source_path.parent
    return parent / f"{stem}_pca18.h5", parent / f"{stem}_pca18_transform.npz"


def fit_pca_transform(train_features: np.ndarray, *, dataset_path: str | Path) -> dict[str, Any]:
    """Fit the standardize -> PCA(18) transform on 27D training features."""
    features = np.asarray(train_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Training features must have shape (N, 27), got {features.shape}.")
    if features.shape[1] != RAW_FEATURE_DIM_V3:
        raise ValueError(
            f"PCA-18 requires 27D (V2 phi9+nx9+ny9) features, got raw_feature_dim={features.shape[1]}."
        )
    if features.shape[0] == 0:
        raise ValueError("Cannot fit a PCA transform on an empty training split.")
    if np.any(~np.isfinite(features)):
        raise ValueError("Training features contain non-finite values; cannot fit PCA transform.")

    n_samples = int(features.shape[0])
    mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)

    standardized = ((features - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32, copy=False)
    # full_matrices=False keeps the thin decomposition; the right singular
    # vectors (Vt rows) are the principal axes of the standardized features.
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    explained_variance_all = (singular_values.astype(np.float64) ** 2) / max(n_samples - 1, 1)
    total_variance = float(explained_variance_all.sum())
    if not (total_variance > 0.0):
        raise ValueError("PCA fit produced non-positive total variance; the training features are degenerate.")
    evr_all = explained_variance_all / total_variance

    components = np.asarray(vt[:PCA_DIM], dtype=np.float32)
    explained_variance = explained_variance_all[:PCA_DIM].astype(np.float32)
    explained_variance_ratio = evr_all[:PCA_DIM].astype(np.float32)
    cumulative_evr = float(evr_all[:PCA_DIM].sum())
    if cumulative_evr < PCA_MIN_EVR:
        raise ValueError(
            f"PCA-18 retains only cumulative EVR={cumulative_evr:.6f}, below the required "
            f"PCA_MIN_EVR={PCA_MIN_EVR}. The 27D features may be less correlated than expected; "
            "review the source dataset before generating a PCA-18 dataset."
        )

    return {
        "transform_schema_version": PCA_TRANSFORM_SCHEMA_VERSION,
        "transform_kind": TRANSFORM_KIND_V3,
        "feature_version": 3,
        "raw_feature_dim": RAW_FEATURE_DIM_V3,
        "output_dim": PCA_DIM,
        "feature_order": FEATURE_ORDER_V3,
        "source_split": "train",
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else "",
        "mean": mean,
        "std": std,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_evr": cumulative_evr,
    }


def apply_pca_transform(features: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Project 27D raw features to 18D: standardize then rotate by ``components``."""
    values = np.asarray(features, dtype=np.float32)
    raw_dim = int(transform["raw_feature_dim"])
    if values.ndim != 2 or values.shape[1] != raw_dim:
        raise ValueError(f"Feature matrix must have shape (N, {raw_dim}), got {values.shape}.")
    mean = np.asarray(transform["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(transform["std"], dtype=np.float32).reshape(1, -1)
    components = np.asarray(transform["components"], dtype=np.float32)
    standardized = (values - mean) / std
    return (standardized @ components.T).astype(np.float32, copy=False)


def save_pca_transform_npz(path: str | Path, transform: dict[str, Any], *, source_path: str | Path) -> Path:
    """Persist the full V3 transform plus source provenance for freshness checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path)
    src_stat = source_path.stat() if source_path.exists() else None
    np.savez(
        path,
        transform_schema_version=np.int64(transform["transform_schema_version"]),
        transform_kind=str(transform["transform_kind"]),
        feature_version=np.int64(transform["feature_version"]),
        raw_feature_dim=np.int64(transform["raw_feature_dim"]),
        output_dim=np.int64(transform["output_dim"]),
        feature_order=str(transform["feature_order"]),
        source_split=str(transform["source_split"]),
        dataset_path=str(transform["dataset_path"]),
        pca_dim=np.int64(PCA_DIM),
        pca_min_evr=np.float64(PCA_MIN_EVR),
        mean=np.asarray(transform["mean"], dtype=np.float32),
        std=np.asarray(transform["std"], dtype=np.float32),
        components=np.asarray(transform["components"], dtype=np.float32),
        explained_variance=np.asarray(transform["explained_variance"], dtype=np.float32),
        explained_variance_ratio=np.asarray(transform["explained_variance_ratio"], dtype=np.float32),
        cumulative_evr=np.float64(transform["cumulative_evr"]),
        source_dataset_path=str(source_path.resolve()),
        source_dataset_size=np.int64(src_stat.st_size if src_stat is not None else -1),
        source_dataset_mtime_ns=np.int64(src_stat.st_mtime_ns if src_stat is not None else -1),
    )
    return path


def load_pca_transform_npz(path: str | Path) -> dict[str, Any]:
    """Restore the canonical V3 feature_transform dict from an NPZ sidecar."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"PCA transform NPZ not found: {npz_path.resolve()}")
    with np.load(npz_path, allow_pickle=False) as data:
        transform = {
            "transform_schema_version": int(data["transform_schema_version"]),
            "transform_kind": str(data["transform_kind"]),
            "feature_version": int(data["feature_version"]),
            "raw_feature_dim": int(data["raw_feature_dim"]),
            "output_dim": int(data["output_dim"]),
            "feature_order": str(data["feature_order"]),
            "source_split": str(data["source_split"]),
            "dataset_path": str(data["dataset_path"]),
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "std": np.asarray(data["std"], dtype=np.float32),
            "components": np.asarray(data["components"], dtype=np.float32),
            "explained_variance": np.asarray(data["explained_variance"], dtype=np.float32),
            "explained_variance_ratio": np.asarray(data["explained_variance_ratio"], dtype=np.float32),
            "cumulative_evr": float(data["cumulative_evr"]),
        }
    return transform


def is_pca_dataset_fresh(npz_path: str | Path, source_path: str | Path) -> bool:
    """Return True when the cached PCA-18 dataset matches the current source."""
    npz_path = Path(npz_path)
    source_path = Path(source_path)
    h5_path, _ = pca_dataset_paths(source_path)
    if not (npz_path.exists() and h5_path.exists() and source_path.exists()):
        return False
    try:
        src_stat = source_path.stat()
        with np.load(npz_path, allow_pickle=False) as data:
            checks: list[tuple[Any, Any]] = [
                (str(data["source_dataset_path"]), str(source_path.resolve())),
                (int(data["source_dataset_size"]), int(src_stat.st_size)),
                (int(data["source_dataset_mtime_ns"]), int(src_stat.st_mtime_ns)),
                (int(data["transform_schema_version"]), PCA_TRANSFORM_SCHEMA_VERSION),
                (str(data["transform_kind"]), TRANSFORM_KIND_V3),
                (int(data["feature_version"]), 3),
                (int(data["raw_feature_dim"]), RAW_FEATURE_DIM_V3),
                (int(data["output_dim"]), PCA_DIM),
                (str(data["feature_order"]), FEATURE_ORDER_V3),
                (int(data["pca_dim"]), PCA_DIM),
                (float(data["pca_min_evr"]), PCA_MIN_EVR),
            ]
    except (KeyError, ValueError, OSError):
        return False
    return all(actual == expected for actual, expected in checks)


def save_pca_training_dataset_hdf5(
    pca_bundle: dict[str, Any],
    *,
    source_bundle: dict[str, Any],
    path: str | Path,
    transform: dict[str, Any],
    npz_path: str | Path,
) -> Path:
    """Write the 18D PCA dataset, copying all source metadata and V3 attrs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_cfg: DataConfig = source_bundle["config"]
    generation_cfg: GenerationConfig = normalize_generation_config(source_bundle["generation_config"])

    with h5py.File(output_path, "w") as handle:
        # ── shared dataset metadata (mirrors io.save_training_dataset_hdf5) ──
        handle.attrs["dataset_format_version"] = DATASET_FORMAT_VERSION
        handle.attrs["geometry_seed"] = int(data_cfg.geometry_seed)
        handle.attrs["variations"] = int(data_cfg.variations)
        handle.attrs["initial_field_types_json"] = json.dumps(list(data_cfg.initial_field_types))
        handle.attrs["augment_sign_flip"] = bool(data_cfg.augment_sign_flip)
        handle.attrs["augment_gradient"] = bool(data_cfg.augment_gradient)
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
        handle.attrs["augment_scale_alpha_json"] = json.dumps(list(data_cfg.augment_scale_alpha))

        # ── V3 feature contract ──
        handle.attrs["feature_version"] = 3
        handle.attrs["feature_dim_raw"] = RAW_FEATURE_DIM_V3       # original source dim (provenance)
        handle.attrs["feature_dim_model"] = PCA_DIM                # actual stored training-input dim
        handle.attrs["feature_order"] = FEATURE_ORDER_V3
        handle.attrs["feature_transform"] = TRANSFORM_KIND_V3
        handle.attrs["transform_schema_version"] = PCA_TRANSFORM_SCHEMA_VERSION
        handle.attrs["pca_transform_path"] = str(Path(npz_path).resolve())
        handle.attrs["pca_cum_evr18"] = float(transform["cumulative_evr"])

        # ── grids / blueprints / splits provenance ──
        handle.create_dataset("resolutions", data=np.asarray(data_cfg.resolutions, dtype=np.int32))
        handle.create_dataset(
            "blueprints_json", data=json.dumps(source_bundle.get("blueprints", [])).encode("utf-8")
        )
        split_group = handle.create_group("split_blueprint_indices")
        for split_name, indices in source_bundle.get("split_blueprint_indices", {}).items():
            split_group.create_dataset(split_name, data=np.asarray(indices, dtype=np.int32))

        # ── per-split arrays (phi9 + hkappa unchanged, features = 18D) ──
        for split_name, split in pca_bundle["splits"].items():
            group = handle.create_group(split_name)
            for field_name in FIELD_ORDER:
                if field_name not in split:
                    continue
                group.create_dataset(
                    field_name,
                    data=np.asarray(split[field_name], dtype=FIELD_DTYPES[field_name]),
                    compression="gzip",
                )
            if "alpha_scale" in split:
                group.create_dataset(
                    "alpha_scale",
                    data=np.asarray(split["alpha_scale"], dtype=np.float32),
                    compression="gzip",
                )

    save_dataset_manifest({**pca_bundle, "generation_config": generation_cfg}, output_path)
    return output_path


def generate_pca_dataset(source_path: str | Path) -> tuple[Path, Path]:
    """Load a V2 27D dataset, fit/apply PCA-18, and persist h5 + npz sidecars."""
    source_path = Path(source_path)
    h5_path, npz_path = pca_dataset_paths(source_path)

    source_bundle = load_training_arrays_from_hdf5(source_path)
    if int(source_bundle["feature_version"]) != 2 or int(source_bundle["raw_feature_dim"]) != RAW_FEATURE_DIM_V3:
        raise ValueError(
            f"PCA-18 source must be a V2 27D dataset (phi9+nx9+ny9). "
            f"Got feature_version={source_bundle['feature_version']}, "
            f"raw_feature_dim={source_bundle['raw_feature_dim']} from {source_path.resolve()}."
        )

    if "train" not in source_bundle["splits"]:
        raise ValueError(f"Source dataset {source_path.resolve()} has no train split to fit PCA on.")
    transform = fit_pca_transform(source_bundle["splits"]["train"]["features"], dataset_path=source_path)

    pca_splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split in source_bundle["splits"].items():
        pca_split: dict[str, np.ndarray] = {
            "phi9": np.asarray(split["phi9"], dtype=np.float32),
            "features": apply_pca_transform(split["features"], transform),
            "hkappa_target": np.asarray(split["hkappa_target"], dtype=np.float32),
        }
        if "alpha_scale" in split:
            pca_split["alpha_scale"] = np.asarray(split["alpha_scale"], dtype=np.float32)
        pca_splits[split_name] = pca_split

    pca_bundle = {
        **source_bundle,
        "splits": pca_splits,
        "sizes": {name: int(split["features"].shape[0]) for name, split in pca_splits.items()},
    }

    save_pca_transform_npz(npz_path, transform, source_path=source_path)
    save_pca_training_dataset_hdf5(
        pca_bundle, source_bundle=source_bundle, path=h5_path, transform=transform, npz_path=npz_path
    )
    return h5_path, npz_path


def ensure_pca_dataset(source_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    """Top-level entry: reuse a fresh cached PCA-18 dataset or regenerate it."""
    source_path = Path(source_path)
    h5_path, npz_path = pca_dataset_paths(source_path)
    if is_pca_dataset_fresh(npz_path, source_path):
        print(f"Reusing existing PCA-18 dataset: {h5_path.resolve()}")
    else:
        print(f"Generating PCA-18 dataset from {source_path.resolve()} ...")
        generate_pca_dataset(source_path)
        print(f"Wrote PCA-18 dataset: {h5_path.resolve()}")
    transform = load_pca_transform_npz(npz_path)
    return h5_path, npz_path, transform
