from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ModelType, create_train_config, filter_train_config_overrides
from model.model import create_model


CHECKPOINT_FORMAT_VERSION = 1
MODEL_CONFIG_VERSION = 1
LEGACY_V1_BASELINE_KEYMAP = {
    "net.1.weight": "net.0.weight",
    "net.1.bias": "net.0.bias",
    "net.3.weight": "net.2.weight",
    "net.3.bias": "net.2.bias",
    "net.5.weight": "net.4.weight",
    "net.5.bias": "net.4.bias",
    "net.7.weight": "net.6.weight",
    "net.7.bias": "net.6.bias",
    "net.9.weight": "net.8.weight",
    "net.9.bias": "net.8.bias",
}


def normalization_csv_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_phi9.csv")


def normalization_csv_candidates(model_path: str | Path) -> list[Path]:
    model_path = Path(model_path)
    return [
        normalization_csv_path(model_path),
        model_path.with_name(f"{model_path.stem}.csv"),
        model_path.with_name(f"{model_path.stem}_phi9_normalization.csv"),
    ]


def feature_transform_npz_candidates(model_path: str | Path) -> list[Path]:
    """V3 PCA-18 sidecar locations checked next to a checkpoint."""
    model_path = Path(model_path)
    return [
        model_path.with_name(f"{model_path.stem}_pca18_transform.npz"),
        model_path.with_name(f"{model_path.stem}.npz"),
    ]


def _as_float32_array(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass config, got {type(config).__name__}.")
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def normalize_checkpoint_model_config(
    *,
    model_type: ModelType,
    raw_model_config: dict[str, Any],
    model_config_version: int,
) -> dict[str, Any]:
    raw = dict(raw_model_config)
    migrated = dict(raw)

    if int(model_config_version) == 0 and model_type == "mlp":
        # Historical MLP checkpoints could persist an activation choice even though
        # the current architecture hard-codes ReLU in model/model.py.
        migrated.pop("activation", None)

    filtered_overrides, dropped_keys = filter_train_config_overrides(model_type, migrated)
    normalized_config = _config_to_dict(create_train_config(model_type, **filtered_overrides))
    return {
        "raw_model_config": raw,
        "normalized_model_config": normalized_config,
        "dropped_model_config_keys": sorted(set(dropped_keys + [key for key in raw.keys() if key not in migrated])),
        "model_config_version": int(model_config_version),
    }


def save_phi9_normalization_csv(path: str | Path, *, mean: np.ndarray, std: np.ndarray, dataset_path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path).resolve()
    mean = _as_float32_array(mean)
    std = _as_float32_array(std)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi_index", "mean", "std", "variance", "source_split", "feature_count", "dataset_path"])
        feature_count = len(mean)
        for idx in range(feature_count):
            writer.writerow([idx + 1, float(mean[idx]), float(std[idx]), float(std[idx] ** 2), "train", feature_count, str(dataset_path)])
    return path


def load_phi9_normalization_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Normalization CSV not found: {csv_path.resolve()}")

    means: list[float] = []
    stds: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "phi_index",
            "mean",
            "std",
            "variance",
            "source_split",
            "feature_count",
            "dataset_path",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"Normalization CSV {csv_path.resolve()} must have columns {sorted(required)}, got {reader.fieldnames!r}."
            )
        source_splits: set[str] = set()
        feature_counts: set[int] = set()
        for row_idx, row in enumerate(reader, start=1):
            phi_index = int(row["phi_index"])
            if phi_index != row_idx:
                raise ValueError(
                    f"Normalization CSV {csv_path.resolve()} has non-sequential phi_index at row {row_idx}: "
                    f"expected {row_idx}, got {phi_index}."
                )
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
            source_splits.add(str(row["source_split"]))
            feature_counts.add(int(row["feature_count"]))
    mean = np.asarray(means, dtype=np.float32)
    std = np.asarray(stds, dtype=np.float32)
    if mean.ndim != 1 or mean.size not in (9, 27):
        raise ValueError(
            f"Normalization CSV {csv_path.resolve()} must describe 9 (V1) or 27 (V2) features, "
            f"got {mean.size} entries."
        )
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError(f"Normalization CSV {csv_path.resolve()} contains non-finite values.")
    if np.any(std <= 0.0):
        raise ValueError(f"Normalization CSV {csv_path.resolve()} contains non-positive std values.")
    if source_splits != {"train"}:
        raise ValueError(f"Normalization CSV {csv_path.resolve()} must record source_split=train, got {sorted(source_splits)}.")
    if feature_counts not in ({9}, {27}):
        raise ValueError(
            f"Normalization CSV {csv_path.resolve()} must record feature_count in {{9, 27}}, got {sorted(feature_counts)}."
        )
    return mean, std


def build_legacy_feature_transform(mean: np.ndarray, std: np.ndarray, *, dataset_path: str | Path | None = None) -> dict[str, Any]:
    mean_arr = _as_float32_array(mean)
    std_arr = _as_float32_array(std)
    raw_dim = int(mean_arr.size)
    if raw_dim == 9:
        fv, fo = 1, "phi9"
    elif raw_dim == 27:
        fv, fo = 2, "phi9+nx9+ny9"
    else:
        raise ValueError(f"build_legacy_feature_transform: expected 9 (V1) or 27 (V2) entries, got {raw_dim}.")
    return {
        "transform_kind": "standardize",
        "feature_version": fv,
        "raw_feature_dim": raw_dim,
        "output_dim": raw_dim,
        "feature_order": fo,
        "source_split": "train",
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else "",
        "mean": mean_arr,
        "std": std_arr,
    }


def validate_feature_transform(transform: dict[str, Any]) -> dict[str, Any]:
    required = {
        "transform_kind",
        "feature_version",
        "raw_feature_dim",
        "output_dim",
        "feature_order",
        "source_split",
        "dataset_path",
        "mean",
        "std",
    }
    missing = required - set(transform.keys())
    if missing:
        raise ValueError(f"Feature transform is missing keys: {sorted(missing)}.")
    out = dict(transform)
    out["feature_version"] = int(out["feature_version"])
    out["raw_feature_dim"] = int(out["raw_feature_dim"])
    out["output_dim"] = int(out["output_dim"])
    out["feature_order"] = str(out["feature_order"])
    out["source_split"] = str(out["source_split"])
    out["dataset_path"] = str(out["dataset_path"])
    out["transform_kind"] = str(out["transform_kind"])
    out["mean"] = _as_float32_array(out["mean"])
    out["std"] = _as_float32_array(out["std"])
    if out["mean"].shape != (out["raw_feature_dim"],) or out["std"].shape != (out["raw_feature_dim"],):
        raise ValueError("Feature transform mean/std shapes do not match raw_feature_dim.")
    if np.any(out["std"] <= 0.0):
        raise ValueError("Feature transform std must be strictly positive.")

    # ── V3: standardize(27D) -> PCA project to 18D ──
    if out["transform_kind"] == "standardize_pca18":
        _v3_ok = (
            out["feature_version"] == 3
            and out["raw_feature_dim"] == 27
            and out["output_dim"] == 18
            and out["feature_order"] == "pca18(phi9+nx9+ny9)"
        )
        if not _v3_ok:
            raise ValueError(
                "Invalid V3 feature transform: expected feature_version=3, raw_feature_dim=27, "
                f"output_dim=18, feature_order='pca18(phi9+nx9+ny9)'. Got feature_version={out['feature_version']}, "
                f"raw_feature_dim={out['raw_feature_dim']}, output_dim={out['output_dim']}, "
                f"feature_order={out['feature_order']!r}."
            )
        components = out.get("components")
        if components is None:
            raise ValueError("V3 feature transform is missing the PCA 'components' matrix.")
        components = np.asarray(components, dtype=np.float32)
        if components.shape != (18, 27):
            raise ValueError(f"V3 PCA components must have shape (18, 27), got {components.shape}.")
        if np.any(~np.isfinite(components)):
            raise ValueError("V3 PCA components contain non-finite values.")
        result: dict[str, Any] = {
            "transform_schema_version": int(out.get("transform_schema_version", 1)),
            "transform_kind": "standardize_pca18",
            "feature_version": 3,
            "raw_feature_dim": 27,
            "output_dim": 18,
            "feature_order": "pca18(phi9+nx9+ny9)",
            "source_split": out["source_split"],
            "dataset_path": out["dataset_path"],
            "mean": out["mean"],
            "std": out["std"],
            "components": components,
        }
        for opt_key in ("explained_variance", "explained_variance_ratio", "cumulative_evr"):
            if out.get(opt_key) is not None:
                result[opt_key] = out[opt_key]
        return result

    # ── V1/V2: pure standardization ──
    legacy_components = np.asarray(
        out.get("components", np.zeros((0, out["raw_feature_dim"]), dtype=np.float32)), dtype=np.float32
    )
    legacy_eigenvalues = np.asarray(out.get("eigenvalues", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    _v1_ok = (
        out["transform_kind"] == "standardize"
        and out["feature_version"] == 1
        and out["raw_feature_dim"] == 9
        and out["output_dim"] == 9
        and out["feature_order"] == "phi9"
        and legacy_components.size == 0
        and legacy_eigenvalues.size == 0
    )
    _v2_ok = (
        out["transform_kind"] == "standardize"
        and out["feature_version"] == 2
        and out["raw_feature_dim"] == 27
        and out["output_dim"] == 27
        and out["feature_order"] == "phi9+nx9+ny9"
        and legacy_components.size == 0
        and legacy_eigenvalues.size == 0
    )
    if not (_v1_ok or _v2_ok):
        raise ValueError(
            "Only V1 (9D phi9 standardization), V2 (27D phi9+nx9+ny9 standardization), and "
            "V3 (standardize_pca18: 27D -> 18D) transforms are supported."
        )
    return {
        "transform_kind": "standardize",
        "feature_version": out["feature_version"],
        "raw_feature_dim": out["raw_feature_dim"],
        "output_dim": out["output_dim"],
        "feature_order": out["feature_order"],
        "source_split": out["source_split"],
        "dataset_path": out["dataset_path"],
        "mean": out["mean"],
        "std": out["std"],
        "components": np.zeros((0, out["raw_feature_dim"]), dtype=np.float32),
    }


def fit_feature_transform(
    train_features: np.ndarray,
    *,
    dataset_path: str | Path,
) -> dict[str, Any]:
    features = np.asarray(train_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Training features must have shape (N, D), got {features.shape}.")
    if features.shape[0] == 0:
        raise ValueError("Cannot fit feature transform on an empty training split.")
    if np.any(~np.isfinite(features)):
        raise ValueError("Training features contain non-finite values; cannot fit feature transform.")
    raw_dim = int(features.shape[1])
    if raw_dim not in (9, 27):
        raise ValueError(
            f"Only 9D (V1 phi9) and 27D (V2 phi9+nx9+ny9) feature dimensions are supported "
            f"for fitting the transform. Got raw_feature_dim={raw_dim}."
        )
    mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)
    return validate_feature_transform(
        build_legacy_feature_transform(mean, std, dataset_path=dataset_path)
    )

def apply_feature_transform(features: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    state = validate_feature_transform(transform)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != state["raw_feature_dim"]:
        raise ValueError(
            f"Feature matrix must have shape (N, {state['raw_feature_dim']}), got {values.shape}."
        )
    standardized = (values - state["mean"].reshape(1, -1)) / state["std"].reshape(1, -1)
    if state["transform_kind"] == "standardize_pca18":
        return (standardized @ state["components"].T).astype(np.float32, copy=False)
    return standardized.astype(np.float32, copy=False)


def save_feature_stats(path: str | Path, *, transform: dict[str, Any], dataset_path: str | Path) -> Path:
    state = validate_feature_transform({**transform, "dataset_path": str(Path(dataset_path).resolve())})
    if state["transform_kind"] == "standardize_pca18":
        raise ValueError(
            "V3 (standardize_pca18) transforms cannot be saved as a CSV sidecar; the PCA components "
            "matrix is persisted via the *_pca18_transform.npz file and embedded in the checkpoint. "
            "Do not call save_feature_stats for the PCA pipeline."
        )
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Normalization output must be a CSV file for the V1/V2 pipeline, got {output_path.resolve()}."
        )
    return save_phi9_normalization_csv(output_path, mean=state["mean"], std=state["std"], dataset_path=dataset_path)


def load_feature_transform(path: str | Path) -> dict[str, Any]:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature stats not found: {feature_path.resolve()}")
    suffix = feature_path.suffix.lower()
    if suffix == ".npz":
        from train_generate.pca_dataset import load_pca_transform_npz

        return validate_feature_transform(load_pca_transform_npz(feature_path))
    if suffix == ".csv":
        mean, std = load_phi9_normalization_csv(feature_path)
        return build_legacy_feature_transform(mean, std)
    raise ValueError(
        f"Unsupported feature-transform sidecar {feature_path.resolve()}. "
        "Expected .csv (V1/V2 standardization) or .npz (V3 PCA-18)."
    )


def resolve_feature_transform(
    *,
    model_path: str | Path,
    explicit_path: str | Path | None,
    checkpoint_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    # 1. explicit path (dispatched by suffix inside load_feature_transform)
    if explicit_path:
        path = Path(explicit_path)
        return load_feature_transform(path), str(path.resolve())
    # 2. embedded checkpoint transform (default for V3)
    checkpoint_transform = checkpoint_meta.get("feature_transform")
    if checkpoint_transform is not None:
        return validate_feature_transform(checkpoint_transform), "checkpoint"
    # 3. auto sidecar discovery: V3 .npz first, then V1/V2 .csv
    npz_candidates = feature_transform_npz_candidates(model_path)
    for npz_path in npz_candidates:
        if npz_path.exists():
            return load_feature_transform(npz_path), str(npz_path.resolve())
    csv_candidates = normalization_csv_candidates(model_path)
    for csv_path in csv_candidates:
        if csv_path.exists():
            return load_feature_transform(csv_path), str(csv_path.resolve())
    checked_paths = [str(path.resolve()) for path in (*npz_candidates, *csv_candidates)]
    raise FileNotFoundError(
        f"No feature transform found for checkpoint {Path(model_path).resolve()}. "
        f"Checked embedded checkpoint metadata and sidecars: {', '.join(checked_paths)}."
    )


def decode_phi9_to_patch(phi9: np.ndarray) -> np.ndarray:
    phi9 = np.asarray(phi9, dtype=np.float32)
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {phi9.shape}.")
    return phi9.reshape(-1, 3, 3).transpose(0, 2, 1)[:, :, ::-1]


def central_difference_hkappa_from_phi9(phi9: np.ndarray) -> np.ndarray:
    patch = decode_phi9_to_patch(phi9).astype(np.float64, copy=False)
    phi_x = 0.5 * (patch[:, 2, 1] - patch[:, 0, 1])
    phi_y = 0.5 * (patch[:, 1, 2] - patch[:, 1, 0])
    phi_xx = patch[:, 2, 1] - 2.0 * patch[:, 1, 1] + patch[:, 0, 1]
    phi_yy = patch[:, 1, 2] - 2.0 * patch[:, 1, 1] + patch[:, 1, 0]
    phi_xy = 0.25 * (patch[:, 2, 2] - patch[:, 2, 0] - patch[:, 0, 2] + patch[:, 0, 0])
    grad_sq = phi_x**2 + phi_y**2
    denom = np.power(grad_sq, 1.5)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
        bad = int(np.count_nonzero((~np.isfinite(denom)) | (denom <= 0.0)))
        raise ValueError(f"Central-difference baseline encountered {bad} non-positive or non-finite denominators.")
    hkappa = (phi_xx * phi_y**2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / denom
    if np.any(~np.isfinite(hkappa)):
        bad = int(np.count_nonzero(~np.isfinite(hkappa)))
        raise ValueError(f"Central-difference baseline produced {bad} non-finite h*kappa values.")
    return hkappa.astype(np.float64, copy=False)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")
    diff = prediction - target
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "maxae": float(np.max(np.abs(diff))),
    }

def serialize_feature_transform_for_checkpoint(transform: dict[str, Any]) -> dict[str, Any]:
    """Return a checkpoint-safe copy of feature_transform with numpy arrays converted to lists.

    torch.save with weights_only=True (PyTorch >= 2.4 default) rejects numpy arrays.
    validate_feature_transform() already converts lists back to arrays on load, so the
    round-trip is lossless.
    """
    out = validate_feature_transform(transform)
    for key in ("mean", "std", "components", "explained_variance", "explained_variance_ratio", "cumulative_evr"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key], dtype=np.float32).tolist()
    return out


def save_checkpoint_bundle(
    model: torch.nn.Module,
    *,
    model_type: str,
    model_config: Any,
    path: str | Path,
    feature_transform: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config_version": MODEL_CONFIG_VERSION,
        "model_type": str(model_type),
        "model_config": _config_to_dict(model_config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if feature_transform is not None:
        payload["feature_transform"] = serialize_feature_transform_for_checkpoint(feature_transform)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint_bundle(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path.resolve()}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path.resolve()} must be a dict payload.")
    required = {"checkpoint_format_version", "model_type", "model_config", "state_dict"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path.resolve()} is missing keys {sorted(missing)}. "
            "Old bare state_dict checkpoints are no longer supported."
        )
    if int(payload["checkpoint_format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_format_version={payload['checkpoint_format_version']} in {checkpoint_path.resolve()}."
        )
    if not isinstance(payload["state_dict"], dict):
        raise ValueError(f"Checkpoint {checkpoint_path.resolve()} has an invalid state_dict payload.")
    payload["state_dict"], payload["state_dict_compatibility"] = remap_checkpoint_state_dict(
        model_type=str(payload["model_type"]),
        state_dict=payload["state_dict"],
        checkpoint_path=checkpoint_path,
    )
    if "feature_transform" in payload and payload["feature_transform"] is not None:
        payload["feature_transform"] = validate_feature_transform(payload["feature_transform"])
    return payload


def remap_checkpoint_state_dict(
    *,
    model_type: str,
    state_dict: dict[str, Any],
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], str]:
    if str(model_type) != "mlp":
        return state_dict, "native"
    checkpoint_path = Path(checkpoint_path)
    key_set = set(str(key) for key in state_dict.keys())
    native_key_set = set(LEGACY_V1_BASELINE_KEYMAP.values())
    legacy_key_set = set(LEGACY_V1_BASELINE_KEYMAP.keys())
    if key_set == native_key_set:
        return state_dict, "native"
    if key_set == legacy_key_set:
        return (
            {
                LEGACY_V1_BASELINE_KEYMAP[str(key)]: value
                for key, value in state_dict.items()
            },
            "legacy_v1_baseline_remapped",
        )
    if key_set & legacy_key_set:
        raise ValueError(
            f"Checkpoint {checkpoint_path.resolve()} uses an unsupported historical MLP state_dict layout. "
            "Supported layouts are the current native keys "
            f"{sorted(native_key_set)} or the legacy V1 baseline keys {sorted(legacy_key_set)}."
        )
    return state_dict, "native"


def load_model_from_checkpoint(model_path: str | Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint_bundle(model_path)
    model_type = str(payload["model_type"])
    raw_model_config = dict(payload["model_config"])
    normalized = normalize_checkpoint_model_config(
        model_type=model_type,
        raw_model_config=raw_model_config,
        model_config_version=int(payload.get("model_config_version", 0)),
    )
    normalized_model_config = dict(normalized["normalized_model_config"])
    config = create_train_config(model_type, **normalized_model_config)
    model = create_model(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, {
        "model_type": model_type,
        "model_config": raw_model_config,
        "raw_model_config": normalized["raw_model_config"],
        "normalized_model_config": normalized_model_config,
        "dropped_model_config_keys": list(normalized["dropped_model_config_keys"]),
        "model_config_version": int(normalized["model_config_version"]),
        "feature_transform": payload.get("feature_transform"),
        "state_dict_compatibility": str(payload.get("state_dict_compatibility", "native")),
    }


@torch.inference_mode()
def predict_hkappa_full_batch(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    transform: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    transformed = apply_feature_transform(features, transform)
    batch = torch.from_numpy(transformed).to(device)
    prediction = model(batch).detach().cpu().numpy()
    return prediction.astype(np.float64, copy=False).reshape(-1)


@torch.inference_mode()
def predict_hkappa_one_by_one(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    transform: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    transformed = apply_feature_transform(features, transform)
    outputs = np.empty(transformed.shape[0], dtype=np.float64)
    for idx in range(transformed.shape[0]):
        sample = torch.from_numpy(transformed[idx : idx + 1]).to(device)
        pred = model(sample).detach().cpu().numpy().reshape(-1)
        outputs[idx] = float(pred[0])
    return outputs


def csv_to_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def add_swanlab_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard SwanLab CLI argument group to a parser."""
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="PINN")
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-description", type=str, default="")
    parser.add_argument("--swanlab-tags", type=str, default="")
    parser.add_argument("--swanlab-group", type=str, default="")
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        choices=("cloud", "local", "offline", "disabled"),
        default="cloud",
    )


def init_swanlab_run(
    *,
    project: str | None,
    experiment_name: str | None,
    description: str | None,
    tags: list[str] | None,
    group: str | None,
    workspace: str | None,
    logdir: str | None,
    mode: str | None,
    config: dict[str, Any],
):
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("SwanLab logging was requested, but `swanlab` is not installed.") from exc

    init_kwargs: dict[str, Any] = {
        "project": project,
        "workspace": workspace,
        "experiment_name": experiment_name,
        "description": description,
        "tags": tags,
        "group": group,
        "logdir": logdir,
        "mode": mode,
        "config": config,
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return swanlab.init(**init_kwargs)


def group_metric_rows(
    *,
    labels: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    numeric: np.ndarray,
    label_name: str,
    label_map: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).reshape(-1)
    rows: list[dict[str, Any]] = []
    for raw_value in np.unique(labels):
        mask = labels == raw_value
        label_value = int(raw_value) if np.issubdtype(labels.dtype, np.integer) else raw_value.item()
        display_value = label_map.get(int(label_value), str(label_value)) if label_map else str(label_value)
        rows.append(
            {
                label_name: label_value,
                "display": display_value,
                "sample_count": int(np.count_nonzero(mask)),
                "numeric_vs_analytic": compute_metrics(numeric[mask], target[mask]),
                "model_vs_analytic": compute_metrics(prediction[mask], target[mask]),
            }
        )
    return rows


ANGLE_BIN_FIELDNAMES = (
    "rho_model",
    "case_id",
    "case_label",
    "iter",
    "case_key",
    "angle_bin_start_deg",
    "angle_bin_end_deg",
    "sample_count",
    "numeric_mse",
    "numeric_mae",
    "numeric_maxae",
    "model_mse",
    "model_mae",
    "model_maxae",
)


def sanitize_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw))


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv_rows(
    output_path: str | Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0].keys()) if rows else tuple()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_swanlab_image(swanlab_module: Any, image_path: str | Path) -> Any:
    image_path = Path(image_path).resolve()
    last_error: Exception | None = None
    constructor_candidates: list[tuple[Any, str]] = []
    if hasattr(swanlab_module, "Image"):
        constructor_candidates.append((getattr(swanlab_module, "Image"), str(image_path)))
    media_namespace = getattr(swanlab_module, "media", None)
    if media_namespace is not None and hasattr(media_namespace, "Image"):
        constructor_candidates.append((getattr(media_namespace, "Image"), str(image_path)))
    data_namespace = getattr(swanlab_module, "data", None)
    if data_namespace is not None and hasattr(data_namespace, "Image"):
        constructor_candidates.append((getattr(data_namespace, "Image"), str(image_path)))
    for constructor, value in constructor_candidates:
        try:
            return constructor(value)
        except Exception as exc:  # pragma: no cover - defensive SDK fallback
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Unable to construct a SwanLab image object for {image_path}: {last_error}") from last_error
    raise RuntimeError("The installed SwanLab SDK does not expose an Image constructor.")


def build_swanlab_table_payload(
    swanlab_module: Any,
    rows: list[dict[str, Any]] | list[list[Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> Any:
    if rows and isinstance(rows[0], dict):
        dict_rows = [dict(item) for item in rows]  # shallow copy for deterministic order
        headers = list(fieldnames or dict_rows[0].keys())
        table_rows: list[list[Any]] = [headers]
        for row in dict_rows:
            table_rows.append([row.get(name, "") for name in headers])
    else:
        table_rows = list(rows)
    echarts_namespace = getattr(swanlab_module, "echarts", None)
    if echarts_namespace is not None and hasattr(echarts_namespace, "table"):
        return echarts_namespace.table(table_rows)
    text_type = getattr(swanlab_module, "Text", None)
    if text_type is not None:
        return text_type("\n".join(" | ".join(str(item) for item in row) for row in table_rows))
    return table_rows


def metric_summary(metric: dict[str, float]) -> dict[str, float]:
    mse = float(metric["mse"])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(metric["mae"]),
        "max_abs_err": float(metric["maxae"]),
    }


def build_case_summary_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        case_rows,
        key=lambda row: (float(row["summary"]["mae"]), str(row["case_key"]), int(row["iter"]), int(row["case_id"])),
    )
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        summary = row["summary"]
        numeric_summary = row["numeric_summary"]
        rows.append(
            {
                "rank": int(rank),
                "rho_model": int(row.get("rho_model", -1)),
                "case_key": str(row["case_key"]),
                "case_id": int(row["case_id"]),
                "case_label": str(row["case_label"]),
                "iter": int(row["iter"]),
                "sample_count": int(row["sample_count"]),
                "rmse": float(summary["rmse"]),
                "mae": float(summary["mae"]),
                "max_abs_err": float(summary["max_abs_err"]),
                "numeric_rmse": float(numeric_summary["rmse"]),
                "numeric_mae": float(numeric_summary["mae"]),
                "numeric_max_abs_err": float(numeric_summary["max_abs_err"]),
            }
        )
    return rows


def _empty_metric_row() -> dict[str, float]:
    return {"mse": float("nan"), "mae": float("nan"), "maxae": float("nan")}


def validate_angle_bin_deg(bin_deg: float) -> int:
    bin_deg_float = float(bin_deg)
    if not np.isfinite(bin_deg_float) or bin_deg_float <= 0.0:
        raise ValueError(f"bin_deg must be a positive finite value, got {bin_deg!r}.")
    bin_deg_int = int(round(bin_deg_float))
    if not np.isclose(bin_deg_float, float(bin_deg_int)):
        raise ValueError(f"bin_deg must be an integer number of degrees, got {bin_deg!r}.")
    if 360 % bin_deg_int != 0:
        raise ValueError(f"bin_deg must divide 360 exactly, got {bin_deg_int}.")
    return bin_deg_int


def build_angle_bin_rows(case_rows: list[dict[str, Any]], *, bin_deg: float) -> list[dict[str, Any]]:
    bin_deg_int = validate_angle_bin_deg(bin_deg)
    n_bins = 360 // bin_deg_int
    edges_deg = np.arange(n_bins + 1, dtype=np.float64) * float(bin_deg_int)
    rows: list[dict[str, Any]] = []

    for case_row in case_rows:
        theta = np.asarray(case_row["theta"], dtype=np.float64)
        pred_hkappa = np.asarray(case_row["pred_hkappa"], dtype=np.float64)
        true_hkappa = np.asarray(case_row["true_hkappa"], dtype=np.float64)
        numeric_hkappa = np.asarray(case_row["numeric_hkappa"], dtype=np.float64)
        theta_deg = np.mod(np.degrees(theta), 360.0)
        bin_idx = np.minimum(np.floor(theta_deg / float(bin_deg_int)).astype(np.int64), n_bins - 1)
        for idx in range(n_bins):
            mask = bin_idx == idx
            if int(np.count_nonzero(mask)) > 0:
                numeric_metric = compute_metrics(numeric_hkappa[mask], true_hkappa[mask])
                model_metric = compute_metrics(pred_hkappa[mask], true_hkappa[mask])
            else:
                numeric_metric = _empty_metric_row()
                model_metric = _empty_metric_row()
            rows.append(
                {
                    "rho_model": int(case_row.get("rho_model", -1)),
                    "case_id": int(case_row["case_id"]),
                    "case_label": str(case_row["case_label"]),
                    "iter": int(case_row["iter"]),
                    "case_key": str(case_row["case_key"]),
                    "angle_bin_start_deg": int(edges_deg[idx]),
                    "angle_bin_end_deg": int(edges_deg[idx + 1]),
                    "sample_count": int(np.count_nonzero(mask)),
                    "numeric_mse": float(numeric_metric["mse"]),
                    "numeric_mae": float(numeric_metric["mae"]),
                    "numeric_maxae": float(numeric_metric["maxae"]),
                    "model_mse": float(model_metric["mse"]),
                    "model_mae": float(model_metric["mae"]),
                    "model_maxae": float(model_metric["maxae"]),
                }
            )
    return rows


def _sorted_case_arrays(case_entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(case_entry["theta"], dtype=np.float64).reshape(-1)
    xy = np.asarray(case_entry["xy"], dtype=np.float64)
    analytic = np.asarray(case_entry["true_hkappa"], dtype=np.float64).reshape(-1)
    numeric = np.asarray(case_entry["numeric_hkappa"], dtype=np.float64).reshape(-1)
    prediction = np.asarray(case_entry["pred_hkappa"], dtype=np.float64).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy with shape (N, 2), got {xy.shape}.")
    if not (theta.shape[0] == xy.shape[0] == analytic.shape[0] == numeric.shape[0] == prediction.shape[0]):
        raise ValueError("Case arrays do not have matching lengths.")
    order = np.argsort(theta, kind="mergesort")
    return (
        theta[order],
        xy[order],
        analytic[order],
        numeric[order],
        prediction[order],
    )


def _closed_xy(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return xy
    return np.vstack([xy, xy[:1]])


def render_curvature_overview(
    case_entries: list[dict[str, Any]],
    output_path: str | Path,
    *,
    suptitle: str | None = None,
    layout: Literal["case_columns", "case_rows"] = "case_columns",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not case_entries:
        raise ValueError("At least one case entry is required to render a curvature overview.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if layout == "case_rows":
        nrows = len(case_entries)
        fig, axes = plt.subplots(nrows, 3, figsize=(16.5, 4.5 * nrows),
                                 squeeze=False, constrained_layout=True)
        for row_idx, case_entry in enumerate(case_entries):
            theta, xy, analytic, numeric, prediction = _sorted_case_arrays(case_entry)
            label = str(case_entry.get("title") or case_entry["case_key"])

            ax_shape = axes[row_idx, 0]
            curve_xy = _closed_xy(xy)
            ax_shape.plot(curve_xy[:, 0], curve_xy[:, 1], color="0.80", linewidth=1.0)
            scatter = ax_shape.scatter(xy[:, 0], xy[:, 1], c=analytic, cmap="coolwarm", s=10, linewidths=0.0)
            fig.colorbar(scatter, ax=ax_shape, fraction=0.046, pad=0.04)
            ax_shape.set_aspect("equal")
            ax_shape.set_xticks([]); ax_shape.set_yticks([])
            ax_shape.set_title(f"{label}\nBoundary curvature", fontsize=10)

            ax_curve = axes[row_idx, 1]
            ax_curve.plot(theta, analytic, label="analytic", linewidth=1.6)
            ax_curve.plot(theta, numeric, label="numeric", linewidth=1.4, linestyle="--")
            ax_curve.plot(theta, prediction, label="model", linewidth=1.4)
            ax_curve.set_title(f"{label}\nh*kappa(theta)", fontsize=10)
            ax_curve.set_xlabel("theta"); ax_curve.set_ylabel("h*kappa")
            ax_curve.grid(True, alpha=0.25)
            if row_idx == 0:
                ax_curve.legend(loc="best")

            ax_err = axes[row_idx, 2]
            ax_err.plot(theta, np.abs(numeric - analytic), label="|numeric-analytic|", linewidth=1.4, linestyle="--")
            ax_err.plot(theta, np.abs(prediction - analytic), label="|model-analytic|", linewidth=1.4)
            ax_err.set_title(f"{label}\nAbsolute error", fontsize=10)
            ax_err.set_xlabel("theta"); ax_err.set_ylabel("|error|")
            ax_err.grid(True, alpha=0.25)
            if row_idx == 0:
                ax_err.legend(loc="best")
    else:
        ncols = len(case_entries)
        fig, axes = plt.subplots(3, ncols, figsize=(5.5 * ncols, 11.0), squeeze=False, constrained_layout=True)

        for col_idx, case_entry in enumerate(case_entries):
            theta, xy, analytic, numeric, prediction = _sorted_case_arrays(case_entry)
            label = str(case_entry.get("title") or case_entry["case_key"])

            ax_shape = axes[0, col_idx]
            curve_xy = _closed_xy(xy)
            ax_shape.plot(curve_xy[:, 0], curve_xy[:, 1], color="0.80", linewidth=1.0)
            scatter = ax_shape.scatter(xy[:, 0], xy[:, 1], c=analytic, cmap="coolwarm", s=10, linewidths=0.0)
            fig.colorbar(scatter, ax=ax_shape, fraction=0.046, pad=0.04)
            ax_shape.set_aspect("equal")
            ax_shape.set_xticks([])
            ax_shape.set_yticks([])
            ax_shape.set_title(f"{label}\nBoundary curvature", fontsize=10)

            ax_curve = axes[1, col_idx]
            ax_curve.plot(theta, analytic, label="analytic", linewidth=1.6)
            ax_curve.plot(theta, numeric, label="numeric", linewidth=1.4, linestyle="--")
            ax_curve.plot(theta, prediction, label="model", linewidth=1.4)
            ax_curve.set_title("h*kappa(theta)", fontsize=10)
            ax_curve.set_xlabel("theta")
            ax_curve.set_ylabel("h*kappa")
            ax_curve.grid(True, alpha=0.25)
            if col_idx == 0:
                ax_curve.legend(loc="best")

            ax_err = axes[2, col_idx]
            ax_err.plot(theta, np.abs(numeric - analytic), label="|numeric-analytic|", linewidth=1.4, linestyle="--")
            ax_err.plot(theta, np.abs(prediction - analytic), label="|model-analytic|", linewidth=1.4)
            ax_err.set_title("Absolute error", fontsize=10)
            ax_err.set_xlabel("theta")
            ax_err.set_ylabel("|error|")
            ax_err.grid(True, alpha=0.25)
            if col_idx == 0:
                ax_err.legend(loc="best")

    if suptitle:
        fig.suptitle(str(suptitle), fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
