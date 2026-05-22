from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.config import ModelType, create_train_config, filter_train_config_overrides
    from model.model import create_model
else:
    from model.config import ModelType, create_train_config, filter_train_config_overrides
    from model.model import create_model


CHECKPOINT_FORMAT_VERSION = 1
MODEL_CONFIG_VERSION = 1


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
        for idx in range(9):
            writer.writerow([idx + 1, float(mean[idx]), float(std[idx]), float(std[idx] ** 2), "train", 9, str(dataset_path)])
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
    if mean.shape != (9,) or std.shape != (9,):
        raise ValueError(
            f"Normalization CSV {csv_path.resolve()} must describe exactly 9 stencil entries, "
            f"got mean shape {mean.shape} and std shape {std.shape}."
        )
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError(f"Normalization CSV {csv_path.resolve()} contains non-finite values.")
    if np.any(std <= 0.0):
        raise ValueError(f"Normalization CSV {csv_path.resolve()} contains non-positive std values.")
    if source_splits != {"train"}:
        raise ValueError(f"Normalization CSV {csv_path.resolve()} must record source_split=train, got {sorted(source_splits)}.")
    if feature_counts != {9}:
        raise ValueError(f"Normalization CSV {csv_path.resolve()} must record feature_count=9, got {sorted(feature_counts)}.")
    return mean, std


def build_legacy_feature_transform(mean: np.ndarray, std: np.ndarray, *, dataset_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "transform_kind": "standardize",
        "feature_version": 1,
        "raw_feature_dim": 9,
        "output_dim": 9,
        "feature_order": "phi9",
        "source_split": "train",
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else "",
        "mean": _as_float32_array(mean),
        "std": _as_float32_array(std),
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
    legacy_pca_enabled = bool(out.get("pca_enabled", False))
    legacy_pca_dim = int(out.get("pca_dim", out["output_dim"]))
    legacy_components = np.asarray(out.get("components", np.zeros((0, out["raw_feature_dim"]), dtype=np.float32)), dtype=np.float32)
    legacy_eigenvalues = np.asarray(out.get("eigenvalues", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if out["mean"].shape != (out["raw_feature_dim"],) or out["std"].shape != (out["raw_feature_dim"],):
        raise ValueError("Feature transform mean/std shapes do not match raw_feature_dim.")
    if np.any(out["std"] <= 0.0):
        raise ValueError("Feature transform std must be strictly positive.")
    if (
        out["transform_kind"] != "standardize"
        or out["feature_version"] != 1
        or out["raw_feature_dim"] != 9
        or out["output_dim"] != 9
        or out["feature_order"] != "phi9"
        or legacy_pca_enabled
        or legacy_pca_dim != 9
        or legacy_components.size != 0
        or legacy_eigenvalues.size != 0
    ):
        raise ValueError(
            "Only the V1 phi9 standardization transform is supported. "
            "PCA, 27D features, and NPZ-based V2 workflows are no longer supported."
        )
    return {
        "transform_kind": "standardize",
        "feature_version": 1,
        "raw_feature_dim": 9,
        "output_dim": 9,
        "feature_order": "phi9",
        "source_split": out["source_split"],
        "dataset_path": out["dataset_path"],
        "mean": out["mean"],
        "std": out["std"],
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
    if raw_dim != 9:
        raise ValueError(
            f"V1 training requires 9D phi9 features. Got raw_feature_dim={raw_dim}; "
            "regenerate the dataset with the V1-only pipeline."
        )
    mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)
    return validate_feature_transform(
        {
            **build_legacy_feature_transform(mean, std, dataset_path=dataset_path),
            "raw_feature_dim": raw_dim,
            "output_dim": raw_dim,
        }
    )

def apply_feature_transform(features: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    state = validate_feature_transform(transform)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != state["raw_feature_dim"]:
        raise ValueError(
            f"Feature matrix must have shape (N, {state['raw_feature_dim']}), got {values.shape}."
        )
    standardized = (values - state["mean"].reshape(1, -1)) / state["std"].reshape(1, -1)
    return standardized.astype(np.float32, copy=False)


def save_feature_stats(path: str | Path, *, transform: dict[str, Any], dataset_path: str | Path) -> Path:
    state = validate_feature_transform({**transform, "dataset_path": str(Path(dataset_path).resolve())})
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Normalization output must be a CSV file for the V1 pipeline, got {output_path.resolve()}."
        )
    return save_phi9_normalization_csv(output_path, mean=state["mean"], std=state["std"], dataset_path=dataset_path)


def load_feature_transform(path: str | Path) -> dict[str, Any]:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature stats not found: {feature_path.resolve()}")
    if feature_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Only CSV normalization sidecars are supported in the V1 pipeline. "
            f"Got {feature_path.resolve()}."
        )
    mean, std = load_phi9_normalization_csv(feature_path)
    return build_legacy_feature_transform(mean, std)


def resolve_feature_transform(
    *,
    model_path: str | Path,
    explicit_path: str | Path | None,
    checkpoint_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if explicit_path:
        path = Path(explicit_path)
        return load_feature_transform(path), str(path.resolve())
    checkpoint_transform = checkpoint_meta.get("feature_transform")
    if checkpoint_transform is not None:
        return validate_feature_transform(checkpoint_transform), "checkpoint"
    csv_candidates = normalization_csv_candidates(model_path)
    for csv_path in csv_candidates:
        if csv_path.exists():
            return load_feature_transform(csv_path), str(csv_path.resolve())
    checked_paths = [str(path.resolve()) for path in csv_candidates]
    raise FileNotFoundError(
        f"No V1 normalization transform found for checkpoint {Path(model_path).resolve()}. "
        f"Checked embedded checkpoint metadata and CSV sidecars: {', '.join(checked_paths)}."
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
        payload["feature_transform"] = validate_feature_transform(feature_transform)
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
    if "feature_transform" in payload and payload["feature_transform"] is not None:
        payload["feature_transform"] = validate_feature_transform(payload["feature_transform"])
    return payload


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


def csv_to_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


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
                "model_vs_numeric": compute_metrics(prediction[mask], numeric[mask]),
            }
        )
    return rows
