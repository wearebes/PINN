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
    from model.config import ModelType, create_train_config
    from model.model import create_model
else:
    from model.config import ModelType, create_train_config
    from model.model import create_model


CHECKPOINT_FORMAT_VERSION = 1


def normalization_csv_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_phi9.csv")


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


def _config_to_dict(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass config, got {type(config).__name__}.")
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def save_checkpoint_bundle(model: torch.nn.Module, *, model_type: str, model_config: Any, path: str | Path) -> Path:
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": str(model_type),
        "model_config": _config_to_dict(model_config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
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
    return payload


def load_model_from_checkpoint(model_path: str | Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint_bundle(model_path)
    model_type = str(payload["model_type"])
    model_config = dict(payload["model_config"])
    config = create_train_config(model_type, **model_config)  # type: ignore[arg-type]
    model = create_model(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, {"model_type": model_type, "model_config": model_config}


@torch.inference_mode()
def predict_hkappa_full_batch(
    model: torch.nn.Module,
    phi9: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    phi9 = np.asarray(phi9, dtype=np.float32)
    normalized = ((phi9 - mean.reshape(1, 9)) / std.reshape(1, 9)).astype(np.float32, copy=False)
    batch = torch.from_numpy(normalized).to(device)
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
