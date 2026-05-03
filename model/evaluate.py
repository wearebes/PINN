from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.config import default_dataset_path, default_output_model_path
    from model.model import HKappaStencilNet
    from model.train import normalization_csv_path
    from traingenerate.io import load_training_arrays_from_hdf5
else:
    from .config import default_dataset_path, default_output_model_path
    from .model import HKappaStencilNet
    from .train import normalization_csv_path
    from traingenerate.io import load_training_arrays_from_hdf5


SPLIT_NAMES = ("train", "val", "test")


def _available_hdf5_files(root: str | Path = "dataset") -> list[Path]:
    dataset_root = Path(root)
    if not dataset_root.exists():
        return []
    return sorted(path.resolve() for path in dataset_root.glob("*.h5"))


def load_phi9_normalization_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Normalization CSV not found: {csv_path.resolve()}")

    means: list[float] = []
    stds: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"phi_index", "mean", "std", "variance"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"Normalization CSV {csv_path.resolve()} must have columns {sorted(required)}, "
                f"got {reader.fieldnames!r}."
            )
        for row_idx, row in enumerate(reader, start=1):
            phi_index = int(row["phi_index"])
            if phi_index != row_idx:
                raise ValueError(
                    f"Normalization CSV {csv_path.resolve()} has non-sequential phi_index at row {row_idx}: "
                    f"expected {row_idx}, got {phi_index}."
                )
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))

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

    # These scaled differences produce h*kappa directly, so no explicit grid spacing is needed.
    hkappa = (phi_xx * phi_y**2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / denom
    if np.any(~np.isfinite(hkappa)):
        bad = int(np.count_nonzero(~np.isfinite(hkappa)))
        raise ValueError(f"Central-difference baseline produced {bad} non-finite h*kappa values.")
    return hkappa.astype(np.float32, copy=False)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")
    diff = prediction - target
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
    }


def infer_hidden_units_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("net.0.weight")
    if weight is None or weight.ndim != 2 or int(weight.shape[1]) != 9:
        raise ValueError("Checkpoint is missing net.0.weight with expected shape [hidden_units, 9].")
    return int(weight.shape[0])


def load_model(
    model_path: str | Path,
    *,
    device: torch.device,
) -> HKappaStencilNet:
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path.resolve()}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path.resolve()} does not contain a valid state_dict.")
    hidden_units = infer_hidden_units_from_state_dict(state_dict)
    model = HKappaStencilNet(hidden_units=hidden_units)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_hkappa(
    model: HKappaStencilNet,
    phi9: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    normalized = ((np.asarray(phi9, dtype=np.float32) - mean.reshape(1, 9)) / std.reshape(1, 9)).astype(
        np.float32,
        copy=False,
    )

    outputs: list[np.ndarray] = []
    for start in range(0, normalized.shape[0], batch_size):
        end = min(start + batch_size, normalized.shape[0])
        batch = torch.from_numpy(normalized[start:end]).to(device)
        pred = model(batch).detach().cpu().numpy()
        outputs.append(pred.astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0).reshape(-1)


def evaluate_split(
    *,
    dataset_path: str | Path,
    model_path: str | Path,
    normalization_path: str | Path,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split={split_name!r}; expected one of {SPLIT_NAMES}.")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        available = _available_hdf5_files(dataset_path.parent)
        available_text = ", ".join(str(path) for path in available) if available else "none"
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path.resolve()}\n"
            f"Available HDF5 files under {dataset_path.parent.resolve()}: {available_text}\n"
            "Pass --dataset-output explicitly or update model/config.py default_dataset_path()."
        )

    bundle = load_training_arrays_from_hdf5(dataset_path)
    split = bundle["splits"][split_name]
    phi9 = np.asarray(split["phi9"], dtype=np.float32)
    hkappa_target = np.asarray(split["hkappa_target"], dtype=np.float32).reshape(-1)
    if phi9.shape[0] == 0:
        raise ValueError(f"Split {split_name!r} is empty in dataset {Path(dataset_path).resolve()}.")

    mean, std = load_phi9_normalization_csv(normalization_path)
    model = load_model(model_path, device=device)
    numeric_hkappa = central_difference_hkappa_from_phi9(phi9)
    model_hkappa = predict_hkappa(model, phi9, mean=mean, std=std, device=device, batch_size=batch_size)

    return {
        "split": split_name,
        "sample_count": int(phi9.shape[0]),
        "dataset_path": str(Path(dataset_path).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "normalization_path": str(Path(normalization_path).resolve()),
        "numeric_vs_analytic": compute_metrics(numeric_hkappa, hkappa_target),
        "model_vs_analytic": compute_metrics(model_hkappa, hkappa_target),
        "model_vs_numeric": compute_metrics(model_hkappa, numeric_hkappa),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    default_model = default_output_model_path()
    parser = argparse.ArgumentParser(description="Evaluate held-out h*kappa predictions on a dataset split.")
    parser.add_argument("--dataset-output", type=str, default=str(default_dataset_path()))
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--normalization-csv", type=str, default=str(normalization_csv_path(default_model)))
    parser.add_argument("--split", type=str, choices=SPLIT_NAMES, default="test")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_split(
        dataset_path=args.dataset_output,
        model_path=args.model_path,
        normalization_path=args.normalization_csv,
        split_name=args.split,
        device=device,
        batch_size=args.batch_size,
    )

    print("Task: 3x3 phi stencil -> h*kappa")
    print(f"Dataset file: {result['dataset_path']}")
    print(f"Split: {result['split']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Checkpoint: {result['model_path']}")
    print(f"Normalization CSV: {result['normalization_path']}")
    print(f"Device: {device}")
    for metric_name in ("numeric_vs_analytic", "model_vs_analytic", "model_vs_numeric"):
        metric = result[metric_name]
        print(f"{metric_name}: MSE={metric['mse']:.6e} | MAE={metric['mae']:.6e}")


if __name__ == "__main__":
    main()
