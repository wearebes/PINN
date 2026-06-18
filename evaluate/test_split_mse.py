from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    compute_metrics,
    load_model_from_checkpoint,
    predict_hkappa_full_batch,
    resolve_feature_transform,
)
from train_generate.io import load_training_arrays_from_hdf5


def _render_split_diagnostics(
    *,
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = np.asarray(result["predictions"], dtype=np.float64).reshape(-1)
    targets = np.asarray(result["targets"], dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {predictions.shape} vs {targets.shape}."
        )
    if predictions.size == 0:
        raise ValueError("Cannot render split diagnostics for an empty split.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = predictions - targets
    abs_errors = np.abs(residuals)
    metric = result["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    ax_scatter = axes[0]
    ax_scatter.scatter(targets, predictions, s=8, alpha=0.35, linewidths=0.0)
    lo = float(min(np.min(targets), np.min(predictions)))
    hi = float(max(np.max(targets), np.max(predictions)))
    ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color="0.35", linewidth=1.2)
    ax_scatter.set_title("Prediction vs target")
    ax_scatter.set_xlabel("target h*kappa")
    ax_scatter.set_ylabel("predicted h*kappa")
    ax_scatter.grid(True, alpha=0.25)

    ax_hist = axes[1]
    ax_hist.hist(residuals, bins=60, color="tab:blue", alpha=0.85)
    ax_hist.set_title("Residual histogram")
    ax_hist.set_xlabel("prediction - target")
    ax_hist.set_ylabel("count")
    ax_hist.grid(True, alpha=0.25)

    ax_abs = axes[2]
    ax_abs.scatter(targets, abs_errors, s=8, alpha=0.35, linewidths=0.0, color="tab:orange")
    ax_abs.set_title("Absolute error vs target")
    ax_abs.set_xlabel("target h*kappa")
    ax_abs.set_ylabel("|error|")
    ax_abs.grid(True, alpha=0.25)

    fig.suptitle(
        "Split diagnostics"
        f" | split={result['split']}"
        f" | N={result['n']:,}"
        f" | MSE={metric['mse']:.3e}"
        f" | MAE={metric['mae']:.3e}"
        f" | MaxAE={metric['maxae']:.3e}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_split(
    model_path: str | Path,
    data_path: str | Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    data = load_training_arrays_from_hdf5(data_path)
    if split not in data["splits"]:
        available = sorted(data["splits"].keys())
        raise ValueError(f"Split {split!r} not found in dataset. Available: {available}")

    features = data["splits"][split]["features"]
    targets = data["splits"][split]["hkappa_target"]
    n = int(features.shape[0])
    stored_feature_dim = int(features.shape[1])

    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    transform, transform_source = resolve_feature_transform(
        model_path=model_path,
        explicit_path=None,
        checkpoint_meta=checkpoint_meta,
    )

    transform_raw_dim = int(transform["raw_feature_dim"])
    if transform_raw_dim != stored_feature_dim:
        raise ValueError(
            f"Feature dim mismatch: dataset split {split!r} has {stored_feature_dim}D features, "
            f"but the checkpoint's feature transform expects {transform_raw_dim}D input. "
            f"Make sure --data matches the checkpoint version "
            f"(V1/V2: 9D dataset/256/256.h5 or 256_h.h5; V3: 27D dataset/256/256_hgradient.h5)."
        )

    predictions = predict_hkappa_full_batch(model, features, transform=transform, device=device)
    metrics = compute_metrics(predictions, targets)

    return {
        "dataset_path": str(Path(data_path).resolve()),
        "split": split,
        "n": n,
        "stored_feature_dim": stored_feature_dim,
        "model_path": str(Path(model_path).resolve()),
        "model_type": str(checkpoint_meta["model_type"]),
        "transform_kind": str(transform["transform_kind"]),
        "transform_source": transform_source,
        "raw_feature_dim": transform_raw_dim,
        "output_dim": int(transform["output_dim"]),
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MSE/MAE/MaxAE on a training dataset split.")
    parser.add_argument("--model", default="out/best_stencil_hkappa.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--data", default="dataset/256/256.h5", help="Path to training HDF5 dataset")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test). Note: val was used for early stopping/model selection.",
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Default: cuda if available.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("out") / "curvature_viz" / "split"),
        help="Directory for the split diagnostic plot.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Filename stem for the diagnostic plot. Defaults to <modelstem>_<datasetstem>_<split>.",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))

    result = evaluate_split(
        model_path=args.model,
        data_path=args.data,
        split=args.split,
        device=device,
    )

    print(f"dataset         : {result['dataset_path']}")
    print(f"split           : {result['split']}  (val = model-selection reference only; test = held-out)")
    print(f"N               : {result['n']:,}")
    print(f"stored feat dim : {result['stored_feature_dim']}")
    print(f"model           : {result['model_path']}")
    print(f"model type      : {result['model_type']}")
    print(f"transform kind  : {result['transform_kind']}")
    print(f"transform source: {result['transform_source']}")
    print(f"raw feat dim    : {result['raw_feature_dim']}")
    print(f"output dim      : {result['output_dim']}")
    print()
    m = result["metrics"]
    print(f"MSE   : {m['mse']:.6e}")
    print(f"MAE   : {m['mae']:.6e}")
    print(f"MaxAE : {m['maxae']:.6e}")

    default_name = (
        f"{Path(result['model_path']).stem}_{Path(result['dataset_path']).stem}_{result['split']}"
    )
    name = args.name or default_name
    plot_path = _render_split_diagnostics(
        result=result,
        output_path=output_dir / f"{name}.png",
    )
    print(f"Plot  : {plot_path}")


if __name__ == "__main__":
    main()
