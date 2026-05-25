from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    add_swanlab_args,
    central_difference_hkappa_from_phi9,
    compute_metrics,
    csv_to_list,
    init_swanlab_run,
    load_model_from_checkpoint,
    predict_hkappa_full_batch,
    resolve_feature_transform,
)
from model.config import default_dataset_path, default_output_model_path
from train_generate.io import load_training_arrays_from_hdf5


SPLIT_NAMES = ("train", "val", "test")


def evaluate_split(*, dataset_path: str | Path, split_name: str, model_path: str | Path, normalization_csv_path: str | Path | None, device: torch.device) -> dict[str, object]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split={split_name!r}; expected one of {SPLIT_NAMES}.")
    bundle = load_training_arrays_from_hdf5(dataset_path)
    if split_name not in bundle["splits"]:
        raise ValueError(f"Split {split_name!r} is not present in dataset {Path(dataset_path).resolve()}.")
    split = bundle["splits"][split_name]
    phi9 = np.asarray(split["phi9"], dtype=np.float32)
    features = np.asarray(split["features"], dtype=np.float32)
    hkappa_target = np.asarray(split["hkappa_target"], dtype=np.float64).reshape(-1)
    raw_feature_dim = int(bundle["raw_feature_dim"])
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"Split {split_name!r} phi9 must have shape (N, 9), got {phi9.shape}.")
    if features.ndim != 2 or features.shape[1] != raw_feature_dim:
        raise ValueError(f"Split {split_name!r} features must have shape (N, {raw_feature_dim}), got {features.shape}.")
    if phi9.shape[0] == 0:
        raise ValueError(f"Split {split_name!r} is empty in dataset {Path(dataset_path).resolve()}.")
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    feature_transform, normalization_source = resolve_feature_transform(
        model_path=model_path,
        explicit_path=normalization_csv_path,
        checkpoint_meta=checkpoint_meta,
    )
    numeric = central_difference_hkappa_from_phi9(phi9)
    prediction = predict_hkappa_full_batch(model, features, transform=feature_transform, device=device)
    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "split": split_name,
        "sample_count": int(phi9.shape[0]),
        "model_path": str(Path(model_path).resolve()),
        "normalization_source": normalization_source,
        "model_type": checkpoint_meta["model_type"],
        "feature_version": int(feature_transform["feature_version"]),
        "raw_feature_dim": int(feature_transform["raw_feature_dim"]),
        "model_input_dim": int(feature_transform["output_dim"]),
        "numeric_vs_analytic": compute_metrics(numeric, hkappa_target),
        "model_vs_analytic": compute_metrics(prediction, hkappa_target),
        "model_vs_numeric": compute_metrics(prediction, numeric),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    default_model = default_output_model_path()
    parser = argparse.ArgumentParser(description="Evaluate train/val/test splits for the 3x3 stencil feature -> h*kappa task.")
    parser.add_argument("--data", type=str, default=str(default_dataset_path()))
    parser.add_argument("--split", type=str, choices=SPLIT_NAMES, default="test")
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--normalization-csv", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    add_swanlab_args(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_split(
        dataset_path=args.data,
        split_name=args.split,
        model_path=args.model_path,
        normalization_csv_path=args.normalization_csv or None,
        device=device,
    )
    print("Task: 3x3 stencil features -> h*kappa")
    print(f"Dataset file: {result['dataset_path']}")
    print(f"Split: {result['split']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Checkpoint: {result['model_path']}")
    print(f"Normalization source: {result['normalization_source']}")
    print(f"Model type: {result['model_type']}")
    print(f"Feature version: {result['feature_version']}")
    print(f"Raw feature dim: {result['raw_feature_dim']}")
    print(f"Model input dim: {result['model_input_dim']}")
    print(f"Device: {device}")
    for metric_name in ("numeric_vs_analytic", "model_vs_analytic", "model_vs_numeric"):
        metric = result[metric_name]
        print(f"{metric_name}: MSE={metric['mse']:.6e} | MAE={metric['mae']:.6e}")
    if args.use_swanlab:
        run = init_swanlab_run(
            project=args.swanlab_project or None,
            experiment_name=args.swanlab_experiment_name or None,
            description=args.swanlab_description or None,
            tags=csv_to_list(args.swanlab_tags),
            group=args.swanlab_group or None,
            workspace=args.swanlab_workspace or None,
            logdir=args.swanlab_logdir or None,
            mode=args.swanlab_mode or None,
            config={
                "task": "split evaluation",
                "split": result["split"],
                "dataset_path": result["dataset_path"],
                "model_path": result["model_path"],
                "normalization_source": result["normalization_source"],
                "model_type": result["model_type"],
                "sample_count": result["sample_count"],
                "device": str(device),
            },
        )
        run.log({
            "eval/numeric_vs_analytic_mse": result["numeric_vs_analytic"]["mse"],
            "eval/numeric_vs_analytic_mae": result["numeric_vs_analytic"]["mae"],
            "eval/model_vs_analytic_mse": result["model_vs_analytic"]["mse"],
            "eval/model_vs_analytic_mae": result["model_vs_analytic"]["mae"],
            "eval/model_vs_numeric_mse": result["model_vs_numeric"]["mse"],
            "eval/model_vs_numeric_mae": result["model_vs_numeric"]["mae"],
        })
        run.finish()


if __name__ == "__main__":
    main()
