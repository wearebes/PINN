from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate.shared import (
        central_difference_hkappa_from_phi9,
        compute_metrics,
        group_metric_rows,
        csv_to_list,
        init_swanlab_run,
        load_model_from_checkpoint,
        load_phi9_normalization_csv,
        normalization_csv_path,
        predict_hkappa_full_batch,
    )
    from model.config import default_output_model_path
else:
    from .shared import (
        central_difference_hkappa_from_phi9,
        compute_metrics,
        group_metric_rows,
        csv_to_list,
        init_swanlab_run,
        load_model_from_checkpoint,
        load_phi9_normalization_csv,
        normalization_csv_path,
        predict_hkappa_full_batch,
    )
    from model.config import default_output_model_path


REQUIRED_FIELDS = ("phi9", "hkappa_target", "case_id", "iter", "rho_model", "h")
PRIMARY_COMPARISONS = ("numeric_vs_analytic", "model_vs_analytic")
AUXILIARY_COMPARISONS = ("model_vs_numeric",)


def load_flower_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Flower test dataset not found: {dataset_path.resolve()}. "
            "Generate a resolution-matched dataset with `python -m testdata_generate.generate --rho-model <rho_model>` "
            "and pass it explicitly via --data."
        )
    with h5py.File(dataset_path, "r") as handle:
        missing = [name for name in REQUIRED_FIELDS if name not in handle]
        if missing:
            raise ValueError(f"Flower test dataset {dataset_path.resolve()} is missing fields: {missing}.")
        arrays = {name: np.asarray(handle[name][:]) for name in handle.keys()}
        attrs = {key: handle.attrs[key] for key in handle.attrs.keys()}
    phi9 = np.asarray(arrays["phi9"], dtype=np.float32)
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"Flower dataset phi9 must have shape (N, 9), got {phi9.shape}.")
    n = int(phi9.shape[0])
    if n == 0:
        raise ValueError(f"Flower dataset {dataset_path.resolve()} is empty.")
    for name in REQUIRED_FIELDS[1:]:
        if int(np.asarray(arrays[name]).shape[0]) != n:
            raise ValueError(f"Field {name!r} first dimension does not match phi9 length {n}.")
    case_label_map: dict[int, str] = {}
    if "scenarios_json" in attrs:
        raw = attrs["scenarios_json"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        for idx, scenario in enumerate(json.loads(str(raw))):
            case_label_map[idx] = str(scenario.get("exp_id", idx))
    return {
        "dataset_path": str(dataset_path.resolve()),
        "arrays": arrays,
        "attrs": attrs,
        "case_label_map": case_label_map,
        "rho_models": tuple(sorted(int(item) for item in np.unique(np.asarray(arrays["rho_model"]).reshape(-1)))),
    }


def evaluate_flower(*, dataset_path: str | Path, model_path: str | Path, normalization_path: str | Path, device: torch.device) -> dict[str, Any]:
    bundle = load_flower_dataset(dataset_path)
    arrays = bundle["arrays"]
    phi9 = np.asarray(arrays["phi9"], dtype=np.float32)
    hkappa_target = np.asarray(arrays["hkappa_target"], dtype=np.float64).reshape(-1)
    mean, std = load_phi9_normalization_csv(normalization_path)
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    numeric = central_difference_hkappa_from_phi9(phi9)
    prediction = predict_hkappa_full_batch(model, phi9, mean=mean, std=std, device=device)
    return {
        "dataset_path": bundle["dataset_path"],
        "sample_count": int(phi9.shape[0]),
        "rho_models": bundle["rho_models"],
        "model_path": str(Path(model_path).resolve()),
        "normalization_path": str(Path(normalization_path).resolve()),
        "model_type": checkpoint_meta["model_type"],
        "numeric_vs_analytic": compute_metrics(numeric, hkappa_target),
        "model_vs_analytic": compute_metrics(prediction, hkappa_target),
        "model_vs_numeric": compute_metrics(prediction, numeric),
        "by_iter": group_metric_rows(
            labels=np.asarray(arrays["iter"]),
            prediction=prediction,
            target=hkappa_target,
            numeric=numeric,
            label_name="iter",
        ),
        "by_case_id": group_metric_rows(
            labels=np.asarray(arrays["case_id"]),
            prediction=prediction,
            target=hkappa_target,
            numeric=numeric,
            label_name="case_id",
            label_map=bundle["case_label_map"],
        ),
        "by_rho_model": group_metric_rows(
            labels=np.asarray(arrays["rho_model"]),
            prediction=prediction,
            target=hkappa_target,
            numeric=numeric,
            label_name="rho_model",
        ),
    }


def _format_metric(metric: dict[str, float], *, include_maxae: bool) -> str:
    summary = f"MSE={metric['mse']:.6e} | MAE={metric['mae']:.6e}"
    if include_maxae:
        summary += f" | MaxAE={metric['maxae']:.6e}"
    return summary


def _print_grouped_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    for row in rows:
        key_name = next(key for key in row.keys() if key in {"iter", "case_id", "rho_model"})
        print(f"  {key_name}={row['display']} samples={row['sample_count']}")
        for metric_name in PRIMARY_COMPARISONS:
            metric = row[metric_name]
            print(f"    {metric_name}: {_format_metric(metric, include_maxae=True)}")
        for metric_name in AUXILIARY_COMPARISONS:
            metric = row[metric_name]
            print(f"    {metric_name}: {_format_metric(metric, include_maxae=False)}")


def build_arg_parser() -> argparse.ArgumentParser:
    default_model = default_output_model_path()
    parser = argparse.ArgumentParser(
        description="Evaluate a resolution-specific flower test_data HDF5 with the trained 3x3 phi stencil -> h*kappa model."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to a flower HDF5 generated for the target model's rho_model. Pass it explicitly; the evaluator does not infer it.",
    )
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--normalization-csv", type=str, default=str(normalization_csv_path(default_model)))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="PINN")
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-description", type=str, default="")
    parser.add_argument("--swanlab-tags", type=str, default="")
    parser.add_argument("--swanlab-group", type=str, default="")
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument("--swanlab-mode", type=str, choices=("cloud", "local", "offline", "disabled"), default="cloud")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_flower(
        dataset_path=args.data,
        model_path=args.model_path,
        normalization_path=args.normalization_csv,
        device=device,
    )
    print("Task: flower test_data evaluation for 3x3 phi stencil -> h*kappa")
    print(f"Dataset file: {result['dataset_path']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Dataset rho_model values: {', '.join(str(item) for item in result['rho_models'])}")
    print(f"Checkpoint: {result['model_path']}")
    print(f"Normalization CSV: {result['normalization_path']}")
    print(f"Model type: {result['model_type']}")
    print(f"Device: {device}")
    print("Primary comparisons against analytic h*kappa:")
    for metric_name in PRIMARY_COMPARISONS:
        metric = result[metric_name]
        print(f"{metric_name}: {_format_metric(metric, include_maxae=True)}")
    print("Auxiliary agreement check:")
    for metric_name in AUXILIARY_COMPARISONS:
        metric = result[metric_name]
        print(f"{metric_name}: {_format_metric(metric, include_maxae=False)}")
    _print_grouped_rows("By iter:", result["by_iter"])
    _print_grouped_rows("By case_id:", result["by_case_id"])
    _print_grouped_rows("By rho_model:", result["by_rho_model"])
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
                "task": "flower evaluation",
                "dataset_path": result["dataset_path"],
                "rho_models": list(result["rho_models"]),
                "model_path": result["model_path"],
                "normalization_path": result["normalization_path"],
                "model_type": result["model_type"],
                "sample_count": result["sample_count"],
                "device": str(device),
            },
        )
        run.log({
            "eval/numeric_vs_analytic_mse": result["numeric_vs_analytic"]["mse"],
            "eval/numeric_vs_analytic_mae": result["numeric_vs_analytic"]["mae"],
            "eval/numeric_vs_analytic_maxae": result["numeric_vs_analytic"]["maxae"],
            "eval/model_vs_analytic_mse": result["model_vs_analytic"]["mse"],
            "eval/model_vs_analytic_mae": result["model_vs_analytic"]["mae"],
            "eval/model_vs_analytic_maxae": result["model_vs_analytic"]["maxae"],
            "eval/model_vs_numeric_mse": result["model_vs_numeric"]["mse"],
            "eval/model_vs_numeric_mae": result["model_vs_numeric"]["mae"],
        })
        run.finish()


if __name__ == "__main__":
    main()
