from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    ANGLE_BIN_FIELDNAMES,
    add_swanlab_args,
    build_angle_bin_rows,
    build_case_summary_rows,
    build_swanlab_image as _build_swanlab_image,
    build_swanlab_table_payload,
    central_difference_hkappa_from_phi9,
    compute_metrics,
    csv_to_list,
    ensure_output_dir,
    group_metric_rows,
    init_swanlab_run,
    load_model_from_checkpoint,
    metric_summary as _metric_summary,
    predict_hkappa_full_batch,
    render_curvature_overview,
    resolve_feature_transform,
    sanitize_name as _sanitize_name,
    validate_angle_bin_deg,
)
from model.config import default_output_model_path
from testdata_generate.generate import find_projection_theta


REQUIRED_FIELDS = ("phi9", "xy", "hkappa_target", "case_id", "iter", "rho_model", "h")
PRIMARY_COMPARISONS = ("numeric_vs_analytic", "model_vs_analytic")


def _decode_json_attr(attrs: dict[str, Any], key: str) -> Any | None:
    raw = attrs.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _case_sort_key(case_entry: dict[str, Any]) -> tuple[str, int, int]:
    return (str(case_entry["case_label"]), int(case_entry["iter"]), int(case_entry["case_id"]))


def _select_overview_cases(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        family = str(row["case_label"]).rsplit("_", 1)[0]
        by_family.setdefault(family, []).append(row)
    result = []
    for family in ("smooth", "acute"):
        if family in by_family:
            result.append(max(by_family[family], key=lambda r: int(r["iter"])))
    return result


def _build_hk_metric_payload(prefix: str, metric: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}/MSE_hk": float(metric["mse"]),
        f"{prefix}/MAE_hk": float(metric["mae"]),
        f"{prefix}/MaxAE_hk": float(metric["maxae"]),
    }


def _build_case_summary_table_payload(
    swanlab_module: Any,
    case_rows: list[dict[str, Any]],
    failed_case_slices: list[dict[str, Any]],
) -> Any:
    rows: list[dict[str, Any]] = build_case_summary_rows(case_rows)
    if failed_case_slices:
        rows.append({
            "rank": "-",
            "rho_model": "-",
            "case_key": "FAILED_CASE_SLICES",
            "case_id": "-",
            "case_label": "-",
            "iter": "-",
            "sample_count": int(len(failed_case_slices)),
            "rmse": "-",
            "mae": "-",
            "max_abs_err": "-",
            "numeric_rmse": "-",
            "numeric_mae": "-",
            "numeric_max_abs_err": "-",
        })
        for failed in failed_case_slices:
            rows.append({
                "rank": "-",
                "rho_model": "-",
                "case_key": str(failed["case_key"]),
                "case_id": int(failed["case_id"]),
                "case_label": str(failed["case_label"]),
                "iter": int(failed["iter"]),
                "sample_count": 0,
                "rmse": str(failed["reason"]),
                "mae": "",
                "max_abs_err": "",
                "numeric_rmse": "",
                "numeric_mae": "",
                "numeric_max_abs_err": "",
            })
    return build_swanlab_table_payload(swanlab_module, rows)


def _log_swanlab_step_series(run: Any, result: dict[str, Any]) -> None:
    payloads_by_step: dict[int, dict[str, float | int]] = {}
    for case_row in sorted(result["cases"], key=_case_sort_key):
        step = int(case_row["iter"])
        case_key = _sanitize_name(case_row["case_label"])
        payload = payloads_by_step.setdefault(step, {})
        payload.update(
            _build_hk_metric_payload(
                f"flower_eval/numeric/case_{case_key}",
                case_row["numeric_vs_analytic"],
            )
        )
        payload.update(
            _build_hk_metric_payload(
                f"flower_eval/model/case_{case_key}",
                case_row["model_vs_analytic"],
            )
        )
        payload[f"flower_eval/case_{case_key}/N_samples"] = int(case_row["sample_count"])

    for step in sorted(payloads_by_step):
        run.log(payloads_by_step[step], step=int(step))


def _build_case_curve_row(
    *,
    case_id: int,
    iteration: int,
    rho_model: int,
    case_label: str,
    scenario: dict[str, Any],
    xy: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    numeric: np.ndarray,
) -> dict[str, Any]:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy with shape (N, 2), got {xy.shape}.")
    if prediction.shape[0] == 0:
        raise ValueError("Case slice is empty.")
    theta = find_projection_theta(
        np.asarray(xy, dtype=np.float64),
        float(scenario["a"]),
        float(scenario["b"]),
        int(scenario["p"]),
    )
    if theta.shape[0] != prediction.shape[0]:
        raise ValueError("theta length does not match prediction length.")
    order = np.argsort(theta, kind="mergesort")
    theta_sorted = np.asarray(theta[order], dtype=np.float64)
    pred_sorted = np.asarray(prediction[order], dtype=np.float64)
    target_sorted = np.asarray(target[order], dtype=np.float64)
    numeric_sorted = np.asarray(numeric[order], dtype=np.float64)
    abs_err = np.abs(pred_sorted - target_sorted)
    model_vs_analytic = compute_metrics(pred_sorted, target_sorted)
    numeric_vs_analytic = compute_metrics(numeric_sorted, target_sorted)
    return {
        "rho_model": int(rho_model),
        "case_id": int(case_id),
        "case_label": str(case_label),
        "iter": int(iteration),
        "case_key": f"{case_label}/iter_{int(iteration)}",
        "sample_count": int(theta_sorted.shape[0]),
        "theta": theta_sorted,
        "xy": np.asarray(xy[order], dtype=np.float64),
        "pred_hkappa": pred_sorted,
        "true_hkappa": target_sorted,
        "numeric_hkappa": numeric_sorted,
        "abs_err": abs_err,
        "summary": _metric_summary(model_vs_analytic),
        "numeric_summary": _metric_summary(numeric_vs_analytic),
        "model_vs_analytic": model_vs_analytic,
        "numeric_vs_analytic": numeric_vs_analytic,
    }


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
    features = np.asarray(arrays.get("features", arrays["phi9"]), dtype=np.float32)
    feature_version = int(attrs.get("feature_version", 1))
    raw_feature_dim = int(attrs.get("feature_dim_raw", features.shape[1] if features.ndim == 2 else 9))
    feature_order = str(attrs.get("feature_order", "phi9"))
    _v1_ok = feature_version == 1 and raw_feature_dim == 9 and feature_order == "phi9"
    _v2_ok = feature_version == 2 and raw_feature_dim == 27 and feature_order == "phi9+nx9+ny9"
    if not (_v1_ok or _v2_ok):
        raise ValueError(
            f"Flower dataset {dataset_path.resolve()} has unsupported feature metadata. "
            f"Got feature_version={feature_version}, feature_dim_raw={raw_feature_dim}, feature_order={feature_order!r}. "
            "Expected V1 (feature_version=1, feature_dim_raw=9, feature_order='phi9') or "
            "V2 (feature_version=2, feature_dim_raw=27, feature_order='phi9+nx9+ny9')."
        )
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"Flower dataset phi9 must have shape (N, 9), got {phi9.shape}.")
    if features.ndim != 2 or features.shape[1] != raw_feature_dim:
        raise ValueError(f"Flower dataset features must have shape (N, {raw_feature_dim}), got {features.shape}.")
    n = int(phi9.shape[0])
    if n == 0:
        raise ValueError(f"Flower dataset {dataset_path.resolve()} is empty.")
    for name in REQUIRED_FIELDS[1:]:
        if int(np.asarray(arrays[name]).shape[0]) != n:
            raise ValueError(f"Field {name!r} first dimension does not match phi9 length {n}.")
    case_label_map: dict[int, str] = {}
    scenario_map: dict[int, dict[str, Any]] = {}
    scenarios = _decode_json_attr(attrs, "scenarios_json") or []
    for idx, scenario in enumerate(scenarios):
        scenario_map[idx] = dict(scenario)
        case_label_map[idx] = str(scenario.get("exp_id", idx))
    arrays["features"] = features
    return {
        "dataset_path": str(dataset_path.resolve()),
        "arrays": arrays,
        "attrs": attrs,
        "case_label_map": case_label_map,
        "scenario_map": scenario_map,
        "rho_models": tuple(sorted(int(item) for item in np.unique(np.asarray(arrays["rho_model"]).reshape(-1)))),
    }


def evaluate_flower(
    *,
    dataset_path: str | Path,
    model_path: str | Path,
    normalization_csv_path: str | Path | None,
    device: torch.device,
    angle_bin_deg: float = 30.0,
) -> dict[str, Any]:
    bin_deg = validate_angle_bin_deg(angle_bin_deg)
    bundle = load_flower_dataset(dataset_path)
    arrays = bundle["arrays"]
    phi9 = np.asarray(arrays["phi9"], dtype=np.float32)
    features = np.asarray(arrays["features"], dtype=np.float32)
    xy = np.asarray(arrays["xy"], dtype=np.float64)
    hkappa_target = np.asarray(arrays["hkappa_target"], dtype=np.float64).reshape(-1)
    case_ids = np.asarray(arrays["case_id"]).reshape(-1)
    iterations = np.asarray(arrays["iter"]).reshape(-1)
    rho_models = np.asarray(arrays["rho_model"]).reshape(-1)
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    feature_transform, normalization_source = resolve_feature_transform(
        model_path=model_path,
        explicit_path=normalization_csv_path,
        checkpoint_meta=checkpoint_meta,
    )
    numeric = central_difference_hkappa_from_phi9(phi9)
    prediction = predict_hkappa_full_batch(model, features, transform=feature_transform, device=device)

    case_rows: list[dict[str, Any]] = []
    failed_case_slices: list[dict[str, Any]] = []
    for case_id in sorted(int(item) for item in np.unique(case_ids)):
        case_mask = case_ids == case_id
        case_label = bundle["case_label_map"].get(case_id, str(case_id))
        scenario = bundle["scenario_map"].get(case_id)
        if scenario is None:
            failed_case_slices.append(
                {
                    "case_id": int(case_id),
                    "case_label": str(case_label),
                    "iter": -1,
                    "case_key": f"{case_label}/iter_unknown",
                    "reason": "scenario metadata missing from dataset attrs.scenarios_json",
                }
            )
            continue
        for iteration in sorted(int(item) for item in np.unique(iterations[case_mask])):
            mask = case_mask & (iterations == iteration)
            try:
                case_rows.append(
                    _build_case_curve_row(
                        case_id=case_id,
                        iteration=iteration,
                        rho_model=int(np.unique(rho_models[mask])[0]),
                        case_label=case_label,
                        scenario=scenario,
                        xy=xy[mask],
                        prediction=prediction[mask],
                        target=hkappa_target[mask],
                        numeric=numeric[mask],
                    )
                )
            except Exception as exc:
                failed_case_slices.append(
                    {
                        "case_id": int(case_id),
                        "case_label": str(case_label),
                        "iter": int(iteration),
                        "case_key": f"{case_label}/iter_{int(iteration)}",
                        "reason": str(exc),
                    }
                )
    if not case_rows:
        raise ValueError("Flower evaluation did not produce any valid case slices.")

    numeric_vs_analytic = compute_metrics(numeric, hkappa_target)
    model_vs_analytic = compute_metrics(prediction, hkappa_target)
    angle_bin_rows = build_angle_bin_rows(case_rows, bin_deg=bin_deg)

    return {
        "dataset_path": bundle["dataset_path"],
        "sample_count": int(phi9.shape[0]),
        "rho_models": bundle["rho_models"],
        "model_path": str(Path(model_path).resolve()),
        "normalization_source": normalization_source,
        "model_type": checkpoint_meta["model_type"],
        "state_dict_compatibility": str(checkpoint_meta.get("state_dict_compatibility", "native")),
        "feature_version": int(feature_transform["feature_version"]),
        "raw_feature_dim": int(feature_transform["raw_feature_dim"]),
        "model_input_dim": int(feature_transform["output_dim"]),
        "angle_bin_deg": int(bin_deg),
        "summary": _metric_summary(model_vs_analytic),
        "numeric_summary": _metric_summary(numeric_vs_analytic),
        "numeric_vs_analytic": numeric_vs_analytic,
        "model_vs_analytic": model_vs_analytic,
        "cases": case_rows,
        "angle_bin_rows": angle_bin_rows,
        "failed_case_slices": failed_case_slices,
        "failed_case_count": int(len(failed_case_slices)),
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


def _print_case_iter_pivot(
    case_rows: list[dict[str, Any]],
    *,
    sources: tuple[str, ...] = ("numeric_vs_analytic", "model_vs_analytic"),
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["case_label"])].append(row)

    print("\nBy case_id (numerical / model across iters):")
    for case_label in sorted(grouped):
        rows = sorted(grouped[case_label], key=lambda r: int(r["iter"]))
        total_samples = sum(int(r["sample_count"]) for r in rows)
        print(f"\n{case_label} ({len(rows)} iters, total samples={total_samples}):")
        for src in sources:
            print(f"  {src}")
            for r in rows:
                m = r[src]
                print(
                    f"    iter={int(r['iter'])}  "
                    f"MSE={float(m['mse']):.6e}  "
                    f"MAE={float(m['mae']):.6e}  "
                    f"MaxAE={float(m['maxae']):.6e}"
                )


def _print_case_angle_bins(
    angle_bin_rows: list[dict[str, Any]],
    *,
    bin_deg: int = 30,
) -> None:
    print(f"\nBy case slice x angle (bin width = {bin_deg} deg):")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in angle_bin_rows:
        grouped[str(row["case_key"])].append(row)
    for case_key in sorted(grouped):
        rows = sorted(grouped[case_key], key=lambda row: int(row["angle_bin_start_deg"]))
        total_samples = sum(int(row["sample_count"]) for row in rows)
        print(f"\n{case_key} (total samples={total_samples}):")
        header = (
            f"  {'angle bin':<14}{'N':>7}    "
            f"{'numeric_vs_analytic (MSE / MAE / MaxAE)':<46}    "
            f"{'model_vs_analytic (MSE / MAE / MaxAE)'}"
        )
        print(header)
        for row in rows:
            lo = int(row["angle_bin_start_deg"])
            hi = int(row["angle_bin_end_deg"])
            bin_label = f"[{lo:>3d},{hi:>4d})"
            if int(row["sample_count"]) == 0:
                print(f"  {bin_label:<14}{0:>7}    {'(empty)':<46}    (empty)")
                continue
            numeric_cell = (
                f"MSE={float(row['numeric_mse']):.3e} "
                f"MAE={float(row['numeric_mae']):.3e} "
                f"MaxAE={float(row['numeric_maxae']):.3e}"
            )
            model_cell = (
                f"MSE={float(row['model_mse']):.3e} "
                f"MAE={float(row['model_mae']):.3e} "
                f"MaxAE={float(row['model_maxae']):.3e}"
            )
            print(f"  {bin_label:<14}{int(row['sample_count']):>7}    {numeric_cell:<46}    {model_cell}")


def build_arg_parser() -> argparse.ArgumentParser:
    default_model = default_output_model_path()
    parser = argparse.ArgumentParser(
        description="Evaluate a resolution-specific flower test_data HDF5 with the trained 3x3 stencil feature -> h*kappa model."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to a flower HDF5 generated for the target model's rho_model. Pass it explicitly; the evaluator does not infer it.",
    )
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--normalization-csv", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--angle-bin-deg", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("out") / "curvature_viz" / "flower"),
    )
    add_swanlab_args(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_flower(
        dataset_path=args.data,
        model_path=args.model_path,
        normalization_csv_path=args.normalization_csv or None,
        device=device,
        angle_bin_deg=args.angle_bin_deg,
    )
    print("Task: flower test_data evaluation for 3x3 stencil features -> h*kappa")
    print(f"Dataset file: {result['dataset_path']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Dataset rho_model values: {', '.join(str(item) for item in result['rho_models'])}")
    print(f"Checkpoint: {result['model_path']}")
    print(f"Normalization source: {result['normalization_source']}")
    print(f"Model type: {result['model_type']}")
    print(f"Checkpoint state_dict compatibility: {result['state_dict_compatibility']}")
    print(f"Feature version: {result['feature_version']}")
    print(f"Raw feature dim: {result['raw_feature_dim']}")
    print(f"Model input dim: {result['model_input_dim']}")
    print(f"Device: {device}")
    print(f"Angle bin width: {result['angle_bin_deg']} deg")
    print(
        "Flower summary (model vs analytic): "
        f"RMSE={result['summary']['rmse']:.6e} | "
        f"MAE={result['summary']['mae']:.6e} | "
        f"MaxAE={result['summary']['max_abs_err']:.6e}"
    )
    if len(result["rho_models"]) != 1:
        raise SystemExit(
            f"flower.py expects exactly one rho_model in the dataset, "
            f"got: {result['rho_models']}"
        )
    rho = int(result["rho_models"][0])
    output_dir = ensure_output_dir(args.output_dir)
    overview_cases = _select_overview_cases(result["cases"])
    overview_path = render_curvature_overview(
        overview_cases,
        output_dir / f"flower_curvature_overview_rho{rho}.png",
        suptitle=f"model_{rho} - data_{rho}",
        layout="case_rows",
    )
    print(f"Overview image: {overview_path}")
    print(f"Failed case slices: {result['failed_case_count']}")
    print("Primary comparisons against analytic h*kappa:")
    for metric_name in PRIMARY_COMPARISONS:
        metric = result[metric_name]
        print(f"{metric_name}: {_format_metric(metric, include_maxae=True)}")
    _print_case_iter_pivot(result["cases"])
    _print_case_angle_bins(result["angle_bin_rows"], bin_deg=int(result["angle_bin_deg"]))
    if args.use_swanlab:
        import swanlab

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
                "normalization_source": result["normalization_source"],
                "model_type": result["model_type"],
                "sample_count": result["sample_count"],
                "angle_bin_deg": int(result["angle_bin_deg"]),
                "device": str(device),
            },
        )
        run.log({
            "flower_eval/mse": result["summary"]["mse"],
            "flower_eval/rmse": result["summary"]["rmse"],
            "flower_eval/mae": result["summary"]["mae"],
            "flower_eval/max_abs_err": result["summary"]["max_abs_err"],
            "flower_eval/numeric_mse": result["numeric_summary"]["mse"],
            "flower_eval/numeric_rmse": result["numeric_summary"]["rmse"],
            "flower_eval/numeric_mae": result["numeric_summary"]["mae"],
            "flower_eval/numeric_max_abs_err": result["numeric_summary"]["max_abs_err"],
            "flower_eval/failed_case_count": result["failed_case_count"],
            "flower_eval/overview": _build_swanlab_image(swanlab, overview_path),
            "flower_eval/case_summary_table": _build_case_summary_table_payload(
                swanlab,
                result["cases"],
                result["failed_case_slices"],
            ),
            "flower_eval/angle_bin_table": build_swanlab_table_payload(
                swanlab,
                result["angle_bin_rows"],
                fieldnames=ANGLE_BIN_FIELDNAMES,
            ),
        })
        _log_swanlab_step_series(run, result)
        run.finish()


if __name__ == "__main__":
    main()
