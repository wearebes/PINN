from __future__ import annotations

import argparse
import json
import math
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
        central_difference_hkappa_from_phi9,
        compute_metrics,
        csv_to_list,
        group_metric_rows,
        init_swanlab_run,
        load_model_from_checkpoint,
        predict_hkappa_full_batch,
        resolve_feature_transform,
    )
    from model.config import default_output_model_path
    from testdata_generate.generate import find_projection_theta
else:
    from .shared import (
        central_difference_hkappa_from_phi9,
        compute_metrics,
        csv_to_list,
        group_metric_rows,
        init_swanlab_run,
        load_model_from_checkpoint,
        predict_hkappa_full_batch,
        resolve_feature_transform,
    )
    from model.config import default_output_model_path
    from testdata_generate.generate import find_projection_theta


REQUIRED_FIELDS = ("phi9", "xy", "hkappa_target", "case_id", "iter", "rho_model", "h")
PRIMARY_COMPARISONS = ("numeric_vs_analytic", "model_vs_analytic")
AUXILIARY_COMPARISONS = ("model_vs_numeric",)


def _decode_json_attr(attrs: dict[str, Any], key: str) -> Any | None:
    raw = attrs.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _metric_summary(metric: dict[str, float]) -> dict[str, float]:
    mse = float(metric["mse"])
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(metric["mae"]),
        "max_abs_err": float(metric["maxae"]),
    }


def _case_sort_key(case_entry: dict[str, Any]) -> tuple[str, int, int]:
    return (str(case_entry["case_label"]), int(case_entry["iter"]), int(case_entry["case_id"]))


def _sanitize_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw))


def _build_hk_metric_payload(prefix: str, metric: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}/MSE_hk": float(metric["mse"]),
        f"{prefix}/MAE_hk": float(metric["mae"]),
        f"{prefix}/MaxAE_hk": float(metric["maxae"]),
    }


def _render_representative_curve(case_entry: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.asarray(case_entry["theta"], dtype=np.float64)
    pred_hkappa = np.asarray(case_entry["pred_hkappa"], dtype=np.float64)
    true_hkappa = np.asarray(case_entry["true_hkappa"], dtype=np.float64)
    abs_err = np.asarray(case_entry["abs_err"], dtype=np.float64)
    metrics = case_entry["summary"]

    output_dir = Path(tempfile.mkdtemp(prefix="flower_eval_curve_"))
    output_path = output_dir / f"{_sanitize_name(case_entry['case_key'])}.png"

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(theta, pred_hkappa, label="pred_hkappa", linewidth=1.6)
    axes[0].plot(theta, true_hkappa, label="true_hkappa", linewidth=1.6)
    axes[0].set_ylabel("h*kappa")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(theta, abs_err, color="tab:red", label="abs_err", linewidth=1.6)
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("|pred-true|")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"{case_entry['case_key']} | RMSE={metrics['rmse']:.6e} | "
        f"MAE={metrics['mae']:.6e} | MaxAE={metrics['max_abs_err']:.6e}"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_swanlab_image(swanlab_module: Any, image_path: Path) -> Any:
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


def _build_case_summary_table_payload(
    swanlab_module: Any,
    case_rows: list[dict[str, Any]],
    failed_case_slices: list[dict[str, Any]],
) -> Any:
    rows: list[list[Any]] = [[
        "rank",
        "case_key",
        "case_id",
        "case_label",
        "iter",
        "sample_count",
        "rmse",
        "mae",
        "max_abs_err",
        "numeric_rmse",
        "numeric_mae",
        "numeric_max_abs_err",
    ]]
    sorted_rows = sorted(case_rows, key=lambda row: (float(row["summary"]["mae"]), _case_sort_key(row)))
    for rank, row in enumerate(sorted_rows, start=1):
        summary = row["summary"]
        numeric_summary = row["numeric_summary"]
        rows.append([
            int(rank),
            str(row["case_key"]),
            int(row["case_id"]),
            str(row["case_label"]),
            int(row["iter"]),
            int(row["sample_count"]),
            float(summary["rmse"]),
            float(summary["mae"]),
            float(summary["max_abs_err"]),
            float(numeric_summary["rmse"]),
            float(numeric_summary["mae"]),
            float(numeric_summary["max_abs_err"]),
        ])
    if failed_case_slices:
        rows.append(["-", "FAILED_CASE_SLICES", "-", "-", "-", int(len(failed_case_slices)), "-", "-", "-", "-", "-", "-"])
        for failed in failed_case_slices:
            rows.append([
                "-",
                str(failed["case_key"]),
                int(failed["case_id"]),
                str(failed["case_label"]),
                int(failed["iter"]),
                0,
                str(failed["reason"]),
                "",
                "",
                "",
                "",
                "",
            ])
    echarts_namespace = getattr(swanlab_module, "echarts", None)
    if echarts_namespace is not None and hasattr(echarts_namespace, "table"):
        return echarts_namespace.table(rows)
    text_type = getattr(swanlab_module, "Text", None)
    if text_type is not None:
        return text_type("\n".join(" | ".join(str(item) for item in row) for row in rows))
    return rows


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
    model_vs_numeric = compute_metrics(pred_sorted, numeric_sorted)
    return {
        "case_id": int(case_id),
        "case_label": str(case_label),
        "iter": int(iteration),
        "case_key": f"{case_label}/iter_{int(iteration)}",
        "sample_count": int(theta_sorted.shape[0]),
        "theta": theta_sorted,
        "pred_hkappa": pred_sorted,
        "true_hkappa": target_sorted,
        "numeric_hkappa": numeric_sorted,
        "abs_err": abs_err,
        "summary": _metric_summary(model_vs_analytic),
        "numeric_summary": _metric_summary(numeric_vs_analytic),
        "model_vs_analytic": model_vs_analytic,
        "numeric_vs_analytic": numeric_vs_analytic,
        "model_vs_numeric": model_vs_numeric,
    }


def _select_representative_case(case_rows: list[dict[str, Any]], preferred_case_id: str | None) -> dict[str, Any]:
    ordered_rows = sorted(case_rows, key=_case_sort_key)
    if not ordered_rows:
        raise ValueError("No flower case slices are available for representative visualization.")
    if not preferred_case_id:
        return ordered_rows[0]
    target = str(preferred_case_id).strip()
    if not target:
        return ordered_rows[0]
    numeric_target = int(target) if target.isdigit() else None
    matches = [
        row
        for row in ordered_rows
        if row["case_key"] == target
        or row["case_label"] == target
        or (numeric_target is not None and int(row["case_id"]) == numeric_target)
    ]
    if matches:
        return matches[0]
    available = ", ".join(row["case_key"] for row in ordered_rows)
    raise ValueError(f"Representative case {target!r} was not found. Available case slices: {available}.")


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
    if feature_version != 1 or raw_feature_dim != 9 or feature_order != "phi9":
        raise ValueError(
            f"Flower dataset {dataset_path.resolve()} is not a supported V1 phi9 dataset. "
            f"Got feature_version={feature_version}, feature_dim_raw={raw_feature_dim}, feature_order={feature_order!r}."
        )
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"Flower dataset phi9 must have shape (N, 9), got {phi9.shape}.")
    if features.ndim != 2 or features.shape[1] != 9:
        raise ValueError(f"Flower dataset features must have shape (N, 9), got {features.shape}.")
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
    representative_case_id: str | None = None,
) -> dict[str, Any]:
    bundle = load_flower_dataset(dataset_path)
    arrays = bundle["arrays"]
    phi9 = np.asarray(arrays["phi9"], dtype=np.float32)
    features = np.asarray(arrays["features"], dtype=np.float32)
    xy = np.asarray(arrays["xy"], dtype=np.float64)
    hkappa_target = np.asarray(arrays["hkappa_target"], dtype=np.float64).reshape(-1)
    case_ids = np.asarray(arrays["case_id"]).reshape(-1)
    iterations = np.asarray(arrays["iter"]).reshape(-1)
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
    model_vs_numeric = compute_metrics(prediction, numeric)
    representative_case = _select_representative_case(case_rows, representative_case_id)

    return {
        "dataset_path": bundle["dataset_path"],
        "sample_count": int(phi9.shape[0]),
        "rho_models": bundle["rho_models"],
        "model_path": str(Path(model_path).resolve()),
        "normalization_source": normalization_source,
        "model_type": checkpoint_meta["model_type"],
        "feature_version": int(feature_transform["feature_version"]),
        "raw_feature_dim": int(feature_transform["raw_feature_dim"]),
        "model_input_dim": int(feature_transform["output_dim"]),
        "summary": _metric_summary(model_vs_analytic),
        "numeric_summary": _metric_summary(numeric_vs_analytic),
        "model_vs_numeric_summary": _metric_summary(model_vs_numeric),
        "numeric_vs_analytic": numeric_vs_analytic,
        "model_vs_analytic": model_vs_analytic,
        "model_vs_numeric": model_vs_numeric,
        "cases": case_rows,
        "representative_case": representative_case,
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
    from collections import defaultdict

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
    case_rows: list[dict[str, Any]],
    *,
    bin_deg: int = 60,
    sources: tuple[str, ...] = ("numeric_vs_analytic", "model_vs_analytic"),
) -> None:
    from collections import defaultdict
    import numpy as np

    if 360 % bin_deg != 0:
        raise ValueError(f"bin_deg must divide 360, got {bin_deg}.")
    n_bins = 360 // bin_deg
    edges_deg = np.arange(n_bins + 1, dtype=np.float64) * float(bin_deg)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["case_label"])].append(row)

    print(f"\nBy case_id x angle (bin width = {bin_deg} deg, aggregated across iters):")

    for case_label in sorted(grouped):
        rows = grouped[case_label]
        theta_all = np.concatenate([np.asarray(r["theta"], dtype=np.float64) for r in rows])
        pred_all  = np.concatenate([np.asarray(r["pred_hkappa"], dtype=np.float64) for r in rows])
        true_all  = np.concatenate([np.asarray(r["true_hkappa"], dtype=np.float64) for r in rows])
        numer_all = np.concatenate([np.asarray(r["numeric_hkappa"], dtype=np.float64) for r in rows])

        theta_deg = np.mod(np.degrees(theta_all), 360.0)
        bin_idx = np.minimum(
            np.floor(theta_deg / float(bin_deg)).astype(np.int64),
            n_bins - 1,
        )

        total_samples = int(theta_all.shape[0])
        print(f"\n{case_label} ({len(rows)} iters, total samples={total_samples}):")
        header = (
            f"  {'angle bin':<14}{'N':>7}    "
            f"{'numeric_vs_analytic (MSE / MAE / MaxAE)':<46}    "
            f"{'model_vs_analytic (MSE / MAE / MaxAE)'}"
        )
        print(header)

        for b in range(n_bins):
            mask = bin_idx == b
            n_in = int(np.count_nonzero(mask))
            lo = int(edges_deg[b])
            hi = int(edges_deg[b + 1])
            bin_label = f"[{lo:>3d},{hi:>4d})"
            if n_in == 0:
                print(f"  {bin_label:<14}{n_in:>7}    {'(empty)':<46}    (empty)")
                continue

            stats_per_src: dict[str, tuple[float, float, float]] = {}
            for src in sources:
                if src == "numeric_vs_analytic":
                    pred_arr, target_arr = numer_all[mask], true_all[mask]
                elif src == "model_vs_analytic":
                    pred_arr, target_arr = pred_all[mask], true_all[mask]
                elif src == "model_vs_numeric":
                    pred_arr, target_arr = pred_all[mask], numer_all[mask]
                else:
                    raise ValueError(f"Unsupported source: {src!r}")
                m = compute_metrics(pred_arr, target_arr)
                stats_per_src[src] = (float(m["mse"]), float(m["mae"]), float(m["maxae"]))

            cells = []
            for src in sources:
                mse, mae, maxae = stats_per_src[src]
                cells.append(f"MSE={mse:.3e} MAE={mae:.3e} MaxAE={maxae:.3e}")
            print(f"  {bin_label:<14}{n_in:>7}    {cells[0]:<46}    {cells[1]}")


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
    parser.add_argument(
        "--representative-case-id",
        type=str,
        default="",
        help="Optional case selector for the SwanLab representative curve. Accepts case_id, exp_id, or case_key like smooth_256/iter_1.",
    )
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
        normalization_csv_path=args.normalization_csv or None,
        device=device,
        representative_case_id=args.representative_case_id or None,
    )
    print("Task: flower test_data evaluation for 3x3 stencil features -> h*kappa")
    print(f"Dataset file: {result['dataset_path']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Dataset rho_model values: {', '.join(str(item) for item in result['rho_models'])}")
    print(f"Checkpoint: {result['model_path']}")
    print(f"Normalization source: {result['normalization_source']}")
    print(f"Model type: {result['model_type']}")
    print(f"Feature version: {result['feature_version']}")
    print(f"Raw feature dim: {result['raw_feature_dim']}")
    print(f"Model input dim: {result['model_input_dim']}")
    print(f"Device: {device}")
    print(
        "Flower summary (model vs analytic): "
        f"RMSE={result['summary']['rmse']:.6e} | "
        f"MAE={result['summary']['mae']:.6e} | "
        f"MaxAE={result['summary']['max_abs_err']:.6e}"
    )
    print(
        "Representative case slice: "
        f"{result['representative_case']['case_key']} "
        f"(samples={result['representative_case']['sample_count']})"
    )
    print(f"Failed case slices: {result['failed_case_count']}")
    print("Primary comparisons against analytic h*kappa:")
    for metric_name in PRIMARY_COMPARISONS:
        metric = result[metric_name]
        print(f"{metric_name}: {_format_metric(metric, include_maxae=True)}")
    print("Auxiliary agreement check:")
    for metric_name in AUXILIARY_COMPARISONS:
        metric = result[metric_name]
        print(f"{metric_name}: {_format_metric(metric, include_maxae=False)}")
    _print_case_iter_pivot(result["cases"])
    _print_case_angle_bins(result["cases"])
    if args.use_swanlab:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("SwanLab logging was requested, but `swanlab` is not installed.") from exc
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
                "representative_case": result["representative_case"]["case_key"],
                "device": str(device),
            },
        )
        representative_curve_path = _render_representative_curve(result["representative_case"])
        run.log({
            "flower_eval/mse": result["summary"]["mse"],
            "flower_eval/rmse": result["summary"]["rmse"],
            "flower_eval/mae": result["summary"]["mae"],
            "flower_eval/max_abs_err": result["summary"]["max_abs_err"],
            "flower_eval/numeric_mse": result["numeric_summary"]["mse"],
            "flower_eval/numeric_rmse": result["numeric_summary"]["rmse"],
            "flower_eval/numeric_mae": result["numeric_summary"]["mae"],
            "flower_eval/numeric_max_abs_err": result["numeric_summary"]["max_abs_err"],
            "flower_eval/model_vs_numeric_mse": result["model_vs_numeric_summary"]["mse"],
            "flower_eval/model_vs_numeric_rmse": result["model_vs_numeric_summary"]["rmse"],
            "flower_eval/model_vs_numeric_mae": result["model_vs_numeric_summary"]["mae"],
            "flower_eval/model_vs_numeric_max_abs_err": result["model_vs_numeric_summary"]["max_abs_err"],
            "flower_eval/failed_case_count": result["failed_case_count"],
            "flower_eval/representative_curve": _build_swanlab_image(swanlab, representative_curve_path),
            "flower_eval/case_summary_table": _build_case_summary_table_payload(
                swanlab,
                result["cases"],
                result["failed_case_slices"],
            ),
        })
        _log_swanlab_step_series(run, result)
        run.finish()


if __name__ == "__main__":
    main()
