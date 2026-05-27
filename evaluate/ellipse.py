from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.curvature_plotting import (
    build_case_summary_rows,
    build_swanlab_image,
    build_swanlab_table_payload,
    ensure_output_dir,
    metric_summary,
    render_curvature_overview,
    write_csv_rows,
)
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
from train_generate.generate import (
    build_grid,
    build_phi0_grid,
    build_raw_features,
    compute_hkappa_targets,
    ellipse_local_coordinates,
    interface_indices,
    project_theta_to_axis_aligned_ellipse,
    project_theta_to_axis_aligned_ellipse_high_precision,
)
from train_generate.io import load_training_metadata_from_hdf5


@dataclass(frozen=True)
class EllipseCase:
    a: float
    b: float

    @property
    def label(self) -> str:
        return f"a{self.a:.2f}_b{self.b:.2f}"


# ============================================================
# 椭圆 case 参数 —— 本文件唯一需要编辑的区域
#
# 一次运行的所有 case 共享命令行传入的 --rho-model；
# center=(0.5, 0.5) 和 psi=0.0 写死在 _build_blueprint 里，
# 本次改造不暴露成可配项。
# ============================================================
ELLIPSE_CASES = (
    EllipseCase(0.10, 0.09),
    EllipseCase(0.22, 0.12),
    EllipseCase(0.18, 0.16),
    EllipseCase(0.34, 0.18),
    EllipseCase(0.24, 0.12),
)
# ============================================================


def _build_blueprint(case: EllipseCase, *, rho_model: int) -> dict[str, Any]:
    h = 1.0 / float(rho_model - 1)
    return {
        "meta": {
            "shape_type": "ellipse",
            "resolution": int(rho_model),
            "blueprint_id": f"rho{rho_model}_{case.label}",
        },
        "params": {
            "center": [0.5, 0.5],
            "a": float(case.a),
            "b": float(case.b),
            "psi": 0.0,
            "h": float(h),
        },
    }


def _theta_for_case(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    case: EllipseCase,
    data_config: Any,
) -> np.ndarray:
    u, v = ellipse_local_coordinates(
        x_values,
        y_values,
        cx=0.5,
        cy=0.5,
        psi=0.0,
    )
    theta_seed = project_theta_to_axis_aligned_ellipse(
        u,
        v,
        a=float(case.a),
        b=float(case.b),
        max_iter=int(data_config.ellipse_sdf_newton_max_iter),
        tol=float(data_config.ellipse_sdf_newton_tol),
    )
    return project_theta_to_axis_aligned_ellipse_high_precision(
        u,
        v,
        a=float(case.a),
        b=float(case.b),
        dps=int(data_config.ellipse_hp_dps),
        max_iter=int(data_config.ellipse_hp_newton_max_iter),
        initial_theta=theta_seed,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate static axis-aligned ellipse SDF cases against analytic curvature."
    )
    parser.add_argument(
        "--rho-model",
        type=int,
        required=True,
        help="Test resolution (grid size). Determines default dataset/model paths and output labels.",
    )
    parser.add_argument("--model-path", type=str, default="", help="Path to model checkpoint (.pt). Default: out/baseline_<rho>.pt")
    parser.add_argument("--normalization-csv", type=str, default="", help="Path to normalization CSV. Default: resolved from checkpoint or candidates.")
    parser.add_argument("--dataset-path", type=str, default="", help="Path to HDF5 dataset for metadata. Default: dataset/<rho>/<rho>.h5")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory. Default: out/curvature_viz/ellipse/rho<rho>")
    parser.add_argument("--device", type=str, default="")
    add_swanlab_args(parser)
    return parser


def run_ellipse_evaluation(
    *,
    device: torch.device,
    rho_model: int,
    output_dir: str | Path,
    model_path_override: str | Path | None = None,
    normalization_csv_override: str | Path | None = None,
    dataset_path_override: str | Path | None = None,
) -> dict[str, Any]:
    output_dir_path = ensure_output_dir(output_dir)

    # ---- 1. dataset / data_config ----
    dataset_path = (
        Path(dataset_path_override)
        if dataset_path_override is not None
        else Path("dataset") / str(rho_model) / f"{rho_model}.h5"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Ellipse evaluation requires dataset metadata for rho={rho_model}, "
            f"but '{dataset_path.resolve()}' does not exist. "
            f"Generate it with `python -m train_generate` for this resolution, "
            f"or pass an explicit --dataset-path."
        )
    metadata = load_training_metadata_from_hdf5(dataset_path)
    data_config = metadata["config"]

    # ---- 2. model + feature transform ----
    model_path = (
        Path(model_path_override)
        if model_path_override is not None
        else Path("out") / f"baseline_{rho_model}.pt"
    )
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    normalization_path = (
        Path(normalization_csv_override)
        if normalization_csv_override is not None
        else None
    )
    feature_transform, normalization_source = resolve_feature_transform(
        model_path=model_path,
        explicit_path=normalization_path,
        checkpoint_meta=checkpoint_meta,
    )

    # ---- 3. per-case loop ----
    case_rows: list[dict[str, Any]] = []
    for case_id, case in enumerate(ELLIPSE_CASES):
        blueprint = _build_blueprint(case, rho_model=rho_model)
        X, Y = build_grid(int(rho_model))
        phi0 = build_phi0_grid(blueprint, "sdf", data_config=data_config, X=X, Y=Y)
        indices = interface_indices(phi0)
        if indices.size == 0:
            raise RuntimeError(
                f"Static ellipse case rho{rho_model}_{case.label} produced no interface nodes."
            )
        h_value = float(blueprint["params"]["h"])
        phi9, features = build_raw_features(
            phi0, indices, scale_h=bool(data_config.scale_h), h=h_value
        )
        analytic = compute_hkappa_targets(
            blueprint, indices, data_config=data_config, X=X, Y=Y
        ).reshape(-1)
        numeric = central_difference_hkappa_from_phi9(phi9)
        prediction = predict_hkappa_full_batch(
            model, features, transform=feature_transform, device=device
        )
        rows, cols = indices[:, 0], indices[:, 1]
        xy = np.column_stack((X[rows, cols], Y[rows, cols])).astype(np.float64)
        theta = _theta_for_case(
            x_values=X[rows, cols],
            y_values=Y[rows, cols],
            case=case,
            data_config=data_config,
        )
        order = np.argsort(theta, kind="mergesort")
        analytic_sorted = np.asarray(analytic[order], dtype=np.float64)
        numeric_sorted = np.asarray(numeric[order], dtype=np.float64)
        prediction_sorted = np.asarray(prediction[order], dtype=np.float64)
        model_vs_analytic = compute_metrics(prediction_sorted, analytic_sorted)
        numeric_vs_analytic = compute_metrics(numeric_sorted, analytic_sorted)
        case_key = f"rho{rho_model}_{case.label}"
        case_rows.append(
            {
                "rho_model": int(rho_model),
                "case_id": int(case_id),
                "case_label": case.label,
                "iter": 0,
                "case_key": case_key,
                "sample_count": int(indices.shape[0]),
                "theta": np.asarray(theta[order], dtype=np.float64),
                "xy": np.asarray(xy[order], dtype=np.float64),
                "pred_hkappa": prediction_sorted,
                "true_hkappa": analytic_sorted,
                "numeric_hkappa": numeric_sorted,
                "abs_err": np.abs(prediction_sorted - analytic_sorted),
                "summary": metric_summary(model_vs_analytic),
                "numeric_summary": metric_summary(numeric_vs_analytic),
                "model_vs_analytic": model_vs_analytic,
                "numeric_vs_analytic": numeric_vs_analytic,
                "a": float(case.a),
                "b": float(case.b),
                "model_path": str(model_path.resolve()),
                "normalization_source": str(normalization_source),
                "title": f"rho={rho_model}\na={case.a:.2f}, b={case.b:.2f}",
            }
        )

    # ---- 4. render overview + summary CSV ----
    overview_path = render_curvature_overview(
        case_rows,
        output_dir_path / "ellipse_curvature_error_overview.png",
        suptitle="Static ellipse curvature error overview",
    )
    summary_rows = build_case_summary_rows(case_rows)
    summary_lookup = {row["case_key"]: row for row in summary_rows}
    for case_row in case_rows:
        summary_lookup[str(case_row["case_key"])].update(
            {
                "a": float(case_row["a"]),
                "b": float(case_row["b"]),
                "model_path": str(case_row["model_path"]),
                "normalization_source": str(case_row["normalization_source"]),
            }
        )
    summary_path = write_csv_rows(
        output_dir_path / "ellipse_case_summary.csv",
        list(summary_lookup.values()),
    )
    return {
        "cases": case_rows,
        "overview_path": overview_path,
        "summary_path": summary_path,
        "summary_rows": list(summary_lookup.values()),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    rho_model = int(args.rho_model)
    output_dir = args.output_dir or str(
        Path("out") / "curvature_viz" / "ellipse" / f"rho{rho_model}"
    )
    result = run_ellipse_evaluation(
        device=device,
        rho_model=rho_model,
        output_dir=output_dir,
        model_path_override=args.model_path or None,
        normalization_csv_override=args.normalization_csv or None,
        dataset_path_override=args.dataset_path or None,
    )
    print("Task: static ellipse curvature evaluation")
    print(f"Device: {device}")
    print(f"Rho (test resolution): {rho_model}")
    print(f"Overview image: {result['overview_path']}")
    print(f"Summary CSV: {result['summary_path']}")
    for row in result["summary_rows"]:
        print(
            f"{row['case_key']} "
            f"RMSE={float(row['rmse']):.6e} "
            f"MAE={float(row['mae']):.6e} "
            f"normalization={row['normalization_source']}"
        )
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
                "task": "ellipse evaluation",
                "case_count": len(result["cases"]),
                "rho_model": rho_model,
                "device": str(device),
            },
        )
        run.log({
            "ellipse_eval/curvature_error_overview": build_swanlab_image(swanlab, result["overview_path"]),
            "ellipse_eval/case_summary_table": build_swanlab_table_payload(swanlab, result["summary_rows"]),
        })
        run.finish()


if __name__ == "__main__":
    main()
