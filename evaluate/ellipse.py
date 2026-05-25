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
    rho_model: int
    a: float
    b: float

    @property
    def label(self) -> str:
        return f"rho{self.rho_model}_a{self.a:.2f}_b{self.b:.2f}"


ELLIPSE_CASES = (
    EllipseCase(256, 0.10, 0.09),
    EllipseCase(256, 0.22, 0.12),
    EllipseCase(266, 0.18, 0.16),
    EllipseCase(276, 0.34, 0.18),
    EllipseCase(276, 0.24, 0.12),
)


def _build_blueprint(case: EllipseCase) -> dict[str, Any]:
    h = 1.0 / float(case.rho_model - 1)
    return {
        "meta": {
            "shape_type": "ellipse",
            "resolution": int(case.rho_model),
            "blueprint_id": case.label,
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
    parser = argparse.ArgumentParser(description="Evaluate static axis-aligned ellipse SDF cases against analytic curvature.")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=str(Path("out") / "curvature_viz" / "ellipse"))
    add_swanlab_args(parser)
    return parser


def run_ellipse_evaluation(*, device: torch.device, output_dir: str | Path) -> dict[str, Any]:
    output_dir_path = ensure_output_dir(output_dir)
    rho_metadata = {
        rho_model: load_training_metadata_from_hdf5(Path("dataset") / f"{rho_model}.h5")
        for rho_model in sorted({case.rho_model for case in ELLIPSE_CASES})
    }
    rho_model_runtime: dict[int, dict[str, Any]] = {}
    for rho_model, metadata in rho_metadata.items():
        model_path = Path("out") / f"baseline_{rho_model}.pt"
        model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
        feature_transform, normalization_source = resolve_feature_transform(
            model_path=model_path,
            explicit_path=Path("out") / f"baseline_{rho_model}.csv",
            checkpoint_meta=checkpoint_meta,
        )
        rho_model_runtime[rho_model] = {
            "data_config": metadata["config"],
            "model": model,
            "model_path": str(model_path.resolve()),
            "feature_transform": feature_transform,
            "normalization_source": normalization_source,
        }

    case_rows: list[dict[str, Any]] = []
    for case_id, case in enumerate(ELLIPSE_CASES):
        runtime = rho_model_runtime[int(case.rho_model)]
        data_config = runtime["data_config"]
        blueprint = _build_blueprint(case)
        X, Y = build_grid(int(case.rho_model))
        phi0 = build_phi0_grid(blueprint, "sdf", data_config=data_config, X=X, Y=Y)
        indices = interface_indices(phi0)
        if indices.size == 0:
            raise RuntimeError(f"Static ellipse case {case.label} produced no interface nodes.")
        h_value = float(blueprint["params"]["h"])
        phi9, features = build_raw_features(phi0, indices, scale_h=bool(data_config.scale_h), h=h_value)
        analytic = compute_hkappa_targets(blueprint, indices, data_config=data_config, X=X, Y=Y).reshape(-1)
        numeric = central_difference_hkappa_from_phi9(phi9)
        prediction = predict_hkappa_full_batch(
            runtime["model"],
            features,
            transform=runtime["feature_transform"],
            device=device,
        )
        rows = indices[:, 0]
        cols = indices[:, 1]
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
        case_rows.append(
            {
                "rho_model": int(case.rho_model),
                "case_id": int(case_id),
                "case_label": case.label,
                "iter": 0,
                "case_key": case.label,
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
                "model_vs_numeric": compute_metrics(prediction_sorted, numeric_sorted),
                "a": float(case.a),
                "b": float(case.b),
                "model_path": str(runtime["model_path"]),
                "normalization_source": str(runtime["normalization_source"]),
                "title": f"rho={case.rho_model}\na={case.a:.2f}, b={case.b:.2f}",
            }
        )

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
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_ellipse_evaluation(device=device, output_dir=args.output_dir)
    print("Task: static ellipse curvature evaluation")
    print(f"Device: {device}")
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
                "case_count": len(ELLIPSE_CASES),
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
