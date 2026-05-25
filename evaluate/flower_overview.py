from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.curvature_plotting import (
    ANGLE_BIN_FIELDNAMES,
    build_swanlab_image,
    build_swanlab_table_payload,
    ensure_output_dir,
    render_curvature_overview,
    write_csv_rows,
)
from evaluate.flower import evaluate_flower
from evaluate.shared import add_swanlab_args, csv_to_list, init_swanlab_run


RHO_MODELS = (256, 266, 276)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a three-resolution flower curvature overview using rho256/rho266/rho276 test datasets."
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--angle-bin-deg", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default=str(Path("out") / "curvature_viz" / "flower"))
    add_swanlab_args(parser)
    return parser


def run_flower_overview(*, device: torch.device, angle_bin_deg: float, output_dir: str | Path) -> dict[str, Any]:
    output_dir_path = ensure_output_dir(output_dir)
    overview_cases: list[dict[str, Any]] = []
    angle_bin_rows: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []

    for rho_model in RHO_MODELS:
        result = evaluate_flower(
            dataset_path=Path("test_data") / f"rho{rho_model}.h5",
            model_path=Path("out") / f"baseline_{rho_model}.pt",
            normalization_csv_path=Path("out") / f"baseline_{rho_model}.csv",
            device=device,
            angle_bin_deg=angle_bin_deg,
        )
        representative_case = dict(result["representative_case"])
        representative_case["title"] = f"rho={rho_model}\n{representative_case['case_key']}"
        overview_cases.append(representative_case)
        angle_bin_rows.extend(result["angle_bin_rows"])
        run_results.append(result)

    overview_path = render_curvature_overview(
        overview_cases,
        output_dir_path / "flower_curvature_error_overview.png",
        suptitle="Flower curvature error overview",
    )
    angle_csv_path = write_csv_rows(
        output_dir_path / "flower_angle_bins.csv",
        angle_bin_rows,
        fieldnames=ANGLE_BIN_FIELDNAMES,
    )
    return {
        "results": run_results,
        "overview_path": overview_path,
        "angle_csv_path": angle_csv_path,
        "angle_bin_rows": angle_bin_rows,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_flower_overview(device=device, angle_bin_deg=args.angle_bin_deg, output_dir=args.output_dir)
    print("Task: flower curvature overview")
    print(f"Device: {device}")
    print(f"Overview image: {result['overview_path']}")
    print(f"Angle-bin CSV: {result['angle_csv_path']}")
    for run_result in result["results"]:
        representative = run_result["representative_case"]
        print(
            f"rho={int(representative['rho_model'])} "
            f"case={representative['case_key']} "
            f"RMSE={run_result['summary']['rmse']:.6e} "
            f"MAE={run_result['summary']['mae']:.6e}"
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
                "task": "flower overview",
                "rho_models": list(RHO_MODELS),
                "angle_bin_deg": float(args.angle_bin_deg),
                "device": str(device),
            },
        )
        run.log({
            "flower_overview/curvature_error_overview": build_swanlab_image(swanlab, result["overview_path"]),
            "flower_overview/angle_bin_table": build_swanlab_table_payload(
                swanlab,
                result["angle_bin_rows"],
                fieldnames=ANGLE_BIN_FIELDNAMES,
            ),
        })
        run.finish()


if __name__ == "__main__":
    main()
