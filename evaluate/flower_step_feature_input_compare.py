from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate.flower import evaluate_flower
else:
    from .flower import evaluate_flower


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _case_family(case_label: str) -> str:
    return str(case_label).split("_")[0]


def _collect_case_step_mse(result: dict[str, Any], *, source_key: str) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for row in result["cases"]:
        family = _case_family(str(row["case_label"]))
        step = int(row["iter"])
        out[(family, step)] = float(row[source_key]["mse"])
    return out


def _collect_case_step_count(result: dict[str, Any]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for row in result["cases"]:
        family = _case_family(str(row["case_label"]))
        step = int(row["iter"])
        out[(family, step)] = int(row["sample_count"])
    return out


def _render_plot(
    *,
    output_path: Path,
    series: dict[str, dict[str, list[float]]],
    rho_model: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.4), sharex=True, constrained_layout=True)
    family_order = ("smooth", "acute")
    line_specs = (
        ("phi_over_h_grad", "Model @ (phi/h, nx, ny)", "#1f77b4", "o", "-"),
        ("phi_raw_grad", "Model @ (phi, nx, ny)", "#d62728", "s", "-"),
        ("fd", "FD", "#222222", "x", "--"),
    )

    for axis, family in zip(axes, family_order, strict=True):
        payload = series[family]
        for key, label, color, marker, linestyle in line_specs:
            axis.plot(
                payload["steps"],
                payload[key],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.5,
            )
        axis.set_title(f"{family}, rho_train=rho_test={rho_model}", fontsize=11)
        axis.set_ylabel("MSE")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Reinit step")
    fig.suptitle("Flower reinit-step comparison for one 27D checkpoint", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_csv(
    *,
    output_path: Path,
    series: dict[str, dict[str, list[float]]],
    sample_counts: dict[tuple[str, int], int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "step",
                "sample_count",
                "mse_model_phi_over_h_grad",
                "mse_model_phi_raw_grad",
                "mse_fd",
            ]
        )
        for family in ("smooth", "acute"):
            payload = series[family]
            for idx, step in enumerate(payload["steps"]):
                writer.writerow(
                    [
                        family,
                        step,
                        sample_counts[(family, int(step))],
                        payload["phi_over_h_grad"][idx],
                        payload["phi_raw_grad"][idx],
                        payload["fd"][idx],
                    ]
                )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Compare one 27D model on two flower feature contracts: "
            "(phi/h,nx,ny) vs (phi,nx,ny), plus the FD baseline."
        )
    )
    parser.add_argument("--rho-model", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--repo-root", type=str, default=str(repo_root))
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Defaults to out/{rho}/baseline_{rho}_hgradient.pt under repo root.",
    )
    parser.add_argument(
        "--hgrad-data",
        type=str,
        default="",
        help="Defaults to test_data/flower_rho{rho}_hgradient.h5 under repo root.",
    )
    parser.add_argument(
        "--rawgrad-data",
        type=str,
        default="",
        help="Defaults to test_data/flower_rho{rho}_rawgradient.h5 under repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Defaults to out/{rho}/flower_input_compare under repo root.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    rho = int(args.rho_model)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path).resolve() if args.model_path else (
        repo_root / "out" / str(rho) / f"baseline_{rho}_hgradient.pt"
    ).resolve()
    hgrad_data = Path(args.hgrad_data).resolve() if args.hgrad_data else (
        repo_root / "test_data" / f"flower_rho{rho}_hgradient.h5"
    ).resolve()
    rawgrad_data = Path(args.rawgrad_data).resolve() if args.rawgrad_data else (
        repo_root / "test_data" / f"flower_rho{rho}_rawgradient.h5"
    ).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        repo_root / "out" / str(rho) / "flower_input_compare"
    ).resolve()

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((output_dir if args.output_dir else Path(tempfile.mkdtemp(prefix="mpl_"))).resolve()),
    )

    print("Flower feature-input step comparison")
    print(f"Repo root : {repo_root}")
    print(f"Device    : {device}")
    print(f"Model     : {model_path}")
    print(f"h+grad    : {hgrad_data}")
    print(f"raw+grad  : {rawgrad_data}")
    print(f"Output dir: {output_dir}")

    hgrad_result = evaluate_flower(
        dataset_path=hgrad_data,
        model_path=model_path,
        normalization_csv_path=None,
        device=device,
    )
    rawgrad_result = evaluate_flower(
        dataset_path=rawgrad_data,
        model_path=model_path,
        normalization_csv_path=None,
        device=device,
    )

    model_hgrad = _collect_case_step_mse(hgrad_result, source_key="model_vs_analytic")
    model_rawgrad = _collect_case_step_mse(rawgrad_result, source_key="model_vs_analytic")
    numeric_hgrad = _collect_case_step_mse(hgrad_result, source_key="numeric_vs_analytic")
    numeric_rawgrad = _collect_case_step_mse(rawgrad_result, source_key="numeric_vs_analytic")
    sample_counts = _collect_case_step_count(hgrad_result)

    keys = sorted(model_hgrad)
    if keys != sorted(model_rawgrad):
        raise ValueError("Case/step keys do not match between h+grad and raw+grad flower evaluations.")
    if keys != sorted(numeric_hgrad) or keys != sorted(numeric_rawgrad):
        raise ValueError("Numeric baseline keys do not match the model-evaluation keys.")

    max_numeric_delta = max(abs(numeric_hgrad[key] - numeric_rawgrad[key]) for key in keys)
    if max_numeric_delta > 1.0e-12:
        raise ValueError(
            f"FD baseline differs between h+grad and raw+grad datasets (max delta={max_numeric_delta:.3e})."
        )

    series: dict[str, dict[str, list[float]]] = {}
    for family in ("smooth", "acute"):
        steps = [step for fam, step in keys if fam == family]
        series[family] = {
            "steps": steps,
            "phi_over_h_grad": [model_hgrad[(family, step)] for step in steps],
            "phi_raw_grad": [model_rawgrad[(family, step)] for step in steps],
            "fd": [numeric_hgrad[(family, step)] for step in steps],
        }

    plot_path = _render_plot(
        output_path=output_dir / f"flower_rho{rho}_feature_input_compare.png",
        series=series,
        rho_model=rho,
    )
    csv_path = _write_csv(
        output_path=output_dir / f"flower_rho{rho}_feature_input_compare.csv",
        series=series,
        sample_counts=sample_counts,
    )

    print(f"Plot: {plot_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
