from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate.flower import evaluate_flower
else:
    from .flower import evaluate_flower


METRIC_SPECS = (
    ("mse", "MSE"),
    ("mae", "MAE"),
    ("maxae", "MaxAE"),
)


@dataclass(frozen=True)
class EvalSpec:
    rho_model: int
    data_path: Path
    model_path: Path
    normalization_csv_path: Path | None


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_normalization_csv(out_dir: Path, rho_model: int) -> Path | None:
    candidates = (
        out_dir / f"baseline_{rho_model}.csv",
        out_dir / f"baseline_{rho_model}_phi9_normalization.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_eval_spec(repo_root: Path, rho_model: int) -> EvalSpec:
    test_data_dir = repo_root / "test_data"
    out_dir = repo_root / "out"
    data_path = (test_data_dir / f"rho{rho_model}.h5").resolve()
    model_path = (out_dir / f"baseline_{rho_model}.pt").resolve()
    normalization_csv_path = _resolve_normalization_csv(out_dir.resolve(), rho_model)
    if not data_path.exists():
        raise FileNotFoundError(f"Flower test dataset not found for rho_model={rho_model}: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for rho_model={rho_model}: {model_path}")
    return EvalSpec(
        rho_model=int(rho_model),
        data_path=data_path,
        model_path=model_path,
        normalization_csv_path=normalization_csv_path,
    )


def _parse_case_label(case_label: str) -> tuple[str, int | None]:
    text = str(case_label)
    if "_" not in text:
        return text, None
    family, rho_text = text.rsplit("_", 1)
    try:
        return family, int(rho_text)
    except ValueError:
        return family, None


def _series_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
    _, payload = item
    family = str(payload["family"])
    source = str(payload["source"])
    family_rank = 0 if family == "smooth" else 1 if family == "acute" else 2
    source_rank = 0 if source == "numeric" else 1 if source == "model" else 2
    rho_model = int(payload["rho_model"]) if payload["rho_model"] is not None else 0
    return family_rank, source_rank, rho_model, payload["label"]


def _build_comparison_series(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for case_row in result["cases"]:
        family, rho_model = _parse_case_label(str(case_row["case_label"]))
        for source_name, metric_key in (("numeric", "numeric_vs_analytic"), ("model", "model_vs_analytic")):
            label = f"{family}_{source_name}"
            metrics = case_row[metric_key]
            entry = grouped.setdefault(
                label,
                {
                    "label": label,
                    "family": family,
                    "source": source_name,
                    "rho_model": rho_model,
                    "points": [],
                },
            )
            entry["points"].append(
                {
                    "step": int(case_row["iter"]),
                    "sample_count": int(case_row["sample_count"]),
                    "mse": float(metrics["mse"]),
                    "mae": float(metrics["mae"]),
                    "maxae": float(metrics["maxae"]),
                }
            )
    for entry in grouped.values():
        entry["points"].sort(key=lambda row: int(row["step"]))
    return grouped


def _render_comparison_figure(
    *,
    rho_model: int,
    series_map: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series_map:
        raise ValueError("No comparison series are available for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(METRIC_SPECS), 1, figsize=(10.5, 11.5), sharex=True, constrained_layout=True)
    color_map = {
        "smooth": "tab:blue",
        "acute": "tab:orange",
    }
    linestyle_map = {
        "numeric": "--",
        "model": "-",
    }
    marker_map = {
        "numeric": "s",
        "model": "o",
    }

    for label, payload in sorted(series_map.items(), key=_series_sort_key):
        family = str(payload["family"])
        source = str(payload["source"])
        steps = [int(point["step"]) for point in payload["points"]]
        for axis, (metric_key, metric_label) in zip(axes, METRIC_SPECS, strict=True):
            values = [float(point[metric_key]) for point in payload["points"]]
            axis.plot(
                steps,
                values,
                label=label,
                color=color_map.get(family),
                linestyle=linestyle_map.get(source, "-"),
                marker=marker_map.get(source, "o"),
                linewidth=1.8,
                markersize=4.5,
            )
            axis.set_ylabel(metric_label)
            axis.grid(True, alpha=0.3)
            axis.set_title(f"rho={rho_model} {metric_label} vs step", fontsize=11)

    axes[-1].set_xlabel("step")
    axes[0].legend(loc="best", ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_plot_csv(*, output_path: Path, series_map: dict[str, dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "family", "source", "rho_model", "step", "sample_count", "mse", "mae", "maxae"])
        for _, payload in sorted(series_map.items(), key=_series_sort_key):
            for point in payload["points"]:
                writer.writerow(
                    [
                        payload["label"],
                        payload["family"],
                        payload["source"],
                        payload["rho_model"],
                        point["step"],
                        point["sample_count"],
                        point["mse"],
                        point["mae"],
                        point["maxae"],
                    ]
                )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one flower rho_model checkpoint locally and render step-based "
            "MSE/MAE/MaxAE plots that compare numeric and model errors for smooth and acute cases."
        )
    )
    parser.add_argument("--rho-model", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for rendered plots and CSV. Defaults to a temporary folder.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(repo_root),
        help="PINN repo root that contains test_data/ and out/.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="flower_step_plots_")
    ).resolve()
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    spec = _resolve_eval_spec(repo_root, int(args.rho_model))

    print("Flower single-resolution step comparison")
    print(f"Repo root: {repo_root}")
    print(f"Device: {device}")
    print(
        f"Evaluating rho_model={spec.rho_model} "
        f"data={spec.data_path} model={spec.model_path} normalization={spec.normalization_csv_path or 'checkpoint'}"
    )
    result = evaluate_flower(
        dataset_path=spec.data_path,
        model_path=spec.model_path,
        normalization_csv_path=spec.normalization_csv_path,
        device=device,
    )
    series_map = _build_comparison_series(result)
    plot_path = _render_comparison_figure(
        rho_model=spec.rho_model,
        series_map=series_map,
        output_path=output_dir / f"flower_rho{spec.rho_model}_model_vs_numeric_step_metrics.png",
    )
    csv_path = _write_plot_csv(
        output_path=output_dir / f"flower_rho{spec.rho_model}_model_vs_numeric_step_metrics.csv",
        series_map=series_map,
    )

    print(f"Comparison plot: {plot_path}")
    print(f"Metric CSV: {csv_path}")


if __name__ == "__main__":
    main()
