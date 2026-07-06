from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate.flower_prediction_snapshot_plate import (
    DEFAULT_STEPS,
    _family_from_case_label,
    _save_figure,
    _update_manifest,
    _write_source_csv,
    render_prediction_snapshot_plate,
)
from evaluate.flower import evaluate_flower
from evaluate.flower_hgradient_cross_resolution import _ensure_flower_dataset, _setup_matplotlib


def _load_step_cases(
    *,
    data_path: Path,
    model_path: Path,
    rho: int,
    step: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    _ensure_flower_dataset(path=data_path, rho=rho, test_iters=DEFAULT_STEPS, overwrite=False)
    result = evaluate_flower(
        dataset_path=data_path,
        model_path=model_path,
        normalization_csv_path=None,
        device=device,
        angle_bin_deg=30.0,
    )
    cases = [
        case
        for case in result["cases"]
        if int(case["iter"]) == int(step)
        and _family_from_case_label(str(case["case_label"])) in {"smooth", "acute"}
    ]
    found = {_family_from_case_label(str(case["case_label"])) for case in cases}
    if found != {"smooth", "acute"}:
        raise ValueError(f"Expected smooth and acute cases at step {step}, got {sorted(found)}")
    return sorted(cases, key=lambda case: ("smooth", "acute").index(_family_from_case_label(str(case["case_label"]))))


def _write_angle_profile_csv(cases: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "step",
                "theta_deg",
                "nn_pred_hkappa",
                "analytic_hkappa",
                "fd_hkappa",
                "nn_abs_error",
                "fd_abs_error",
            ]
        )
        for case in cases:
            family = _family_from_case_label(str(case["case_label"]))
            step = int(case["iter"])
            theta_deg = np.mod(np.degrees(np.asarray(case["theta"], dtype=float)), 360.0)
            pred = np.asarray(case["pred_hkappa"], dtype=float)
            analytic = np.asarray(case["true_hkappa"], dtype=float)
            fd = np.asarray(case["numeric_hkappa"], dtype=float)
            order = np.argsort(theta_deg, kind="mergesort")
            for idx in order:
                writer.writerow(
                    [
                        family,
                        step,
                        f"{theta_deg[idx]:.10g}",
                        f"{pred[idx]:.10g}",
                        f"{analytic[idx]:.10g}",
                        f"{fd[idx]:.10g}",
                        f"{abs(pred[idx] - analytic[idx]):.10g}",
                        f"{abs(fd[idx] - analytic[idx]):.10g}",
                    ]
                )


def _plot_closed_angle_curve(ax: mpl.axes.Axes, theta_deg: np.ndarray, values: np.ndarray, **kwargs: Any) -> None:
    order = np.argsort(theta_deg, kind="mergesort")
    x = np.asarray(theta_deg[order], dtype=float)
    y = np.asarray(values[order], dtype=float)
    if x.size:
        x = np.r_[x, x[0] + 360.0]
        y = np.r_[y, y[0]]
    ax.plot(x, y, **kwargs)


def render_angle_profile_figure(
    *,
    cases: list[dict[str, Any]],
    output_stem: Path,
    source_csv_path: Path,
    step: int,
) -> dict[str, str]:
    _setup_matplotlib()
    mpl.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    _write_angle_profile_csv(cases, source_csv_path)

    color_nn = "#B64342"
    color_fd = "#6F6F6F"
    color_true = "#111111"
    by_family = {_family_from_case_label(str(case["case_label"])): case for case in cases}
    all_values = np.concatenate(
        [
            np.asarray(case["pred_hkappa"], dtype=float)
            for case in cases
        ]
        + [
            np.asarray(case["true_hkappa"], dtype=float)
            for case in cases
        ]
        + [
            np.asarray(case["numeric_hkappa"], dtype=float)
            for case in cases
        ]
    )
    y_min, y_max = np.nanpercentile(all_values, [0.2, 99.8])
    y_pad = 0.06 * max(float(y_max - y_min), 1.0e-6)

    all_errors = []
    for case in cases:
        true = np.asarray(case["true_hkappa"], dtype=float)
        all_errors.append(np.abs(np.asarray(case["pred_hkappa"], dtype=float) - true))
        all_errors.append(np.abs(np.asarray(case["numeric_hkappa"], dtype=float) - true))
    error_values = np.concatenate(all_errors)
    positive_errors = error_values[error_values > 0.0]
    err_min = max(float(np.nanpercentile(positive_errors, 1.0)) if positive_errors.size else 1.0e-8, 1.0e-7)
    err_max = float(np.nanpercentile(error_values, 99.5))
    if not np.isfinite(err_max) or err_max <= err_min:
        err_max = err_min * 10.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 3.9),
        sharex=True,
        gridspec_kw={"hspace": 0.18, "wspace": 0.20, "height_ratios": [1.35, 1.0]},
    )

    for col_index, family in enumerate(("smooth", "acute")):
        case = by_family[family]
        theta_deg = np.mod(np.degrees(np.asarray(case["theta"], dtype=float)), 360.0)
        pred = np.asarray(case["pred_hkappa"], dtype=float)
        analytic = np.asarray(case["true_hkappa"], dtype=float)
        fd = np.asarray(case["numeric_hkappa"], dtype=float)

        ax_curve = axes[0, col_index]
        _plot_closed_angle_curve(ax_curve, theta_deg, analytic, color=color_true, linewidth=1.15, label="analytic")
        _plot_closed_angle_curve(ax_curve, theta_deg, pred, color=color_nn, linewidth=1.65, label="NN")
        _plot_closed_angle_curve(
            ax_curve,
            theta_deg,
            fd,
            color=color_fd,
            linewidth=1.25,
            linestyle=(0, (3.0, 2.2)),
            label="FD",
        )
        ax_curve.axhline(0.0, color="#D5D5D5", linewidth=0.6, zorder=0)
        ax_curve.set_title(family, pad=3)
        ax_curve.set_xlim(0.0, 360.0)
        ax_curve.set_ylim(float(y_min - y_pad), float(y_max + y_pad))
        if col_index == 0:
            ax_curve.set_ylabel(r"$h\kappa$")
        else:
            ax_curve.set_ylabel("")
        ax_error = axes[1, col_index]
        nn_error = np.maximum(np.abs(pred - analytic), 1.0e-8)
        fd_error = np.maximum(np.abs(fd - analytic), 1.0e-8)
        _plot_closed_angle_curve(ax_error, theta_deg, nn_error, color=color_nn, linewidth=1.45, label="NN")
        _plot_closed_angle_curve(
            ax_error,
            theta_deg,
            fd_error,
            color=color_fd,
            linewidth=1.20,
            linestyle=(0, (3.0, 2.2)),
            label="FD",
        )
        ax_error.set_yscale("log")
        ax_error.set_ylim(err_min, err_max * 1.25)
        ax_error.set_xlabel(r"$\theta$ (deg)")
        if col_index == 0:
            ax_error.set_ylabel("absolute error")
        else:
            ax_error.set_ylabel("")

    for ax in axes.ravel():
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.tick_params(width=0.7, length=3)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=3,
        handlelength=2.4,
        columnspacing=1.2,
        frameon=False,
    )
    fig.text(0.012, 0.968, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.430, "b", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.090, right=0.985, top=0.865, bottom=0.145)
    outputs = _save_figure(fig, output_stem)
    plt.close(fig)
    outputs["source_csv"] = str(source_csv_path.resolve())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render angle-resolved h*kappa and error profiles on the final flower level set."
    )
    parser.add_argument("--rho", type=int, default=256)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--model", type=Path, default=Path("out/256/baseline_256_hgradient.pt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/flower_hgradient_cross_resolution/research_figures/nature_style"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir
    step_tag = "-".join(str(step) for step in DEFAULT_STEPS)
    data_path = output_dir / "test_data" / f"flower_rho{args.rho}_hgradient_cfl0p5_iters{step_tag}.h5"
    snapshot_stem = output_dir / f"flower_nn_prediction_snapshots_train256_test{args.rho}_cfl0p5_steps{step_tag}_nature"
    snapshot_csv = snapshot_stem.with_suffix(".csv")
    if not snapshot_csv.exists():
        render_prediction_snapshot_plate(
            model_path=args.model,
            data_path=data_path,
            output_stem=snapshot_stem,
            source_csv_path=snapshot_csv,
            rho=int(args.rho),
            steps=DEFAULT_STEPS,
            device=torch.device(args.device),
        )

    cases = _load_step_cases(
        data_path=data_path,
        model_path=args.model,
        rho=int(args.rho),
        step=int(args.step),
        device=torch.device(args.device),
    )
    stem = output_dir / f"flower_angle_profile_step{args.step}_train256_test{args.rho}_cfl0p5_nature"
    source_csv = stem.with_suffix(".csv")
    outputs = render_angle_profile_figure(
        cases=cases,
        output_stem=stem,
        source_csv_path=source_csv,
        step=int(args.step),
    )
    outputs["data"] = str(data_path.resolve())
    _update_manifest(
        output_dir / "flower_reinit_nature_manifest.json",
        {
            "name": stem.name,
            "figure_type": "quantitative grid",
            "claim": "At the final reinitialization step, angle-resolved profiles expose where NN and FD curvature estimates deviate from the analytic target.",
            "rho_model": int(args.rho),
            "train_rho": 256,
            "test_rho": int(args.rho),
            "cfl": 0.5,
            "step": int(args.step),
            "outputs": outputs,
        },
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
