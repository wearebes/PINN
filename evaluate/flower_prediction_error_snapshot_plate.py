from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np
import torch

from evaluate.flower import evaluate_flower
from evaluate.flower_hgradient_cross_resolution import _ensure_flower_dataset, _setup_matplotlib
from evaluate.flower_prediction_snapshot_plate import (
    DEFAULT_STEPS,
    _curve_segments,
    _family_from_case_label,
    _save_figure,
    _update_manifest,
)


def _write_error_source_csv(cases: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "step",
                "theta_rad",
                "x",
                "y",
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
            for theta, (x_coord, y_coord), pred, target, numeric in zip(
                np.asarray(case["theta"], dtype=float),
                np.asarray(case["xy"], dtype=float),
                np.asarray(case["pred_hkappa"], dtype=float),
                np.asarray(case["true_hkappa"], dtype=float),
                np.asarray(case["numeric_hkappa"], dtype=float),
            ):
                writer.writerow(
                    [
                        family,
                        step,
                        f"{theta:.10g}",
                        f"{x_coord:.10g}",
                        f"{y_coord:.10g}",
                        f"{pred:.10g}",
                        f"{target:.10g}",
                        f"{numeric:.10g}",
                        f"{abs(pred - target):.10g}",
                        f"{abs(numeric - target):.10g}",
                    ]
                )


def render_prediction_error_snapshot_plate(
    *,
    model_path: Path,
    data_path: Path,
    output_stem: Path,
    source_csv_path: Path,
    rho: int,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    device: torch.device,
) -> dict[str, str]:
    _setup_matplotlib()
    mpl.rcParams.update(
        {
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.bottom": False,
            "ytick.left": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "savefig.transparent": False,
        }
    )

    _ensure_flower_dataset(path=data_path, rho=rho, test_iters=steps, overwrite=False)
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
        if _family_from_case_label(str(case["case_label"])) in {"smooth", "acute"}
        and int(case["iter"]) in set(steps)
    ]
    expected = {("smooth", step) for step in steps} | {("acute", step) for step in steps}
    found = {(_family_from_case_label(str(case["case_label"])), int(case["iter"])) for case in cases}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing expected family/step slices: {missing}")

    _write_error_source_csv(cases, source_csv_path)

    # Error spans several orders of magnitude (near-zero on smooth flanks, ~1e-1 at
    # concave tips), so a log-normalized sequential map is needed to show structure
    # at both ends -- a linear scale would just show the tips and wash out the rest.
    abs_errors = np.concatenate(
        [
            np.abs(np.asarray(case["pred_hkappa"], dtype=float) - np.asarray(case["true_hkappa"], dtype=float))
            for case in cases
        ]
    )
    # Tighter than a 1-99 percentile window: the bulk of the curve sits within a
    # narrow error band, so bracketing close to it (and letting the rare tip
    # outliers clip/saturate) spends more of the colormap's range on the part of
    # the curve that actually varies, instead of stretching it across 3+ decades.
    positive = abs_errors[abs_errors > 0.0]
    vmin = max(float(np.nanpercentile(positive, 5.0)) if positive.size else 1.0e-7, 1.0e-7)
    vmax = float(np.nanpercentile(abs_errors, 95.0))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps["YlOrRd"]

    all_xy = np.vstack([np.asarray(case["xy"], dtype=float) for case in cases])
    x_min, y_min = np.nanmin(all_xy, axis=0)
    x_max, y_max = np.nanmax(all_xy, axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min) * 1.12

    by_key = {(_family_from_case_label(str(case["case_label"])), int(case["iter"])): case for case in cases}
    fig, axes = plt.subplots(
        2,
        len(steps),
        figsize=(7.2, 3.45),
        gridspec_kw={"wspace": 0.06, "hspace": 0.10},
        constrained_layout=False,
    )

    for row_index, family in enumerate(("smooth", "acute")):
        for col_index, step in enumerate(steps):
            ax = axes[row_index, col_index]
            case = by_key[(family, step)]
            xy = np.asarray(case["xy"], dtype=float)
            pred = np.asarray(case["pred_hkappa"], dtype=float)
            analytic = np.asarray(case["true_hkappa"], dtype=float)
            abs_error = np.maximum(np.abs(pred - analytic), 1.0e-12)
            segment_values = np.r_[
                0.5 * (abs_error[:-1] + abs_error[1:]), 0.5 * (abs_error[-1] + abs_error[0])
            ]

            ax.plot(
                np.r_[xy[:, 0], xy[0, 0]],
                np.r_[xy[:, 1], xy[0, 1]],
                color="#252525",
                linewidth=2.35,
                solid_capstyle="round",
                zorder=1,
            )
            collection = LineCollection(
                _curve_segments(xy),
                cmap=cmap,
                norm=norm,
                linewidths=2.9,
                zorder=2,
            )
            collection.set_array(segment_values)
            ax.add_collection(collection)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_center - half_span, x_center + half_span)
            ax.set_ylim(y_center - half_span, y_center + half_span)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(str(step), fontsize=8, pad=4)
            if col_index == 0:
                ax.text(
                    -0.16,
                    0.5,
                    family,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=8,
                )

    fig.text(0.50, 0.988, "step", ha="center", va="top", fontsize=8)
    cbar_ax = fig.add_axes([0.918, 0.18, 0.014, 0.64])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cbar_ax, extend="both")
    cbar.set_label(
        r"absolute error $|h\kappa_{\mathrm{NN}} - h\kappa_{\mathrm{analytic}}|$", fontsize=8, labelpad=5
    )
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.085, right=0.895, top=0.90, bottom=0.08)
    outputs = _save_figure(fig, output_stem)
    plt.close(fig)
    outputs["source_csv"] = str(source_csv_path.resolve())
    outputs["data"] = str(data_path.resolve())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the 256/CFL=0.5 smooth/acute NN prediction ERROR snapshot plate "
        "(interface colored by |NN - analytic| across reinitialization steps)."
    )
    parser.add_argument("--rho", type=int, default=256)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--model", type=Path, default=Path("out/256/baseline_256_hgradient.pt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/flower_hgradient_cross_resolution/research_figures/nature_style"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    steps = tuple(int(step) for step in args.steps)
    output_dir = args.output_dir
    step_tag = "-".join(str(step) for step in steps)
    stem = (
        output_dir
        / f"flower_nn_prediction_error_snapshots_train256_test{args.rho}_cfl0p5_steps{step_tag}_nature"
    )
    source_csv = stem.with_suffix(".csv")
    data_path = (
        output_dir
        / "test_data"
        / f"flower_rho{args.rho}_hgradient_cfl0p5_iters{step_tag}.h5"
    )

    outputs = render_prediction_error_snapshot_plate(
        model_path=args.model,
        data_path=data_path,
        output_stem=stem,
        source_csv_path=source_csv,
        rho=int(args.rho),
        steps=steps,
        device=torch.device(args.device),
    )
    _update_manifest(
        output_dir / "flower_reinit_nature_manifest.json",
        {
            "name": stem.name,
            "figure_type": "image plate + quant",
            "claim": "NN curvature prediction error (|NN - analytic|) on smooth and acute flowers can be "
            "inspected spatially and across reinitialization steps.",
            "rho_model": int(args.rho),
            "train_rho": 256,
            "test_rho": int(args.rho),
            "cfl": 0.5,
            "steps": list(steps),
            "outputs": outputs,
        },
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
