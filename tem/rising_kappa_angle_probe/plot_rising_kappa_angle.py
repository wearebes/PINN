#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/private/tmp/mplconfig"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    ROOT
    / "basilisk_reference_gate/rising/results/raw/"
    / "phase2_hgrad_resolution_matched_20260629/N128_L7/full"
)
DEFAULT_BENCHMARK = ROOT / "cfd_applications_cleanroom/vendor/basilisk_clean/src/test/c1g3l4.txt"

METHOD_FILES = {
    "CLSVOF-LS": "CLSVOF-LS.csv",
    "NN single": "baseline_hgradient_single.csv",
    "NN D4 avg": "baseline_hgradient_D4_average.csv",
}

MPL_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Experiment/report figure_style_guide.md palette: near-black for the
# reference/native curve, muted red / deep blue for the two NN variants,
# held fixed across all three probe figures (same method -> same color).
COLOR_NATIVE = "#272727"
COLOR_NN_RAW = "#B64342"
COLOR_NN_D4 = "#0F4D92"
COLOR_ZERO_LINE = "#9A9A9A"
COLOR_CORNER_LABEL = "#4D4D4D"

# METHOD_FILES keys stay as-is (they flow into the CSV "method" column);
# this maps them to short legend text only.
LEGEND_LABEL = {
    "CLSVOF-LS": "CLSVOF-LS",
    "NN single": "NN RAW",
    "NN D4 avg": "NN D4",
}
BENCHMARK_STYLE = {
    "CLSVOF-LS": {"color": COLOR_NATIVE, "ls": "-", "lw": 1.1, "zorder": 3},
    "NN single": {"color": COLOR_NN_RAW, "ls": "--", "lw": 1.3, "zorder": 4},
    "NN D4 avg": {"color": COLOR_NN_D4, "ls": (0, (1, 1)), "lw": 1.3, "zorder": 5},
}


def read_float_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, float | str]] = []
    for row in rows:
        converted: dict[str, float | str] = {}
        for key, value in row.items():
            if key == "method":
                converted[key] = value
            else:
                converted[key] = float(value)
        out.append(converted)
    return out


def read_benchmark_trace(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append(
                {
                    "method": "MooNMD/Hysing",
                    "t": float(parts[0]),
                    "vol_rel": float(parts[1]),
                    "circ": float(parts[2]),
                    "yc": float(parts[3]),
                    "vc": float(parts[4]),
                }
            )
    return rows


def interp_trace(rows: list[dict[str, float | str]], t: float, field: str) -> float:
    numeric = sorted((float(r["t"]), float(r[field])) for r in rows)
    if t <= numeric[0][0]:
        return numeric[0][1]
    if t >= numeric[-1][0]:
        return numeric[-1][1]
    for (t0, v0), (t1, v1) in zip(numeric, numeric[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            w = (t - t0) / (t1 - t0)
            return v0 + w * (v1 - v0)
    raise ValueError(f"t={t} outside trace range")


def mirrored_theta_rows(
    rows: list[dict[str, float | str]], trace_rows: list[dict[str, float | str]]
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in rows:
        t = float(row["t"])
        center_x = interp_trace(trace_rows, t, "yc")
        x_rel = float(row["x"]) - center_x
        y_abs = abs(float(row["y"]))
        delta = float(row["Delta"])

        for sign in (1.0, -1.0):
            y_rel = sign * y_abs
            theta = math.degrees(math.atan2(y_rel, x_rel)) % 360.0
            out.append(
                {
                    "t": t,
                    "theta_deg": theta,
                    "kappa_ref": float(row["hk_ref"]) / delta,
                    "kappa_cd": float(row["hk_cd"]) / delta,
                    "kappa_nn_raw": float(row["hk_nn_raw"]) / delta,
                    "kappa_nn_d4": float(row["hk_nn_d4"]) / delta,
                    "hk_ref": float(row["hk_ref"]),
                    "hk_nn_raw": float(row["hk_nn_raw"]),
                    "hk_nn_d4": float(row["hk_nn_d4"]),
                }
            )
    return sorted(out, key=lambda r: (r["t"], r["theta_deg"]))


def write_angle_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t",
        "theta_deg",
        "kappa_ref",
        "kappa_cd",
        "kappa_nn_raw",
        "kappa_nn_d4",
        "hk_ref",
        "hk_nn_raw",
        "hk_nn_d4",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.17g}" for k in fields})


def write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["t"]].append(row)

    fields = [
        "t",
        "n_points_full_mirrored",
        "mean_abs_kappa_nn_raw_minus_ref",
        "max_abs_kappa_nn_raw_minus_ref",
        "mean_abs_kappa_nn_d4_minus_ref",
        "max_abs_kappa_nn_d4_minus_ref",
        "mean_abs_hk_nn_raw_minus_ref",
        "max_abs_hk_nn_raw_minus_ref",
        "mean_abs_hk_nn_d4_minus_ref",
        "max_abs_hk_nn_d4_minus_ref",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in sorted(grouped):
            vals = grouped[t]
            raw_k = [abs(v["kappa_nn_raw"] - v["kappa_ref"]) for v in vals]
            d4_k = [abs(v["kappa_nn_d4"] - v["kappa_ref"]) for v in vals]
            raw_h = [abs(v["hk_nn_raw"] - v["hk_ref"]) for v in vals]
            d4_h = [abs(v["hk_nn_d4"] - v["hk_ref"]) for v in vals]
            writer.writerow(
                {
                    "t": f"{t:.17g}",
                    "n_points_full_mirrored": len(vals),
                    "mean_abs_kappa_nn_raw_minus_ref": f"{sum(raw_k)/len(raw_k):.17g}",
                    "max_abs_kappa_nn_raw_minus_ref": f"{max(raw_k):.17g}",
                    "mean_abs_kappa_nn_d4_minus_ref": f"{sum(d4_k)/len(d4_k):.17g}",
                    "max_abs_kappa_nn_d4_minus_ref": f"{max(d4_k):.17g}",
                    "mean_abs_hk_nn_raw_minus_ref": f"{sum(raw_h)/len(raw_h):.17g}",
                    "max_abs_hk_nn_raw_minus_ref": f"{max(raw_h):.17g}",
                    "mean_abs_hk_nn_d4_minus_ref": f"{sum(d4_h)/len(d4_h):.17g}",
                    "max_abs_hk_nn_d4_minus_ref": f"{max(d4_h):.17g}",
                }
            )


def write_benchmark_error_csv(
    path: Path,
    method_rows: dict[str, list[dict[str, float | str]]],
    benchmark_rows: list[dict[str, float | str]],
) -> None:
    fields = [
        "method",
        "field",
        "mae_vs_benchmark",
        "rmse_vs_benchmark",
        "max_abs_vs_benchmark",
        "signed_error_final_vs_benchmark_last",
        "value_final",
        "benchmark_last",
        "method_t_final",
        "benchmark_t_last",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, rows in method_rows.items():
            for field in ("yc", "vc", "circ", "vol_rel"):
                diffs = [
                    float(row[field]) - interp_trace(benchmark_rows, float(row["t"]), field)
                    for row in rows
                    if float(row["t"]) >= float(benchmark_rows[0]["t"])
                    and float(row["t"]) <= float(benchmark_rows[-1]["t"])
                ]
                if not diffs:
                    continue
                final = rows[-1]
                benchmark_final = benchmark_rows[-1]
                final_value = float(final[field])
                benchmark_last = float(benchmark_final[field])
                writer.writerow(
                    {
                        "method": method,
                        "field": field,
                        "mae_vs_benchmark": f"{sum(abs(v) for v in diffs) / len(diffs):.17g}",
                        "rmse_vs_benchmark": f"{math.sqrt(sum(v * v for v in diffs) / len(diffs)):.17g}",
                        "max_abs_vs_benchmark": f"{max(abs(v) for v in diffs):.17g}",
                        "signed_error_final_vs_benchmark_last": f"{final_value - benchmark_last:.17g}",
                        "value_final": f"{final_value:.17g}",
                        "benchmark_last": f"{benchmark_last:.17g}",
                        "method_t_final": f"{float(final['t']):.17g}",
                        "benchmark_t_last": f"{float(benchmark_final['t']):.17g}",
                    }
                )


def plot_benchmark_error(
    path: Path,
    method_rows: dict[str, list[dict[str, float | str]]],
    benchmark_rows: list[dict[str, float | str]],
) -> None:
    # Native drawn first (lowest zorder) as the thin reference; NN methods sit
    # almost exactly on top of it, so distinct dash patterns (not just color)
    # are what makes the near-total overlap legible.
    panels = [
        ("yc", "centroid height", r"$y_c - y_{c,\mathrm{MooNMD}}$"),
        ("vc", "rise velocity", r"$v_c - v_{c,\mathrm{MooNMD}}$"),
        ("circ", "circularity", r"$c - c_{\mathrm{MooNMD}}$"),
        ("vol_rel", "relative volume drift", r"$\Delta V/V_0 - (\Delta V/V_0)_{\mathrm{MooNMD}}$"),
    ]

    with plt.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
        flat_axes = axes.ravel()
        for ax, (field, label, ylabel) in zip(flat_axes, panels):
            ax.axhline(0.0, color=COLOR_ZERO_LINE, lw=0.8, zorder=1)
            for method, rows in method_rows.items():
                xs: list[float] = []
                ys: list[float] = []
                for row in rows:
                    t = float(row["t"])
                    if t < float(benchmark_rows[0]["t"]) or t > float(benchmark_rows[-1]["t"]):
                        continue
                    xs.append(t)
                    ys.append(float(row[field]) - interp_trace(benchmark_rows, t, field))
                s = BENCHMARK_STYLE[method]
                ax.plot(
                    xs, ys,
                    color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["zorder"],
                    label=LEGEND_LABEL[method], solid_capstyle="round",
                )
            ax.set_title(label, loc="left", fontsize=8.6, color=COLOR_CORNER_LABEL, pad=4, fontweight="normal")
            ax.set_ylabel(ylabel)
            ax.margins(x=0.01)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

        for ax in flat_axes[2:]:
            ax.set_xlabel("time")

        fig.tight_layout(rect=(0, 0, 1, 0.93))

        handles, labels_ = flat_axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels_, loc="upper center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
        )
        fig.savefig(path)
        plt.close(fig)


def plot_panels(path: Path, rows: list[dict[str, float]], times: list[float]) -> None:
    plt.rcParams.update(MPL_STYLE)
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["t"]].append(row)

    fig, axes = plt.subplots(len(times), 1, figsize=(8.6, 2.05 * len(times)), sharex=True)
    if len(times) == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        vals = grouped[t]
        theta = [v["theta_deg"] for v in vals]
        ax.plot(theta, [v["kappa_ref"] for v in vals], color=COLOR_NATIVE, lw=1.1, label="CLSVOF-LS")
        ax.plot(theta, [v["kappa_nn_raw"] for v in vals], color=COLOR_NN_RAW, lw=1.3, ls="--", label="NN RAW")
        ax.plot(theta, [v["kappa_nn_d4"] for v in vals], color=COLOR_NN_D4, lw=1.3, ls=(0, (1, 1)), label="NN D4")
        ax.set_xlim(0, 360)
        ax.set_ylabel(r"$\kappa$")
        ax.text(
            0.012, 0.88, f"t = {t:g}", transform=ax.transAxes,
            fontsize=8.6, color=COLOR_CORNER_LABEL, ha="left", va="top",
        )
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
    )
    axes[-1].set_xlabel(r"$\theta$ (deg)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path)
    plt.close(fig)


def plot_error_panels(path: Path, rows: list[dict[str, float]], times: list[float]) -> None:
    plt.rcParams.update(MPL_STYLE)
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["t"]].append(row)

    fig, axes = plt.subplots(len(times), 1, figsize=(8.6, 1.75 * len(times)), sharex=True)
    if len(times) == 1:
        axes = [axes]
    for ax, t in zip(axes, times):
        vals = grouped[t]
        theta = [v["theta_deg"] for v in vals]
        ax.axhline(0, color=COLOR_ZERO_LINE, lw=0.8)
        ax.plot(theta, [v["kappa_nn_raw"] - v["kappa_ref"] for v in vals], color=COLOR_NN_RAW, lw=1.2, label="NN RAW")
        ax.plot(theta, [v["kappa_nn_d4"] - v["kappa_ref"] for v in vals], color=COLOR_NN_D4, lw=1.2, ls=(0, (1, 1)), label="NN D4")
        ax.set_xlim(0, 360)
        ax.set_ylabel(r"$\Delta\kappa$")
        ax.text(
            0.012, 0.86, f"t = {t:g}", transform=ax.transAxes,
            fontsize=8.6, color=COLOR_CORNER_LABEL, ha="left", va="top",
        )
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
    )
    axes[-1].set_xlabel(r"$\theta$ (deg)")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--times", default="0,1,2,3")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tem/rising_kappa_angle_probe")
    args = parser.parse_args()

    method_rows = {label: read_float_csv(args.run_dir / filename) for label, filename in METHOD_FILES.items()}
    benchmark_rows = read_benchmark_trace(args.benchmark)
    probe = read_float_csv(args.run_dir / "diagnostic_nn_probe.probe.csv")
    trace = read_float_csv(args.run_dir / "diagnostic_nn_probe.csv")
    all_rows = mirrored_theta_rows(probe, trace)
    requested = [float(v) for v in args.times.split(",") if v.strip()]
    available = sorted({row["t"] for row in all_rows})
    missing = [t for t in requested if t not in available]
    if missing:
        raise SystemExit(f"requested times not in probe data: {missing}; available={available}")

    selected = [row for row in all_rows if row["t"] in requested]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_angle_csv(args.out_dir / "rising_kappa_angle_N128_L7_full_t0_1_2_3.csv", selected)
    write_summary_csv(args.out_dir / "rising_kappa_angle_N128_L7_full_t0_1_2_3_summary.csv", selected)
    write_benchmark_error_csv(
        args.out_dir / "rising_benchmark_error_N128_L7_full_summary.csv",
        method_rows,
        benchmark_rows,
    )
    plot_benchmark_error(
        args.out_dir / "rising_benchmark_error_N128_L7_full.png",
        method_rows,
        benchmark_rows,
    )
    plot_panels(args.out_dir / "rising_kappa_angle_N128_L7_full_t0_1_2_3.png", selected, requested)
    plot_error_panels(args.out_dir / "rising_kappa_angle_error_N128_L7_full_t0_1_2_3.png", selected, requested)
    print(args.out_dir / "rising_benchmark_error_N128_L7_full.png")
    print(args.out_dir / "rising_benchmark_error_N128_L7_full_summary.csv")
    print(args.out_dir / "rising_kappa_angle_N128_L7_full_t0_1_2_3.png")
    print(args.out_dir / "rising_kappa_angle_error_N128_L7_full_t0_1_2_3.png")


if __name__ == "__main__":
    main()
