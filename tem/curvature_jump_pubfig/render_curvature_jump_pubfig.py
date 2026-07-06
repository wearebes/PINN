"""Publication-grade re-rendering of the stationary curvature-jump diagnostic.

Reads the audited cleanroom artifacts (source_data CSV + summary JSON) for a
given curvature-jump run and re-renders the figure only. No pipeline numbers
are recomputed without being cross-checked against the audited summary.

Panels
  A  Survival distribution (CCDF, log-y) of grid-edge jump magnitudes in the
     force band: native h*kappa field vs NN / NND4 correction increments.
  B  Tail statistics p95 / p99 / max as a slopegraph (same data as the audited
     tail table, drawn instead of typeset).
  C  Arc-neighbour correction jumps vs interface angle, measured quadrant only
     (0-90 deg), scatter + binned p95. No symmetry expansion is drawn.

Outputs (out/ next to this script): PNG/PDF/SVG figure, plotted-data CSV,
provenance JSON with input hashes and cross-check results.

Usage:
  python tem/curvature_jump_pubfig/render_curvature_jump_pubfig.py \
      --run-id stationary_curvature_20260705T132529Z [--panel-a ecdf]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "cfd_applications_cleanroom" / "results"

MM = 1.0 / 25.4
SCALE = 1e3  # draw jumps in units of 1e-3

COLORS = {
    "native": "#7a7a7a",
    "NN27_RAW": "#0072B2",
    "NN27_D4": "#D55E00",
}
LABELS = {
    "native": "CLSVOF-LS native",
    "NN27_RAW": "NN correction",
    "NN27_D4": "NND4 correction",
}
SHORT = {"native": "native", "NN27_RAW": "NN", "NN27_D4": "NND4"}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def jumps(rows, *, method, band, neighbor, signal) -> np.ndarray:
    vals = [
        float(r["jump_abs"])
        for r in rows
        if r["method"] == method
        and r["band"] == band
        and r["neighbor_type"] == neighbor
        and r["signal"] == signal
    ]
    return np.asarray(vals, dtype=float)


def thetas_and_jumps(rows, *, method, band, neighbor, signal):
    pts = [
        (float(r["theta_mid_deg"]), float(r["jump_abs"]))
        for r in rows
        if r["method"] == method
        and r["band"] == band
        and r["neighbor_type"] == neighbor
        and r["signal"] == signal
    ]
    theta = np.asarray([p[0] for p in pts])
    jump = np.asarray([p[1] for p in pts])
    return theta, jump


def ccdf_steps(values: np.ndarray):
    """P(X >= x) evaluated at the sorted sample points, prefixed with (0, 1)."""
    x = np.sort(values)
    n = x.size
    y = (n - np.arange(n)) / n
    return np.concatenate([[0.0], x]), np.concatenate([[1.0], y])


def ecdf_steps(values: np.ndarray):
    x = np.sort(values)
    n = x.size
    y = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], y])


def crosscheck_tails(summary: dict, series_values: dict[str, np.ndarray]) -> dict:
    """Recomputed p95/p99/max must equal the audited summary values."""
    checks = {}
    for key, values in series_values.items():
        method, signal = key
        stats = summary["methods"][method]["force_band"]["grid_edge"][signal]
        for stat, official in (
            ("p95", stats["jump_p95"]),
            ("p99", stats["jump_p99"]),
            ("max", stats["jump_max"]),
        ):
            mine = (
                float(np.max(values))
                if stat == "max"
                else float(np.percentile(values, {"p95": 95, "p99": 99}[stat]))
            )
            ok = math.isclose(mine, official, rel_tol=1e-9, abs_tol=0.0)
            checks[f"{method}.{signal}.{stat}"] = {
                "recomputed": mine,
                "audited": official,
                "match": ok,
            }
            if not ok:
                raise SystemExit(
                    f"cross-check failed: {method}.{signal}.{stat} "
                    f"recomputed={mine!r} audited={official!r}"
                )
        if int(stats["pair_count"]) != values.size:
            raise SystemExit(f"cross-check failed: {method}.{signal} pair_count")
    return checks


def binned_p95(theta: np.ndarray, jump: np.ndarray, edges: np.ndarray):
    centers, values, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (theta >= lo) & (theta < hi) if hi < edges[-1] else (theta >= lo) & (theta <= hi)
        if not mask.any():
            continue
        centers.append(0.5 * (lo + hi))
        values.append(float(np.percentile(jump[mask], 95)))
        counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(values), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-a", choices=("ccdf", "ecdf"), default="ccdf")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "out"))
    args = parser.parse_args()

    run_id = args.run_id
    source_csv = RESULTS / "source_data" / f"{run_id}_curvature_jump_source_data.csv"
    summary_json = RESULTS / "reports" / f"{run_id}_curvature_jump_summary.json"
    manifest_json = RESULTS / "source_data" / f"{run_id}_curvature_jump_manifest.json"
    for path in (source_csv, summary_json):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    rows = load_source_rows(source_csv)
    summary = json.loads(summary_json.read_text())
    if summary.get("overall_status") != "PASS":
        raise SystemExit("refusing to plot: audited summary overall_status != PASS")

    # ---- Panel A/B series (force band, 4-neighbour grid edges) -------------
    native = jumps(rows, method="NN27_RAW", band="force_band", neighbor="grid_edge", signal="hk_native")
    native_d4_copy = jumps(rows, method="NN27_D4", band="force_band", neighbor="grid_edge", signal="hk_native")
    if not np.allclose(np.sort(native), np.sort(native_d4_copy)):
        raise SystemExit("native grid-edge jumps differ between method runs; single-snapshot assumption broken")
    nn = jumps(rows, method="NN27_RAW", band="force_band", neighbor="grid_edge", signal="delta_hk")
    d4 = jumps(rows, method="NN27_D4", band="force_band", neighbor="grid_edge", signal="delta_hk")

    checks = crosscheck_tails(
        summary,
        {
            ("NN27_RAW", "hk_native"): native,
            ("NN27_RAW", "delta_hk"): nn,
            ("NN27_D4", "delta_hk"): d4,
        },
    )

    tail_stats = {}
    for series_key, method, signal in (
        ("native", "NN27_RAW", "hk_native"),
        ("NN27_RAW", "NN27_RAW", "delta_hk"),
        ("NN27_D4", "NN27_D4", "delta_hk"),
    ):
        stats = summary["methods"][method]["force_band"]["grid_edge"][signal]
        tail_stats[series_key] = [stats["jump_p95"], stats["jump_p99"], stats["jump_max"]]

    # ---- Panel C series (force band, arc-order neighbours, measured 0-90) --
    theta_nn, arc_nn = thetas_and_jumps(rows, method="NN27_RAW", band="force_band", neighbor="theta_order", signal="delta_hk")
    theta_d4, arc_d4 = thetas_and_jumps(rows, method="NN27_D4", band="force_band", neighbor="theta_order", signal="delta_hk")
    theta_nat, arc_nat = thetas_and_jumps(rows, method="NN27_RAW", band="force_band", neighbor="theta_order", signal="hk_native")
    if theta_nn.min() < 0.0 or theta_nn.max() > 90.0:
        raise SystemExit("measured theta outside 0-90 quadrant")
    edges = np.arange(0.0, 90.0 + 1e-9, 15.0)
    cen_nn, p95_nn, n_nn = binned_p95(theta_nn, arc_nn, edges)
    cen_d4, p95_d4, n_d4 = binned_p95(theta_d4, arc_d4, edges)
    cen_nat, p95_nat, n_nat = binned_p95(theta_nat, arc_nat, edges)

    # ---- Figure -------------------------------------------------------------
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(183 * MM, 64 * MM))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.30, 0.80, 1.10],
        left=0.062, right=0.988, bottom=0.175, top=0.865, wspace=0.52,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plotted_rows: list[dict] = []

    # Panel A ---------------------------------------------------------------
    step_fn = ccdf_steps if args.panel_a == "ccdf" else ecdf_steps
    for key, values, z in (("native", native, 2.0), ("NN27_RAW", nn, 3.0), ("NN27_D4", d4, 3.1)):
        x, y = step_fn(values)
        ax_a.step(x * SCALE, y, where="post", color=COLORS[key], lw=1.6, zorder=z, label=LABELS[key])
        for xi, yi in zip(x, y):
            plotted_rows.append({
                "panel": "A", "series": SHORT[key], "element": f"{args.panel_a}_step",
                "x": xi * SCALE, "y": yi, "n": values.size,
            })
    ax_a.set_xlim(0, 2.8)
    ax_a.set_xlabel(r"Grid-edge jump magnitude ($\times 10^{-3}$)")
    if args.panel_a == "ccdf":
        ax_a.set_yscale("log")
        ax_a.set_ylim(2.8e-3, 1.15)
        ax_a.set_ylabel("Fraction of edges $\\geq$ jump")
        for frac, tag in ((0.05, "p95"), (0.01, "p99")):
            ax_a.axhline(frac, color="0.75", lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
            ax_a.text(2.74, frac * 1.18, tag, ha="right", va="bottom", fontsize=6.2, color="0.45")
        ax_a.legend(loc="lower left", frameon=False, handlelength=1.5, borderaxespad=0.2)
    else:
        ax_a.set_ylim(0, 1.02)
        ax_a.set_ylabel("ECDF")
        ax_a.legend(loc="lower right", frameon=False, handlelength=1.5)
    ax_a.set_title("Grid-edge jump distribution", pad=5)

    # Panel B ---------------------------------------------------------------
    xs = np.array([0.0, 1.0, 2.0])
    for key in ("native", "NN27_RAW", "NN27_D4"):
        vals = np.asarray(tail_stats[key]) * SCALE
        ax_b.plot(xs, vals, "-o", color=COLORS[key], lw=1.4, ms=3.6, mew=0, zorder=3)
        ax_b.annotate(
            f"{vals[2]:.2f}", (xs[2], vals[2]), xytext=(5, 0),
            textcoords="offset points", va="center", ha="left",
            fontsize=6.4, color=COLORS[key],
        )
        for stat, v in zip(("p95", "p99", "max"), tail_stats[key]):
            plotted_rows.append({
                "panel": "B", "series": SHORT[key], "element": f"tail_{stat}",
                "x": stat, "y": v * SCALE, "n": native.size,
            })
    ax_b.set_xlim(-0.35, 2.95)
    ax_b.set_xticks(xs, ["p95", "p99", "max"])
    ax_b.set_ylim(1.65, 2.75)
    ax_b.set_ylabel(r"Jump magnitude ($\times 10^{-3}$)")
    ax_b.set_title("Tail statistics", pad=5)

    # Panel C ---------------------------------------------------------------
    for key, theta, arc in (("NN27_RAW", theta_nn, arc_nn), ("NN27_D4", theta_d4, arc_d4)):
        ax_c.scatter(theta, arc * SCALE, s=5, color=COLORS[key], alpha=0.28, linewidths=0, zorder=2)
        for ti, ji in zip(theta, arc):
            plotted_rows.append({
                "panel": "C", "series": SHORT[key], "element": "arc_jump_point",
                "x": ti, "y": ji * SCALE, "n": theta.size,
            })
    ax_c.plot(cen_nat, p95_nat * SCALE, color=COLORS["native"], lw=1.2, ls=(0, (4, 2)), zorder=3, label="native p95")
    ax_c.plot(cen_nn, p95_nn * SCALE, "-o", color=COLORS["NN27_RAW"], lw=1.5, ms=3.4, mew=0, zorder=4, label="NN p95")
    ax_c.plot(cen_d4, p95_d4 * SCALE, "-o", color=COLORS["NN27_D4"], lw=1.5, ms=3.4, mew=0, zorder=4, label="NND4 p95")
    for series, cen, p95, cnt in (
        ("native", cen_nat, p95_nat, n_nat), ("NN", cen_nn, p95_nn, n_nn), ("NND4", cen_d4, p95_d4, n_d4),
    ):
        for c, v, n in zip(cen, p95, cnt):
            plotted_rows.append({
                "panel": "C", "series": series, "element": "binned_p95_15deg",
                "x": c, "y": v * SCALE, "n": n,
            })
    ax_c.set_xlim(0, 90)
    ax_c.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_c.set_ylim(0, 6.4)
    ax_c.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax_c.set_xlabel(r"Interface angle $\theta$ (deg)")
    ax_c.set_ylabel(r"Arc-neighbour jump ($\times 10^{-3}$)")
    ax_c.set_title("Jumps along the interface", pad=5)
    ax_c.legend(
        loc="upper center", frameon=False, ncols=3, handlelength=1.3,
        columnspacing=0.9, handletextpad=0.5, borderaxespad=0.1, fontsize=6.4,
    )

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        ax.text(-0.20, 1.09, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")

    # ---- Outputs ------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{run_id}_curvature_jump_pubfig"
    if args.panel_a == "ecdf":
        stem = out_dir / f"{run_id}_curvature_jump_pubfig_ecdfA"
    fig.savefig(f"{stem}.png", dpi=600)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)

    plotted_csv = Path(f"{stem}_plotted_data.csv")
    with plotted_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "series", "element", "x", "y", "n"])
        writer.writeheader()
        writer.writerows(plotted_rows)

    provenance = {
        "run_id": run_id,
        "panel_a_style": args.panel_a,
        "inputs": {
            "source_data_csv": str(source_csv.relative_to(REPO)),
            "source_data_sha256": sha256_file(source_csv),
            "summary_json": str(summary_json.relative_to(REPO)),
            "summary_sha256": sha256_file(summary_json),
            "manifest_json": str(manifest_json.relative_to(REPO)) if manifest_json.exists() else None,
        },
        "series_counts": {
            "grid_edge_native": int(native.size),
            "grid_edge_nn": int(nn.size),
            "grid_edge_nnd4": int(d4.size),
            "arc_nn": int(theta_nn.size),
            "arc_nnd4": int(theta_d4.size),
        },
        "angle_domain": "measured_quadrant_0_90_deg_only_no_symmetry_expansion",
        "tail_crosscheck": checks,
        "evidence_level": summary.get("evidence_level"),
        "plotted_rows": len(plotted_rows),
    }
    Path(f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"figure={stem}.png")
    print(f"plotted_data={plotted_csv}")
    print(f"crosscheck=PASS ({len(checks)} values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
