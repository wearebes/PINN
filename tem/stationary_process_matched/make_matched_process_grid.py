from __future__ import annotations

import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
(OUT_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT_DIR / "xdg-cache"))

import matplotlib as mpl

SOURCE_DATA = ROOT / "cfd_applications_cleanroom/results/source_data"

CASES = [
    {
        "label": "Circle",
        "short": "circle",
        "path": SOURCE_DATA
        / "stationary_curvature_process_20260703T134504Z_stationary_curvature_process_plate_source_data.csv",
    },
    {
        "label": "Ellipse E1 (5:4)",
        "short": "e1",
        "path": SOURCE_DATA
        / "stationary_ellipse_curvature_process_E1_20260703T170939Z_stationary_ellipse_curvature_process_e1_plate_source_data.csv",
    },
    {
        "label": "Ellipse E2 (3:2)",
        "short": "e2",
        "path": SOURCE_DATA
        / "stationary_ellipse_curvature_process_E2_20260703T170955Z_stationary_ellipse_curvature_process_e2_plate_source_data.csv",
    },
]

SNAPSHOT_LABELS = {
    0: "t/T = 0",
    1: "t/T = 1/3",
    2: "t/T = 2/3",
    3: "t/T = 1",
}

COLORS = {
    "native": "#4c566a",
    "nn": "#0072b2",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value == "" else float(value)


def tail_max(rows: list[dict[str, str]]) -> float:
    values = [as_float(row, "Ca") for row in rows if as_float(row, "Ca") > 0]
    if not values:
        return math.nan
    tail_start = max(0, int(len(values) * 0.8))
    return max(values[tail_start:] or values)


def trace_xy(rows: list[dict[str, str]], row_type: str, *, positive_only: bool) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if row["row_type"] == row_type]
    x_values: list[float] = []
    y_values: list[float] = []
    for row in selected:
        y = as_float(row, "Ca")
        if positive_only and y <= 0:
            continue
        x_values.append(as_float(row, "tau"))
        y_values.append(y)
    return x_values, y_values


def render(*, ca_scale: str) -> Path:
    if ca_scale not in {"linear", "log"}:
        raise ValueError(f"Unsupported Ca scale: {ca_scale}")

    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "matplotlib"))
    (OUT_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    case_rows = [(case, read_rows(case["path"])) for case in CASES]

    all_positive_ca = [
        as_float(row, "Ca")
        for _, rows in case_rows
        for row in rows
        if row["row_type"] in {"surface_tension_trace", "clsvof_ls_trace"}
        and as_float(row, "Ca") > 0
    ]
    if ca_scale == "log":
        ca_ylim = (min(all_positive_ca) * 0.72, max(all_positive_ca) * 1.25)
    else:
        ca_ylim = (0.0, max(all_positive_ca) * 1.08)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(10.2, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(
        len(CASES),
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.28],
        wspace=0.12,
        hspace=0.12,
    )

    snapshot_axes = []
    trace_axes = []
    for row_index, (case, rows) in enumerate(case_rows):
        curvature_rows = [row for row in rows if row["row_type"] == "curvature"]
        h_values = [
            value
            for row in curvature_rows
            for value in (as_float(row, "hk_native"), as_float(row, "hk_nn"))
        ]
        h_min, h_max = min(h_values), max(h_values)
        h_pad = max((h_max - h_min) * 0.08, 1e-4)

        for snapshot_index in range(4):
            ax = fig.add_subplot(grid[row_index, snapshot_index])
            snapshot_axes.append(ax)
            rows_for_snapshot = [
                row
                for row in curvature_rows
                if int(row["snapshot_index"]) == snapshot_index
            ]
            rows_for_snapshot.sort(key=lambda row: as_float(row, "theta_deg_360"))
            theta = [as_float(row, "theta_deg_360") for row in rows_for_snapshot]
            native = [as_float(row, "hk_native") for row in rows_for_snapshot]
            nn = [as_float(row, "hk_nn") for row in rows_for_snapshot]
            ax.scatter(
                theta,
                native,
                s=4,
                color=COLORS["native"],
                alpha=0.42,
                linewidths=0,
                label="CLSVOF-LS",
            )
            ax.scatter(
                theta,
                nn,
                s=4,
                color=COLORS["nn"],
                alpha=0.58,
                linewidths=0,
                label="NN",
            )
            ax.set_xlim(0, 360)
            ax.set_ylim(h_min - h_pad, h_max + h_pad)
            ax.set_xticks([0, 180, 360])
            if row_index == 0:
                ax.set_title(SNAPSHOT_LABELS[snapshot_index])
            if row_index == len(CASES) - 1:
                ax.set_xlabel("degree")
            else:
                ax.set_xticklabels([])
            if snapshot_index == 0:
                ax.set_ylabel(f"{case['label']}\nh*kappa")
            else:
                ax.set_yticklabels([])

        trace_ax = fig.add_subplot(grid[row_index, 4])
        trace_axes.append(trace_ax)
        positive_only = ca_scale == "log"
        nn_x, nn_y = trace_xy(rows, "surface_tension_trace", positive_only=positive_only)
        native_x, native_y = trace_xy(rows, "clsvof_ls_trace", positive_only=positive_only)
        trace_ax.plot(
            nn_x,
            nn_y,
            color=COLORS["nn"],
            linewidth=1.2,
            label=f"NN tail {tail_max([row for row in rows if row['row_type'] == 'surface_tension_trace']):.2e}",
        )
        trace_ax.plot(
            native_x,
            native_y,
            color=COLORS["native"],
            linewidth=1.2,
            label=f"CLSVOF-LS tail {tail_max([row for row in rows if row['row_type'] == 'clsvof_ls_trace']):.2e}",
        )
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            trace_ax.axvline(frac, color="#9aa1a8", linewidth=0.7, alpha=0.45)
        trace_ax.set_xlim(0, 1.0)
        trace_ax.set_ylim(*ca_ylim)
        trace_ax.set_yscale(ca_scale)
        if row_index == 0:
            trace_ax.set_title(f"Ca max ({ca_scale})")
        if row_index == len(CASES) - 1:
            trace_ax.set_xlabel("t/T")
        else:
            trace_ax.set_xticklabels([])
        trace_ax.set_ylabel(r"$Ca_{max}$")
        trace_ax.legend(loc="best", fontsize=6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"stationary_process_matched_grid_{ca_scale}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return Path(f"{stem}.png")


def main() -> None:
    for ca_scale in ("log", "linear"):
        print(render(ca_scale=ca_scale))


if __name__ == "__main__":
    main()
