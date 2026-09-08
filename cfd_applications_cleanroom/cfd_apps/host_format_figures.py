"""Publication figure for the VOF-HF host-format migration decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from cfd_applications_cleanroom.cfd_apps.host_format import classify_application
from cfd_applications_cleanroom.cfd_apps.paths import FIGURES, REPORTS, SOURCE_DATA, ensure_result_dirs, route_relative


CLSVOF_PARITY = REPORTS / "stationary_canary_20260629T201944Z_canary_summary.json"
NN_CANARY = REPORTS / "stationary_canary_20260630T005436Z_canary_summary.json"
METHOD_DISPLAY_LABELS = {
    "NN27_RAW": "NN",
}


def _latest_stock_report() -> Path:
    reports = sorted(REPORTS.glob("stationary_stock_*_stock_reference_summary.json"))
    if not reports:
        raise FileNotFoundError("missing_stationary_stock_reference_summary")
    return reports[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _max_ca_by_method(report: dict[str, Any], method: str) -> float:
    values = [float(row["Ca_tail_max"]) for row in report["summaries"] if row["method"] == method]
    if not values:
        raise ValueError(f"missing_method:{method}")
    return max(values)


def _build_source_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    stock_path = _latest_stock_report()
    stock = _load_json(stock_path)
    clsvof = _load_json(CLSVOF_PARITY)
    nn = _load_json(NN_CANARY)

    method_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for level, data in sorted(stock["levels"].items(), key=lambda item: int(item[0])):
        method_rows.append(
            {
                "display": f"VOF-HF L{level}",
                "method": "VOF_HF_NATIVE",
                "level": int(level),
                "Ca_tail_max": float(data["max_Ca_tail_max"]),
                "gate": 1.0e-12,
                "status": "PASS",
                "evidence": stock["run_id"],
            }
        )
    for row in stock["summaries"]:
        repeat_rows.append(
            {
                "method": "VOF_HF_NATIVE",
                "level": int(row["level"]),
                "repeat_id": row["repeat_id"],
                "Ca_tail_max": float(row["Ca_tail_max"]),
                "evidence": stock["run_id"],
            }
        )

    method_rows.append(
        {
            "display": "CLSVOF-LS native",
            "method": "CLSVOF_LS_NATIVE",
            "level": 6,
            "Ca_tail_max": _max_ca_by_method(clsvof, "CLSVOF_LS_NATIVE"),
            "gate": 1.0e-12,
            "status": "HOST_NOT_MIGRATED",
            "evidence": clsvof["run_id"],
        }
    )
    for method in ("NN27_RAW",):
        method_rows.append(
            {
                "display": METHOD_DISPLAY_LABELS[method],
                "method": method,
                "level": 6,
                "Ca_tail_max": _max_ca_by_method(nn, method),
                "gate": 1.0e-12,
                "status": nn["method_labels"][method],
                "evidence": nn["run_id"],
            }
        )

    gate_rows: list[dict[str, Any]] = []
    for application in ("stationary_bubble", "clsvof_ls_nn_static_bubble"):
        decision = classify_application(application)
        for gate, passed in decision["gates"].items():
            gate_rows.append(
                {
                    "application": application,
                    "gate": gate,
                    "passed": bool(passed),
                    "status": decision["status"],
                    "recommendation": decision["recommendation"],
                }
            )

    metadata = {
        "stock_report": route_relative(stock_path),
        "clsvof_report": route_relative(CLSVOF_PARITY),
        "nn_report": route_relative(NN_CANARY),
    }
    return method_rows + gate_rows, repeat_rows, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib() -> None:
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
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def build_figure() -> dict[str, str]:
    ensure_result_dirs()
    _configure_matplotlib()
    combined_rows, repeat_rows, metadata = _build_source_tables()
    method_rows = [row for row in combined_rows if "Ca_tail_max" in row]
    gate_rows = [row for row in combined_rows if "passed" in row]

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    method_csv = SOURCE_DATA / "vof_hf_host_format_methods.csv"
    gate_csv = SOURCE_DATA / "vof_hf_host_format_gates.csv"
    repeat_csv = SOURCE_DATA / "vof_hf_host_format_repeats.csv"
    _write_csv(method_csv, method_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(repeat_csv, repeat_rows)

    colors = {
        "VOF_HF_NATIVE": "#2A9D8F",
        "CLSVOF_LS_NATIVE": "#69707A",
        "NN27_RAW": "#D95F02",
    }
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 0.92])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, 1:])

    xs = list(range(len(method_rows)))
    ax_a.bar(
        xs,
        [row["Ca_tail_max"] for row in method_rows],
        color=[colors[row["method"]] for row in method_rows],
        width=0.72,
        edgecolor="#222222",
        linewidth=0.35,
    )
    ax_a.axhline(1.0e-12, color="#1F2937", linewidth=0.9, linestyle="--")
    ax_a.text(len(method_rows) - 0.15, 1.22e-12, "gate 1e-12", ha="right", va="bottom", color="#1F2937")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1.0e-16, 1.0e-4)
    ax_a.set_ylabel("tail maximum capillary number")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([row["display"] for row in method_rows], rotation=45, ha="right")
    ax_a.set_title("A  Stationary-bubble force balance", loc="left", fontweight="bold")
    for idx, row in enumerate(method_rows):
        if row["status"] == "PASS":
            ax_a.text(idx, 1.6e-14, "PASS", ha="center", va="bottom", fontsize=6, rotation=90, color=colors[row["method"]])
        else:
            ax_a.text(idx, 6.0e-5, "blocked", ha="center", va="top", fontsize=6, rotation=90, color="white")

    applications = ["stationary_bubble", "clsvof_ls_nn_static_bubble"]
    gates = ["HF-G0", "HF-G1", "HF-G2", "HF-G3", "HF-G4"]
    matrix = []
    for application in applications:
        by_gate = {row["gate"]: row["passed"] for row in gate_rows if row["application"] == application}
        matrix.append([1 if by_gate[gate] else 0 for gate in gates])
    ax_b.imshow(matrix, cmap=ListedColormap(["#F1C7BF", "#B8DCC7"]), vmin=0, vmax=1, aspect="auto")
    ax_b.set_xticks(range(len(gates)))
    ax_b.set_xticklabels(gates)
    ax_b.set_yticks(range(len(applications)))
    ax_b.set_yticklabels(["VOF-HF candidate", "CLSVOF-LS + NN"])
    ax_b.set_title("B  Host-format gates", loc="left", fontweight="bold")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax_b.text(x, y, "pass" if value else "block", ha="center", va="center", fontsize=6)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.tick_params(length=0)

    for level in sorted({row["level"] for row in repeat_rows}):
        vals = [row["Ca_tail_max"] for row in repeat_rows if row["level"] == level]
        ax_c.scatter([level] * len(vals), vals, s=28, color="#2A9D8F", edgecolor="#222222", linewidth=0.35, zorder=3)
        ax_c.plot([level - 0.18, level + 0.18], [max(vals), max(vals)], color="#2A9D8F", linewidth=1.0)
    ax_c.axhline(1.0e-12, color="#1F2937", linewidth=0.9, linestyle="--")
    ax_c.set_yscale("log")
    ax_c.set_ylim(1.0e-16, 2.0e-12)
    ax_c.set_xticks([5, 6, 7])
    ax_c.set_xlabel("Basilisk level")
    ax_c.set_ylabel("VOF-HF repeat Ca tail max")
    ax_c.set_title("C  Repeat-3 VOF-HF evidence", loc="left", fontweight="bold")

    fig.suptitle(
        "VOF-HF succeeds as a host-format positive control, not as an NN curvature repair",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    base = FIGURES / "vof_hf_host_format_publication_plate"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": route_relative(base.with_suffix(".png")),
        "source_data_methods": route_relative(method_csv),
        "source_data_gates": route_relative(gate_csv),
        "source_data_repeats": route_relative(repeat_csv),
        **metadata,
    }
    manifest_path = SOURCE_DATA / "vof_hf_host_format_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest"] = route_relative(manifest_path)
    return manifest


def main() -> int:
    manifest = build_figure()
    print(f"figure={manifest['figure']}")
    print(f"source_data_methods={manifest['source_data_methods']}")
    print(f"source_data_gates={manifest['source_data_gates']}")
    print(f"source_data_repeats={manifest['source_data_repeats']}")
    print(f"manifest={manifest['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
