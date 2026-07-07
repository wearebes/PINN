from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import (
    render_curvature_process_plate,
    write_curvature_process_source_data,
)
from cfd_applications_cleanroom.cfd_apps.stationary import run_canary, run_curvature_process


ROOT = Path("/Users/jcy/research/PINN")
REPORT_DIR = ROOT / "Experiment/report/stationary bubble"
RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def last_trace_value(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    return rows[-1]


def max_trace_ca(path: Path) -> float:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    return max(float(row["Ca"]) for row in rows)


def clsvof_native_summary_path(run_id: str, level: int) -> Path:
    return RAW / run_id / f"CLSVOF_LS_NATIVE_L{level}_r0" / "summary.json"


def load_nn_process_summaries(run_id: str, levels: tuple[int, ...]) -> list[dict[str, object]]:
    run_root = RAW / run_id
    summaries: list[dict[str, object]] = []
    for level in levels:
        summary_path = run_root / f"NN27_RAW_L{level}" / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"missing NN27_RAW summary for level {level}: {summary_path}")
        summaries.append(json.loads(summary_path.read_text()))
    return summaries


def clsvof_native_traces(run_id: str, levels: tuple[int, ...]) -> dict[int, Path]:
    run_root = RAW / run_id
    native_by_level: dict[int, Path] = {}
    for level in levels:
        native_trace = run_root / f"CLSVOF_LS_NATIVE_L{level}_r0" / "stationary_trace.csv"
        if not native_trace.exists():
            raise RuntimeError(f"missing CLSVOF-LS native trace for level {level}: {native_trace}")
        native_by_level[level] = native_trace
    return native_by_level


def build_plot_data(process_report: dict, native_report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    levels = (6, 7, 8)
    summaries = load_nn_process_summaries(process_report["run_id"], levels)
    native_by_level = clsvof_native_traces(native_report["run_id"], levels)

    fieldnames = [
        "row_type",
        "grid_n",
        "level",
        "benchmark_family",
        "case_id",
        "geometry_family",
        "geometry_type",
        "geometry_motion",
        "geometry_parameters",
        "geometry_role",
        "method",
        "series",
        "run_id",
        "source_csv",
        "report_png",
        "status",
        "report_ready",
        "mirror_index",
        "snapshot_index",
        "snapshot_fraction",
        "t",
        "tau",
        "theta_deg_360",
        "hk_native",
        "hk_nn",
        "delta_hk",
        "Ca",
        "sigma",
        "method_final_Ca",
        "method_max_Ca",
        "method_current_Ca",
        "method_current_tau",
        "nn_final_Ca",
        "nn_max_Ca",
        "nn_tail_mean_Ca",
        "nn_tail_max_Ca",
        "clsvof_ls_final_Ca",
        "clsvof_ls_max_Ca",
        "clsvof_ls_current_Ca",
        "clsvof_ls_current_tau",
        "clsvof_ls_status",
        "clsvof_ls_reached_final_time",
        "clsvof_ls_repeat_count",
        "clsvof_ls_repeat_consistency",
        "clsvof_ls_trace_csv",
        "source_note",
    ]
    all_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    report_names = {
        64: "stationarybuubble_64.png",
        128: "stationarybubble_128.png",
        256: "stationarybubble_256.png",
    }
    for summary in summaries:
        level = int(summary["level"])
        grid_n = int(summary["grid_n"])
        process_csv = Path(summary["curvature_process_csv"])
        trace_csv = Path(summary["surface_tension_trace_csv"])
        native_trace = native_by_level.get(level)
        if native_trace is None or not native_trace.exists():
            raise RuntimeError(f"missing CLSVOF-LS native trace for level {level}")
        native_summary = clsvof_native_summary_path(native_report["run_id"], level)
        native_complete = native_summary.exists()
        expanded = write_curvature_process_source_data(
            source_data_csv=ROOT / "tem/stationary_source_rerun" / f"expanded_L{level}.csv",
            process_records=read_csv(process_csv),
            trace_records=read_csv(trace_csv),
            clsvof_trace_records=read_csv(native_trace),
        )
        report_png = f"Experiment/report/stationary bubble/{report_names[grid_n]}"
        native_last = last_trace_value(native_trace)
        native_max_ca = max_trace_ca(native_trace)
        native_tau = float(native_last["tau"])
        native_status = "complete" if native_complete else "partial_trace_no_summary"
        row_status = "complete" if native_complete else "partial_clsvof_ls_trace"
        report_ready = "true" if native_complete else "false"
        clsvof_final_ca = native_last["Ca"] if native_complete else ""
        clsvof_max_ca = native_max_ca if native_complete else ""
        source_note = (
            "complete plot source data; can redraw the report plate from this CSV alone"
            if native_complete
            else "interim plot source data; CLSVOF-LS r0 is still running and only current/partial values are populated"
        )
        base_meta = {
            "grid_n": grid_n,
            "level": level,
            "benchmark_family": "stationary_bubble",
            "case_id": "stationary_curvature_process",
            "geometry_family": "circle",
            "geometry_type": "single_stationary_circle",
            "geometry_motion": "stationary",
            "geometry_parameters": "radius=0.4; center=(0.5,0.5)",
            "geometry_role": "baseline_circle_for_stationary_bubble_resolution_sweep",
            "run_id": summary["run_id"],
            "report_png": report_png,
            "status": row_status,
            "report_ready": report_ready,
            "nn_final_Ca": summary["Ca_final"],
            "nn_max_Ca": summary["Ca_max"],
            "nn_tail_mean_Ca": summary["Ca_tail_mean"],
            "nn_tail_max_Ca": summary["Ca_tail_max"],
            "clsvof_ls_final_Ca": clsvof_final_ca,
            "clsvof_ls_max_Ca": clsvof_max_ca,
            "clsvof_ls_current_Ca": native_last["Ca"],
            "clsvof_ls_current_tau": native_last["tau"],
            "clsvof_ls_status": native_status,
            "clsvof_ls_reached_final_time": str(native_complete).lower(),
            "clsvof_ls_repeat_count": 1,
            "clsvof_ls_repeat_consistency": "not_evaluated",
            "clsvof_ls_trace_csv": str(native_trace.relative_to(ROOT)),
            "source_note": source_note,
        }
        index_rows.append(
            {
                **base_meta,
                "row_type": "index",
                "method": summary["method"],
                "series": "metadata",
                "source_csv": str(process_csv.relative_to(ROOT)),
                "method_final_Ca": summary["Ca_final"],
                "method_max_Ca": summary["Ca_max"],
                "method_current_Ca": summary["Ca_final"],
                "method_current_tau": summary["final_tau"],
                "mirror_index": "",
                "snapshot_index": "",
                "snapshot_fraction": "",
                "t": summary["final_t"],
                "tau": summary["final_tau"],
                "theta_deg_360": "",
                "hk_native": "",
                "hk_nn": "",
                "delta_hk": "",
                "Ca": summary["Ca_final"],
                "sigma": summary["sigma"],
            }
        )
        index_rows.append(
            {
                **base_meta,
                "row_type": "index",
                "method": "CLSVOF_LS_NATIVE",
                "series": "metadata",
                "source_csv": str(native_trace.relative_to(ROOT)),
                "method_final_Ca": native_last["Ca"] if native_complete else "",
                "method_max_Ca": native_max_ca if native_complete else "",
                "method_current_Ca": native_last["Ca"],
                "method_current_tau": native_last["tau"],
                "mirror_index": "",
                "snapshot_index": "",
                "snapshot_fraction": "",
                "t": native_last["t"],
                "tau": native_last["tau"],
                "theta_deg_360": "",
                "hk_native": "",
                "hk_nn": "",
                "delta_hk": "",
                "Ca": native_last["Ca"],
                "sigma": "",
            }
        )
        for row in expanded:
            series = {
                "curvature": "angle_resolved_curvature",
                "surface_tension_trace": "NN",
                "clsvof_ls_trace": "CLSVOF-LS",
            }[row["row_type"]]
            method = summary["method"] if row["row_type"] != "clsvof_ls_trace" else "CLSVOF_LS_NATIVE"
            method_final_ca = summary["Ca_final"] if method == summary["method"] else (native_last["Ca"] if native_complete else "")
            method_max_ca = summary["Ca_max"] if method == summary["method"] else (native_max_ca if native_complete else "")
            method_current_ca = summary["Ca_final"] if method == summary["method"] else native_last["Ca"]
            method_current_tau = summary["final_tau"] if method == summary["method"] else native_last["tau"]
            source_csv = native_trace if row["row_type"] == "clsvof_ls_trace" else trace_csv
            if row["row_type"] == "curvature":
                source_csv = process_csv
            all_rows.append(
                {
                    **base_meta,
                    "row_type": row["row_type"],
                    "method": method,
                    "series": series,
                    "source_csv": str(source_csv.relative_to(ROOT)),
                    "method_final_Ca": method_final_ca,
                    "method_max_Ca": method_max_ca,
                    "method_current_Ca": method_current_ca,
                    "method_current_tau": method_current_tau,
                    "mirror_index": row["mirror_index"],
                    "snapshot_index": row["snapshot_index"],
                    "snapshot_fraction": row["snapshot_fraction"],
                    "t": row["t"],
                    "tau": row["tau"],
                    "theta_deg_360": row["theta_deg_360"],
                    "hk_native": row["hk_native"],
                    "hk_nn": row["hk_nn"],
                    "delta_hk": row["delta_hk"],
                    "Ca": row["Ca"],
                    "sigma": row["sigma"],
                }
            )
        figure = render_curvature_process_plate(
            run_id=f"{summary['run_id']}_L{level}",
            summaries=[summary],
            artifact_prefix="stationary_curvature_process",
            native_trace_csv_override=native_trace,
        )
        shutil.copyfile(figure["png"], ROOT / report_png)

    with (REPORT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows + all_rows)
    with (REPORT_DIR / "summary_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)
    manifest = {
        "process_run_id": process_report["run_id"],
        "native_run_id": native_report["run_id"],
        "clsvof_ls_repeat_count": 1,
        "clsvof_ls_repeat_ids_used": ["0"],
        "clsvof_ls_repeat_consistency": "not_evaluated",
        "clsvof_ls_repeat_note": "Report plates and plot-source CSV use r0 only; additional repeats are canary consistency checks, not required for the circle figure source data.",
        "status_by_grid": {
            str(int(summary["grid_n"])): {
                "nn_status": "complete",
                "clsvof_ls_status": "complete"
                if clsvof_native_summary_path(native_report["run_id"], int(summary["level"])).exists()
                else "partial_trace_no_summary",
            }
            for summary in summaries
        },
        "interim_report": any(
            not clsvof_native_summary_path(native_report["run_id"], int(summary["level"])).exists()
            for summary in summaries
        ),
        "summary_csv": str((REPORT_DIR / "summary.csv").relative_to(ROOT)),
        "summary_index_csv": str((REPORT_DIR / "summary_index.csv").relative_to(ROOT)),
    }
    (REPORT_DIR / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    process_report = run_curvature_process(methods=["NN27_RAW"], levels=[6, 7, 8])
    native_report = run_canary(methods=["CLSVOF_LS_NATIVE"], levels=[6, 7, 8], repeat=3)
    build_plot_data(process_report, native_report)


if __name__ == "__main__":
    main()
