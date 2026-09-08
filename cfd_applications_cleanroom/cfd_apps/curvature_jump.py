"""Source-data-backed curvature jump diagnostics for stationary field CSVs."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.hashes import sha256_file
from cfd_applications_cleanroom.cfd_apps.paths import MANIFESTS, RAW, RESULTS, ROOT, route_relative


class CurvatureJumpError(RuntimeError):
    pass


@dataclass(frozen=True)
class FieldRow:
    run_id: str
    benchmark: str
    case_id: str
    method: str
    level: int
    grid_n: int
    x: float
    y: float
    theta: float
    d: float
    abs_d_over_delta: float
    hk_native: float
    hk_nn: float
    delta_hk: float
    native_kappa: float
    kappa_nn: float
    delta_kappa: float
    kappa_nn_from_hk_over_delta: float
    scale_identity_error: float
    grid_i: int
    grid_j: int


@dataclass(frozen=True)
class Pair:
    a: FieldRow
    b: FieldRow
    neighbor_type: str
    is_true_grid_adjacency: bool


REQUIRED_COLUMNS = {
    "run_id",
    "benchmark",
    "case_id",
    "method",
    "level",
    "grid_n",
    "x",
    "y",
    "theta",
    "d",
    "abs_d_over_delta",
    "hk_native",
    "hk_nn",
    "delta_hk",
    "native_kappa",
    "kappa_nn",
    "delta_kappa",
    "kappa_nn_from_hk_over_delta",
    "scale_identity_error",
}
NUMERIC_COLUMNS = {
    "x",
    "y",
    "theta",
    "d",
    "abs_d_over_delta",
    "hk_native",
    "hk_nn",
    "delta_hk",
    "native_kappa",
    "kappa_nn",
    "delta_kappa",
    "kappa_nn_from_hk_over_delta",
    "scale_identity_error",
}
BANDS = {"force_band": 2.0, "interface_band": 1.0}
SIGNALS = ("hk_native", "hk_nn", "delta_hk")
STATS = ("mean", "rms", "p95", "p99", "max", "total_variation")
PAPER_DEFAULT_METHODS = ("NN27_RAW",)
DISPLAY_LABELS = {
    "NN27_RAW": "NN",
}
MODEL_LINES = {
    "NN27_RAW": "baseline_hgradient",
}
METHOD_DEFINITIONS = {
    "NN27_RAW": "baseline_hgradient single prediction",
}
ALL_GATE_IDS = (
    "JG-D1",
    "JG-D2",
    "JG-D3",
    "JG-D4",
    "JG-D5",
    "JG-D6",
    "JG-D7",
    "JG-D8",
    "JG-D9",
    "JG-D10",
    "JG-D11",
    "JG-D12",
    "JG-D13",
    "JG-D14",
    "JG-N1",
    "JG-N2",
    "JG-N3",
    "JG-N4",
    "JG-N5",
    "JG-N6",
    "JG-N7",
    "JG-N8",
    "JG-N9",
    "JG-F1",
    "JG-F2",
    "JG-F3",
    "JG-F4",
    "JG-F5",
    "JG-R1",
    "JG-R2",
    "JG-R3",
    "JG-R4",
    "JG-R5",
    "JG-R6",
    "JG-R7",
    "JG-R8",
    "JG-R9",
    "JG-R10",
    "JG-R11",
    "JG-R12",
)
SOURCE_FIELDS = [
    "run_id",
    "method",
    "band",
    "neighbor_type",
    "signal",
    "pair_index",
    "x0",
    "y0",
    "x1",
    "y1",
    "theta0_deg",
    "theta1_deg",
    "theta_mid_deg",
    "grid_i0",
    "grid_j0",
    "grid_i1",
    "grid_j1",
    "theta_gap_deg",
    "abs_d_over_delta0",
    "abs_d_over_delta1",
    "abs_d_over_delta_gap",
    "value0",
    "value1",
    "jump_abs",
    "jump_signed",
    "is_true_grid_adjacency",
]
PLOTTED_SOURCE_FIELDS = [
    "run_id",
    "panel",
    "plotted_quantity",
    "method",
    "method_label",
    "source_method",
    "source_signal",
    "band",
    "neighbor_type",
    "pair_index",
    "statistic",
    "x_value",
    "y_value",
    "value",
    "angle_source_deg",
    "angle_plot_deg",
    "symmetry_copy_id",
    "angle_bin_start_deg",
    "angle_bin_end_deg",
    "source_pair_count",
    "source_row_role",
    "angle_domain",
    "trace_note",
]


def load_curvature_field_csv(path: Path, *, run_id: str, method: str) -> list[FieldRow]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise CurvatureJumpError(f"JG-D2:missing_columns:{','.join(missing)}")
        rows = list(reader)
    if not rows:
        raise CurvatureJumpError("JG-N1:empty_curvature_field")

    parsed: list[FieldRow] = []
    levels: set[int] = set()
    grid_ns: set[int] = set()
    for raw in rows:
        if raw["run_id"] != run_id or raw["method"] != method:
            raise CurvatureJumpError("JG-D4:run_or_method_mismatch")
        if raw["benchmark"] != "stationary":
            raise CurvatureJumpError("JG-D4:benchmark_mismatch")
        if raw["case_id"] != "stationary_curvature_field":
            raise CurvatureJumpError("JG-D9:case_id_mismatch")
        try:
            level = int(raw["level"])
            grid_n = int(raw["grid_n"])
        except ValueError as exc:
            raise CurvatureJumpError("JG-D4:invalid_level_or_grid_n") from exc
        levels.add(level)
        grid_ns.add(grid_n)
        values: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            try:
                value = float(raw[column])
            except ValueError as exc:
                raise CurvatureJumpError(f"JG-D3:non_numeric:{column}") from exc
            if not math.isfinite(value):
                raise CurvatureJumpError(f"JG-D3:non_finite:{column}")
            values[column] = value
        if grid_n != 2**level:
            raise CurvatureJumpError("JG-D5:grid_n_level_mismatch")
        delta = 1.0 / grid_n
        grid_i = round(values["x"] / delta - 0.5)
        grid_j = round(values["y"] / delta - 0.5)
        if not (0 <= grid_i < grid_n and 0 <= grid_j < grid_n):
            raise CurvatureJumpError("JG-D6:grid_index_out_of_bounds")
        x_expected = (grid_i + 0.5) * delta
        y_expected = (grid_j + 0.5) * delta
        if abs(values["x"] - x_expected) > 1e-9 or abs(values["y"] - y_expected) > 1e-9:
            raise CurvatureJumpError("JG-D6:grid_round_trip_failed")
        if abs((values["hk_nn"] - values["hk_native"]) - values["delta_hk"]) > 1e-10:
            raise CurvatureJumpError("JG-D7:delta_hk_mismatch")
        if abs(values["hk_nn"] / delta - values["kappa_nn"]) > 1e-10:
            raise CurvatureJumpError("JG-D7:kappa_nn_mismatch")
        if abs(values["hk_nn"] / delta - values["kappa_nn_from_hk_over_delta"]) > 1e-10:
            raise CurvatureJumpError("JG-D7:kappa_nn_from_hk_mismatch")
        if abs((values["kappa_nn"] - values["native_kappa"]) - values["delta_kappa"]) > 1e-10:
            raise CurvatureJumpError("JG-D7:delta_kappa_mismatch")
        if abs(values["scale_identity_error"]) > 1e-12:
            raise CurvatureJumpError("JG-D8:scale_identity_error")
        parsed.append(
            FieldRow(
                run_id=raw["run_id"],
                benchmark=raw["benchmark"],
                case_id=raw["case_id"],
                method=raw["method"],
                level=level,
                grid_n=grid_n,
                x=values["x"],
                y=values["y"],
                theta=values["theta"],
                d=values["d"],
                abs_d_over_delta=values["abs_d_over_delta"],
                hk_native=values["hk_native"],
                hk_nn=values["hk_nn"],
                delta_hk=values["delta_hk"],
                native_kappa=values["native_kappa"],
                kappa_nn=values["kappa_nn"],
                delta_kappa=values["delta_kappa"],
                kappa_nn_from_hk_over_delta=values["kappa_nn_from_hk_over_delta"],
                scale_identity_error=values["scale_identity_error"],
                grid_i=grid_i,
                grid_j=grid_j,
            )
        )

    if len(levels) != 1 or len(grid_ns) != 1:
        raise CurvatureJumpError("JG-D4:level_or_grid_n_not_unique")
    return parsed


def grid_edge_pairs(rows: list[FieldRow], *, band: str) -> list[Pair]:
    filtered = _band_rows(rows, band)
    by_grid: dict[tuple[int, int], FieldRow] = {}
    for row in filtered:
        key = (row.grid_i, row.grid_j)
        if key in by_grid:
            raise CurvatureJumpError("JG-N9:duplicate_grid_row")
        by_grid[key] = row
    pairs: list[Pair] = []
    for grid_i, grid_j in sorted(by_grid):
        a = by_grid[(grid_i, grid_j)]
        for neighbor in ((grid_i + 1, grid_j), (grid_i, grid_j + 1)):
            b = by_grid.get(neighbor)
            if b is not None:
                pairs.append(Pair(a=a, b=b, neighbor_type="grid_edge", is_true_grid_adjacency=True))
    return pairs


def theta_order_pairs(rows: list[FieldRow], *, band: str) -> list[Pair]:
    ordered = sorted(_band_rows(rows, band), key=lambda row: (row.theta, row.x, row.y))
    return [
        Pair(a=ordered[index], b=ordered[index + 1], neighbor_type="theta_order", is_true_grid_adjacency=False)
        for index in range(max(0, len(ordered) - 1))
    ]


def summarize_jumps(rows: list[FieldRow], *, method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"method": method, "bands": list(BANDS)}
    source_rows: list[dict[str, Any]] = []
    for band in BANDS:
        summary[band] = {}
        builders = {
            "grid_edge": grid_edge_pairs(rows, band=band),
            "theta_order": theta_order_pairs(rows, band=band),
        }
        for neighbor_type, pairs in builders.items():
            block: dict[str, Any] = {}
            signal_stats: dict[str, dict[str, Any]] = {}
            for signal in SIGNALS:
                values = [_jump(pair, signal) for pair in pairs]
                stats = _stats(values)
                block[signal] = stats
                signal_stats[signal] = stats
                source_rows.extend(
                    _source_row(
                        pair=pair,
                        method=method,
                        band=band,
                        signal=signal,
                        pair_index=pair_index,
                    )
                    for pair_index, pair in enumerate(pairs)
                )
            block.update(_ratio_stats(signal_stats))
            summary[band][neighbor_type] = block
    return summary, source_rows


def write_curvature_jump_package(
    *,
    run_id: str,
    methods: list[str],
    raw_root: Path | None = None,
    results_root: Path | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    methods = list(methods) if methods else list(PAPER_DEFAULT_METHODS)
    raw_root = Path(raw_root) if raw_root is not None else RAW / "stationary" / run_id
    results_root = Path(results_root) if results_root is not None else RESULTS
    reports = results_root / "reports"
    figures = results_root / "figures"
    source_data_dir = results_root / "source_data"
    for directory in (reports, figures, source_data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = _artifact_paths(run_id=run_id, reports=reports, figures=figures, source_data_dir=source_data_dir)
    gate_results = {gate: "PASS" for gate in ALL_GATE_IDS}
    failed_gates: list[str] = []

    try:
        inputs = _resolve_inputs(run_id=run_id, methods=methods, raw_root=raw_root, level=level)
        _check_manifest(run_id=run_id, inputs=inputs)
        method_summaries: dict[str, Any] = {}
        all_source_rows: list[dict[str, Any]] = []
        levels = {int(data["level"]) for data in inputs.values()}
        if len(levels) != 1:
            raise CurvatureJumpError("JG-D10:multiple_levels_selected")
        for method, data in inputs.items():
            rows = load_curvature_field_csv(Path(data["path"]), run_id=run_id, method=method)
            _check_band_and_neighbor_gates(rows)
            summary, source_rows = summarize_jumps(rows, method=method)
            summary["level"] = rows[0].level
            summary["grid_n"] = rows[0].grid_n
            summary["input_csv"] = route_relative(Path(data["path"]))
            summary["display_label"] = _display_label(method)
            summary["model_line"] = _model_line(method)
            method_summaries[method] = summary
            all_source_rows.extend(source_rows)
        expected_source_rows = _expected_source_row_count(method_summaries)
        if len(all_source_rows) != expected_source_rows:
            raise CurvatureJumpError("JG-R10:source_row_count_mismatch")
        _write_source_data(paths["source_data_csv"], all_source_rows)
        plotted_source_rows = _build_plotted_source_rows(
            run_id=run_id,
            summaries=method_summaries,
            source_rows=all_source_rows,
            methods=list(method_summaries),
        )
        _write_plotted_source_data(paths["plotted_source_data_csv"], plotted_source_rows)
        figures_dict, png_variance, png_shape = _render_plate(
            run_id=run_id,
            summaries=method_summaries,
            source_rows=all_source_rows,
            plotted_source_rows=plotted_source_rows,
            figure_stem=paths["figure_stem"],
        )
        if png_variance <= 1e-8:
            raise CurvatureJumpError("JG-R6:blank_png")
        manifest = _write_manifest(
            run_id=run_id,
            inputs=inputs,
            paths=paths,
            figures_dict=figures_dict,
        )
        report = _summary_payload(
            run_id=run_id,
            methods=method_summaries,
            figures=figures_dict,
            paths=paths,
            manifest=manifest,
            gate_results=gate_results,
            failed_gates=[],
            overall_status="PASS",
            png_variance=png_variance,
            png_shape=png_shape,
            expected_source_rows=expected_source_rows,
            plotted_source_rows=len(plotted_source_rows),
        )
        _write_json(paths["summary_json"], report)
        _write_markdown(paths["summary_md"], report)
        return _return_payload(report)
    except Exception as exc:
        if isinstance(exc, CurvatureJumpError):
            failed_gates = [_gate_from_error(exc)]
        else:
            failed_gates = ["JG-F4"]
        if "JG-D1" in failed_gates:
            failed_gates.append("JG-F1")
        if "JG-D10" in failed_gates:
            failed_gates.append("JG-F2")
        if "JG-R10" in failed_gates:
            failed_gates.append("JG-F3")
        for gate in failed_gates:
            gate_results[gate] = "FAIL"
        failure = _summary_payload(
            run_id=run_id,
            methods={},
            figures={"png": "", "pdf": "", "svg": ""},
            paths=paths,
            manifest={},
            gate_results=gate_results,
            failed_gates=sorted(set(failed_gates)),
            overall_status="FAIL",
            png_variance=0.0,
            png_shape=[],
            expected_source_rows=0,
            plotted_source_rows=0,
            failure_reason=f"{type(exc).__name__}:{exc}",
        )
        _write_json(paths["summary_json"], failure)
        _write_markdown(paths["summary_md"], failure)
        return _return_payload(failure)


def _band_rows(rows: list[FieldRow], band: str) -> list[FieldRow]:
    try:
        limit = BANDS[band]
    except KeyError as exc:
        raise CurvatureJumpError(f"JG-N1:unknown_band:{band}") from exc
    return [row for row in rows if row.abs_d_over_delta <= limit]


def _jump(pair: Pair, signal: str) -> float:
    return abs(getattr(pair.b, signal) - getattr(pair.a, signal))


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "pair_count": 0,
            "jump_mean": None,
            "jump_rms": None,
            "jump_p50": None,
            "jump_p95": None,
            "jump_p99": None,
            "jump_max": None,
            "jump_total_variation": 0.0,
        }
    return {
        "pair_count": len(values),
        "jump_mean": sum(values) / len(values),
        "jump_rms": math.sqrt(sum(value * value for value in values) / len(values)),
        "jump_p50": _percentile(values, 0.50),
        "jump_p95": _percentile(values, 0.95),
        "jump_p99": _percentile(values, 0.99),
        "jump_max": max(values),
        "jump_total_variation": sum(values),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _ratio_stats(signal_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ratios: dict[str, Any] = {"ratio_null_reasons": []}
    for stat in STATS:
        native = signal_stats["hk_native"][f"jump_{stat}"]
        for signal, prefix in (
            ("hk_nn", "jump_ratio_nn_to_native"),
            ("delta_hk", "jump_ratio_delta_to_native"),
        ):
            numerator = signal_stats[signal][f"jump_{stat}"]
            key = f"{prefix}_{stat}"
            if native in (None, 0.0) or numerator is None:
                ratios[key] = None
                ratios["ratio_null_reasons"].append(f"{key}:zero_or_null_denominator")
            else:
                ratios[key] = numerator / native
    return ratios


def _source_row(*, pair: Pair, method: str, band: str, signal: str, pair_index: int) -> dict[str, Any]:
    value0 = getattr(pair.a, signal)
    value1 = getattr(pair.b, signal)
    theta0 = math.degrees(pair.a.theta)
    theta1 = math.degrees(pair.b.theta)
    return {
        "run_id": pair.a.run_id,
        "method": method,
        "band": band,
        "neighbor_type": pair.neighbor_type,
        "signal": signal,
        "pair_index": pair_index,
        "x0": pair.a.x,
        "y0": pair.a.y,
        "x1": pair.b.x,
        "y1": pair.b.y,
        "theta0_deg": theta0,
        "theta1_deg": theta1,
        "theta_mid_deg": 0.5 * (theta0 + theta1),
        "grid_i0": pair.a.grid_i,
        "grid_j0": pair.a.grid_j,
        "grid_i1": pair.b.grid_i,
        "grid_j1": pair.b.grid_j,
        "theta_gap_deg": abs(theta1 - theta0),
        "abs_d_over_delta0": pair.a.abs_d_over_delta,
        "abs_d_over_delta1": pair.b.abs_d_over_delta,
        "abs_d_over_delta_gap": abs(pair.b.abs_d_over_delta - pair.a.abs_d_over_delta),
        "value0": value0,
        "value1": value1,
        "jump_abs": abs(value1 - value0),
        "jump_signed": value1 - value0,
        "is_true_grid_adjacency": "true" if pair.is_true_grid_adjacency else "false",
    }


def _resolve_inputs(
    *, run_id: str, methods: list[str], raw_root: Path, level: int | None
) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for method in methods:
        pattern = f"{method}_L{level}/curvature_field.csv" if level is not None else f"{method}_L*/curvature_field.csv"
        matches = sorted(raw_root.glob(pattern))
        if not matches:
            raise CurvatureJumpError(f"JG-D1:missing_csv:{method}")
        if len(matches) != 1:
            levels = {_level_from_method_dir(path.parent.name) for path in matches}
            if level is None and len(levels) > 1:
                raise CurvatureJumpError(f"JG-D10:multiple_levels:{method}")
            raise CurvatureJumpError(f"JG-D11:multiple_csvs:{method}")
        selected = matches[0]
        selected_level = _level_from_method_dir(selected.parent.name)
        inputs[method] = {
            "method": method,
            "level": selected_level,
            "path": selected,
            "sha256": sha256_file(selected),
        }
    if len({int(data["level"]) for data in inputs.values()}) != 1:
        raise CurvatureJumpError("JG-D10:methods_have_different_levels")
    return inputs


def _level_from_method_dir(name: str) -> int:
    try:
        return int(name.rsplit("_L", 1)[1])
    except (IndexError, ValueError) as exc:
        raise CurvatureJumpError(f"JG-D11:cannot_parse_level:{name}") from exc


def _check_manifest(*, run_id: str, inputs: dict[str, dict[str, Any]]) -> None:
    manifest_path = MANIFESTS / f"{run_id}.manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("rows", [])
    for method, data in inputs.items():
        expected_path = Path(data["path"]).resolve()
        matches = [
            row for row in rows
            if row.get("method") == method
            and int(row.get("level", -1)) == int(data["level"])
            and Path(str(row.get("field_csv", ""))).resolve() == expected_path
            and row.get("evidence_level") == "controlled_diagnostic"
        ]
        if len(matches) != 1:
            raise CurvatureJumpError(f"JG-D12:manifest_row_mismatch:{method}")


def _check_band_and_neighbor_gates(rows: list[FieldRow]) -> None:
    for band, limit in BANDS.items():
        band_rows = [row for row in rows if row.abs_d_over_delta <= limit]
        if not band_rows:
            raise CurvatureJumpError("JG-N1:empty_force_band" if band == "force_band" else "JG-N2:empty_interface_band")
        theta_pairs = theta_order_pairs(rows, band=band)
        if len(theta_pairs) != len(band_rows) - 1:
            raise CurvatureJumpError("JG-N3:theta_pair_count")
        edge_pairs = grid_edge_pairs(rows, band=band)
        if not edge_pairs:
            raise CurvatureJumpError("JG-N5:no_grid_edges")
        for pair in edge_pairs:
            distance = abs(pair.a.grid_i - pair.b.grid_i) + abs(pair.a.grid_j - pair.b.grid_j)
            if distance != 1:
                raise CurvatureJumpError("JG-N4:not_manhattan_adjacent")
        if any(pair.is_true_grid_adjacency for pair in theta_pairs):
            raise CurvatureJumpError("JG-N8:theta_marked_as_grid")


def _write_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_plotted_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLOTTED_SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _build_plotted_source_rows(
    *,
    run_id: str,
    summaries: dict[str, Any],
    source_rows: list[dict[str, Any]],
    methods: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_ecdf_plotted_rows(run_id=run_id, source_rows=source_rows, methods=methods))
    rows.extend(_tail_table_plotted_rows(run_id=run_id, summaries=summaries, methods=methods))
    rows.extend(_angle_profile_plotted_rows(run_id=run_id, source_rows=source_rows, methods=methods))
    return rows


def _comparison_series(methods: list[str]) -> list[dict[str, str]]:
    if not methods:
        return []
    series = [
        {
            "method": "CLSVOF-LS",
            "method_label": "CLSVOF-LS",
            "source_method": methods[0],
            "source_signal": "hk_native",
            "trace_note": "same-host CLSVOF-LS native h*kappa jump",
        }
    ]
    for method in methods:
        series.append({
            "method": method,
            "method_label": _short_display_label(method),
            "source_method": method,
            "source_signal": "delta_hk",
            "trace_note": "NN correction jump relative to same-host CLSVOF-LS native",
        })
    return series


def _ecdf_plotted_rows(
    *, run_id: str, source_rows: list[dict[str, Any]], methods: list[str]
) -> list[dict[str, Any]]:
    plotted: list[dict[str, Any]] = []
    for series in _comparison_series(methods):
        jump_rows = _source_jump_rows(
            source_rows=source_rows,
            method=series["source_method"],
            band="force_band",
            neighbor_type="grid_edge",
            signal=series["source_signal"],
        )
        ordered = sorted(jump_rows, key=lambda row: (float(row["jump_abs"]), int(row["pair_index"])))
        count = len(ordered)
        for index, row in enumerate(ordered):
            value = float(row["jump_abs"])
            plotted.append(_plotted_row(
                run_id=run_id,
                panel="A",
                plotted_quantity="grid_neighbor_jump_ecdf",
                method=series["method"],
                method_label=series["method_label"],
                source_method=series["source_method"],
                source_signal=series["source_signal"],
                band="force_band",
                neighbor_type="grid_edge",
                pair_index=row["pair_index"],
                x_value=value,
                y_value=(index + 1) / count if count else "",
                value=value,
                source_pair_count=count,
                source_row_role="plotted",
                trace_note=series["trace_note"],
            ))
    return plotted


def _tail_table_plotted_rows(
    *, run_id: str, summaries: dict[str, Any], methods: list[str]
) -> list[dict[str, Any]]:
    plotted: list[dict[str, Any]] = []
    for series in _comparison_series(methods):
        summary = summaries[series["source_method"]]
        stats = summary["force_band"]["grid_edge"][series["source_signal"]]
        for statistic in ("p95", "p99", "max"):
            value = stats[f"jump_{statistic}"]
            plotted.append(_plotted_row(
                run_id=run_id,
                panel="B",
                plotted_quantity="grid_neighbor_tail_value",
                method=series["method"],
                method_label=series["method_label"],
                source_method=series["source_method"],
                source_signal=series["source_signal"],
                band="force_band",
                neighbor_type="grid_edge",
                statistic=statistic,
                value=value,
                source_pair_count=stats["pair_count"],
                source_row_role="plotted",
                trace_note=series["trace_note"],
            ))
    return plotted


def _angle_profile_plotted_rows(
    *, run_id: str, source_rows: list[dict[str, Any]], methods: list[str]
) -> list[dict[str, Any]]:
    plotted: list[dict[str, Any]] = []
    bin_size = 30.0
    for method in methods:
        label = _short_display_label(method)
        source = _source_jump_rows(
            source_rows=source_rows,
            method=method,
            band="force_band",
            neighbor_type="theta_order",
            signal="delta_hk",
        )
        expanded: list[dict[str, Any]] = []
        for row in source:
            theta_source = float(row["theta_mid_deg"])
            value = float(row["jump_abs"])
            for copy_id, theta_plot in enumerate(_expanded_thetas(theta_source)):
                record = {
                    "pair_index": row["pair_index"],
                    "theta_source": theta_source,
                    "theta_plot": theta_plot,
                    "copy_id": copy_id,
                    "value": value,
                }
                expanded.append(record)
                plotted.append(_plotted_row(
                    run_id=run_id,
                    panel="C",
                    plotted_quantity="angle_profile_underlying_expanded_point",
                    method=method,
                    method_label=label,
                    source_method=method,
                    source_signal="delta_hk",
                    band="force_band",
                    neighbor_type="theta_order",
                    pair_index=row["pair_index"],
                    x_value=theta_plot,
                    y_value=value,
                    value=value,
                    angle_source_deg=theta_source,
                    angle_plot_deg=theta_plot,
                    symmetry_copy_id=copy_id,
                    source_pair_count=len(source),
                    source_row_role="underlying",
                    angle_domain="quadrant_symmetry_expanded_0_360_degrees",
                    trace_note="theta -> theta, 180-theta, 180+theta, 360-theta",
                ))
        for start in [float(value) for value in range(0, 360, int(bin_size))]:
            end = start + bin_size
            in_bin = [
                record["value"]
                for record in expanded
                if start <= record["theta_plot"] < end
                or (end == 360.0 and record["theta_plot"] == 360.0)
            ]
            if not in_bin:
                continue
            value = _percentile([float(item) for item in in_bin], 0.95)
            plotted.append(_plotted_row(
                run_id=run_id,
                panel="C",
                plotted_quantity="angle_profile_binned_p95",
                method=method,
                method_label=label,
                source_method=method,
                source_signal="delta_hk",
                band="force_band",
                neighbor_type="theta_order",
                statistic="p95",
                x_value=0.5 * (start + end),
                y_value=value,
                value=value,
                angle_bin_start_deg=start,
                angle_bin_end_deg=end,
                source_pair_count=len(in_bin),
                source_row_role="plotted",
                angle_domain="quadrant_symmetry_expanded_0_360_degrees",
                trace_note="binned p95 of symmetry-expanded angle-order correction jumps",
            ))
    return plotted


def _plotted_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in PLOTTED_SOURCE_FIELDS}
    row.update(values)
    row.setdefault("angle_domain", "")
    return row


def _render_plate(
    *,
    run_id: str,
    summaries: dict[str, Any],
    source_rows: list[dict[str, Any]],
    plotted_source_rows: list[dict[str, Any]],
    figure_stem: Path,
) -> tuple[dict[str, str], float, list[int]]:
    mpl_config_dir = ROOT / "cfd_applications_cleanroom/results/tmp/matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    methods = list(summaries)
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_ecdf = fig.add_subplot(grid[:, 0])
    ax_tail = fig.add_subplot(grid[0, 1])
    ax_angle = fig.add_subplot(grid[1, 1])
    colors = {
        "CLSVOF-LS": "#5f6368",
        "NN": "#2b6ea6",
    }

    for series in _comparison_series(methods):
        rows = _plotted_panel_rows(
            plotted_source_rows,
            panel="A",
            plotted_quantity="grid_neighbor_jump_ecdf",
            method=series["method"],
        )
        if not rows:
            continue
        label = series["method_label"]
        legend_label = "CLSVOF-LS native" if label == "CLSVOF-LS" else f"{label} correction"
        ax_ecdf.step(
            [float(row["x_value"]) for row in rows],
            [float(row["y_value"]) for row in rows],
            where="post",
            linewidth=1.5,
            color=colors.get(label, "#333333"),
            label=legend_label,
        )
    ax_ecdf.set_title("A  Grid-neighbor jump distribution", loc="left", fontweight="bold")
    ax_ecdf.set_xlabel(r"Jump magnitude in $h\kappa$")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax_ecdf.legend(loc="lower right", fontsize=6.5, handlelength=1.8)
    ax_ecdf.grid(axis="x", color="#e6e6e6", linewidth=0.6)

    ax_tail.set_axis_off()
    ax_tail.set_title("B  Tail values (grid neighbors)", loc="left", fontweight="bold")
    ax_tail.text(0.58, 0.86, "p95", ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax_tail.text(0.78, 0.86, "p99", ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax_tail.text(0.96, 0.86, "max", ha="right", va="center", fontsize=6.5, fontweight="bold")
    row_y = {"CLSVOF-LS": 0.76, "NN": 0.60}
    for series in _comparison_series(methods):
        label = series["method_label"]
        rows = {
            row["statistic"]: row for row in _plotted_panel_rows(
                plotted_source_rows,
                panel="B",
                plotted_quantity="grid_neighbor_tail_value",
                method=series["method"],
            )
        }
        y = row_y.get(label, 0.12)
        ax_tail.plot([0.03, 0.11], [y, y], color=colors.get(label, "#333333"), linewidth=2.0, solid_capstyle="round")
        ax_tail.text(0.13, y, label, ha="left", va="center", fontsize=6.8)
        for x, statistic in ((0.58, "p95"), (0.78, "p99"), (0.96, "max")):
            value = rows.get(statistic, {}).get("value", "")
            ax_tail.text(x, y, _panel_fmt(value), ha="right", va="center", fontsize=6.5)
    ax_tail.set_xlim(0.0, 1.0)
    ax_tail.set_ylim(0.0, 1.0)

    for method in methods:
        label = _short_display_label(method)
        rows = _plotted_panel_rows(
            plotted_source_rows,
            panel="C",
            plotted_quantity="angle_profile_binned_p95",
            method=method,
        )
        if not rows:
            continue
        ax_angle.plot(
            [float(row["x_value"]) for row in rows],
            [float(row["y_value"]) for row in rows],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
            color=colors.get(label, "#333333"),
            label=label,
        )
    ax_angle.set_title("C  Angular correction tails", loc="left", fontweight="bold")
    ax_angle.set_xlabel("Angle (deg)")
    ax_angle.set_ylabel(r"p95 jump in $h\kappa$")
    ax_angle.set_xlim(0.0, 360.0)
    ax_angle.set_xticks([0, 90, 180, 270, 360])
    ax_angle.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax_angle.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax_angle.legend(loc="upper right", fontsize=6.5)
    ax_angle.text(
        0.0,
        -0.38,
        "0-360 deg symmetry expansion from measured 0-90 deg quadrant.",
        transform=ax_angle.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color="#444444",
    )

    exports = {
        "png": str(figure_stem.with_suffix(".png")),
        "pdf": str(figure_stem.with_suffix(".pdf")),
        "svg": str(figure_stem.with_suffix(".svg")),
    }
    fig.savefig(exports["svg"], bbox_inches="tight")
    fig.savefig(exports["pdf"], bbox_inches="tight")
    fig.savefig(exports["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    image = mpimg.imread(exports["png"])
    return {key: route_relative(Path(value)) for key, value in exports.items()}, float(image.var()), list(image.shape)


def _source_jump_rows(
    *,
    source_rows: list[dict[str, Any]],
    method: str,
    band: str,
    neighbor_type: str,
    signal: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in source_rows
        if row["method"] == method
        and row["band"] == band
        and row["neighbor_type"] == neighbor_type
        and row["signal"] == signal
    ]


def _plotted_panel_rows(
    rows: list[dict[str, Any]], *, panel: str, plotted_quantity: str, method: str
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["panel"] == panel
        and row["plotted_quantity"] == plotted_quantity
        and row["method"] == method
        and row["source_row_role"] == "plotted"
    ]


def _expanded_thetas(theta: float) -> tuple[float, float, float, float]:
    return (
        theta,
        180.0 - theta,
        180.0 + theta,
        360.0 - theta,
    )


def _write_manifest(
    *,
    run_id: str,
    inputs: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    figures_dict: dict[str, str],
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "artifact": "curvature-jump",
        "evidence_level": "controlled_diagnostic",
        "angle_domain": "quadrant_symmetry_expanded_0_360_degrees",
        "measured_angle_domain": "measured_quadrant_0_90_degrees",
        "angle_expansion_rule": "theta -> theta, 180-theta, 180+theta, 360-theta",
        "angle_domain_note": "The source CSV is measured on the quadrant host; the figure is a full-circle symmetry expansion, not an independent 360-degree full-domain export.",
        "method_definitions": {
            method: METHOD_DEFINITIONS.get(method, _model_line(method))
            for method in inputs
        },
        "input_csvs": [
            {
                "method": method,
                "display_label": _display_label(method),
                "model_line": _model_line(method),
                "level": data["level"],
                "path": route_relative(Path(data["path"])),
                "sha256": data["sha256"],
            }
            for method, data in inputs.items()
        ],
        "source_data_csv": route_relative(paths["source_data_csv"]),
        "source_data_sha256": sha256_file(paths["source_data_csv"]),
        "plotted_source_data_csv": route_relative(paths["plotted_source_data_csv"]),
        "plotted_source_data_sha256": sha256_file(paths["plotted_source_data_csv"]),
        "figure_contract": {
            "core_conclusion": "Under the same CLSVOF-LS host, NN correction jumps should be judged by grid-neighbor tail distributions.",
            "panel_a": "Grid-neighbor ECDF compares same-host CLSVOF-LS native h*kappa jumps against NN correction jumps.",
            "panel_b": "p95, p99, and max tail values for the grid-neighbor evidence; no mean statistic is used in the visible figure.",
            "panel_c": "Binned p95 angle profile for NN correction jumps after deterministic quadrant-symmetry expansion.",
        },
        "summary_json": route_relative(paths["summary_json"]),
        "summary_md": route_relative(paths["summary_md"]),
        "exports": figures_dict,
        "export_sha256": {
            key: sha256_file(ROOT / value) if not Path(value).is_absolute() else sha256_file(Path(value))
            for key, value in figures_dict.items()
        },
    }
    _write_json(paths["source_data_manifest"], manifest)
    return manifest


def _summary_payload(
    *,
    run_id: str,
    methods: dict[str, Any],
    figures: dict[str, str],
    paths: dict[str, Path],
    manifest: dict[str, Any],
    gate_results: dict[str, str],
    failed_gates: list[str],
    overall_status: str,
    png_variance: float,
    png_shape: list[int],
    expected_source_rows: int,
    plotted_source_rows: int,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark": "stationary",
        "artifact": "curvature-jump",
        "evidence_level": "controlled_diagnostic",
        "angle_domain": "quadrant_symmetry_expanded_0_360_degrees",
        "measured_angle_domain": "measured_quadrant_0_90_degrees",
        "angle_expansion_rule": "theta -> theta, 180-theta, 180+theta, 360-theta",
        "overall_status": overall_status,
        "failed_gates": failed_gates,
        "failure_reason": failure_reason,
        "gate_results": gate_results,
        "methods": methods,
        "figures": figures,
        "source_data_csv": route_relative(paths["source_data_csv"]),
        "plotted_source_data_csv": route_relative(paths["plotted_source_data_csv"]),
        "source_data_manifest": route_relative(paths["source_data_manifest"]),
        "summary_json": route_relative(paths["summary_json"]),
        "summary_md": route_relative(paths["summary_md"]),
        "png_variance": png_variance,
        "png_shape": png_shape,
        "expected_source_rows": expected_source_rows,
        "plotted_source_rows": plotted_source_rows,
        "manifest": manifest,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stationary Curvature Jump Diagnostic",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- evidence_level: `{report['evidence_level']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- summary_json: `{report['summary_json']}`",
        f"- source_data_csv: `{report['source_data_csv']}`",
        f"- plotted_source_data_csv: `{report['plotted_source_data_csv']}`",
        f"- source_data_manifest: `{report['source_data_manifest']}`",
        f"- figure_png: `{report['figures'].get('png', '')}`",
        "",
        "This is controlled_diagnostic evidence for local curvature-field roughness.",
        "It does not replace same-host stationary-bubble Ca evidence.",
        "VOF-HF remains a positive-control reference, not the research target replacement.",
        "Angle-domain note: the source CSV is measured on the quadrant host (0-90 degrees). The figure uses quadrant symmetry expansion to display 0-360 degrees; it is not an independent full-domain export.",
        "Publication figure note: the visible plate is a three-panel tail-evidence figure: grid-neighbor ECDF, p95/p99/max table, and binned angular p95 profile.",
        "",
        "## Gate Table",
        "",
        "| gate | status |",
        "|---|---|",
    ]
    for gate, status in report["gate_results"].items():
        lines.append(f"| {gate} | {status} |")
    if report["overall_status"] != "PASS":
        lines.extend([
            "",
            "No interpretation is reported because quality gates failed.",
            f"Failure reason: `{report.get('failure_reason', '')}`",
        ])
        path.write_text("\n".join(lines) + "\n")
        return
    lines.extend([
        "",
        "## Jump Summary",
        "",
        "| method | band | neighbor | signal | pairs | mean | rms | p50 | p95 | p99 | max | total variation | ratio to native mean | ratio to native p95 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method, summary in report["methods"].items():
        display = _short_display_label(method)
        for band in BANDS:
            for neighbor in ("grid_edge", "theta_order"):
                block = summary[band][neighbor]
                for signal in SIGNALS:
                    stats = block[signal]
                    if signal == "hk_native":
                        ratio_mean = "1.000000e+00"
                        ratio_p95 = "1.000000e+00"
                    elif signal == "hk_nn":
                        ratio_mean = _format_optional(block["jump_ratio_nn_to_native_mean"])
                        ratio_p95 = _format_optional(block["jump_ratio_nn_to_native_p95"])
                    else:
                        ratio_mean = _format_optional(block["jump_ratio_delta_to_native_mean"])
                        ratio_p95 = _format_optional(block["jump_ratio_delta_to_native_p95"])
                    lines.append(
                        f"| {display} | {band} | {neighbor} | {signal} | {stats['pair_count']} | "
                        f"{_fmt(stats['jump_mean'])} | {_fmt(stats['jump_rms'])} | {_fmt(stats['jump_p50'])} | "
                        f"{_fmt(stats['jump_p95'])} | {_fmt(stats['jump_p99'])} | {_fmt(stats['jump_max'])} | "
                        f"{_fmt(stats['jump_total_variation'])} | {ratio_mean} | {ratio_p95} |"
                    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This table compares the selected NN curvature closures against the same CLSVOF-LS native curvature field. Larger `delta_hk` jump ratios indicate stronger local roughness in the NN correction. This is a controlled diagnostic only; it must be read beside same-host CLSVOF-LS Ca evidence.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _display_label(method: str) -> str:
    return DISPLAY_LABELS.get(method, method)


def _short_display_label(method: str) -> str:
    return _display_label(method)


def _model_line(method: str) -> str:
    return MODEL_LINES.get(method, "unknown")


def _signal_label(signal: str, display: str) -> str:
    if signal == "hk_native":
        return "native"
    if signal == "hk_nn":
        return display
    return f"{display}-native"


def _delta_metric_text(*, summary: dict[str, Any], method: str) -> str:
    stats = summary["interface_band"]["theta_order"]["delta_hk"]
    label = _short_display_label(method)
    return "\n".join([
        "CLSVOF-LS p95=0",
        f"{label} p95={_panel_fmt(stats['jump_p95'])}",
        f"{label} p99={_panel_fmt(stats['jump_p99'])}",
        f"{label} max={_panel_fmt(stats['jump_max'])}",
    ])


def _ecdf_metric_text(
    *, summaries: dict[str, Any], methods: list[str], neighbor_type: str
) -> str:
    if not methods:
        return ""
    first = summaries[methods[0]]["force_band"][neighbor_type]["hk_native"]
    lines = [
        f"CLSVOF-LS p95={_panel_fmt(first['jump_p95'])}",
        f"CLSVOF-LS p99={_panel_fmt(first['jump_p99'])}",
        f"CLSVOF-LS max={_panel_fmt(first['jump_max'])}",
    ]
    for method in methods:
        stats = summaries[method]["force_band"][neighbor_type]["delta_hk"]
        label = _short_display_label(method)
        lines.extend([
            f"{label} p95={_panel_fmt(stats['jump_p95'])}",
            f"{label} p99={_panel_fmt(stats['jump_p99'])}",
            f"{label} max={_panel_fmt(stats['jump_max'])}",
        ])
    return "\n".join(lines)


def _bottom_metric_text(*, summary: dict[str, Any], method: str) -> str:
    native = summary["force_band"]["theta_order"]["hk_native"]
    correction = summary["force_band"]["theta_order"]["delta_hk"]
    label = _short_display_label(method)
    return "\n".join([
        f"CLSVOF-LS p95={_panel_fmt(native['jump_p95'])}",
        f"CLSVOF-LS p99={_panel_fmt(native['jump_p99'])}",
        f"CLSVOF-LS max={_panel_fmt(native['jump_max'])}",
        f"{label} p95={_panel_fmt(correction['jump_p95'])}",
        f"{label} p99={_panel_fmt(correction['jump_p99'])}",
        f"{label} max={_panel_fmt(correction['jump_max'])}",
    ])


def _panel_fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) < 1e-2:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.6e}"


def _format_optional(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.6e}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _artifact_paths(*, run_id: str, reports: Path, figures: Path, source_data_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": reports / f"{run_id}_curvature_jump_summary.json",
        "summary_md": reports / f"{run_id}_curvature_jump_summary.md",
        "source_data_csv": source_data_dir / f"{run_id}_curvature_jump_source_data.csv",
        "plotted_source_data_csv": source_data_dir / f"{run_id}_curvature_jump_plotted_source_data.csv",
        "source_data_manifest": source_data_dir / f"{run_id}_curvature_jump_manifest.json",
        "figure_stem": figures / f"{run_id}_curvature_jump_plate",
    }


def _return_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run_id"],
        "overall_status": report["overall_status"],
        "failed_gates": report["failed_gates"],
        "evidence_level": report["evidence_level"],
        "figures": report["figures"],
        "source_data_csv": report["source_data_csv"],
        "plotted_source_data_csv": report["plotted_source_data_csv"],
        "source_data_manifest": report["source_data_manifest"],
        "summary_json": report["summary_json"],
        "summary_md": report["summary_md"],
    }


def _expected_source_row_count(method_summaries: dict[str, Any]) -> int:
    count = 0
    for summary in method_summaries.values():
        for band in BANDS:
            for neighbor in ("grid_edge", "theta_order"):
                count += summary[band][neighbor]["hk_native"]["pair_count"] * len(SIGNALS)
    return count


def _gate_from_error(exc: CurvatureJumpError) -> str:
    message = str(exc)
    if message.startswith("JG-"):
        return message.split(":", 1)[0]
    return "JG-F3"
