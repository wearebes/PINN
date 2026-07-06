from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from cfd_applications_cleanroom.cfd_apps.curvature_jump import (
    CurvatureJumpError,
    grid_edge_pairs,
    load_curvature_field_csv,
    summarize_jumps,
    theta_order_pairs,
    write_curvature_jump_package,
)


def _write_field_csv(path: Path, *, method: str = "NN27_RAW") -> None:
    rows = [
        # grid_n=4, h=0.25, cell centers are 0.125 and 0.375.
        (
            "r1",
            "stationary",
            "stationary_curvature_field",
            method,
            2,
            4,
            0.125,
            0.125,
            0.10,
            0.01,
            0.04,
            0.10,
            0.11,
            0.01,
            0.40,
            0.44,
            0.04,
            0.44,
            0.0,
            0.176,
        ),
        (
            "r1",
            "stationary",
            "stationary_curvature_field",
            method,
            2,
            4,
            0.375,
            0.125,
            0.20,
            0.02,
            0.08,
            0.20,
            0.24,
            0.04,
            0.80,
            0.96,
            0.16,
            0.96,
            0.0,
            0.768,
        ),
        (
            "r1",
            "stationary",
            "stationary_curvature_field",
            method,
            2,
            4,
            0.375,
            0.375,
            0.30,
            0.03,
            0.12,
            0.30,
            0.33,
            0.03,
            1.20,
            1.32,
            0.12,
            1.32,
            0.0,
            1.584,
        ),
    ]
    header = [
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
        "sign_product",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _replace_first(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_grid_edge_pairs_use_only_four_neighbor_edges(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    rows = load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")

    pairs = grid_edge_pairs(rows, band="force_band")

    endpoints = {
        ((pair.a.grid_i, pair.a.grid_j), (pair.b.grid_i, pair.b.grid_j))
        for pair in pairs
    }
    assert endpoints == {((0, 0), (1, 0)), ((1, 0), (1, 1))}
    assert all(pair.is_true_grid_adjacency for pair in pairs)


def test_theta_order_pairs_do_not_wrap_quadrant_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    rows = load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")

    pairs = theta_order_pairs(rows, band="force_band")

    assert len(pairs) == 2
    assert pairs[0].a.theta < pairs[0].b.theta
    assert pairs[-1].b.theta == pytest.approx(0.30)
    assert pairs[0].neighbor_type == "theta_order"
    assert pairs[0].is_true_grid_adjacency is False


def test_loader_rejects_wrong_case_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    _replace_first(csv_path, "stationary_curvature_field", "stationary_curvature_process")

    with pytest.raises(CurvatureJumpError, match="JG-D9"):
        load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")


def test_loader_rejects_grid_round_trip_failure(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    _replace_first(csv_path, ",0.375,0.125,", ",0.376,0.125,")

    with pytest.raises(CurvatureJumpError, match="JG-D6"):
        load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")


def test_loader_rejects_scale_identity_failure(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    _replace_first(csv_path, ",0.0,0.176", ",1e-6,0.176")

    with pytest.raises(CurvatureJumpError, match="JG-D8"):
        load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")


def test_summary_reports_tail_jump_statistics(tmp_path: Path) -> None:
    csv_path = tmp_path / "curvature_field.csv"
    _write_field_csv(csv_path)
    rows = load_curvature_field_csv(csv_path, run_id="r1", method="NN27_RAW")

    summary, source_rows = summarize_jumps(rows, method="NN27_RAW")

    row = summary["force_band"]["grid_edge"]["hk_nn"]
    assert row["pair_count"] == 2
    assert row["jump_mean"] == pytest.approx(0.11)
    assert row["jump_rms"] == pytest.approx(((0.13**2 + 0.09**2) / 2) ** 0.5)
    assert row["jump_p50"] == pytest.approx(0.11)
    assert row["jump_p95"] == pytest.approx(0.128)
    assert row["jump_p99"] == pytest.approx(0.1296)
    assert row["jump_max"] == pytest.approx(0.13)
    assert row["jump_total_variation"] == pytest.approx(0.22)
    assert summary["force_band"]["grid_edge"]["jump_ratio_nn_to_native_mean"] == pytest.approx(1.1)
    assert source_rows
    assert {
        "theta_gap_deg",
        "abs_d_over_delta_gap",
        "is_true_grid_adjacency",
    }.issubset(source_rows[0])


def test_write_curvature_jump_package_outputs_source_backed_nonblank_figures(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "stationary" / "r1"
    method_dir = input_root / "NN27_RAW_L2"
    method_dir.mkdir(parents=True)
    _write_field_csv(method_dir / "curvature_field.csv", method="NN27_RAW")

    result = write_curvature_jump_package(
        run_id="r1",
        methods=["NN27_RAW"],
        raw_root=input_root,
        results_root=tmp_path / "results",
    )

    assert result["overall_status"] == "PASS"
    for key in ("summary_json", "summary_md", "source_data_csv", "plotted_source_data_csv", "source_data_manifest"):
        assert Path(result[key]).exists()
    for key in ("png", "pdf", "svg"):
        path = Path(result["figures"][key])
        assert path.exists()
        assert path.stat().st_size > 1000

    image = mpimg.imread(result["figures"]["png"])
    assert image.size > 0
    assert float(image.var()) > 1e-8

    summary = json.loads(Path(result["summary_json"]).read_text())
    manifest = json.loads(Path(result["source_data_manifest"]).read_text())
    assert summary["png_variance"] > 1e-8
    assert summary["gate_results"]["JG-R6"] == "PASS"
    assert manifest["source_data_sha256"]
    assert manifest["plotted_source_data_sha256"]
    assert manifest["input_csvs"][0]["sha256"]


def test_write_curvature_jump_package_defaults_to_paper_nn_and_nnd4(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "stationary" / "r1"
    method_dir = input_root / "NN27_RAW_L2"
    method_dir.mkdir(parents=True)
    _write_field_csv(method_dir / "curvature_field.csv", method="NN27_RAW")

    result = write_curvature_jump_package(
        run_id="r1",
        methods=[],
        raw_root=input_root,
        results_root=tmp_path / "results",
    )

    assert result["overall_status"] == "FAIL"
    assert "JG-D1" in result["failed_gates"]
    assert "missing_csv:NN27_D4" in Path(result["summary_json"]).read_text()
    assert "PART2_D4" not in Path(result["summary_json"]).read_text()


def test_curvature_jump_plate_uses_publication_labels(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "stationary" / "r1"
    for method in ("NN27_RAW", "NN27_D4"):
        method_dir = input_root / f"{method}_L2"
        method_dir.mkdir(parents=True)
        _write_field_csv(method_dir / "curvature_field.csv", method=method)

    result = write_curvature_jump_package(
        run_id="r1",
        methods=["NN27_RAW", "NN27_D4"],
        raw_root=input_root,
        results_root=tmp_path / "results",
    )

    assert result["overall_status"] == "PASS"
    svg_text = Path(result["figures"]["svg"]).read_text()
    assert "Grid-neighbor jump distribution" in svg_text
    assert "Tail values (grid neighbors)" in svg_text
    assert "Angular correction tails" in svg_text
    assert "D4 = 8-transform avg" in svg_text
    assert "CLSVOF-LS" in svg_text
    assert "360" in svg_text
    assert "mean=" not in svg_text
    assert "p95" in svg_text
    assert "p99" in svg_text
    assert "max" in svg_text
    assert "NN correction jumps" not in svg_text
    assert "NND4 correction jumps" not in svg_text
    assert "NN27_RAW:" not in svg_text
    assert "NN27_D4:" not in svg_text
    assert "curvature jump diagnostic" not in svg_text
    assert "theta_order" not in svg_text
    assert "raw quadrant" not in svg_text
    report_text = Path(result["summary_md"]).read_text()
    assert "| NN |" in report_text
    assert "| NND4 |" in report_text
    assert "| NN27_RAW |" not in report_text
    assert "| NN27_D4 |" not in report_text
    summary = json.loads(Path(result["summary_json"]).read_text())
    assert summary["angle_domain"] == "quadrant_symmetry_expanded_0_360_degrees"
    assert summary["measured_angle_domain"] == "measured_quadrant_0_90_degrees"
    assert summary["plotted_source_rows"] > 0
    manifest = json.loads(Path(result["source_data_manifest"]).read_text())
    assert manifest["angle_domain"] == "quadrant_symmetry_expanded_0_360_degrees"
    assert manifest["measured_angle_domain"] == "measured_quadrant_0_90_degrees"
    assert manifest["method_definitions"]["NN27_D4"] == "baseline_hgradient D4 consensus: average of 8 transformed predictions"
    assert manifest["figure_contract"]["panel_a"].startswith("Grid-neighbor ECDF")

    plotted_rows = list(csv.DictReader(Path(result["plotted_source_data_csv"]).open()))
    assert {row["panel"] for row in plotted_rows} == {"A", "B", "C"}
    assert "angle_profile_underlying_expanded_point" in {row["plotted_quantity"] for row in plotted_rows}
    assert any(row["angle_plot_deg"] and float(row["angle_plot_deg"]) > 90.0 for row in plotted_rows)
    assert any(row["statistic"] == "p95" for row in plotted_rows)


def test_curvature_jump_package_renders_relax_public_labels(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "stationary" / "r1"
    method_dir = input_root / "NN27_RAW_RELAX_L2"
    method_dir.mkdir(parents=True)
    _write_field_csv(method_dir / "curvature_field.csv", method="NN27_RAW_RELAX")

    result = write_curvature_jump_package(
        run_id="r1",
        methods=["NN27_RAW_RELAX"],
        raw_root=input_root,
        results_root=tmp_path / "results",
    )

    assert result["overall_status"] == "PASS"
    manifest = json.loads(Path(result["source_data_manifest"]).read_text())
    assert manifest["method_definitions"]["NN27_RAW_RELAX"].startswith(
        "baseline_hgradient single prediction followed by"
    )
    report_text = Path(result["summary_md"]).read_text()
    assert "| NN+LR |" in report_text
    assert "| NN27_RAW_RELAX |" not in report_text


def test_write_curvature_jump_package_missing_method_writes_finite_failure_report(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "stationary" / "r1"
    method_dir = input_root / "NN27_RAW_L2"
    method_dir.mkdir(parents=True)
    _write_field_csv(method_dir / "curvature_field.csv", method="NN27_RAW")

    result = write_curvature_jump_package(
        run_id="r1",
        methods=["NN27_RAW", "NN27_D4"],
        raw_root=input_root,
        results_root=tmp_path / "results",
    )

    assert result["overall_status"] == "FAIL"
    assert "JG-D1" in result["failed_gates"]
    summary_text = Path(result["summary_json"]).read_text()
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    report_text = Path(result["summary_md"]).read_text()
    assert "No interpretation is reported because quality gates failed." in report_text
