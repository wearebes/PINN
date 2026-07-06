from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STATIONARY = ROOT / "cfd_applications_cleanroom/cfd_apps/stationary.py"
CURVATURE_SHARED = ROOT / "cfd_applications_cleanroom/cfd_apps/curvature_diagnostic_shared.py"
STOCK_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_vof_hf_stock_single.c"
NATIVE_CLSVOF_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_clsvof_native.c"
NN_CLSVOF_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_clsvof_nn.c"
CURVATURE_FIELD_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_curvature_field.c"
CURVATURE_PROCESS_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_curvature_process.c"
CONFIG = ROOT / "cfd_applications_cleanroom/configs/stationary_bubble.yaml"
ACCEPTED_MATRIX_CONFIG = ROOT / "cfd_applications_cleanroom/configs/accepted_matrix.yaml"
METHODS_CONFIG = ROOT / "cfd_applications_cleanroom/configs/methods.yaml"
CLI = ROOT / "cfd_applications_cleanroom/cfd_apps/cli.py"


def test_stock_compile_uses_case_default_for_string_case_id() -> None:
    runner = STATIONARY.read_text()
    case_source = STOCK_CASE.read_text()

    assert '# define CASE_ID "stationary_vof_hf_stock_single"' in case_source
    assert "-DCASE_ID" not in runner


def test_stock_repeat_metadata_is_runtime_not_binary_specific() -> None:
    runner = STATIONARY.read_text()
    case_source = STOCK_CASE.read_text()

    assert "CLEANROOM_RUN_ID" in case_source
    assert "CLEANROOM_REPEAT_ID" in case_source
    assert "-DRUN_ID" not in runner
    assert "-DREPEAT_ID" not in runner


def test_stock_reference_logging_keeps_schema_without_per_row_flush() -> None:
    case_source = STOCK_CASE.read_text()

    assert "stationary_trace.csv" in case_source
    assert "primary_metric_name,primary_metric_value" in case_source
    assert "i % 4096 == 0" in case_source


def test_stationary_canary_cli_dispatch_exists() -> None:
    cli = CLI.read_text()
    runner = STATIONARY.read_text()

    assert 'args.benchmark == "stationary" and args.tier == "canary"' in cli
    assert "run_canary" in cli
    assert "def run_canary" in runner
    assert "CLSVOF_LS_NATIVE" in runner
    assert "NN_DISABLE" in runner
    assert "NN_PROBE_ONLY" in runner


def test_stationary_nn_canary_methods_are_dispatchable() -> None:
    cli = CLI.read_text()
    runner = STATIONARY.read_text()
    shared = CURVATURE_SHARED.read_text()

    assert '"NN27_RAW"' in cli
    assert '"NN27_D4"' in cli
    assert 'PAPER_DEPLOYABLE_METHODS = ("NN27_RAW", "NN27_D4")' in runner
    assert '"NN_DISABLE": "nn27_r128"' in shared
    assert '"NN_PROBE_ONLY": "nn27_r128"' in shared
    assert '"NN27_D4": "nn27_r128"' in shared
    assert '"PART2_D4": "part2_wd0"' not in shared
    assert "stationary_nn_not_worse" in runner
    assert "stationary_nn_worse_than_native" in runner
    assert "stationary_blocked_by_parity" in runner


def test_relax_methods_are_not_paper_defaults() -> None:
    runner = STATIONARY.read_text()
    cli = CLI.read_text()

    assert 'PAPER_DEPLOYABLE_METHODS = ("NN27_RAW", "NN27_D4")' in runner
    # config-equality gate in _resolved_stationary_canary_config depends on this:
    assert "DEPLOYABLE_CANARY_METHODS = PAPER_DEPLOYABLE_METHODS\n" in runner
    assert 'methods = args.methods or ["NN27_RAW", "NN27_D4"]' in cli
    assert '"NN27_RAW_RELAX"' in runner
    assert '"NN27_D4_RELAX"' in runner


def test_reproduce_accepts_relax_lambda() -> None:
    from cfd_applications_cleanroom.cfd_apps.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "reproduce",
        "--benchmark", "stationary",
        "--tier", "curvature-diagnostic",
        "--methods", "NN27_RAW_RELAX",
        "--relax-lambda", "0.25",
    ])

    assert args.relax_lambda == 0.25
    assert args.methods == ["NN27_RAW_RELAX"]


def _relax_mirror(center: float, neighbors: list[float], lam: float) -> float:
    numerator = center + lam * sum(neighbors)
    return numerator / (1.0 + lam * len(neighbors))


def test_relax_formula_spec_invariance_and_identity() -> None:
    assert _relax_mirror(3.1, [3.1, 3.1, 3.1, 3.1], 0.25) == pytest.approx(3.1, abs=1e-15)
    assert _relax_mirror(3.1, [3.1, 3.1], 0.25) == pytest.approx(3.1, abs=1e-15)
    assert _relax_mirror(2.0, [1.0, 3.0, 2.5], 0.0) == 2.0
    values = [2.0, 1.0, 3.0, 2.5, 1.5]
    filtered = _relax_mirror(values[0], values[1:], 0.5)
    assert min(values) <= filtered <= max(values)


def test_stationary_curvature_diagnostic_is_dispatchable() -> None:
    cli = CLI.read_text()
    runner = STATIONARY.read_text()
    shared = CURVATURE_SHARED.read_text()

    assert '"curvature-diagnostic"' in cli
    assert "run_curvature_diagnostic" in cli
    assert "def run_curvature_diagnostic" in runner
    # Report-writing lives in the shared module; the current paper-facing
    # branch contract is baseline_hgradient raw vs D4, with PART2 retained only
    # as an explicit legacy diagnostic.
    assert "curvature_field_summary" in shared
    assert "paper_model_line" in runner
    assert "d4_branch" in shared


def test_curvature_diagnostic_report_renders_publication_method_labels() -> None:
    from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import (
        render_curvature_diagnostic_report,
    )

    force = {
        "row_count": 1,
        "mean_delta_hk": 0.0,
        "force_band_std_kappa_nn": 0.0,
        "max_abs_delta_hk": 0.0,
        "p95_abs_delta_hk": 0.0,
    }
    report = {
        "run_id": "r1",
        "evidence_level": "controlled_diagnostic",
        "primary_branch": "NN27_RAW",
        "d4_branch": "NN27_D4",
        "paper_model_line": "baseline_hgradient",
        "methods": {
            "NN27_RAW": {
                "force_band": force,
                "sign_scale": {"sign_verdict": "same_sign", "scale_identity_max_abs": 0.0},
            },
            "NN27_D4": {
                "force_band": force,
                "sign_scale": {"sign_verdict": "same_sign", "scale_identity_max_abs": 0.0},
            },
        },
        "comparison": {},
    }

    text = render_curvature_diagnostic_report(report)

    assert "| NN |" in text
    assert "| NND4 |" in text
    assert "| NN27_RAW |" not in text
    assert "| NN27_D4 |" not in text


def test_nn_canary_compile_uses_qcc_compatible_inline_include_flag() -> None:
    runner = STATIONARY.read_text()

    assert 'f"-I{NN_DIR}"' in runner
    assert '"-I", str(NN_DIR)' not in runner


def test_method_id_macro_is_passed_as_c_string_literal() -> None:
    runner = STATIONARY.read_text()
    shared = CURVATURE_SHARED.read_text()

    assert r'f"-DMETHOD_ID=\\\"{method}\\\""' in runner
    assert r'f"-DMETHOD_ID=\\\"{method}\\\""' in shared
    assert 'f\'-DMETHOD_ID="{method}"\'' not in runner
    assert 'f\'-DMETHOD_ID="{method}"\'' not in shared


def test_nn_canary_links_forward_inference_as_separate_object() -> None:
    runner = STATIONARY.read_text()
    shared = CURVATURE_SHARED.read_text()

    # This compile step moved to curvature_diagnostic_shared.py (shared-module
    # refactor) and dropped its leading underscore there; the runner still
    # imports and uses it under the original private name.
    assert "def compile_nn_forward_object" in shared
    assert "compile_nn_forward_object as _compile_nn_forward_object" in runner
    assert "nn_forward_clean.c" in shared
    assert "nn_forward_clean.o" in shared
    assert "nn_object_compile_log" in shared


def test_clsvof_stationary_hosts_use_official_integral_stack() -> None:
    native = NATIVE_CLSVOF_CASE.read_text()
    nn = NN_CLSVOF_CASE.read_text()

    for source in (native, nn):
        assert '#include "grid/multigrid.h"' in source
        assert '#include "navier-stokes/centered.h"' in source
        assert '#include "two-phase-clsvof.h"' in source
        assert '#include "integral.h"' in source
        assert "d.sigmaf = sigma" in source
        assert "sqrt (sq(x) + sq(y)) - RADIUS" in source


def test_curvature_field_case_exports_hk_native_hk_nn_and_scale_sign_fields() -> None:
    source = CURVATURE_FIELD_CASE.read_text()

    assert '#include "two-phase-clsvof.h"' in source
    assert '#include "integral.h"' in source
    assert "distance_curvature (point, d)" in source
    assert "hk_native = Delta*native_kappa" in source
    assert "kappa_nn = hk_nn/Delta" in source
    assert "delta_hk = hk_nn - hk_native" in source
    assert "sign_product = kappa_nn*native_kappa" in source
    assert "theta = atan2 (y, x)" in source
    assert "curvature_field.csv" in source


def test_curvature_diagnostic_summary_reports_user_requested_metrics() -> None:
    # The statistics that produce these fields moved to
    # curvature_diagnostic_shared.py (shared-module refactor); the circle
    # runner calls them unmodified via curvature_band_stats.
    shared = CURVATURE_SHARED.read_text()

    for field in (
        "mean_delta_hk",
        "interface_std_kappa_nn",
        "force_band_std_kappa_nn",
        "max_abs_delta_hk",
        "p95_abs_delta_hk",
        "sign_agreement_fraction",
        "rmse_direct",
        "rmse_negated",
        "scale_identity_max_abs",
    ):
        assert field in shared


def test_stationary_curvature_process_exports_requested_time_slices_and_surface_tension() -> None:
    source = CURVATURE_PROCESS_CASE.read_text()
    runner = STATIONARY.read_text()
    cli = CLI.read_text()

    assert '"curvature-process"' in cli
    assert "run_curvature_process" in runner
    assert "CURVATURE_PROCESS_SOURCE" in runner
    assert "curvature_process.csv" in source
    assert "surface_tension_trace.csv" in source
    assert "snapshot_fraction" in source
    assert "snapshot_targets[4] = {0., 1./3., 2./3., 1.}" in source
    assert "theta_deg" in source
    assert "hk_native" in source
    assert "hk_nn" in source
    assert "sigma" in source
    assert "Ca" in source


def test_stationary_curvature_process_figure_contract_is_python_source_data_backed() -> None:
    # Plate rendering moved to curvature_diagnostic_shared.py (shared-module
    # refactor) and the "stationary_curvature_process_plate" figure stem is
    # now assembled at runtime as f"{artifact_prefix}_plate"; the circle
    # runner still binds artifact_prefix="stationary_curvature_process".
    runner = STATIONARY.read_text()
    shared = CURVATURE_SHARED.read_text()

    assert 'artifact_prefix="stationary_curvature_process"' in runner
    assert '{artifact_prefix}_plate' in shared
    assert ".svg" in shared
    assert ".pdf" in shared
    assert ".png" in shared
    assert "source_data" in shared
    assert "curvature_process_csv" in shared
    assert "surface_tension_trace_csv" in shared
    assert "legend_ax = fig.add_subplot(grid[0, :])" in shared
    assert 'ax.legend(loc="best", markerscale=2)' not in shared


def test_stationary_curvature_process_allows_report_resolution_diagnostics() -> None:
    runner = STATIONARY.read_text()

    assert "CURVATURE_PROCESS_LEVELS" in runner
    assert "4, 5, 6, 7, 8, 9" in runner
    assert "stationary_curvature_process_unknown_levels" in runner


def test_curvature_jump_is_figure_artifact_not_solver_tier() -> None:
    import pytest

    from cfd_applications_cleanroom.cfd_apps.cli import build_parser

    cli = CLI.read_text()

    assert '"curvature-jump"' in cli
    assert "--artifact" in cli
    assert "write_curvature_jump_package" in cli

    parser = build_parser()
    args = parser.parse_args([
        "figures",
        "--artifact",
        "curvature-jump",
        "--run-id",
        "stationary_curvature_20260703T132941Z",
        "--methods",
        "NN27_RAW",
        "NN27_D4",
    ])
    assert args.command == "figures"
    assert args.artifact == "curvature-jump"
    assert args.run_id == "stationary_curvature_20260703T132941Z"
    assert args.methods == ["NN27_RAW", "NN27_D4"]

    with pytest.raises(SystemExit):
        parser.parse_args([
            "reproduce",
            "--tier",
            "curvature-jump",
            "--benchmark",
            "stationary",
        ])


def test_canary_config_declares_inert_overlay_gates() -> None:
    text = CONFIG.read_text()

    assert '"canary"' in text
    assert '"CLSVOF_LS_NATIVE"' in text
    assert '"NN_DISABLE"' in text
    assert '"NN_PROBE_ONLY"' in text
    assert '"sb_g4_max_abs_ca_gate": 1e-12' in text
    assert '"sb_g5_max_abs_ca_gate": 1e-12' in text


def test_paper_defaults_do_not_route_accepted_matrix_to_legacy_part2() -> None:
    text = ACCEPTED_MATRIX_CONFIG.read_text()

    assert "methods: [NN27_RAW, NN27_D4]" in text
    assert "PART2_D4" not in text


def test_method_catalog_marks_baseline_hgradient_nnd4_as_deployable() -> None:
    text = METHODS_CONFIG.read_text()

    assert "NN27_D4:" in text
    assert "force_path: D4-averaged baseline_hgradient checkpoint curvature" in text
    assert "PART2_RAW:" not in text
    assert "PART2_D4:" not in text
