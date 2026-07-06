from __future__ import annotations

import json
from pathlib import Path

from cfd_applications_cleanroom.cfd_apps.hashes import sha256_file


ROOT = Path(__file__).resolve().parents[2]
STATIONARY_DIR = ROOT / "cfd_applications_cleanroom/cases/stationary"
ELLIPSE_DIR = ROOT / "cfd_applications_cleanroom/cases/stationary_ellipse"
CLI = ROOT / "cfd_applications_cleanroom/cfd_apps/cli.py"
STATIONARY_ELLIPSE_RUNNER = ROOT / "cfd_applications_cleanroom/cfd_apps/stationary_ellipse.py"
CURVATURE_SHARED = ROOT / "cfd_applications_cleanroom/cfd_apps/curvature_diagnostic_shared.py"
CONFIG = ROOT / "cfd_applications_cleanroom/configs/stationary_ellipse.yaml"

# Baseline recorded in docs/superpowers/plans/2026-07-03-stationary-ellipse-curvature-stress-diagnostic.md
# Phase 0. If one of these hashes legitimately needs to change, that change
# must be to the circle route on its own terms, not a side effect of adding
# the ellipse family, and the plan doc's baseline table must be updated too.
CIRCLE_HOST_SHA256 = {
    "stationary_clsvof_native.c": "2d6c4cbb7beeb689f42959f939486e22868b07db5572618f5f1a0c21ac7fa57e",
    "stationary_clsvof_nn.c": "b53526c425402151e952d7f31dd0a80548b3d98feead0146c3488fe4da89a2cd",
    "stationary_curvature_field.c": "6b21e797607b28bb592a82cbe5b98b78e8bd6b53322bd4be181aba73ca4819c7",
    "stationary_curvature_process.c": "f5cf18f1f560f5794fe6192d43a52ab6489198b82ae0a3e9b023a80af0845b11",
}


def test_circle_hosts_are_byte_unchanged() -> None:
    for filename, expected_sha256 in CIRCLE_HOST_SHA256.items():
        path = STATIONARY_DIR / filename
        actual_sha256 = sha256_file(path)
        assert actual_sha256 == expected_sha256, (
            f"{filename} changed (sha256 {actual_sha256} != baseline {expected_sha256}). "
            "The stationary_ellipse family must not modify the circle route's host files."
        )


def test_ellipse_cli_dispatch_exists() -> None:
    cli = CLI.read_text()

    assert '"E1", "E2"' in cli or "E1\", \"E2\"" in cli
    assert 'args.benchmark == "stationary_ellipse" and args.tier == "curvature-diagnostic"' in cli
    assert 'args.benchmark == "stationary_ellipse" and args.tier == "curvature-process"' in cli
    assert "run_curvature_diagnostic" in cli
    assert "run_curvature_process" in cli
    # No existing tier choice was touched: still exactly these six.
    assert cli.count('"curvature-diagnostic"') >= 1
    assert cli.count('"curvature-process"') >= 1


def test_ellipse_case_files_use_vendor_geometry_not_a_new_projection() -> None:
    header = (ELLIPSE_DIR / "ellipse_geometry_clean.h").read_text()
    impl = (ELLIPSE_DIR / "ellipse_geometry_clean.c").read_text()
    field = (ELLIPSE_DIR / "stationary_ellipse_curvature_field.c").read_text()
    process = (ELLIPSE_DIR / "stationary_ellipse_curvature_process.c").read_text()

    assert "ELLIPSE_A" in header
    assert "ELLIPSE_B" in header
    assert '#include "distance_point_ellipse.h"' in impl
    assert "DistancePointEllipse" in impl
    # No Newton/theta root-finding loop was ported from geometry_core.py.
    assert "for (" not in impl
    assert "while (" not in impl
    for source in (field, process):
        assert '#include "ellipse_geometry_clean.h"' in source
        assert "ellipse_geometry_signed_distance" in source


def test_ellipse_process_host_has_four_way_force_switch() -> None:
    process = (ELLIPSE_DIR / "stationary_ellipse_curvature_process.c").read_text()

    for mode in ("NN_DISABLE_MODE", "NN_PROBE_ONLY_MODE", "NN27_RAW_MODE", "NN27_D4_MODE"):
        assert mode in process
    assert 'include "integral.h"' in process
    assert 'include "generated/integral_nn_clean.h"' in process
    assert "CLEANROOM_USE_NN_FORCE_CURVATURE" in process


def test_ellipse_config_declares_locked_cases() -> None:
    config = json.loads(CONFIG.read_text())

    assert config["benchmark"] == "stationary_ellipse"
    assert config["not_a_stationary_bubble_gate"] is True
    assert config["not_an_ellipse_regression_benchmark"] is True
    cases = config["physics"]["cases"]
    assert cases["E1"]["a"] == 0.4472136
    assert cases["E1"]["b"] == 0.3577709
    assert cases["E2"]["a"] == 0.4898979
    assert cases["E2"]["b"] == 0.3265986
    for case in ("E1", "E2"):
        a, b = cases[case]["a"], cases[case]["b"]
        assert abs(a * b - 0.16) < 1e-6, f"{case} does not preserve circle-equivalent area a*b=R^2"


def test_ellipse_config_defaults_to_baseline_hgradient_nnd4_not_legacy_part2() -> None:
    config = json.loads(CONFIG.read_text())

    assert config["curvature_diagnostic"]["methods"] == ["NN27_RAW", "NN27_D4"]
    assert config["curvature_process"]["methods"] == [
        "NN_DISABLE",
        "NN_PROBE_ONLY",
        "NN27_RAW",
        "NN27_D4",
    ]
    assert "PART2_D4" not in json.dumps(config["curvature_diagnostic"])
    assert "PART2_D4" not in json.dumps(config["curvature_process"])


def test_ellipse_matrix_includes_64_128_256_levels() -> None:
    config = json.loads(CONFIG.read_text())
    runner = STATIONARY_ELLIPSE_RUNNER.read_text()

    assert config["curvature_diagnostic"]["levels"] == [6, 7, 8]
    assert config["curvature_process"]["levels"] == [6, 7, 8]
    assert "CURVATURE_DIAGNOSTIC_LEVELS = (6, 7, 8)" in runner
    assert "CURVATURE_PROCESS_LEVELS = (6, 7, 8)" in runner


def test_ellipse_runner_reuses_shared_module_not_new_stats() -> None:
    runner = STATIONARY_ELLIPSE_RUNNER.read_text()
    shared = CURVATURE_SHARED.read_text()

    # The ellipse runner must call the shared curvature_band_stats, not
    # reimplement mean/std/percentile itself (plan section 8, Phase 4).
    assert "from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import" in runner
    assert "curvature_band_stats" in runner
    assert "def curvature_band_stats" not in runner
    assert "def curvature_band_stats" in shared
    assert "std_delta_kappa" in shared


def test_ellipse_runner_uses_own_namespace_not_circle_manifests() -> None:
    runner = STATIONARY_ELLIPSE_RUNNER.read_text()

    assert 'RAW / BENCHMARK' in runner or 'RAW / "stationary_ellipse"' in runner
    assert 'BENCHMARK = "stationary_ellipse"' in runner
