from __future__ import annotations

from pathlib import Path

import yaml

from cfd_applications_cleanroom.cfd_apps.host_format import classify_application


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cfd_applications_cleanroom" / "configs" / "host_format_migration.yaml"
HOST_FORMAT_FIGURES = ROOT / "cfd_applications_cleanroom" / "cfd_apps" / "host_format_figures.py"


def test_host_format_registry_has_no_ambiguous_candidate() -> None:
    data = yaml.safe_load(REGISTRY.read_text())
    apps = data["applications"]
    first_round = [name for name, app in apps.items() if app["first_round_candidate"]]
    assert first_round == ["stationary_bubble"]


def test_stationary_bubble_is_a_valid_vof_hf_migration_candidate() -> None:
    app = yaml.safe_load(REGISTRY.read_text())["applications"]["stationary_bubble"]
    assert app["host_format"] == "VOF-HF"
    assert app["interface_state"] == "volume_fraction_c"
    assert app["needs_persistent_signed_distance"] is False
    assert app["permits_host_format_change"] is True
    assert app["external_curvature_injection_required"] is False
    assert app["accepted_metric"] == "Ca_tail_max"
    assert float(app["accepted_gate"]) == 1.0e-12
    assert (ROOT / app["source_case"]).exists()
    assert (ROOT / app["stock_reference_report"]).exists()
    assert app["stock_reference_report"].endswith("stationary_stock_20260705T013552Z_stock_reference_summary.md")


def test_clsvof_nn_path_is_not_misclassified_as_vof_hf() -> None:
    app = yaml.safe_load(REGISTRY.read_text())["applications"]["clsvof_ls_nn_static_bubble"]
    assert app["host_format"] == "CLSVOF-LS"
    assert app["needs_persistent_signed_distance"] is True
    assert app["permits_host_format_change"] is False
    assert app["external_curvature_injection_required"] is True
    assert app["first_round_candidate"] is False


def test_host_format_classifies_stationary_as_pass() -> None:
    decision = classify_application("stationary_bubble")
    assert decision["status"] == "PASS"
    assert decision["recommendation"] == "use_vof_hf_native"
    assert all(decision["gates"].values())


def test_host_format_blocks_clsvof_nn_path() -> None:
    decision = classify_application("clsvof_ls_nn_static_bubble")
    assert decision["status"] == "BLOCKED"
    assert decision["recommendation"] == "do_not_migrate_in_this_plan"
    assert decision["gates"]["HF-G0"] is False
    assert decision["gates"]["HF-G2"] is False
    assert decision["gates"]["HF-G3"] is False


def test_host_format_figure_uses_baseline_hgradient_nnd4_not_legacy_part2() -> None:
    text = HOST_FORMAT_FIGURES.read_text()

    assert '("NN27_RAW", "NN27_D4")' in text
    assert '"NN27_D4": "#B2182B"' in text
    assert '("NN27_RAW", "PART2_D4")' not in text
