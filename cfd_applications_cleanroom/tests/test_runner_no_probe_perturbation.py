from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NN_CURVATURE = ROOT / "cfd_applications_cleanroom/nn/nn_curvature_clean.h"
NN_STATIONARY_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_clsvof_nn.c"
RELAX_FILTER_HEADER = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_relax_filter.h"
CURVATURE_FIELD_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_curvature_field.c"
CURVATURE_PROCESS_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_curvature_process.c"
NN_INTEGRAL_OVERLAY = ROOT / "cfd_applications_cleanroom/nn/generated/integral_nn_clean.h"
CONFIGS = [
    ROOT / "cfd_applications_cleanroom/configs/canary_matrix.yaml",
    ROOT / "cfd_applications_cleanroom/configs/accepted_matrix.yaml",
]


def test_probe_and_force_storage_are_declared_disjoint() -> None:
    text = NN_CURVATURE.read_text()

    force = _extract_storage(text, "force_kappa_storage")
    probe = _extract_storage(text, "probe_kappa_storage")

    assert force == "kappa_nn[]"
    assert probe == "kappa_nn_probe[]"
    assert force != probe


def test_probe_mode_rules_are_declared_before_solver_hosts_exist() -> None:
    text = NN_CURVATURE.read_text()

    assert "NN_DISABLE writes neither storage path" in text
    assert "NN_PROBE_ONLY writes probe_kappa_storage only" in text
    assert "force path reads native curvature" in text
    assert "HK_BIAS_PROBE writes CSV diagnostics only" in text
    assert "cannot be active with NN_DISABLE or NN_PROBE_ONLY" in text


def test_no_parity_config_row_has_hk_bias_probe() -> None:
    forbidden = {"NN_DISABLE", "NN_PROBE_ONLY"}
    for path in CONFIGS:
        text = path.read_text()
        for line in text.splitlines():
            if "methods:" not in line:
                continue
            methods = set(re.findall(r"[A-Z0-9_]+", line))
            assert not ("HK_BIAS_PROBE" in methods and methods.intersection(forbidden))


def test_stationary_probe_mode_cannot_perturb_force_path() -> None:
    text = NN_STATIONARY_CASE.read_text()

    assert '#include "integral.h"' in text
    assert "#define distance_curvature" not in text
    assert "NN_PROBE_ONLY_MODE" in text

    probe_block = re.search(
        r"#if NN_MODE == NN_PROBE_ONLY_MODE(?P<body>.*?)#endif /\* NN_MODE == NN_PROBE_ONLY_MODE \*/",
        text,
        re.S,
    )
    assert probe_block, "missing isolated probe-only block"
    body = probe_block.group("body")
    assert "kappa_nn_probe[]" in body
    assert not re.search(r"(?<!probe_)kappa_nn\[\]\s*=", body)
    assert not re.search(r"\bf\[[^\]]*\]\s*=", body)
    assert not re.search(r"\bd\[[^\]]*\]\s*=", body)
    assert not re.search(r"\bu\.[xy]\[[^\]]*\]\s*=", body)
    assert not re.search(r"\ba\.[xy]\[[^\]]*\]\s*=", body)


def test_stationary_probe_uses_constant_solver_stencil_offsets() -> None:
    text = NN_STATIONARY_CASE.read_text()

    assert "d[ii,jj]" not in text
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            assert f"d[{dx},{dy}]" in text


def test_stationary_host_does_not_embed_nn_forward_implementation_in_qcc_unit() -> None:
    text = NN_STATIONARY_CASE.read_text()

    assert '#include "nn_forward_clean.h"' in text
    assert '#include "nn_forward_clean.c"' not in text


def test_generated_integral_overlay_has_force_curvature_hook() -> None:
    text = NN_INTEGRAL_OVERLAY.read_text()

    assert "CLEANROOM_PREFILL_FORCE_KAPPA" in text
    assert "CLEANROOM_FORCE_KAPPA_VALUE" in text
    assert "double ki = CLEANROOM_FORCE_KAPPA_VALUE;" in text


def test_stationary_force_modes_write_force_storage_and_use_overlay_hook() -> None:
    text = NN_STATIONARY_CASE.read_text()

    assert "NN27_RAW_MODE" in text
    assert "NN27_D4_MODE" in text
    assert "CLEANROOM_USE_NN_FORCE_CURVATURE" in text
    assert "CLEANROOM_PREFILL_FORCE_KAPPA" in text
    assert '#include "generated/integral_nn_clean.h"' in text
    assert "kappa_nn[] = cleanroom_nn_force_curvature" in text
    assert "mlp_hkappa_clean" in text
    assert "nn_hkappa_d4_clean" in text
    assert "return native_k*ratio;" in text


def test_relax_modes_are_explicit_and_do_not_relabel_raw_modes() -> None:
    source = NN_STATIONARY_CASE.read_text()

    assert "#define NN27_RAW_RELAX_MODE 4" in source
    assert "#define NN27_D4_RELAX_MODE 5" in source
    assert "CLEANROOM_RELAX_ENABLED" in source
    # LR-G0: existing raw/D4 force path text is untouched.
    assert "kappa_nn[] = cleanroom_nn_force_curvature" in source
    assert "return native_k*ratio;" in source


def test_relax_filter_formula_is_single_sourced_with_finite_mask() -> None:
    header = RELAX_FILTER_HEADER.read_text()

    assert "numerator += CLEANROOM_RELAX_LAMBDA" in header
    assert "!= nodata" in header
    assert "isfinite" in header
    for case in (NN_STATIONARY_CASE, CURVATURE_FIELD_CASE, CURVATURE_PROCESS_CASE):
        assert "numerator += CLEANROOM_RELAX_LAMBDA" not in case.read_text()


def _extract_storage(text: str, name: str) -> str:
    match = re.search(rf"{name}=([A-Za-z0-9_]+\[\])", text)
    assert match, f"missing {name}"
    return match.group(1)
