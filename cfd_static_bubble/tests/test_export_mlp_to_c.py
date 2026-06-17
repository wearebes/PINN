from __future__ import annotations

from pathlib import Path

from cfd_static_bubble.scripts.export_mlp_to_c import export_checkpoint_to_headers


def test_export_checkpoint_to_headers_writes_weights_and_fixtures(tmp_path: Path):
    weights = tmp_path / "mlp_weights.h"
    fixtures = tmp_path / "parity_fixtures.h"
    meta = export_checkpoint_to_headers(64, weights_path=weights, fixtures_path=fixtures, fixture_count=8)
    text = weights.read_text(encoding="utf-8")
    fixture_text = fixtures.read_text(encoding="utf-8")
    assert meta["resolution"] == 64
    assert "#define MLP_IN 27" in text
    assert "#define MLP_H 128" in text
    assert "static const double MEAN[MLP_IN]" in text
    assert "static const double W4[1][MLP_H]" in text
    assert "PARITY_FIXTURE_COUNT" in fixture_text
    assert "PARITY_RAW27" in fixture_text
    assert "PARITY_EXPECTED_HKAPPA" in fixture_text
