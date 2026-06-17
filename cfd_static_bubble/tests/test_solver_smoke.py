from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cfd_static_bubble.scripts.export_mlp_to_c import export_checkpoint_to_headers


def test_mlp_parity_harness_matches_pytorch(tmp_path: Path):
    weights = tmp_path / "mlp_weights.h"
    fixtures = tmp_path / "parity_fixtures.h"
    binary = tmp_path / "mlp_parity"
    export_checkpoint_to_headers(64, weights_path=weights, fixtures_path=fixtures, fixture_count=8)
    subprocess.run(
        [
            "cc",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(Path("cfd_static_bubble/basilisk").resolve()),
            str(Path("cfd_static_bubble/basilisk/mlp_parity.c").resolve()),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["max_rel_error"] < 1e-6
