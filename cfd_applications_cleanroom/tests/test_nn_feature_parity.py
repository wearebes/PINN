from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cfd_applications_cleanroom.nn.export_clean_nn_assets import export_all
from model.config import MLP_TrainConfig
from model.model import create_model


ROOT = Path(__file__).resolve().parents[2]
NN_DIR = ROOT / "cfd_applications_cleanroom/nn"


def test_export_manifest_contains_clean_27d_contract() -> None:
    manifest = export_all()

    assert manifest["input_dim"] == 27
    assert manifest["target_definition"] == "h*kappa"
    assert len(manifest["rows"]) == 2
    for row in manifest["rows"]:
        assert row["raw_feature_dim"] == 27
        assert row["feature_order"] == "phi9+nx9+ny9"
        assert Path(ROOT / row["weights_header"]).exists()


@pytest.mark.parametrize("label", ["nn27_r128", "nn27_r256"])
def test_python_c_forward_and_normalized_feature_parity(label: str, tmp_path: Path) -> None:
    manifest = export_all()
    row = next(item for item in manifest["rows"] if item["label"] == label)
    checkpoint = ROOT / row["checkpoint_path"]
    weights_header = row["weights_header"]
    raw = np.array(
        [
            -0.3,
            -0.2,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.2,
            0.3,
            1.0,
            0.8,
            0.6,
            0.4,
            0.2,
            0.0,
            -0.2,
            -0.4,
            -0.6,
            -0.7,
            -0.5,
            -0.3,
            -0.1,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ],
        dtype=np.float64,
    )
    py_out, py_z = _python_forward_and_z(checkpoint, raw)
    c_out, c_z = _run_forward_probe(tmp_path, weights_header, raw)

    assert abs(c_out - py_out) <= 1e-10
    assert float(np.max(np.abs(c_z - py_z))) <= 1e-8


def test_raw27_feature_parity_against_python_mirror(tmp_path: Path) -> None:
    patch = np.array(
        [
            [0.21, 0.15, 0.11, 0.08, 0.04],
            [0.18, 0.10, 0.04, 0.01, -0.02],
            [0.14, 0.07, 0.00, -0.04, -0.08],
            [0.11, 0.03, -0.03, -0.08, -0.13],
            [0.08, -0.01, -0.08, -0.14, -0.20],
        ],
        dtype=np.float64,
    )
    delta = 0.015625
    sgn = 1.0
    py_raw = _python_raw27(patch, delta, sgn)
    c_raw = _run_feature_probe(tmp_path, patch, delta, sgn)

    assert float(np.max(np.abs(c_raw - py_raw))) <= 1e-12


def _python_forward_and_z(checkpoint: Path, raw: np.ndarray) -> tuple[float, np.ndarray]:
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(bundle["feature_transform"]["mean"], dtype=np.float64)
    std = np.asarray(bundle["feature_transform"]["std"], dtype=np.float64)
    z = (raw - mean) / std
    hidden = int(bundle["model_config"]["hidden_units"])
    model = create_model(MLP_TrainConfig(input_dim=27, hidden_units=hidden))
    model.load_state_dict(bundle["state_dict"])
    model.double().eval()
    with torch.no_grad():
        out = model(torch.from_numpy(z.reshape(1, -1))).cpu().numpy().reshape(-1)[0]
    return float(out), z


def _run_forward_probe(tmp_path: Path, weights_header: str, raw: np.ndarray) -> tuple[float, np.ndarray]:
    raw_init = ", ".join(f"{float(x):.17g}" for x in raw)
    source = tmp_path / "forward_probe.c"
    source.write_text(
        f"""
#include <stdio.h>
#include "nn_forward_clean.h"

int main(void) {{
  double raw[27] = {{{raw_init}}};
  double z[27];
  nn_prepare_input_clean(raw, z);
  printf("out %.17g\\n", mlp_hkappa_clean(raw));
  for (int i = 0; i < 27; ++i)
    printf("z %d %.17g\\n", i, z[i]);
  return 0;
}}
"""
    )
    binary = tmp_path / "forward_probe"
    cmd = [
        "cc",
        "-std=c99",
        "-O2",
        f"-DNN_WEIGHTS_HEADER=\"{weights_header}\"",
        "-I",
        str(ROOT),
        "-I",
        str(NN_DIR),
        str(source),
        str(NN_DIR / "nn_forward_clean.c"),
        "-o",
        str(binary),
        "-lm",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    result = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, check=True)
    out = None
    z = np.zeros(27, dtype=np.float64)
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "out":
            out = float(parts[1])
        elif parts[0] == "z":
            z[int(parts[1])] = float(parts[2])
    assert out is not None
    return out, z


def _python_raw27(D: np.ndarray, delta: float, sgn: float) -> np.ndarray:
    offsets = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    raw = np.zeros(27, dtype=np.float64)
    for k, (dr, dc) in enumerate(offsets):
        a = dr + 2
        b = dc + 2
        dx = sgn * (D[a + 1, b] - D[a - 1, b])
        dy = sgn * (D[a, b + 1] - D[a, b - 1])
        mag = math.hypot(dx, dy) or 1.0
        raw[k] = sgn * D[a, b] / delta
        raw[9 + k] = dx / mag
        raw[18 + k] = dy / mag
    return raw


def _run_feature_probe(tmp_path: Path, patch: np.ndarray, delta: float, sgn: float) -> np.ndarray:
    rows = ["{" + ", ".join(f"{float(v):.17g}" for v in row) + "}" for row in patch]
    source = tmp_path / "feature_probe.c"
    source.write_text(
        f"""
#include <stdio.h>
#include "nn_features_clean.h"

int main(void) {{
  double D[5][5] = {{{', '.join(rows)}}};
  double raw[27];
  nn_build_raw27_clean(D, {delta:.17g}, {sgn:.17g}, raw);
  for (int i = 0; i < 27; ++i)
    printf("%.17g\\n", raw[i]);
  return 0;
}}
"""
    )
    binary = tmp_path / "feature_probe"
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-I",
            str(NN_DIR),
            str(source),
            "-o",
            str(binary),
            "-lm",
        ],
        cwd=ROOT,
        check=True,
    )
    result = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, check=True)
    return np.array([float(line) for line in result.stdout.splitlines()], dtype=np.float64)
