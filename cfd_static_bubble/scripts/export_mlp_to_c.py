from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cfd_static_bubble.scripts.nn27_contract import build_circle_raw27, load_v2_checkpoint_bundle
from evaluate.shared import apply_feature_transform, validate_feature_transform
from model.config import MLP_TrainConfig
from model.model import create_model


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fmt_scalar(value: float) -> str:
    return f"{float(value):.17g}"


def _c_array(name: str, values: np.ndarray, *, size_expr: str | None = None) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    size = size_expr or str(arr.shape[0])
    body = ", ".join(_fmt_scalar(v) for v in arr)
    return f"static const double {name}[{size}] = {{{body}}};"


def _c_matrix(name: str, values: np.ndarray, *, rows_expr: str | None = None, cols_expr: str | None = None) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}.")
    rows = rows_expr or str(arr.shape[0])
    cols = cols_expr or str(arr.shape[1])
    row_chunks = []
    for row in arr:
        row_chunks.append("{" + ", ".join(_fmt_scalar(v) for v in row) + "}")
    return f"static const double {name}[{rows}][{cols}] = {{{', '.join(row_chunks)}}};"


def _build_model(bundle: dict) -> torch.nn.Module:
    hidden_units = int(bundle["model_config"]["hidden_units"])
    model = create_model(MLP_TrainConfig(input_dim=27, hidden_units=hidden_units))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _write_weights_header(path: Path, transform: dict, state_dict: dict[str, torch.Tensor], hidden_units: int) -> None:
    lines = [
        "#pragma once",
        "#define MLP_IN 27",
        f"#define MLP_H {hidden_units}",
        _c_array("MEAN", transform["mean"], size_expr="MLP_IN"),
        _c_array("STD", transform["std"], size_expr="MLP_IN"),
        _c_matrix("W0", _as_numpy(state_dict["net.0.weight"]), rows_expr="MLP_H", cols_expr="MLP_IN"),
        _c_array("b0", _as_numpy(state_dict["net.0.bias"]), size_expr="MLP_H"),
        _c_matrix("W1", _as_numpy(state_dict["net.2.weight"]), rows_expr="MLP_H", cols_expr="MLP_H"),
        _c_array("b1", _as_numpy(state_dict["net.2.bias"]), size_expr="MLP_H"),
        _c_matrix("W2", _as_numpy(state_dict["net.4.weight"]), rows_expr="MLP_H", cols_expr="MLP_H"),
        _c_array("b2", _as_numpy(state_dict["net.4.bias"]), size_expr="MLP_H"),
        _c_matrix("W3", _as_numpy(state_dict["net.6.weight"]), rows_expr="MLP_H", cols_expr="MLP_H"),
        _c_array("b3", _as_numpy(state_dict["net.6.bias"]), size_expr="MLP_H"),
        _c_matrix("W4", _as_numpy(state_dict["net.8.weight"]), rows_expr="1", cols_expr="MLP_H"),
        _c_array("b4", _as_numpy(state_dict["net.8.bias"]), size_expr="1"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def _write_fixtures_header(path: Path, resolution: int, transform: dict, bundle: dict, fixture_count: int) -> None:
    raw27, _ = build_circle_raw27(resolution, radius=0.4, center=(0.5, 0.5))
    if raw27.shape[0] == 0:
        raise ValueError("No interface samples found for parity fixtures.")
    sample_count = min(int(fixture_count), raw27.shape[0])
    sample_idx = np.linspace(0, raw27.shape[0] - 1, num=sample_count, dtype=int)
    sample = raw27[sample_idx]
    z = apply_feature_transform(sample, transform)
    model = _build_model(bundle)
    with torch.no_grad():
        expected = model(torch.from_numpy(z)).detach().cpu().numpy().reshape(-1)
    lines = [
        "#pragma once",
        f"#define PARITY_FIXTURE_COUNT {sample_count}",
        _c_matrix("PARITY_RAW27", sample, rows_expr="PARITY_FIXTURE_COUNT", cols_expr="MLP_IN"),
        _c_array("PARITY_EXPECTED_HKAPPA", expected, size_expr="PARITY_FIXTURE_COUNT"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def export_checkpoint_to_headers(
    resolution: int,
    *,
    weights_path: Path,
    fixtures_path: Path,
    fixture_count: int = 20,
) -> dict[str, int]:
    bundle = load_v2_checkpoint_bundle(resolution)
    transform = validate_feature_transform(bundle["feature_transform"])
    hidden_units = int(bundle["model_config"]["hidden_units"])
    state_dict = bundle["state_dict"]
    _write_weights_header(weights_path, transform, state_dict, hidden_units)
    _write_fixtures_header(fixtures_path, resolution, transform, bundle, fixture_count)
    return {
        "resolution": int(resolution),
        "hidden_units": hidden_units,
        "fixture_count": min(int(fixture_count), int(build_circle_raw27(resolution, radius=0.4, center=(0.5, 0.5))[0].shape[0])),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a V2 NN27 checkpoint to C headers.")
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--weights-out", type=Path, required=True)
    parser.add_argument("--fixtures-out", type=Path, required=True)
    parser.add_argument("--fixture-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = export_checkpoint_to_headers(
        args.resolution,
        weights_path=args.weights_out,
        fixtures_path=args.fixtures_out,
        fixture_count=args.fixture_count,
    )
    print(
        f"exported resolution={meta['resolution']} hidden_units={meta['hidden_units']} fixture_count={meta['fixture_count']}"
    )


if __name__ == "__main__":
    main()
