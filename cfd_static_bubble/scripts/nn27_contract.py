from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evaluate.shared import validate_feature_transform
from train_generate.generate import (
    build_circle_sdf,
    build_grid,
    extract_grad9,
    extract_phi9,
    interface_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "out"

_OFFSETS = [
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]
_OFFSET_TO_INDEX = {offset: i for i, offset in enumerate(_OFFSETS)}
_D4_MATRICES = (
    np.asarray([[1, 0], [0, 1]], dtype=np.int64),
    np.asarray([[0, -1], [1, 0]], dtype=np.int64),
    np.asarray([[-1, 0], [0, -1]], dtype=np.int64),
    np.asarray([[0, 1], [-1, 0]], dtype=np.int64),
    np.asarray([[1, 0], [0, -1]], dtype=np.int64),
    np.asarray([[-1, 0], [0, 1]], dtype=np.int64),
    np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    np.asarray([[0, -1], [-1, 0]], dtype=np.int64),
)


def checkpoint_paths(resolution: int) -> tuple[Path, Path]:
    ckpt = OUT_ROOT / str(int(resolution)) / f"baseline_{int(resolution)}_hgradient.pt"
    csv = OUT_ROOT / str(int(resolution)) / f"baseline_{int(resolution)}_hgradient.csv"
    return ckpt, csv


def load_v2_checkpoint_bundle(resolution: int) -> dict:
    ckpt, csv = checkpoint_paths(resolution)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    if not csv.exists():
        raise FileNotFoundError(f"Missing normalization CSV: {csv}")
    bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_config = dict(bundle["model_config"])
    transform = validate_feature_transform(bundle["feature_transform"])
    if int(model_config["input_dim"]) != 27:
        raise ValueError(f"Expected input_dim=27, got {model_config['input_dim']!r}.")
    if transform["transform_kind"] != "standardize":
        raise ValueError(f"Expected transform_kind='standardize', got {transform['transform_kind']!r}.")
    if transform["feature_order"] != "phi9+nx9+ny9":
        raise ValueError(f"Expected V2 feature order, got {transform['feature_order']!r}.")
    if int(transform["raw_feature_dim"]) != 27 or int(transform["output_dim"]) != 27:
        raise ValueError(
            f"Expected raw/output feature dim 27, got raw={transform['raw_feature_dim']}, output={transform['output_dim']}."
        )
    return {
        **bundle,
        "model_config": model_config,
        "feature_transform": transform,
    }


def build_circle_raw27(
    resolution: int,
    *,
    radius: float,
    center: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | int]]:
    rho = int(resolution)
    h = 1.0 / (rho - 1)
    x, y = build_grid(rho)
    phi = build_circle_sdf(x, y, cx=float(center[0]), cy=float(center[1]), radius=float(radius))
    idx = interface_indices(phi)
    phi9 = extract_phi9(phi, idx)
    grad9 = extract_grad9(phi, idx)
    raw27 = np.concatenate([phi9 / np.float32(h), grad9[:, :, 0], grad9[:, :, 1]], axis=1).astype(np.float32)
    return raw27, {"h": float(h), "interface_count": int(len(idx))}


def d4_average_inputs(raw27: np.ndarray) -> list[np.ndarray]:
    raw = np.asarray(raw27, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 27:
        raise ValueError(f"Expected raw27 with shape (N, 27), got {raw.shape}.")
    phi9 = raw[:, :9]
    nx9 = raw[:, 9:18]
    ny9 = raw[:, 18:27]
    out: list[np.ndarray] = []
    for mat in _D4_MATRICES:
        phi_t = np.empty_like(phi9)
        nx_t = np.empty_like(nx9)
        ny_t = np.empty_like(ny9)
        for dst, offset in enumerate(_OFFSETS):
            src = _OFFSET_TO_INDEX[tuple((mat.T @ np.asarray(offset, dtype=np.int64)).tolist())]
            phi_t[:, dst] = phi9[:, src]
            vectors = np.stack([nx9[:, src], ny9[:, src]], axis=1)
            rotated = vectors @ mat.T.astype(np.float32)
            nx_t[:, dst] = rotated[:, 0]
            ny_t[:, dst] = rotated[:, 1]
        out.append(np.concatenate([phi_t, nx_t, ny_t], axis=1))
    return out
