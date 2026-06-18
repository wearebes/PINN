from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    transform: dict
    transform_source: str
    checkpoint_path: Path
    device: Any
    metadata: dict[str, str | int]

    def predict_hkappa(self, raw_features: np.ndarray) -> np.ndarray:
        from evaluate.shared import predict_hkappa_full_batch

        return predict_hkappa_full_batch(
            self.model,
            np.asarray(raw_features, dtype=np.float32),
            transform=self.transform,
            device=self.device,
        )


def load_nn27_bundle(
    checkpoint_path: str | Path,
    *,
    feature_transform_path: str | Path | None = None,
    device: str | Any = "cpu",
) -> ModelBundle:
    import torch
    from evaluate.shared import load_model_from_checkpoint, resolve_feature_transform

    checkpoint = Path(checkpoint_path)
    torch_device = torch.device(device)
    model, checkpoint_meta = load_model_from_checkpoint(checkpoint, device=torch_device)
    transform, transform_source = resolve_feature_transform(
        model_path=checkpoint,
        explicit_path=feature_transform_path,
        checkpoint_meta=checkpoint_meta,
    )
    if int(transform["raw_feature_dim"]) != 27:
        raise ValueError(
            f"NN27 checkpoint requires raw_feature_dim=27, got {transform['raw_feature_dim']}."
        )
    feature_order = str(transform.get("feature_order", ""))
    if feature_order not in ("phi9+nx9+ny9", "pca18(phi9+nx9+ny9)"):
        raise ValueError(f"Unsupported NN27 feature_order={feature_order!r}.")
    metadata: dict[str, str | int] = {
        "raw_feature_dim": 27,
        "transform_source": str(transform_source),
        "feature_order": feature_order,
        "target_scale": "h*kappa",
        "checkpoint_hash": sha256_file(checkpoint),
        "feature_order_assumed": "phi9_top_to_bottom_training_order",
    }
    return ModelBundle(
        model=model,
        transform=transform,
        transform_source=str(transform_source),
        checkpoint_path=checkpoint,
        device=torch_device,
        metadata=metadata,
    )
