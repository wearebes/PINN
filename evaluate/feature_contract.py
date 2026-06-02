from __future__ import annotations

from typing import Any


def use_scaled_h_features_for_inference(data_config: Any) -> bool:
    # Alpha-augmented training still deploys on the canonical phi / h input,
    # even when the original CLI did not pass --scale-h explicitly.
    return bool(getattr(data_config, "scale_h", False) or getattr(data_config, "augment_scale_alpha", ()))
