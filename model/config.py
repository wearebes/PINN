from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def default_dataset_path(setting: str = "setting1") -> Path:
    return Path("dataset") / f"train_stencil_{setting}.h5"


def default_output_model_path() -> Path:
    return Path("out") / "256circle.pt"


@dataclass(frozen=True)
class TrainConfig:
    hidden_units: int = 128
    lr: float = 1.0e-4
    max_epochs: int = 500
    patience: int = 30
    batch_size: int = 2048
    seed: int = 42
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)
