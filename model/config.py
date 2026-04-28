from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def default_dataset_path(cfl: float = 0.5, setting: str = "setting1") -> Path:
    return Path("dataset") / f"train_{format(float(cfl), 'g')}_{setting}.h5"


def default_output_model_path() -> Path:
    return Path("out") / "pinn-3_curvature.pt"


@dataclass(frozen=True)
class TrainConfig:
    hidden_units: int = 128
    activation: str = "tanh"
    lr: float = 1.0e-4
    max_epochs: int = 500
    patience: int = 30
    lambda_traj: float = 1.0
    lambda_pde: float = 1.0
    lambda_interface: float = 1.0
    use_curvature_loss: bool = True
    lambda_curvature: float = 0.0001
    batch_size: int = 204800
    seed: int = 42
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)
