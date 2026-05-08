from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Union


ModelType = Literal["mlp", "cnn"]


def default_dataset_path() -> Path:
    return Path("dataset") / "256.h5"


def default_output_model_path() -> Path:
    return Path("out") / "best_stencil_hkappa.pt"


@dataclass(frozen=True)
class MLP_TrainConfig:
    hidden_units: int = 128
    activation: str = "relu"
    lr: float = 1.0e-4
    max_epochs: int = 1000
    patience: int = 30
    batch_size: int = 32
    seed: int = 42
    compile_mode: str = "none"
    grad_accum_steps: int = 1
    profile_cuda_timing: bool = False
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)


@dataclass(frozen=True)
class CNN_TrainConfig:
    kernel_size: int = 3
    padding: int = 1
    activation: str = "relu"
    lr: float = 1.0e-4
    max_epochs: int = 1000
    patience: int = 30
    batch_size: int = 204800
    seed: int = 42
    compile_mode: str = "none"
    grad_accum_steps: int = 1
    profile_cuda_timing: bool = False
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)


TrainConfig = Union[MLP_TrainConfig, CNN_TrainConfig]


def _get_field_default(f: dataclasses.Field) -> Any:
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return f.default


def create_train_config(model_type: ModelType, **overrides) -> TrainConfig:
    if model_type == "mlp":
        defaults = {f.name: _get_field_default(f) for f in fields(MLP_TrainConfig)}
        defaults.update(overrides)
        return MLP_TrainConfig(**defaults)
    if model_type == "cnn":
        defaults = {f.name: _get_field_default(f) for f in fields(CNN_TrainConfig)}
        defaults.update(overrides)
        return CNN_TrainConfig(**defaults)
    raise ValueError(f"Unknown model type: {model_type!r}. Expected 'mlp' or 'cnn'.")
