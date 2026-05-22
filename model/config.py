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
    lr: float = 1.0e-4
    max_epochs: int = 1000
    patience: int = 30
    batch_size: int = 256
    seed: int = 42
    compile_mode: str = "reduce-overhead"
    profile_cuda_timing: bool = False
    input_dim: int = 9
    raw_feature_dim: int = 9
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)


@dataclass(frozen=True)
class CNN_TrainConfig:
    kernel_size: int = 3
    padding: int = 1
    lr: float = 1.0e-4
    max_epochs: int = 1000
    patience: int = 30
    batch_size: int = 204800
    seed: int = 42
    compile_mode: str = "reduce-overhead"
    profile_cuda_timing: bool = False
    input_dim: int = 9
    raw_feature_dim: int = 9
    dataset_path: Path = field(default_factory=default_dataset_path)
    output_model_path: Path = field(default_factory=default_output_model_path)


TrainConfig = Union[MLP_TrainConfig, CNN_TrainConfig]


def _get_field_default(f: dataclasses.Field) -> Any:
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return f.default


def _config_class_for_model_type(model_type: ModelType) -> type[MLP_TrainConfig] | type[CNN_TrainConfig]:
    if model_type == "mlp":
        return MLP_TrainConfig
    if model_type == "cnn":
        return CNN_TrainConfig
    raise ValueError(f"Unknown model type: {model_type!r}. Expected 'mlp' or 'cnn'.")


def train_config_field_names(model_type: ModelType) -> set[str]:
    config_cls = _config_class_for_model_type(model_type)
    return {f.name for f in fields(config_cls)}


def filter_train_config_overrides(model_type: ModelType, overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    valid_fields = train_config_field_names(model_type)
    filtered = {key: value for key, value in overrides.items() if key in valid_fields}
    dropped = sorted(key for key in overrides.keys() if key not in valid_fields)
    return filtered, dropped


def create_train_config(model_type: ModelType, **overrides) -> TrainConfig:
    config_cls = _config_class_for_model_type(model_type)
    defaults = {f.name: _get_field_default(f) for f in fields(config_cls)}
    filtered_overrides, _ = filter_train_config_overrides(model_type, dict(overrides))
    defaults.update(filtered_overrides)
    return config_cls(**defaults)
