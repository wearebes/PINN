from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def default_dataset_name(setting: str = "setting1") -> str:
    return f"train_stencil_{setting}.h5"


def default_output_dir() -> Path:
    return Path("dataset")


@dataclass(frozen=True)
class DataConfig:
    resolutions: tuple[int, ...] = (256,)
    geometry_seed: int = 42
    variations: int = 5
    initial_field_types: tuple[str, ...] = ("sdf", "nonsdf")
    n_samples_per_circle: int = 4096
    train_fraction: float = 0.70
    val_fraction: float = 0.15


@dataclass(frozen=True)
class GenerationConfig:
    num_workers: int = 12
    generation_batch_size: int = 8192
    output_dir: Path = field(default_factory=default_output_dir)
    dataset_name: str = ""

    def output_path(self) -> Path:
        return Path(self.output_dir) / (self.dataset_name or default_dataset_name())
