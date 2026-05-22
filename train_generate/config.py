from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from math import pi


def default_dataset_name() -> str:
    return "256.h5"


def default_output_dir() -> Path:
    return Path("dataset")


@dataclass(frozen=True)
class DataConfig:
    resolutions: tuple[int, ...] = (256,)
    geometry_seed: int = 42
    variations: int = 12
    initial_field_types: tuple[str, ...] = ("sdf", "nonsdf")
    augment_sign_flip: bool = True
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    shape_types: tuple[str, ...] = ("circle", "ellipse")
    ellipse_num_a: int = 48
    ellipse_variations_per_a: int = 124
    ellipse_axis_ratio_min: float = 0.50
    ellipse_axis_ratio_max: float = 0.9
    ellipse_rotation_min: float = 0.0
    ellipse_rotation_max: float = pi
    ellipse_a_min_factor: float = 8.0
    ellipse_sdf_newton_max_iter: int = 30
    ellipse_sdf_newton_tol: float = 1.0e-12
    ellipse_hp_dps: int = 80
    ellipse_hp_newton_max_iter: int = 100


@dataclass(frozen=True)
class GenerationConfig:
    num_workers: int = 24
    generation_batch_size: int = 10240
    output_dir: Path = field(default_factory=default_output_dir)
    dataset_name: str = ""

    def output_path(self) -> Path:
        return Path(self.output_dir) / (self.dataset_name or default_dataset_name())
