from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def default_dataset_name(cfl: float = 0.5, setting: str = "setting1") -> str:
    return f"train_{format(float(cfl), 'g')}_{setting}.h5"


def default_output_dir() -> Path:
    return Path("dataset")


@dataclass(frozen=True)
class ReinitConfig:
    cfl: float = 0.5
    eps_weno: float = 1.0e-6
    eps_sign_factor: float = 1.0
    sign_mode: str = "frozen_phi0"
    time_order: int = 3
    space_order: int = 5


@dataclass(frozen=True)
class DataConfig:
    resolutions: tuple[int, ...] = (256, 266, 276)
    geometry_seed: int = 42
    variations: int = 5
    initial_field_type: str = "nonsdf"
    reinit_steps: tuple[int, ...] = (5, 10, 15, 20)
    n_traj_per_time: int = 1024
    n_pde_per_circle: int = 50000
    n_interface_per_time: int = 1024
    band_half_width_cells: float = 3.0
    proposal_half_width_cells: float = 3.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    validation_field_limit: int = 20


@dataclass(frozen=True)
class GenerationConfig:
    num_workers: int = 14
    generation_batch_size: int = 1024
    output_dir: Path = field(default_factory=default_output_dir)
    dataset_name: str = ""
    stored_sample_fields: tuple[str, ...] = ()

    def output_path(self) -> Path:
        return Path(self.output_dir) / (self.dataset_name or default_dataset_name())
