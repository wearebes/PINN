from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

from train_generate.geometry_core import (
    DEFAULT_ELLIPSE_HP_DPS,
    DEFAULT_ELLIPSE_HP_NEWTON_MAX_ITER,
    DEFAULT_ELLIPSE_SDF_NEWTON_MAX_ITER,
    DEFAULT_ELLIPSE_SDF_NEWTON_TOL,
)


D4_NAMES = ("e", "R90", "R180", "R270", "Mx", "My", "Md", "Ma")
D4_MATRICES = {
    "e": np.asarray([[1, 0], [0, 1]], dtype=np.int64),
    "R90": np.asarray([[0, -1], [1, 0]], dtype=np.int64),
    "R180": np.asarray([[-1, 0], [0, -1]], dtype=np.int64),
    "R270": np.asarray([[0, 1], [-1, 0]], dtype=np.int64),
    "Mx": np.asarray([[1, 0], [0, -1]], dtype=np.int64),
    "My": np.asarray([[-1, 0], [0, 1]], dtype=np.int64),
    "Md": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    "Ma": np.asarray([[0, -1], [-1, 0]], dtype=np.int64),
}

REPORT_FILENAMES = (
    "blueprint_inventory.csv",
    "node_count_summary.csv",
    "curvature_geometry_weighted_summary.csv",
    "curvature_sample_weighted_summary.csv",
    "feature_scale_summary.csv",
    "hk_central_sanity_summary.csv",
    "projection_quality_summary.csv",
    "augmentation_contract_summary.json",
    "feature_contract_summary.json",
)


@dataclass(frozen=True)
class Part2Config:
    rho: int = 64
    seed: int = 42
    output_dir: Path = Path("dataset/part2_dryrun/rho64")
    circle_eta_levels: int = 20
    circle_phase_count: int = 10
    ellipse_count: int = 800
    initial_field_types: tuple[str, ...] = ("sdf", "nonsdf")
    epsilon_zero: float = 1.0e-14
    epsilon_projection_report: float = 1.0e-12
    epsilon_distance_report: float = 1.0e-12
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    ellipse_axis_ratio_min: float = 0.50
    ellipse_axis_ratio_max: float = 0.9
    ellipse_rotation_min: float = 0.0
    ellipse_rotation_max: float = pi
    ellipse_a_min_factor: float = 8.0
    ellipse_sdf_newton_max_iter: int = DEFAULT_ELLIPSE_SDF_NEWTON_MAX_ITER
    ellipse_sdf_newton_tol: float = DEFAULT_ELLIPSE_SDF_NEWTON_TOL
    ellipse_hp_dps: int = DEFAULT_ELLIPSE_HP_DPS
    ellipse_hp_newton_max_iter: int = DEFAULT_ELLIPSE_HP_NEWTON_MAX_ITER

    @property
    def h(self) -> float:
        return 1.0 / (int(self.rho) - 1)

    @property
    def r_min(self) -> float:
        return 1.6 * self.h

    @property
    def r_max(self) -> float:
        return 0.5 - 2.0 * self.h

    @property
    def eta_min(self) -> float:
        return self.h / self.r_max

    @property
    def eta_max(self) -> float:
        return self.h / self.r_min

    @property
    def a_min(self) -> float:
        return float(self.ellipse_a_min_factor) * self.h

    @property
    def a_max(self) -> float:
        return 0.5 - 2.0 * self.h

    @property
    def center_min(self) -> float:
        return 0.5 - self.h / 2.0

    @property
    def center_max(self) -> float:
        return 0.5 + self.h / 2.0


def make_part2_config(
    *,
    rho: int = 64,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Part2Config:
    resolved_output = Path(output_dir) if output_dir is not None else Path(f"dataset/part2_dryrun/rho{int(rho)}")
    return Part2Config(rho=int(rho), seed=int(seed), output_dir=resolved_output)
