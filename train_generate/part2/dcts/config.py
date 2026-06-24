"""DCTS (Dimensionless-Curvature Targeted Sampling) configuration.

All design decisions are LOCKED per train_generate/part2/IMPLEMENTATION_PLAN.md
(v2.1, 2026-06-23). This module only encodes those locked values; it does not
re-open any of them.

Single dimensionless dataset (h=1). eta = |h*kappa| in [0.006, 2/3], approximated
by 100 log-uniform fine-bins. circle:ellipse = 30:70 per fine-bin.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi
from pathlib import Path


SPLITS = ("train", "val", "test")


def _mask64(value: int) -> int:
    return value & ((1 << 64) - 1)


def seed_from_parts(seed: int, *parts: int) -> int:
    """Deterministic 32-bit seed from a base seed and integer parts (FNV-1a style).

    Matches the spirit of train_generate.part2.generate._seed_from_parts so the
    "generate-until-quota" loop is exactly reproducible.
    """
    value = _mask64(int(seed) ^ 1469598103934665603)
    for part in parts:
        value = _mask64(value ^ _mask64(int(part) + 0x9E3779B97F4A7C15))
        value = _mask64(value * 1099511628211)
    return int(value % (1 << 32))


@dataclass(frozen=True)
class DctsConfig:
    # --- curvature binning (LOCKED) ---
    eta_min: float = 0.006
    eta_max: float = 2.0 / 3.0
    n_fine_bins: int = 100
    n_coarse_regimes: int = 8

    # --- per-fine-bin pack budget (canonical packs, before augmentation) ---
    per_bin: dict[str, int] = field(default_factory=lambda: {"train": 1000, "val": 200, "test": 200})
    circle_fraction: float = 0.30  # circle:ellipse = 30:70

    # --- ellipse geometry sweep (LOCKED) ---
    q_values: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90)
    psi_values: tuple[float, ...] = (0.0, pi / 12.0, pi / 6.0, pi / 4.0)
    ellipse_eta_max_scan: int = 6
    ellipse_dense_t: int = 4096
    ellipse_cap: dict[str, int] = field(default_factory=lambda: {"train": 24, "val": 12, "test": 12})

    # --- patch sampling ---
    d0_min: float = -0.5
    d0_max: float = 0.5

    # --- ellipse projection (float64 deterministic Newton, NO mpmath) ---
    newton_tol: float = 1.0e-12
    newton_max_iter: int = 30
    coarse_global_samples: int = 720

    # --- augmentation (D4 x sign), config-gated ---
    d4_sign_enabled: bool = True
    augment_splits: tuple[str, ...] = ("train",)  # val/test stored canonical-only

    # --- metadata (Stage 0: SDF-only by default, analytic normals) ---
    sdf_mode: str = "sdf"
    normal_source: str = "analytic"
    h: float = 1.0

    # --- non-SDF field axis (opt-in, OFF by default -- does not affect the
    # canonical "main"/"smoke" presets or anything already generated/trained).
    # When True, every canonical pack gets a second field representation built
    # from the SAME geometry parameters (circle: (dist)^2-r^2; ellipse:
    # u^2/a^2+v^2/b^2-1 -- exactly train_generate.geometry_core's
    # build_circle_nonsdf/build_ellipse_nonsdf convention), expanded as a third
    # row-multiplying axis alongside D4 x sign in augment.py. Target_hk, eta,
    # nx9/ny9 (normal direction is field-mode-invariant) are unchanged; only
    # phi9 and the downstream hk_central/features27 differ per field_mode. ---
    nonsdf_enabled: bool = False

    # --- determinism ---
    base_seed: int = 42

    # --- Gate 11 (mandatory diagnostic) thresholds on min||grad phi||_FD ---
    gate11_thresholds: tuple[float, ...] = (0.5, 0.2, 0.1)

    # --- deficiency policy: record, never silently oversample ---
    allow_deficiency: bool = True

    # --- consistency gate tolerances ---
    eta_consistency_tol: float = 1.0e-12

    # --- output ---
    output_dir: Path = Path("dataset/part2_dcts/main")
    tag: str = "main"

    # ---- derived quotas ----
    def circle_quota(self, split: str) -> int:
        return int(round(self.per_bin[split] * self.circle_fraction))

    def ellipse_quota(self, split: str) -> int:
        return int(self.per_bin[split] - self.circle_quota(split))

    def cap(self, split: str) -> int:
        return int(self.ellipse_cap[split])

    def split_seed(self, split: str) -> int:
        return seed_from_parts(self.base_seed, 7, SPLITS.index(split))

    def split_phase(self, split: str) -> float:
        """Fractional offset in [0,1) for the eta_max log-grid, per split.

        Guarantees train/val/test sample distinct eta_max values, so synthetic
        ellipse shapes never collide across splits (leakage hard gate).
        """
        return (SPLITS.index(split) + 0.5) / float(len(SPLITS))

    # ---- presets ----
    @classmethod
    def main(cls, **overrides) -> "DctsConfig":
        return cls(output_dir=Path("dataset/part2_dcts/main"), tag="main", **overrides)

    @classmethod
    def smoke(cls, **overrides) -> "DctsConfig":
        base = cls(
            per_bin={"train": 100, "val": 20, "test": 20},
            output_dir=Path("dataset/part2_dcts/smoke"),
            tag="smoke",
        )
        return replace(base, **overrides) if overrides else base
