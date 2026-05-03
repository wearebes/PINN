from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


METHOD_CODE = "FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN"
DEFAULT_OUTPUT_DIR = Path("test_data")
DEFAULT_DATASET_NAME = f"test_{METHOD_CODE}.h5"
DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FlowerScenario:
    exp_id: str
    experiment_type: str
    rho_model: int
    L: float
    N: int
    h: float
    a: float
    b: float
    p: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TestDataConfig:
    mode: str = "formula_phi0"
    sign_mode: str = "dynamic_phi"
    cfl: float = 0.5
    eps_weno: float = 1.0e-6
    eps_sign_factor: float = 2.5
    time_order: int = 3
    space_order: int = 5
    test_iters: tuple[int, ...] = tuple(range(1, 21))
    sampling_rule: str = "current_interface_nodes"
    stencil_encoding: str = "training_order"
    target_rule: str = "analytic_projection_current_nodes"
    method_code: str = METHOD_CODE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    dataset_name: str = DEFAULT_DATASET_NAME
    scenarios: tuple[FlowerScenario, ...] = field(
        default_factory=lambda: (
            FlowerScenario("smooth_256", "smooth", 256, 0.207843, 107, 3.921569e-3, 0.05, 0.15, 3),
            FlowerScenario("smooth_266", "smooth", 266, 0.207547, 111, 3.773585e-3, 0.05, 0.15, 3),
            FlowerScenario("smooth_276", "smooth", 276, 0.207339, 114, 3.669724e-3, 0.05, 0.15, 3),
            FlowerScenario("acute_256", "acute", 256, 0.232826, 120, 3.913043e-3, 0.075, 0.15, 3),
            FlowerScenario("acute_266", "acute", 266, 0.232563, 124, 3.781513e-3, 0.075, 0.15, 3),
            FlowerScenario("acute_276", "acute", 276, 0.232258, 129, 3.629032e-3, 0.075, 0.15, 3),
        )
    )

    def output_path(self) -> Path:
        return Path(self.output_dir) / self.dataset_name

