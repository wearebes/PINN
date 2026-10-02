from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


METHOD_CODE = "FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN"
DEFAULT_OUTPUT_DIR = Path("test_data")
DEFAULT_DATASET_NAME = f"test_{METHOD_CODE}_rho<RHO_MODEL>.h5"
DEFAULT_CUSTOM_DATASET_NAME = f"test_{METHOD_CODE}_custom.h5"
DATASET_SCHEMA_VERSION = 2


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


def compute_grid_spacing(*, L: float, N: int) -> float:
    return 2.0 * float(L) / (int(N) - 1)


def normalize_test_iters(values: Any) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"test_iters must be a list of non-negative integers, got {type(values).__name__}.")
    normalized = tuple(sorted({int(item) for item in values}))
    if not normalized:
        raise ValueError("test_iters must contain at least one non-negative integer.")
    if min(normalized) < 0:
        raise ValueError("test_iters must contain only non-negative integers.")
    return normalized


def legacy_flower_scenarios() -> tuple[FlowerScenario, ...]:
    return (
        FlowerScenario("smooth_256", "smooth", 256, 0.207843, 107, compute_grid_spacing(L=0.207843, N=107), 0.05, 0.15, 3),
        FlowerScenario("smooth_266", "smooth", 266, 0.207547, 111, compute_grid_spacing(L=0.207547, N=111), 0.05, 0.15, 3),
        FlowerScenario("smooth_276", "smooth", 276, 0.207339, 114, compute_grid_spacing(L=0.207339, N=114), 0.05, 0.15, 3),
        FlowerScenario("acute_256", "acute", 256, 0.232826, 120, compute_grid_spacing(L=0.232826, N=120), 0.075, 0.15, 3),
        FlowerScenario("acute_266", "acute", 266, 0.232563, 124, compute_grid_spacing(L=0.232563, N=124), 0.075, 0.15, 3),
        FlowerScenario("acute_276", "acute", 276, 0.232258, 129, compute_grid_spacing(L=0.232258, N=129), 0.075, 0.15, 3),
    )


# (exp_type, a, b, p) — the two canonical flower shapes used in all evaluations
_FLOWER_SHAPES: tuple[tuple[str, float, float, int], ...] = (
    ("smooth", 0.05, 0.15, 3),
    ("acute", 0.075, 0.15, 3),
)


def make_scenarios_for_rho_model(rho_model: int) -> tuple[FlowerScenario, ...]:
    """Build smooth + acute flower scenarios for any rho_model.

    Domain parameters are derived from rho_model so that h = 1/(rho_model-1)
    matches the training-data grid spacing exactly (or within one ULP when
    2*r_max*(rho_model-1) is not an integer, e.g. for acute shapes).

    Formula:
        h = 1 / (rho_model - 1)
        L = (b + a) + 2 * h          # interface fits with 2-cell margin
        N = round(2 * L / h) + 1     # grid points consistent with h
    """
    rho = int(rho_model)
    if rho < 4:
        raise ValueError(f"rho_model must be >= 4, got {rho}.")
    h = 1.0 / (rho - 1)
    scenarios: list[FlowerScenario] = []
    for exp_type, a, b, p in _FLOWER_SHAPES:
        r_max = b + a
        L = r_max + 2.0 * h
        N = round(2.0 * L / h) + 1
        scenarios.append(FlowerScenario(
            exp_id=f"{exp_type}_{rho}",
            experiment_type=exp_type,
            rho_model=rho,
            L=L,
            N=N,
            h=compute_grid_spacing(L=L, N=N),
            a=a,
            b=b,
            p=p,
        ))
    return tuple(scenarios)


def default_custom_dataset_name() -> str:
    return DEFAULT_CUSTOM_DATASET_NAME


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
    rho_model: int | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    dataset_name: str = ""
    scenarios: tuple[FlowerScenario, ...] = field(default_factory=legacy_flower_scenarios)
    config_source: str = "legacy_builtin"
    requested_rho_model: int | None = None
    scale_h: bool = False
    augment_gradient: bool = False
    # None preserves the legacy augment_gradient switch; explicit mode takes precedence.
    feature_mode: str | None = None

    def __post_init__(self) -> None:
        from train_generate.features import resolve_feature_mode

        mode = resolve_feature_mode(self.feature_mode, self.augment_gradient)
        object.__setattr__(self, "augment_gradient", mode != "phi9")

    def output_path(self) -> Path:
        dataset_name = self.dataset_name
        if not dataset_name:
            if self.rho_model is not None:
                dataset_name = flower_dataset_name(rho_model=self.rho_model)
            elif len(available_rho_models(self.scenarios)) == 1:
                dataset_name = flower_dataset_name(rho_model=available_rho_models(self.scenarios)[0])
            else:
                dataset_name = default_custom_dataset_name()
        return Path(self.output_dir) / dataset_name


def flower_dataset_name(*, rho_model: int) -> str:
    return f"test_{METHOD_CODE}_rho{int(rho_model)}.h5"


def available_rho_models(scenarios: tuple[FlowerScenario, ...]) -> tuple[int, ...]:
    return tuple(sorted({int(scenario.rho_model) for scenario in scenarios}))


def filter_scenarios_by_rho_model(
    scenarios: tuple[FlowerScenario, ...],
    rho_model: int,
) -> tuple[FlowerScenario, ...]:
    target = int(rho_model)
    return tuple(scenario for scenario in scenarios if int(scenario.rho_model) == target)


