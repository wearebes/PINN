from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


METHOD_CODE = "FP0_DynSign_CFL0.5_EPS2.5_RK3_WENO5_CIN_ST9_APCN"
DEFAULT_OUTPUT_DIR = Path("test_data")
DEFAULT_DATASET_NAME = f"test_{METHOD_CODE}_rho<RHO_MODEL>.h5"
DEFAULT_CUSTOM_DATASET_NAME = f"test_{METHOD_CODE}_custom.h5"
DATASET_SCHEMA_VERSION = 2
_TOP_LEVEL_REQUIRED_KEYS = frozenset({"scenarios"})
_TOP_LEVEL_OPTIONAL_KEYS = frozenset({"dataset_name", "output_dir", "test_iters"})
_SCENARIO_REQUIRED_KEYS = frozenset({"exp_id", "experiment_type", "rho_model", "L", "N", "a", "b", "p"})
_SCENARIO_OPTIONAL_KEYS = frozenset({"h"})


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
        raise ValueError(f"test_iters must be a list of positive integers, got {type(values).__name__}.")
    normalized = tuple(sorted({int(item) for item in values}))
    if not normalized:
        raise ValueError("test_iters must contain at least one positive integer.")
    if min(normalized) < 1:
        raise ValueError("test_iters must contain only positive integers.")
    return normalized


def _resolve_flower_scenario(raw: Mapping[str, Any], *, source: str, index: int) -> FlowerScenario:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Scenario #{index} in {source} must be a JSON object.")
    unknown = set(raw.keys()) - (_SCENARIO_REQUIRED_KEYS | _SCENARIO_OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"Scenario #{index} in {source} contains unsupported keys: {sorted(unknown)}.")
    missing = _SCENARIO_REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Scenario #{index} in {source} is missing required keys: {sorted(missing)}.")

    exp_id = str(raw["exp_id"])
    experiment_type = str(raw["experiment_type"])
    rho_model = int(raw["rho_model"])
    L = float(raw["L"])
    N = int(raw["N"])
    a = float(raw["a"])
    b = float(raw["b"])
    p = int(raw["p"])

    if not exp_id:
        raise ValueError(f"Scenario #{index} in {source} has an empty exp_id.")
    if rho_model <= 0:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have rho_model > 0.")
    if L <= 0.0:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have L > 0.")
    if N < 3:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have N >= 3.")
    if a <= 0.0:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have a > 0.")
    if b <= 0.0:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have b > 0.")
    if p < 1:
        raise ValueError(f"Scenario {exp_id!r} in {source} must have p >= 1.")

    derived_h = compute_grid_spacing(L=L, N=N)
    if "h" in raw:
        provided_h = float(raw["h"])
        if not math.isclose(provided_h, derived_h, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError(
                f"Scenario {exp_id!r} in {source} has inconsistent h={provided_h:.12g}; "
                f"expected {derived_h:.12g} from L={L} and N={N}."
            )

    return FlowerScenario(
        exp_id=exp_id,
        experiment_type=experiment_type,
        rho_model=rho_model,
        L=L,
        N=N,
        h=derived_h,
        a=a,
        b=b,
        p=p,
    )


def legacy_flower_scenarios() -> tuple[FlowerScenario, ...]:
    return (
        FlowerScenario("smooth_256", "smooth", 256, 0.207843, 107, compute_grid_spacing(L=0.207843, N=107), 0.05, 0.15, 3),
        FlowerScenario("smooth_266", "smooth", 266, 0.207547, 111, compute_grid_spacing(L=0.207547, N=111), 0.05, 0.15, 3),
        FlowerScenario("smooth_276", "smooth", 276, 0.207339, 114, compute_grid_spacing(L=0.207339, N=114), 0.05, 0.15, 3),
        FlowerScenario("acute_256", "acute", 256, 0.232826, 120, compute_grid_spacing(L=0.232826, N=120), 0.075, 0.15, 3),
        FlowerScenario("acute_266", "acute", 266, 0.232563, 124, compute_grid_spacing(L=0.232563, N=124), 0.075, 0.15, 3),
        FlowerScenario("acute_276", "acute", 276, 0.232258, 129, compute_grid_spacing(L=0.232258, N=129), 0.075, 0.15, 3),
    )


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


def load_scenario_config(path: str | Path) -> TestDataConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Scenario config {config_path.resolve()} must be a JSON object.")

    unknown = set(raw.keys()) - (_TOP_LEVEL_REQUIRED_KEYS | _TOP_LEVEL_OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"Scenario config {config_path.resolve()} contains unsupported keys: {sorted(unknown)}.")
    missing = _TOP_LEVEL_REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Scenario config {config_path.resolve()} is missing required keys: {sorted(missing)}.")

    scenarios_raw = raw["scenarios"]
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError(f"Scenario config {config_path.resolve()} must define a non-empty scenarios list.")

    scenarios = tuple(
        _resolve_flower_scenario(item, source=str(config_path.resolve()), index=index)
        for index, item in enumerate(scenarios_raw, start=1)
    )
    exp_ids = [scenario.exp_id for scenario in scenarios]
    if len(set(exp_ids)) != len(exp_ids):
        raise ValueError(f"Scenario config {config_path.resolve()} must use unique exp_id values.")

    default_cfg = TestDataConfig()
    test_iters = normalize_test_iters(raw.get("test_iters", default_cfg.test_iters))
    unique_rho_models = available_rho_models(scenarios)
    dataset_name = str(raw.get("dataset_name", "") or "")
    if not dataset_name:
        dataset_name = flower_dataset_name(rho_model=unique_rho_models[0]) if len(unique_rho_models) == 1 else default_custom_dataset_name()

    output_dir_raw = raw.get("output_dir", default_cfg.output_dir)
    output_dir = Path(output_dir_raw)
    rho_model = unique_rho_models[0] if len(unique_rho_models) == 1 else None
    return TestDataConfig(
        test_iters=test_iters,
        rho_model=rho_model,
        output_dir=output_dir,
        dataset_name=dataset_name,
        scenarios=scenarios,
        config_source=str(config_path.resolve()),
        requested_rho_model=None,
    )
