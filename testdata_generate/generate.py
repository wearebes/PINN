from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_generate.features import local_normal_features, FEATURE_MODES, feature_contract, resolve_feature_mode
from testdata_generate.config import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_CUSTOM_DATASET_NAME,
    DEFAULT_DATASET_NAME,
    DEFAULT_OUTPUT_DIR,
    TestDataConfig,
    available_rho_models,
    filter_scenarios_by_rho_model,
    flower_dataset_name,
    legacy_flower_scenarios,
    make_scenarios_for_rho_model,
    normalize_test_iters,
)
from testdata_generate.reinit import LevelSetReinitializer
from train_generate.geometry_core import interface_indices
from train_generate.generate import build_raw_features, extract_grad9

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None

_STORE_SCHEMA: tuple[tuple[str, tuple[int, ...], Any], ...] = (
    ("phi9", (0, 9), np.float32),
    ("features", (0, 9), np.float32),
    ("xy", (0, 2), np.float32),
    ("phi0_center", (0,), np.float32),
    ("hkappa_target", (0,), np.float32),
    ("case_id", (0,), np.int16),
    ("iter", (0,), np.int16),
    ("rho_model", (0,), np.int16),
    ("h", (0,), np.float32),
)


def build_grid(L: float, N: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(-float(L), float(L), int(N), dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    h = 2.0 * float(L) / (int(N) - 1)
    return X, Y, float(h)


def build_flower_phi0(X: np.ndarray, Y: np.ndarray, a: float, b: float, p: int) -> np.ndarray:
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return r - float(a) * np.cos(int(p) * theta) - float(b)


def find_projection_theta(
    xy: np.ndarray,
    a: float,
    b: float,
    p: int,
    *,
    tol: float = 1.0e-12,
    max_iter: int = 80,
) -> np.ndarray:
    two_pi = 2.0 * np.pi
    a = float(a)
    b = float(b)
    p = int(p)
    result = np.empty((xy.shape[0],), dtype=np.float64)

    def terms(theta: float) -> tuple[float, float, float, float, float, float]:
        ct, st = np.cos(theta), np.sin(theta)
        cpt, spt = np.cos(p * theta), np.sin(p * theta)
        r = b + a * cpt
        rp = -a * p * spt
        rpp = -a * p**2 * cpt
        cx = r * ct
        cy = r * st
        cpx = rp * ct - r * st
        cpy = rp * st + r * ct
        cppx = rpp * ct - 2.0 * rp * st - r * ct
        cppy = rpp * st + 2.0 * rp * ct - r * st
        return cx, cy, cpx, cpy, cppx, cppy

    for idx, (x, y) in enumerate(np.asarray(xy, dtype=np.float64)):
        theta = float(np.arctan2(y, x) % two_pi)
        converged = False
        for _ in range(max_iter):
            cx, cy, cpx, cpy, cppx, cppy = terms(theta)
            f = (cx - x) * cpx + (cy - y) * cpy
            fp = cpx * cpx + cpy * cpy + (cx - x) * cppx + (cy - y) * cppy
            if abs(f) <= tol:
                converged = True
                break
            if abs(fp) < 1.0e-18:
                break
            step = f / fp
            theta = float((theta - step) % two_pi)
            if abs(step) <= tol:
                converged = True
                break

        if not converged and _scipy_minimize is not None:
            def dist_sq(theta_array: np.ndarray) -> float:
                t = float(theta_array[0]) % two_pi
                radius = b + a * np.cos(p * t)
                return float((x - radius * np.cos(t)) ** 2 + (y - radius * np.sin(t)) ** 2)

            res = _scipy_minimize(dist_sq, np.array([theta], dtype=np.float64), tol=tol)
            theta = float(res.x[0] % two_pi)

        result[idx] = theta
    return result


def hkappa_analytic(theta_proj: np.ndarray, h: float, a: float, b: float, p: int) -> np.ndarray:
    a = float(a)
    b = float(b)
    p = int(p)
    theta_proj = np.asarray(theta_proj, dtype=np.float64)
    r = b + a * np.cos(p * theta_proj)
    rp = -a * p * np.sin(p * theta_proj)
    rpp = -a * p**2 * np.cos(p * theta_proj)
    kappa = (r**2 + 2.0 * rp**2 - r * rpp) / (r**2 + rp**2) ** 1.5
    return (float(h) * kappa).astype(np.float32, copy=False)


def _resolved_store_schema(*, feature_dim: int = 9) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
    if feature_dim == 9:
        return _STORE_SCHEMA
    return tuple(
        (name, (0, int(feature_dim)), dtype) if name == "features" else (name, shape, dtype)
        for name, shape, dtype in _STORE_SCHEMA
    )


def _empty_store() -> dict[str, list[np.ndarray]]:
    return {name: [] for name, _, _ in _STORE_SCHEMA}


def _append_case_iter(
    store: dict[str, list[np.ndarray]],
    *,
    phi: np.ndarray,
    phi0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    h: float,
    a: float,
    b: float,
    p: int,
    case_id: int,
    iteration: int,
    rho_model: int,
    scale_h: bool = False,
    augment_gradient: bool = False,
    feature_mode: str | None = None,
) -> int:
    indices = interface_indices(phi)
    if indices.size == 0:
        raise RuntimeError(f"No current-interface nodes for case_id={case_id}, iter={iteration}.")

    rows = indices[:, 0]
    cols = indices[:, 1]
    xy = np.column_stack((X[rows, cols], Y[rows, cols])).astype(np.float32, copy=False)
    theta_proj = find_projection_theta(xy, a, b, p)
    count = indices.shape[0]
    phi9, features = build_raw_features(phi, indices, scale_h=scale_h, h=h)
    if feature_mode == "phi9_local_normal":
        features = np.concatenate([features, local_normal_features(phi9)], axis=1)
    elif resolve_feature_mode(feature_mode, augment_gradient) != "phi9":
        grad9 = extract_grad9(
            phi, indices, center_only=feature_mode == "phi9_center_normal",
            cross_only=feature_mode == "phi9_cross_normal"
        )
        # Layout: [phi9_features | nx (9, 5 or 1) | ny (9, 5 or 1)].
        features = np.concatenate(
            [features, grad9[:, :, 0], grad9[:, :, 1]], axis=1
        ).astype(np.float32, copy=False)

    store["phi9"].append(phi9)
    store["features"].append(features)
    store["xy"].append(xy)
    store["phi0_center"].append(phi0[rows, cols].astype(np.float32, copy=False))
    store["hkappa_target"].append(hkappa_analytic(theta_proj, h, a, b, p))
    store["case_id"].append(np.full((count,), int(case_id), dtype=np.int16))
    store["iter"].append(np.full((count,), int(iteration), dtype=np.int16))
    store["rho_model"].append(np.full((count,), int(rho_model), dtype=np.int16))
    store["h"].append(np.full((count,), float(h), dtype=np.float32))
    return count


def _concat_store(store: dict[str, list[np.ndarray]], *, feature_dim: int = 9) -> dict[str, np.ndarray]:
    schema = _resolved_store_schema(feature_dim=feature_dim)
    return {
        name: np.concatenate(store[name], axis=0) if store[name] else np.zeros(shape, dtype=dtype)
        for name, shape, dtype in schema
    }

def _resolve_generation_config(config: TestDataConfig) -> TestDataConfig:
    base = replace(
        config,
        test_iters=normalize_test_iters(config.test_iters),
        scenarios=tuple(config.scenarios),
    )
    if not base.scenarios:
        raise ValueError("At least one flower scenario must be configured.")

    if base.config_source == "legacy_builtin":
        requested_rho_model = base.requested_rho_model if base.requested_rho_model is not None else base.rho_model
        if requested_rho_model is None:
            available = ", ".join(str(item) for item in available_rho_models(base.scenarios))
            raise ValueError(
                "TestDataConfig.rho_model must be set for legacy resolution-specific flower generation. "
                f"Available rho_model values: {available}."
            )
        filtered_scenarios = filter_scenarios_by_rho_model(base.scenarios, requested_rho_model)
        if not filtered_scenarios:
            # Not a legacy rho_model — auto-derive L and N via h = 1/(rho_model-1).
            filtered_scenarios = make_scenarios_for_rho_model(int(requested_rho_model))
            print(
                f"[testdata_generate] rho_model={int(requested_rho_model)} not in legacy set; "
                f"auto-computed scenarios: "
                + ", ".join(
                    f"{s.exp_id}(L={s.L:.6g}, N={s.N}, h={s.h:.6g})"
                    for s in filtered_scenarios
                )
            )
        dataset_name = base.dataset_name or flower_dataset_name(rho_model=int(requested_rho_model))
        return replace(
            base,
            rho_model=int(requested_rho_model),
            requested_rho_model=int(requested_rho_model),
            dataset_name=dataset_name,
            scenarios=filtered_scenarios,
        )

    unique_rho_models = available_rho_models(base.scenarios)
    dataset_name = base.dataset_name or (
        flower_dataset_name(rho_model=unique_rho_models[0]) if len(unique_rho_models) == 1 else DEFAULT_CUSTOM_DATASET_NAME
    )
    resolved_rho_model = unique_rho_models[0] if len(unique_rho_models) == 1 else None
    return replace(
        base,
        rho_model=resolved_rho_model,
        requested_rho_model=None,
        dataset_name=dataset_name,
    )


def generate_test_data(config: TestDataConfig | None = None, *, output: str | Path | None = None) -> Path:
    cfg = _resolve_generation_config(config or TestDataConfig())
    output_path = Path(output) if output is not None else cfg.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, output_dir=output_path.parent, dataset_name=output_path.name)
    scenario_rho_models = available_rho_models(cfg.scenarios)

    reinitializer = LevelSetReinitializer(
        indexing="ij",
        cfl=cfg.cfl,
        eps_weno=cfg.eps_weno,
        eps_sign_factor=cfg.eps_sign_factor,
        sign_mode=cfg.sign_mode,
        time_order=cfg.time_order,
        space_order=cfg.space_order,
    )
    store = _empty_store()
    counts: dict[str, int] = {}

    print(
        "[testdata_generate] "
        f"rho_models={','.join(str(item) for item in scenario_rho_models)} mode={cfg.mode} "
        f"sign_mode={cfg.sign_mode} cfl={cfg.cfl} "
        f"eps_sign_factor={cfg.eps_sign_factor} RK{cfg.time_order} WENO{cfg.space_order}"
    )
    feature_order = feature_contract(cfg.feature_mode, cfg.augment_gradient)[2]
    print(f"[testdata_generate] feature_order={feature_order} output={output_path}")

    for case_id, scenario in enumerate(cfg.scenarios):
        X, Y, h_from_grid = build_grid(scenario.L, scenario.N)
        if not np.isclose(h_from_grid, scenario.h, rtol=0.0, atol=1.0e-8):
            print(f"    using grid h={h_from_grid:.12g}; scenario h metadata={scenario.h:.12g}")
        phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
        phi = phi0.astype(np.float64, copy=True)
        print(f"  case={case_id} {scenario.exp_id}")

        if 0 in cfg.test_iters:
            count = _append_case_iter(
                store,
                phi=phi,
                phi0=phi0,
                X=X,
                Y=Y,
                h=h_from_grid,
                a=scenario.a,
                b=scenario.b,
                p=scenario.p,
                case_id=case_id,
                iteration=0,
                rho_model=scenario.rho_model,
                scale_h=bool(cfg.scale_h),
                augment_gradient=bool(cfg.augment_gradient),
                feature_mode=cfg.feature_mode,
            )
            counts[f"{scenario.exp_id}/iter_0"] = count
            print(f"    iter= 0 samples={count}")

        for iteration in range(1, max(cfg.test_iters) + 1):
            sign_reference = phi0 if cfg.sign_mode == "frozen_phi0" else None
            phi = reinitializer.reinitialize(
                phi,
                h_from_grid,
                1,
                sign_reference=sign_reference,
            )
            if iteration not in cfg.test_iters:
                continue
            count = _append_case_iter(
                store,
                phi=phi,
                phi0=phi0,
                X=X,
                Y=Y,
                h=h_from_grid,
                a=scenario.a,
                b=scenario.b,
                p=scenario.p,
                case_id=case_id,
                iteration=iteration,
                rho_model=scenario.rho_model,
                scale_h=bool(cfg.scale_h),
                augment_gradient=bool(cfg.augment_gradient),
                feature_mode=cfg.feature_mode,
            )
            counts[f"{scenario.exp_id}/iter_{iteration}"] = count
            print(f"    iter={iteration:>2d} samples={count}")

    feature_dim = feature_contract(cfg.feature_mode, cfg.augment_gradient)[1]
    arrays = _concat_store(store, feature_dim=feature_dim)
    _validate_arrays(arrays, cfg)
    _write_hdf5(output_path, arrays, cfg, counts)
    print(f"[testdata_generate] wrote {arrays['features'].shape[0]} samples")
    return output_path


def _validate_arrays(arrays: dict[str, np.ndarray], cfg: TestDataConfig) -> None:
    n = arrays["features"].shape[0]
    if arrays["phi9"].ndim != 2 or arrays["phi9"].shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {arrays['phi9'].shape}.")
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] not in (9, 11, 15, 19, 27):
        raise ValueError(f"features must have shape (N, 9), (N, 11), (N, 15), (N, 19), or (N, 27), got {arrays['features'].shape}.")
    if arrays["xy"].ndim != 2 or arrays["xy"].shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {arrays['xy'].shape}.")
    for key, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"{key} first dimension {value.shape[0]} does not match features length {n}.")
    if set(np.unique(arrays["case_id"]).tolist()) != set(range(len(cfg.scenarios))):
        raise ValueError("case_id does not cover all configured scenarios.")
    if set(np.unique(arrays["iter"]).tolist()) != set(int(item) for item in cfg.test_iters):
        raise ValueError("iter does not cover all configured test iterations.")
    if not np.isfinite(arrays["hkappa_target"]).all():
        raise ValueError("hkappa_target contains non-finite values.")

    max_abs_err = 0.0
    for case_id, scenario in enumerate(cfg.scenarios):
        mask = arrays["case_id"] == case_id
        if not np.any(mask):
            raise ValueError(f"No samples stored for scenario {scenario.exp_id}.")
        xy = arrays["xy"][mask].astype(np.float64)
        expected = build_flower_phi0(xy[:, 0], xy[:, 1], scenario.a, scenario.b, scenario.p)
        err = np.max(np.abs(expected - arrays["phi0_center"][mask].astype(np.float64)))
        max_abs_err = max(max_abs_err, float(err))
    if max_abs_err > 5.0e-7:
        raise ValueError(f"phi0_center formula check failed: max_abs_err={max_abs_err}.")
    actual_rho_models = set(np.unique(arrays["rho_model"]).tolist())
    expected_rho_models = {int(scenario.rho_model) for scenario in cfg.scenarios}
    if actual_rho_models != expected_rho_models:
        raise ValueError(
            f"Generated dataset rho_model values {sorted(actual_rho_models)} "
            f"do not match configured scenarios {sorted(expected_rho_models)}."
        )
    if cfg.requested_rho_model is not None and actual_rho_models != {int(cfg.requested_rho_model)}:
        raise ValueError(
            f"Generated dataset rho_model values {sorted(actual_rho_models)} "
            f"do not match requested rho_model={int(cfg.requested_rho_model)}."
        )

def _feature_version_attrs(raw_feature_dim: int, scale_h: bool) -> dict[str, Any]:
    if raw_feature_dim == 9:
        feature_version = 1
        feature_order = "phi9"
        feature_transform = "phi9_over_h" if scale_h else "phi9"
    elif raw_feature_dim == 11:
        feature_version = 4
        feature_order = "phi9+nx_center+ny_center"
        feature_transform = "phi9nx_centerny_center_over_h" if scale_h else feature_order
    elif raw_feature_dim == 15:
        feature_version, _, feature_order = feature_contract("phi9_local_normal")
        feature_transform = feature_order + ("_over_h" if scale_h else "")
    elif raw_feature_dim == 19:
        feature_version = 5
        feature_order = "phi9+nx_cross5+ny_cross5"
        feature_transform = "phi9nx_cross5ny_cross5_over_h" if scale_h else feature_order
    elif raw_feature_dim == 27:
        feature_version = 2
        feature_order = "phi9+nx9+ny9"
        feature_transform = "phi9nx9ny9_over_h" if scale_h else "phi9+nx9+ny9"
    else:
        raise ValueError(f"Unexpected feature dim {raw_feature_dim}; expected 9, 11, 15, 19, or 27.")
    return {
        "feature_version": feature_version,
        "feature_dim_raw": raw_feature_dim,
        "feature_order": feature_order,
        "scale_h": bool(scale_h),
        "feature_transform": feature_transform,
    }


def _write_hdf5(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    cfg: TestDataConfig,
    counts: dict[str, int],
) -> None:
    config_dict = {
        **{key: value for key, value in asdict(cfg).items() if key not in {"output_dir", "scenarios"}},
        "output_dir": str(cfg.output_dir),
        "scenarios": [scenario.as_dict() for scenario in cfg.scenarios],
    }
    attrs: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "mode": cfg.mode,
        "sign_mode": cfg.sign_mode,
        "cfl": float(cfg.cfl),
        "eps_weno": float(cfg.eps_weno),
        "eps_sign_factor": float(cfg.eps_sign_factor),
        "time_order": int(cfg.time_order),
        "space_order": int(cfg.space_order),
        "sampling_rule": cfg.sampling_rule,
        "stencil_encoding": cfg.stencil_encoding,
        "target_rule": cfg.target_rule,
        "augment_gradient": bool(cfg.augment_gradient),
        "feature_mode": resolve_feature_mode(cfg.feature_mode, cfg.augment_gradient),
        **_feature_version_attrs(arrays["features"].shape[1], cfg.scale_h),
        "method_code": cfg.method_code,
        "output_root": str(cfg.output_dir),
        "config_source": str(cfg.config_source),
        "scenarios_json": json.dumps([scenario.as_dict() for scenario in cfg.scenarios], ensure_ascii=True),
        "test_iters_json": json.dumps([int(item) for item in cfg.test_iters]),
        "sample_counts_json": json.dumps(counts, sort_keys=True),
        "config_json": json.dumps(config_dict, ensure_ascii=True, sort_keys=True),
    }
    if cfg.requested_rho_model is not None:
        attrs["requested_rho_model"] = int(cfg.requested_rho_model)
    with h5py.File(output_path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        for key, value in arrays.items():
            handle.create_dataset(key, data=value, compression="gzip")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if min(values) < 0:
        raise argparse.ArgumentTypeError("test iterations must be >= 0 (0 = raw phi0 before first reinit).")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    available = ", ".join(str(item) for item in available_rho_models(legacy_flower_scenarios()))
    parser = argparse.ArgumentParser(
        description="Generate flower test data for a given rho_model."
    )
    parser.add_argument(
        "--rho-model",
        type=int,
        required=True,
        help=(
            "Target rho_model (integer >= 4). "
            f"Legacy pre-tuned values: {available}. "
            "Any other value auto-computes L and N via h = 1/(rho_model-1) "
            "with a 2-cell margin around the interface."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=f"Full output path. Defaults to {DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_NAME}.",
    )
    parser.add_argument(
        "--test-iters",
        type=_parse_int_tuple,
        default=None,
        help="Comma-separated test iterations. Use 0 for raw phi0 before the first reinit step.",
    )
    parser.add_argument(
        "--scale-h",
        action="store_true",
        default=False,
        help="If set, write features = phi9 / h. Default off; features == phi9.",
    )
    parser.add_argument(
        "--augment-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append per-node normalised gradient directions (nx9, ny9) to phi9 features, "
            "yielding 27D features [phi9 | nx9 | ny9]."
        ),
    )
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=None,
                        help="Input feature mode; overrides --augment-gradient when specified.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    default_cfg = TestDataConfig()
    cfg = TestDataConfig(
        rho_model=int(args.rho_model),
        test_iters=normalize_test_iters(args.test_iters if args.test_iters is not None else default_cfg.test_iters),
        requested_rho_model=int(args.rho_model),
        scale_h=bool(args.scale_h),
        augment_gradient=bool(args.augment_gradient),
        feature_mode=args.feature_mode,
    )
    output = Path(args.output) if args.output else None
    generate_test_data(cfg, output=output)


if __name__ == "__main__":
    main()
