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
    from testdata_generate.config import (
        DATASET_SCHEMA_VERSION,
        DEFAULT_DATASET_NAME,
        DEFAULT_OUTPUT_DIR,
        TestDataConfig,
        available_rho_models,
        filter_scenarios_by_rho_model,
        flower_dataset_name,
    )
    from .reinit import LevelSetReinitializer
else:
    from .config import (
        DATASET_SCHEMA_VERSION,
        DEFAULT_DATASET_NAME,
        DEFAULT_OUTPUT_DIR,
        TestDataConfig,
        available_rho_models,
        filter_scenarios_by_rho_model,
        flower_dataset_name,
    )
    from .reinit import LevelSetReinitializer

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None


_PHI9_ROW_OFFSETS = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)
_PHI9_COL_OFFSETS = np.asarray([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int64)


def build_grid(L: float, N: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(-float(L), float(L), int(N), dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    h = 2.0 * float(L) / (int(N) - 1)
    return X, Y, float(h)


def build_flower_phi0(X: np.ndarray, Y: np.ndarray, a: float, b: float, p: int) -> np.ndarray:
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return r - float(a) * np.cos(int(p) * theta) - float(b)


def interface_indices(phi: np.ndarray) -> np.ndarray:
    scx = phi[:-1, :] * phi[1:, :] <= 0.0
    scy = phi[:, :-1] * phi[:, 1:] <= 0.0
    mask = np.zeros_like(phi, dtype=bool)
    ix, jx = np.where(scx)
    iy, jy = np.where(scy)
    mask[ix, jx] = True
    mask[ix + 1, jx] = True
    mask[iy, jy] = True
    mask[iy, jy + 1] = True
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((rows, cols)).astype(np.int64, copy=False)


def extract_stencil_values(field: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 9), dtype=np.float32)
    rows = indices[:, 0]
    cols = indices[:, 1]
    row_idx = rows[:, None] + _PHI9_ROW_OFFSETS.reshape(1, 9)
    col_idx = cols[:, None] + _PHI9_COL_OFFSETS.reshape(1, 9)
    return np.asarray(field[row_idx, col_idx], dtype=np.float32)


def extract_phi9(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return extract_stencil_values(phi, indices)


def central_difference_gradient(phi: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float64)
    h = float(h)
    phix = np.empty_like(phi, dtype=np.float64)
    phiy = np.empty_like(phi, dtype=np.float64)
    phix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phix[:, 0] = (phi[:, 1] - phi[:, 0]) / h
    phix[:, -1] = (phi[:, -1] - phi[:, -2]) / h
    phiy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    phiy[0, :] = (phi[1, :] - phi[0, :]) / h
    phiy[-1, :] = (phi[-1, :] - phi[-2, :]) / h
    return phix, phiy


def raw_feature_dim(feature_version: int) -> int:
    return 9 if int(feature_version) == 1 else 27


def build_raw_features(phi: np.ndarray, indices: np.ndarray, *, h: float, feature_version: int, gradient_epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    phi9 = extract_phi9(phi, indices)
    if int(feature_version) == 1:
        return phi9, phi9.copy()
    phix, phiy = central_difference_gradient(phi, h)
    phix9 = extract_stencil_values(phix, indices)
    phiy9 = extract_stencil_values(phiy, indices)
    denom = np.sqrt(phix9.astype(np.float64) ** 2 + phiy9.astype(np.float64) ** 2 + float(gradient_epsilon))
    nx9 = (phix9 / denom).astype(np.float32, copy=False)
    ny9 = (phiy9 / denom).astype(np.float32, copy=False)
    features = np.concatenate((phi9, nx9, ny9), axis=1).astype(np.float32, copy=False)
    return phi9, features

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


def _empty_store() -> dict[str, list[np.ndarray]]:
    return {
        "phi9": [],
        "features": [],
        "xy": [],
        "phi0_center": [],
        "hkappa_target": [],
        "case_id": [],
        "iter": [],
        "rho_model": [],
        "h": [],
    }


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
    feature_version: int,
    gradient_epsilon: float,
) -> int:
    indices = interface_indices(phi)
    if indices.size == 0:
        raise RuntimeError(f"No current-interface nodes for case_id={case_id}, iter={iteration}.")

    rows = indices[:, 0]
    cols = indices[:, 1]
    xy = np.column_stack((X[rows, cols], Y[rows, cols])).astype(np.float32, copy=False)
    theta_proj = find_projection_theta(xy, a, b, p)
    count = indices.shape[0]
    phi9, features = build_raw_features(
        phi,
        indices,
        h=h,
        feature_version=feature_version,
        gradient_epsilon=gradient_epsilon,
    )

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


def _concat_store(store: dict[str, list[np.ndarray]], *, feature_dim: int) -> dict[str, np.ndarray]:
    shapes = {
        "phi9": (0, 9),
        "features": (0, feature_dim),
        "xy": (0, 2),
        "phi0_center": (0,),
        "hkappa_target": (0,),
        "case_id": (0,),
        "iter": (0,),
        "rho_model": (0,),
        "h": (0,),
    }
    dtypes = {
        "phi9": np.float32,
        "features": np.float32,
        "xy": np.float32,
        "phi0_center": np.float32,
        "hkappa_target": np.float32,
        "case_id": np.int16,
        "iter": np.int16,
        "rho_model": np.int16,
        "h": np.float32,
    }
    return {
        key: np.concatenate(values, axis=0) if values else np.zeros(shapes[key], dtype=dtypes[key])
        for key, values in store.items()
    }

def _resolve_generation_config(config: TestDataConfig) -> TestDataConfig:
    if config.rho_model is None:
        available = ", ".join(str(item) for item in available_rho_models(config.scenarios))
        raise ValueError(
            "TestDataConfig.rho_model must be set for resolution-specific flower generation. "
            f"Available rho_model values: {available}."
        )
    filtered_scenarios = filter_scenarios_by_rho_model(config.scenarios, config.rho_model)
    if not filtered_scenarios:
        available = ", ".join(str(item) for item in available_rho_models(config.scenarios))
        raise ValueError(
            f"No flower scenarios configured for rho_model={int(config.rho_model)}. "
            f"Available rho_model values: {available}."
        )
    dataset_name = config.dataset_name or flower_dataset_name(rho_model=config.rho_model)
    return replace(
        config,
        rho_model=int(config.rho_model),
        feature_version=int(config.feature_version),
        dataset_name=dataset_name,
        scenarios=filtered_scenarios,
    )


def generate_test_data(config: TestDataConfig | None = None, *, output: str | Path | None = None) -> Path:
    cfg = _resolve_generation_config(config or TestDataConfig())
    output_path = Path(output) if output is not None else cfg.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, output_dir=output_path.parent, dataset_name=output_path.name)

    reinitializer = LevelSetReinitializer(
        indexing="xy",
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
        f"rho_model={cfg.rho_model} mode={cfg.mode} sign_mode={cfg.sign_mode} cfl={cfg.cfl} "
        f"eps_sign_factor={cfg.eps_sign_factor} RK{cfg.time_order} WENO{cfg.space_order}"
    )
    print(f"[testdata_generate] feature_version={cfg.feature_version} output={output_path}")

    for case_id, scenario in enumerate(cfg.scenarios):
        X, Y, h_from_grid = build_grid(scenario.L, scenario.N)
        if not np.isclose(h_from_grid, scenario.h, rtol=0.0, atol=1.0e-8):
            print(f"    using grid h={h_from_grid:.12g}; scenario h metadata={scenario.h:.12g}")
        phi0 = build_flower_phi0(X, Y, scenario.a, scenario.b, scenario.p)
        phi = phi0.astype(np.float64, copy=True)
        print(f"  case={case_id} {scenario.exp_id}")

        for iteration in range(1, max(cfg.test_iters) + 1):
            phi = reinitializer.reinitialize(phi, h_from_grid, 1)
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
                feature_version=cfg.feature_version,
                gradient_epsilon=cfg.gradient_epsilon,
            )
            counts[f"{scenario.exp_id}/iter_{iteration}"] = count
            print(f"    iter={iteration:>2d} samples={count}")

    arrays = _concat_store(store, feature_dim=raw_feature_dim(cfg.feature_version))
    _validate_arrays(arrays, cfg)
    _write_hdf5(output_path, arrays, cfg, counts)
    print(f"[testdata_generate] wrote {arrays['features'].shape[0]} samples")
    return output_path


def _validate_arrays(arrays: dict[str, np.ndarray], cfg: TestDataConfig) -> None:
    n = arrays["features"].shape[0]
    if arrays["phi9"].ndim != 2 or arrays["phi9"].shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {arrays['phi9'].shape}.")
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != raw_feature_dim(cfg.feature_version):
        raise ValueError(f"features must have shape (N, {raw_feature_dim(cfg.feature_version)}), got {arrays['features'].shape}.")
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
    if set(np.unique(arrays["rho_model"]).tolist()) != {int(cfg.rho_model)}:
        raise ValueError(
            f"Generated dataset rho_model values {sorted(np.unique(arrays['rho_model']).tolist())} "
            f"do not match requested rho_model={int(cfg.rho_model)}."
        )

def _write_hdf5(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    cfg: TestDataConfig,
    counts: dict[str, int],
) -> None:
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
        "feature_version": int(cfg.feature_version),
        "feature_dim_raw": int(raw_feature_dim(cfg.feature_version)),
        "feature_order": "phi9" if int(cfg.feature_version) == 1 else "phi9_nx9_ny9",
        "gradient_epsilon": float(cfg.gradient_epsilon),
        "method_code": cfg.method_code,
        "output_root": str(cfg.output_dir),
        "scenarios_json": json.dumps([scenario.as_dict() for scenario in cfg.scenarios], ensure_ascii=True),
        "test_iters_json": json.dumps([int(item) for item in cfg.test_iters]),
        "sample_counts_json": json.dumps(counts, sort_keys=True),
        "config_json": json.dumps(
            {
                **{key: value for key, value in asdict(cfg).items() if key not in {"output_dir", "scenarios"}},
                "output_dir": str(cfg.output_dir),
                "scenarios": [scenario.as_dict() for scenario in cfg.scenarios],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    with h5py.File(output_path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        for key, value in arrays.items():
            handle.create_dataset(key, data=value, compression="gzip")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if min(values) < 1:
        raise argparse.ArgumentTypeError("test iterations must be >= 1.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    default_cfg = TestDataConfig()
    available = ", ".join(str(item) for item in available_rho_models(default_cfg.scenarios))
    parser = argparse.ArgumentParser(description="Generate independent flower test data into test_data/, one rho_model per file.")
    parser.add_argument(
        "--rho-model",
        type=int,
        required=True,
        help=f"Target rho_model to generate. Available values: {available}.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=f"Full output path. Defaults to {DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_NAME} with the selected rho_model.",
    )
    parser.add_argument(
        "--test-iters",
        type=_parse_int_tuple,
        default=default_cfg.test_iters,
        help="Comma-separated test iterations. Defaults to 1,2,...,20.",
    )
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=default_cfg.feature_version)
    parser.add_argument("--gradient-epsilon", type=float, default=default_cfg.gradient_epsilon)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TestDataConfig(
        rho_model=int(args.rho_model),
        test_iters=tuple(sorted(set(args.test_iters))),
        feature_version=int(args.feature_version),
        gradient_epsilon=float(args.gradient_epsilon),
    )
    output = Path(args.output) if args.output else None
    generate_test_data(cfg, output=output)


if __name__ == "__main__":
    main()
