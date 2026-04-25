from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from testdata_generate.config import DATASET_SCHEMA_VERSION, DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, TestDataConfig
    from traingenerate.reinit import LevelSetReinitializer
else:
    from .config import DATASET_SCHEMA_VERSION, DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, TestDataConfig
    from traingenerate.reinit import LevelSetReinitializer

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover - scipy is optional for a rare fallback path.
    _scipy_minimize = None


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


def encode_patch_training_order(patch_2d: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch_2d, dtype=np.float64)
    return patch[:, ::-1].T.reshape(-1)


def extract_phi9(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 9), dtype=np.float32)
    out = np.empty((indices.shape[0], 9), dtype=np.float32)
    for n, (row, col) in enumerate(indices):
        out[n] = encode_patch_training_order(phi[row - 1:row + 2, col - 1:col + 2])
    return out


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
) -> int:
    indices = interface_indices(phi)
    if indices.size == 0:
        raise RuntimeError(f"No current-interface nodes for case_id={case_id}, iter={iteration}.")

    rows = indices[:, 0]
    cols = indices[:, 1]
    xy = np.column_stack((X[rows, cols], Y[rows, cols])).astype(np.float32, copy=False)
    theta_proj = find_projection_theta(xy, a, b, p)
    count = indices.shape[0]

    store["phi9"].append(extract_phi9(phi, indices))
    store["xy"].append(xy)
    store["phi0_center"].append(phi0[rows, cols].astype(np.float32, copy=False))
    store["hkappa_target"].append(hkappa_analytic(theta_proj, h, a, b, p))
    store["case_id"].append(np.full((count,), int(case_id), dtype=np.int16))
    store["iter"].append(np.full((count,), int(iteration), dtype=np.int16))
    store["rho_model"].append(np.full((count,), int(rho_model), dtype=np.int16))
    store["h"].append(np.full((count,), float(h), dtype=np.float32))
    return count


def _concat_store(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    shapes = {
        "phi9": (0, 9),
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


def generate_test_data(config: TestDataConfig | None = None, *, output: str | Path | None = None) -> Path:
    cfg = config or TestDataConfig()
    output_path = Path(output) if output is not None else cfg.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        f"mode={cfg.mode} sign_mode={cfg.sign_mode} cfl={cfg.cfl} "
        f"eps_sign_factor={cfg.eps_sign_factor} RK{cfg.time_order} WENO{cfg.space_order}"
    )
    print(f"[testdata_generate] output={output_path}")

    for case_id, scenario in enumerate(cfg.scenarios):
        X, Y, h_from_grid = build_grid(scenario.L, scenario.N)
        if not np.isclose(h_from_grid, scenario.h, rtol=0.0, atol=1.0e-8):
            print(
                f"    using grid h={h_from_grid:.12g}; scenario h metadata={scenario.h:.12g}"
            )
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
            )
            counts[f"{scenario.exp_id}/iter_{iteration}"] = count
            print(f"    iter={iteration:>2d} samples={count}")

    arrays = _concat_store(store)
    _validate_arrays(arrays, cfg)
    _write_hdf5(output_path, arrays, cfg, counts)
    print(f"[testdata_generate] wrote {arrays['phi9'].shape[0]} samples")
    return output_path


def _validate_arrays(arrays: dict[str, np.ndarray], cfg: TestDataConfig) -> None:
    n = arrays["phi9"].shape[0]
    if arrays["phi9"].ndim != 2 or arrays["phi9"].shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {arrays['phi9'].shape}.")
    if arrays["xy"].ndim != 2 or arrays["xy"].shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {arrays['xy'].shape}.")
    for key, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"{key} first dimension {value.shape[0]} does not match phi9 length {n}.")
    if set(np.unique(arrays["case_id"]).tolist()) != set(range(len(cfg.scenarios))):
        raise ValueError("case_id does not cover all configured scenarios.")
    if set(np.unique(arrays["iter"]).tolist()) != set(int(item) for item in cfg.test_iters):
        raise ValueError("iter does not cover all configured test iterations.")
    if not np.isfinite(arrays["hkappa_target"]).all():
        raise ValueError("hkappa_target contains non-finite values.")

    # Check the stored center phi0 against the analytical formula at the stored center xy.
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
        "method_code": cfg.method_code,
        "output_root": str(DEFAULT_OUTPUT_DIR),
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
    parser = argparse.ArgumentParser(description="Generate independent flower test data into test_data/.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=f"Full output path. Defaults to {DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_NAME}.",
    )
    parser.add_argument(
        "--test-iters",
        type=_parse_int_tuple,
        default=TestDataConfig().test_iters,
        help="Comma-separated test iterations. Defaults to 1,2,...,20.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TestDataConfig(test_iters=tuple(sorted(set(args.test_iters))))
    output = Path(args.output) if args.output else None
    generate_test_data(cfg, output=output)


if __name__ == "__main__":
    main()
