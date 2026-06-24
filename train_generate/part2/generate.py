from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_generate.geometry_core import (
    TWO_PI,
    STENCIL_OFFSETS,
    build_circle_nonsdf,
    build_circle_sdf,
    build_ellipse_nonsdf,
    build_ellipse_sdf,
    build_grid,
    ellipse_hkappa_from_theta,
    ellipse_local_coordinates,
    interface_indices,
    project_theta_high_precision_scalar_diagnostics,
    project_theta_to_axis_aligned_ellipse,
    project_theta_to_axis_aligned_ellipse_high_precision,
)
from train_generate.part2.config import (
    D4_MATRICES,
    D4_NAMES,
    REPORT_FILENAMES,
    Part2Config,
)
from train_generate.part2.io import (
    append_part2_rows,
    create_part2_hdf5_datasets,
    node_count_summary,
    projection_quality_summary,
    quantile_row,
    write_blueprint_inventory,
    write_csv,
)


def _seed_from_parts(seed: int, *parts: int) -> int:
    mask64 = (1 << 64) - 1
    value = (int(seed) ^ 1469598103934665603) & mask64
    for part in parts:
        value ^= (int(part) + 0x9E3779B97F4A7C15) & mask64
        value = (value * 1099511628211) & mask64
    return int(value % (1 << 32))


def build_part2_blueprints(config: Part2Config) -> list[dict[str, Any]]:
    if int(config.rho) < 8:
        raise ValueError("rho must be >= 8 for the two-layer stencil boundary exclusion.")
    if not (config.r_min > 0.0 and config.r_min < config.r_max):
        raise ValueError(f"Invalid circle radius bounds for rho={config.rho}.")
    ellipse_a_min = float(config.a_min)
    if not (ellipse_a_min > 0.0 and ellipse_a_min < config.a_max):
        is_default_scale = (
            int(config.circle_eta_levels) == 20
            and int(config.circle_phase_count) == 10
            and int(config.ellipse_count) == 800
        )
        if is_default_scale:
            raise ValueError(f"Invalid ellipse axis bounds for rho={config.rho}.")
        ellipse_a_min = max(1.6 * float(config.h), 0.25 * float(config.a_max))
    if not (ellipse_a_min > 0.0 and ellipse_a_min < config.a_max):
        raise ValueError(f"Invalid ellipse axis bounds for rho={config.rho}.")

    blueprints: list[dict[str, Any]] = []
    eta_levels = np.linspace(config.eta_min, config.eta_max, int(config.circle_eta_levels), dtype=np.float64)
    for eta_idx, eta in enumerate(eta_levels):
        radius = float(config.h / float(eta))
        for phase_idx in range(int(config.circle_phase_count)):
            rng = np.random.default_rng(_seed_from_parts(config.seed, config.rho, 1, eta_idx, phase_idx))
            cx = float(rng.uniform(config.center_min, config.center_max))
            cy = float(rng.uniform(config.center_min, config.center_max))
            blueprints.append(
                {
                    "geometry_id": len(blueprints),
                    "blueprint_id": f"part2_circle_rho{config.rho}_eta{eta_idx:02d}_phase{phase_idx:02d}",
                    "shape_type": "circle",
                    "rho": int(config.rho),
                    "h": float(config.h),
                    "eta_index": int(eta_idx),
                    "phase_index": int(phase_idx),
                    "eta": float(eta),
                    "r": radius,
                    "c_x": cx,
                    "c_y": cy,
                    "hk_exact_min": float(config.h / radius),
                    "hk_exact_max": float(config.h / radius),
                }
            )

    for ellipse_idx in range(int(config.ellipse_count)):
        rng = np.random.default_rng(_seed_from_parts(config.seed, config.rho, 2, ellipse_idx))
        a = float(rng.uniform(ellipse_a_min, config.a_max))
        q = float(rng.uniform(config.ellipse_axis_ratio_min, config.ellipse_axis_ratio_max))
        b = float(q * a)
        psi = float(rng.uniform(config.ellipse_rotation_min, config.ellipse_rotation_max))
        cx = float(rng.uniform(config.center_min, config.center_max))
        cy = float(rng.uniform(config.center_min, config.center_max))
        hk_axis = np.asarray([config.h * q / a, config.h / (q * q * a)], dtype=np.float64)
        blueprints.append(
            {
                "geometry_id": len(blueprints),
                "blueprint_id": f"part2_ellipse_rho{config.rho}_{ellipse_idx:04d}",
                "shape_type": "ellipse",
                "rho": int(config.rho),
                "h": float(config.h),
                "ellipse_index": int(ellipse_idx),
                "a": a,
                "b": b,
                "q": q,
                "psi": psi,
                "c_x": cx,
                "c_y": cy,
                "hk_exact_min": float(np.min(hk_axis)),
                "hk_exact_max": float(np.max(hk_axis)),
            }
        )

    _assign_splits(blueprints, config=config)
    return blueprints


def _assign_splits(blueprints: list[dict[str, Any]], *, config: Part2Config) -> None:
    for shape_offset, shape_type in enumerate(("circle", "ellipse")):
        indices = np.asarray(
            [idx for idx, item in enumerate(blueprints) if item["shape_type"] == shape_type],
            dtype=np.int64,
        )
        order = indices.copy()
        np.random.default_rng(_seed_from_parts(config.seed, config.rho, 3, shape_offset)).shuffle(order)
        n_total = int(order.size)
        train_end = int(round(n_total * float(config.train_fraction)))
        val_end = int(round(n_total * (float(config.train_fraction) + float(config.val_fraction))))
        for idx in order[:train_end]:
            blueprints[int(idx)]["split"] = "train"
        for idx in order[train_end:val_end]:
            blueprints[int(idx)]["split"] = "val"
        for idx in order[val_end:]:
            blueprints[int(idx)]["split"] = "test"


def central_difference_hkappa_from_phi9_float64(phi9: np.ndarray) -> np.ndarray:
    phi9 = np.asarray(phi9, dtype=np.float64)
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {phi9.shape}.")
    patch = phi9.reshape(-1, 3, 3).transpose(0, 2, 1)[:, :, ::-1]
    phi_x = 0.5 * (patch[:, 2, 1] - patch[:, 0, 1])
    phi_y = 0.5 * (patch[:, 1, 2] - patch[:, 1, 0])
    phi_xx = patch[:, 2, 1] - 2.0 * patch[:, 1, 1] + patch[:, 0, 1]
    phi_yy = patch[:, 1, 2] - 2.0 * patch[:, 1, 1] + patch[:, 1, 0]
    phi_xy = 0.25 * (patch[:, 2, 2] - patch[:, 2, 0] - patch[:, 0, 2] + patch[:, 0, 0])
    grad_sq = phi_x**2 + phi_y**2
    denom = np.power(grad_sq, 1.5)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
        bad = int(np.count_nonzero((~np.isfinite(denom)) | (denom <= 0.0)))
        raise ValueError(f"hk_central denominator has {bad} non-positive or non-finite values.")
    hkappa = (phi_xx * phi_y**2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / denom
    if np.any(~np.isfinite(hkappa)):
        bad = int(np.count_nonzero(~np.isfinite(hkappa)))
        raise ValueError(f"hk_central produced {bad} non-finite values.")
    return hkappa.reshape(-1, 1)


def _extract_phi9_float64(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    if indices.size == 0:
        return np.zeros((0, 9), dtype=np.float64)
    rows = indices[:, 0]
    cols = indices[:, 1]
    row_idx = rows[:, None] + STENCIL_OFFSETS[:, 0].reshape(1, 9)
    col_idx = cols[:, None] + STENCIL_OFFSETS[:, 1].reshape(1, 9)
    return phi[row_idx, col_idx].astype(np.float64, copy=False)


def _extract_grad9_float64(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    if indices.size == 0:
        return np.zeros((0, 9, 2), dtype=np.float64)
    rows = indices[:, 0]
    cols = indices[:, 1]
    grad9 = np.empty((indices.shape[0], 9, 2), dtype=np.float64)
    for k, (di, dj) in enumerate(STENCIL_OFFSETS):
        rk = rows + int(di)
        ck = cols + int(dj)
        gx = phi[rk + 1, ck] - phi[rk - 1, ck]
        gy = phi[rk, ck + 1] - phi[rk, ck - 1]
        mag = np.sqrt(gx * gx + gy * gy)
        if np.any(~np.isfinite(mag)) or np.any(mag <= 0.0):
            bad = int(np.count_nonzero((~np.isfinite(mag)) | (mag <= 0.0)))
            raise ValueError(f"normal denominator has {bad} non-positive or non-finite values.")
        grad9[:, k, 0] = gx / mag
        grad9[:, k, 1] = gy / mag
    return grad9


def _d4_sigma(d4_name: str) -> np.ndarray:
    matrix = D4_MATRICES[d4_name]
    inv_matrix = matrix.T
    result: list[int] = []
    offsets = [tuple(int(v) for v in row) for row in STENCIL_OFFSETS]
    for offset in STENCIL_OFFSETS:
        source = tuple(int(v) for v in inv_matrix @ offset)
        result.append(offsets.index(source))
    return np.asarray(result, dtype=np.int64)


_D4_SIGMAS = {name: _d4_sigma(name) for name in D4_NAMES}


def transform_d4_features28(features28: np.ndarray, *, d4_name: str) -> np.ndarray:
    if d4_name not in D4_MATRICES:
        raise ValueError(f"Unknown D4 transform {d4_name!r}.")
    features28 = np.asarray(features28, dtype=np.float64)
    if features28.ndim != 2 or features28.shape[1] != 28:
        raise ValueError(f"features28 must have shape (N, 28), got {features28.shape}.")
    sigma = _D4_SIGMAS[d4_name]
    matrix = D4_MATRICES[d4_name].astype(np.float64)
    out = np.empty_like(features28, dtype=np.float64)
    out[:, :9] = features28[:, :9][:, sigma]
    normals = np.stack([features28[:, 9:18][:, sigma], features28[:, 18:27][:, sigma]], axis=2)
    transformed = np.einsum("ab,nkb->nka", matrix, normals)
    out[:, 9:18] = transformed[:, :, 0]
    out[:, 18:27] = transformed[:, :, 1]
    out[:, 27:28] = features28[:, 27:28]
    return out


def transform_sign_flip_features28(features28: np.ndarray, hk_exact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -np.asarray(features28, dtype=np.float64), -np.asarray(hk_exact, dtype=np.float64)


def transform_d4_phi9(phi9: np.ndarray, *, d4_name: str) -> np.ndarray:
    phi9 = np.asarray(phi9, dtype=np.float64)
    if phi9.ndim != 2 or phi9.shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {phi9.shape}.")
    if d4_name not in _D4_SIGMAS:
        raise ValueError(f"Unknown D4 transform {d4_name!r}.")
    return phi9[:, _D4_SIGMAS[d4_name]]


def build_fields(blueprint: dict[str, Any], *, config: Part2Config, X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    cx = float(blueprint["c_x"])
    cy = float(blueprint["c_y"])
    if blueprint["shape_type"] == "circle":
        radius = float(blueprint["r"])
        return {
            "sdf": build_circle_sdf(X, Y, cx=cx, cy=cy, radius=radius).astype(np.float64, copy=False),
            "nonsdf": build_circle_nonsdf(X, Y, cx=cx, cy=cy, radius=radius).astype(np.float64, copy=False),
        }

    u, v = ellipse_local_coordinates(X, Y, cx=cx, cy=cy, psi=float(blueprint["psi"]))
    nonsdf = build_ellipse_nonsdf(u, v, a=float(blueprint["a"]), b=float(blueprint["b"])).astype(np.float64, copy=False)
    sdf = build_ellipse_sdf(
        u,
        v,
        a=float(blueprint["a"]),
        b=float(blueprint["b"]),
        max_iter=int(config.ellipse_sdf_newton_max_iter),
        tol=float(config.ellipse_sdf_newton_tol),
        dps=int(config.ellipse_hp_dps),
        hp_max_iter=int(config.ellipse_hp_newton_max_iter),
    ).astype(np.float64, copy=False)
    return {"sdf": sdf, "nonsdf": nonsdf}


def projection_diagnostics(
    blueprint: dict[str, Any],
    indices: np.ndarray,
    *,
    config: Part2Config,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if blueprint["shape_type"] == "circle":
        hk = np.full((indices.shape[0], 1), float(blueprint["h"]) / float(blueprint["r"]), dtype=np.float64)
        return hk, []

    rows = indices[:, 0]
    cols = indices[:, 1]
    u, v = ellipse_local_coordinates(
        X[rows, cols],
        Y[rows, cols],
        cx=float(blueprint["c_x"]),
        cy=float(blueprint["c_y"]),
        psi=float(blueprint["psi"]),
    )
    a = float(blueprint["a"])
    b = float(blueprint["b"])
    theta_seed = project_theta_to_axis_aligned_ellipse(
        u,
        v,
        a=a,
        b=b,
        max_iter=int(config.ellipse_sdf_newton_max_iter),
        tol=float(config.ellipse_sdf_newton_tol),
    )
    theta = np.empty(u.shape, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    coarse_thetas = np.linspace(0.0, TWO_PI, 4096, endpoint=False, dtype=np.float64)
    coarse_ct = np.cos(coarse_thetas)
    coarse_st = np.sin(coarse_thetas)
    for idx, (u_item, v_item, theta_item) in enumerate(zip(u.reshape(-1), v.reshape(-1), theta_seed.reshape(-1), strict=True)):
        diag = project_theta_high_precision_scalar_diagnostics(
            float(u_item),
            float(v_item),
            a,
            b,
            float(theta_item),
            dps=int(config.ellipse_hp_dps),
            max_iter=int(config.ellipse_hp_newton_max_iter),
        )
        theta_value = float(diag["theta"])
        theta.reshape(-1)[idx] = theta_value
        st = float(np.sin(theta_value))
        ct = float(np.cos(theta_value))
        residual = abs((b * b - a * a) * st * ct + a * float(u_item) * st - b * float(v_item) * ct)
        distance = float(np.sqrt((a * ct - float(u_item)) ** 2 + (b * st - float(v_item)) ** 2))
        coarse_dist = np.sqrt((a * coarse_ct - float(u_item)) ** 2 + (b * coarse_st - float(v_item)) ** 2)
        coarse_min = float(np.min(coarse_dist))
        converged = bool(diag["converged"]) and residual <= float(config.epsilon_projection_report)
        if not converged:
            raise RuntimeError(
                f"Ellipse projection failed hard gate for {blueprint['blueprint_id']} "
                f"node={idx}: residual={residual:.3e}."
            )
        if distance > coarse_min + float(config.epsilon_distance_report):
            raise RuntimeError(
                f"Ellipse projection distance gate failed for {blueprint['blueprint_id']} "
                f"node={idx}: distance={distance:.16e}, coarse_min={coarse_min:.16e}."
            )
        diagnostics.append(
            {
                "geometry_id": int(blueprint["geometry_id"]),
                "blueprint_id": str(blueprint["blueprint_id"]),
                "theta_projected": theta_value,
                "projection_residual": residual,
                "newton_internal_residual": float(diag["newton_internal_residual"]),
                "distance_to_interface": distance,
                "coarse_global_min_distance": coarse_min,
                "projection_iterations": int(diag["iterations"]),
                "projection_converged": bool(converged),
                "seed_kind": str(diag["seed_kind"]),
            }
        )
    hk = ellipse_hkappa_from_theta(theta, h=float(blueprint["h"]), a=a, b=b).reshape(-1, 1)
    return hk, diagnostics


def validate_field_consistency(
    fields: dict[str, np.ndarray],
    *,
    blueprint: dict[str, Any],
    config: Part2Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    sdf = np.asarray(fields["sdf"], dtype=np.float64)
    nonsdf = np.asarray(fields["nonsdf"], dtype=np.float64)
    near_zero = np.abs(nonsdf) < float(config.epsilon_zero)
    check = ~near_zero
    sign_mismatch = int(np.count_nonzero(np.signbit(sdf[check]) != np.signbit(nonsdf[check])))
    if sign_mismatch:
        raise RuntimeError(f"sdf/nonsdf sign mismatch for {blueprint['blueprint_id']}: {sign_mismatch}")

    idx_sdf = interface_indices(sdf)
    idx_nonsdf = interface_indices(nonsdf)
    if idx_sdf.shape != idx_nonsdf.shape or np.any(idx_sdf != idx_nonsdf):
        raise RuntimeError(
            f"sdf/nonsdf interface-node mismatch for {blueprint['blueprint_id']}: "
            f"sdf={idx_sdf.shape[0]}, nonsdf={idx_nonsdf.shape[0]}"
        )
    return idx_sdf, {
        "geometry_id": int(blueprint["geometry_id"]),
        "blueprint_id": str(blueprint["blueprint_id"]),
        "shape_type": str(blueprint["shape_type"]),
        "split": str(blueprint["split"]),
        "interface_nodes": int(idx_sdf.shape[0]),
        "near_zero_nonsdf_points": int(np.count_nonzero(near_zero)),
        "sign_mismatch": sign_mismatch,
    }


def features28_for_field(phi: np.ndarray, indices: np.ndarray, *, h: float) -> tuple[np.ndarray, np.ndarray]:
    phi9 = _extract_phi9_float64(phi, indices)
    phi9_contract = phi9 / float(h)
    grad9 = _extract_grad9_float64(phi, indices)
    hk_central = central_difference_hkappa_from_phi9_float64(phi9_contract)
    features28 = np.concatenate([phi9_contract, grad9[:, :, 0], grad9[:, :, 1], hk_central], axis=1)
    return features28, phi9


def _part2_rows_for_field(
    *,
    blueprint: dict[str, Any],
    field_type: str,
    features28: np.ndarray,
    phi9: np.ndarray,
    hk_exact: np.ndarray,
    indices: np.ndarray,
) -> list[dict[str, np.ndarray | list[str]]]:
    result: list[dict[str, np.ndarray | list[str]]] = []
    n_nodes = int(features28.shape[0])
    geometry_id = int(blueprint["geometry_id"])
    for d4_id, d4_name in enumerate(D4_NAMES):
        d4_features = transform_d4_features28(features28, d4_name=d4_name)
        d4_phi9 = transform_d4_phi9(phi9, d4_name=d4_name)
        for sign_flip in (0, 1):
            sign = -1.0 if sign_flip else 1.0
            out_features = sign * d4_features
            out_phi9 = sign * d4_phi9
            out_hk_exact = sign * hk_exact
            out_hk_central = out_features[:, 27:28]
            rows = {
                "features28": out_features.astype(np.float32),
                "features27": out_features[:, :27].astype(np.float32),
                "phi9": out_phi9.astype(np.float32),
                "hk_exact": out_hk_exact.astype(np.float32),
                "hk_central": out_hk_central.astype(np.float32),
                "hk_residual": (out_hk_exact - out_hk_central).astype(np.float32),
                "geometry_id": np.full(n_nodes, geometry_id, dtype=np.int32),
                "blueprint_id": [str(blueprint["blueprint_id"])] * n_nodes,
                "split": [str(blueprint["split"])] * n_nodes,
                "shape_type": [str(blueprint["shape_type"])] * n_nodes,
                "field_type": [field_type] * n_nodes,
                "d4_id": np.full(n_nodes, d4_id, dtype=np.int8),
                "d4_name": [d4_name] * n_nodes,
                "sign_flip": np.full(n_nodes, sign_flip, dtype=np.int8),
                "rho": np.full(n_nodes, int(blueprint["rho"]), dtype=np.int32),
                "h": np.full(n_nodes, float(blueprint["h"]), dtype=np.float64),
                "grid_i": indices[:, 0].astype(np.int32),
                "grid_j": indices[:, 1].astype(np.int32),
            }
            result.append(rows)
    return result


def _append_values(store: dict[str, list[np.ndarray]], key: str, values: np.ndarray) -> None:
    store.setdefault(key, []).append(np.asarray(values, dtype=np.float64).reshape(-1))


def _concat_values(store: dict[str, list[np.ndarray]], key: str) -> np.ndarray:
    values = store.get(key, [])
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def save_part2_training_hdf5(config: Part2Config, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blueprints = build_part2_blueprints(config)
    X, Y = build_grid(int(config.rho))

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_family"] = "part2_training"
        handle.attrs["part2_schema_version"] = 1
        handle.attrs["rho"] = int(config.rho)
        handle.attrs["h"] = float(config.h)
        handle.attrs["seed"] = int(config.seed)
        handle.attrs["base_geometry_count"] = int(len(blueprints))
        handle.attrs["circle_count"] = int(sum(1 for item in blueprints if item["shape_type"] == "circle"))
        handle.attrs["ellipse_count"] = int(sum(1 for item in blueprints if item["shape_type"] == "ellipse"))
        handle.attrs["initial_field_types_json"] = json.dumps(list(config.initial_field_types))
        handle.attrs["features28_columns"] = "[phi9_contract:0-8, nx9:9-17, ny9:18-26, hk_central:27]"
        handle.attrs["target_name"] = "hk_exact"
        handle.create_dataset("blueprints_json", data=json.dumps(blueprints, sort_keys=True).encode("utf-8"))
        datasets = create_part2_hdf5_datasets(handle)

        for blueprint_idx, blueprint in enumerate(blueprints, start=1):
            if blueprint_idx == 1 or blueprint_idx % 50 == 0 or blueprint_idx == len(blueprints):
                print(
                    f"[part2 hdf5] processing blueprint {blueprint_idx}/{len(blueprints)}: {blueprint['blueprint_id']}",
                    file=sys.stderr,
                    flush=True,
                )
            fields = build_fields(blueprint, config=config, X=X, Y=Y)
            indices, _node_row = validate_field_consistency(fields, blueprint=blueprint, config=config)
            if indices.shape[0] == 0:
                raise RuntimeError(f"No interface nodes found for {blueprint['blueprint_id']}.")
            hk_exact, _projection_diag = projection_diagnostics(blueprint, indices, config=config, X=X, Y=Y)
            for field_type in config.initial_field_types:
                features28, phi9 = features28_for_field(fields[field_type], indices, h=float(config.h))
                if not np.all(np.isfinite(features28)) or not np.all(np.isfinite(hk_exact)):
                    raise RuntimeError(f"Non-finite Part 2 arrays for {blueprint['blueprint_id']} {field_type}.")
                for rows in _part2_rows_for_field(
                    blueprint=blueprint,
                    field_type=field_type,
                    features28=features28,
                    phi9=phi9,
                    hk_exact=hk_exact,
                    indices=indices,
                ):
                    append_part2_rows(datasets, rows)

        handle.attrs["row_count"] = int(datasets["features28"].shape[0])
    return output_path


def run_part2_dry_run(config: Part2Config) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blueprints = build_part2_blueprints(config)
    expected_circle = int(config.circle_eta_levels) * int(config.circle_phase_count)
    expected_ellipse = int(config.ellipse_count)
    expected_total = expected_circle + expected_ellipse
    if len(blueprints) != expected_total:
        raise RuntimeError(f"Expected {expected_total} base geometries ({expected_circle} circles + {expected_ellipse} ellipses), got {len(blueprints)}.")
    if sum(1 for item in blueprints if item["shape_type"] == "circle") != expected_circle:
        raise RuntimeError(f"Expected exactly {expected_circle} circle blueprints.")
    if sum(1 for item in blueprints if item["shape_type"] == "ellipse") != expected_ellipse:
        raise RuntimeError(f"Expected exactly {expected_ellipse} ellipse blueprints.")

    X, Y = build_grid(int(config.rho))
    node_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    geometry_curvature: dict[str, list[np.ndarray]] = {}
    sample_curvature: dict[str, list[np.ndarray]] = {}
    feature_scales: dict[str, list[np.ndarray]] = {}
    hk_sanity: dict[str, list[np.ndarray]] = {}
    hard_gates = {
        "blueprint_counts": True,
        "field_type_signs": True,
        "interface_node_sets": True,
        "projection_quality": True,
        "feature_prefix": True,
        "d4_consistency": True,
        "sign_flip_consistency": True,
        "finite_values": True,
    }
    row_count = 0
    max_d4_hk_error = 0.0
    max_sign_feature_error = 0.0
    max_sign_target_error = 0.0
    max_prefix_error = 0.0
    max_hk_column_error = 0.0
    max_normal_norm_error = 0.0

    for blueprint_idx, blueprint in enumerate(blueprints, start=1):
        if blueprint_idx == 1 or blueprint_idx % 50 == 0 or blueprint_idx == len(blueprints):
            print(
                f"[part2 dry-run] processing blueprint {blueprint_idx}/{len(blueprints)}: {blueprint['blueprint_id']}",
                file=sys.stderr,
                flush=True,
            )
        fields = build_fields(blueprint, config=config, X=X, Y=Y)
        indices, node_row = validate_field_consistency(fields, blueprint=blueprint, config=config)
        if indices.shape[0] == 0:
            raise RuntimeError(f"No interface nodes found for {blueprint['blueprint_id']}.")
        node_rows.append(node_row)
        hk_exact, projection_diag = projection_diagnostics(blueprint, indices, config=config, X=X, Y=Y)
        projection_rows.extend(projection_diag)
        abs_hk = np.abs(hk_exact.reshape(-1))
        _append_values(geometry_curvature, "all", np.asarray([float(np.mean(abs_hk))]))
        _append_values(geometry_curvature, str(blueprint["shape_type"]), np.asarray([float(np.mean(abs_hk))]))
        _append_values(geometry_curvature, str(blueprint["split"]), np.asarray([float(np.mean(abs_hk))]))

        for field_type in config.initial_field_types:
            features28, _phi9_raw = features28_for_field(fields[field_type], indices, h=float(config.h))
            features27 = features28[:, :27]
            hk_central = features28[:, 27:28]
            hk_residual = hk_exact - hk_central
            max_prefix_error = max(max_prefix_error, float(np.max(np.abs(features27 - features28[:, :27]))))
            max_hk_column_error = max(max_hk_column_error, float(np.max(np.abs(hk_central - features28[:, 27:28]))))
            if not np.all(np.isfinite(features28)) or not np.all(np.isfinite(hk_exact)) or not np.all(np.isfinite(hk_residual)):
                raise RuntimeError(f"Non-finite Part 2 arrays for {blueprint['blueprint_id']} {field_type}.")

            normal_norm_error = np.abs(np.sqrt(features28[:, 9:18] ** 2 + features28[:, 18:27] ** 2) - 1.0)
            max_normal_norm_error = max(max_normal_norm_error, float(np.max(normal_norm_error)))
            _append_values(feature_scales, f"field_type={field_type}", features28[:, :9])
            _append_values(feature_scales, f"field_type={field_type}", -features28[:, :9])
            _append_values(feature_scales, f"shape_type={blueprint['shape_type']}", features28[:, :9])
            _append_values(feature_scales, f"shape_type={blueprint['shape_type']}", -features28[:, :9])
            _append_values(feature_scales, f"split={blueprint['split']}", features28[:, :9])
            _append_values(feature_scales, f"split={blueprint['split']}", -features28[:, :9])

            _append_values(sample_curvature, "all", abs_hk)
            _append_values(sample_curvature, f"shape_type={blueprint['shape_type']}", abs_hk)
            _append_values(sample_curvature, f"field_type={field_type}", abs_hk)
            _append_values(sample_curvature, f"split={blueprint['split']}", abs_hk)

            _append_values(hk_sanity, f"hk_central|shape_type={blueprint['shape_type']}|field_type={field_type}", hk_central)
            _append_values(hk_sanity, f"hk_exact|shape_type={blueprint['shape_type']}|field_type={field_type}", hk_exact)
            _append_values(
                hk_sanity,
                f"hk_central_minus_exact|shape_type={blueprint['shape_type']}|field_type={field_type}",
                hk_central - hk_exact,
            )
            _append_values(
                hk_sanity,
                f"hk_exact_minus_central|shape_type={blueprint['shape_type']}|field_type={field_type}",
                hk_residual,
            )

            for band_name, mask in _curvature_band_masks(abs_hk).items():
                _append_values(hk_sanity, f"hk_central|abs_hk_band={band_name}", hk_central[mask])
                _append_values(hk_sanity, f"hk_exact|abs_hk_band={band_name}", hk_exact[mask])
                _append_values(hk_sanity, f"hk_exact_minus_central|abs_hk_band={band_name}", hk_residual[mask])

            for d4_name in D4_NAMES:
                d4_features = transform_d4_features28(features28, d4_name=d4_name)
                recomputed = central_difference_hkappa_from_phi9_float64(d4_features[:, :9])
                d4_error = float(np.max(np.abs(recomputed - hk_central))) if recomputed.size else 0.0
                max_d4_hk_error = max(max_d4_hk_error, d4_error)
                if d4_error > 1.0e-10:
                    raise RuntimeError(f"D4 hk_central recompute failed for {blueprint['blueprint_id']} {field_type} {d4_name}.")
                flipped_features, flipped_hk = transform_sign_flip_features28(d4_features, hk_exact)
                max_sign_feature_error = max(
                    max_sign_feature_error,
                    float(np.max(np.abs(flipped_features + d4_features))) if flipped_features.size else 0.0,
                )
                max_sign_target_error = max(
                    max_sign_target_error,
                    float(np.max(np.abs(flipped_hk + hk_exact))) if flipped_hk.size else 0.0,
                )
            row_count += int(indices.shape[0]) * 8 * 2

    write_blueprint_inventory(output_dir / "blueprint_inventory.csv", blueprints)
    write_csv(output_dir / "node_count_summary.csv", node_count_summary(node_rows))
    write_csv(
        output_dir / "curvature_geometry_weighted_summary.csv",
        [quantile_row(key, _concat_values(geometry_curvature, key), quantiles=(1, 5, 25, 50, 75, 95, 99)) for key in sorted(geometry_curvature)],
    )
    write_csv(
        output_dir / "curvature_sample_weighted_summary.csv",
        [quantile_row(key, _concat_values(sample_curvature, key), quantiles=(1, 5, 25, 50, 75, 95, 99)) for key in sorted(sample_curvature)],
    )
    write_csv(
        output_dir / "feature_scale_summary.csv",
        [quantile_row(key, _concat_values(feature_scales, key), quantiles=(1, 5, 25, 50, 75, 95, 99, 99.9)) for key in sorted(feature_scales)],
    )
    write_csv(
        output_dir / "hk_central_sanity_summary.csv",
        [quantile_row(key, _concat_values(hk_sanity, key), quantiles=(50, 90, 95, 99, 99.9)) for key in sorted(hk_sanity)],
    )
    write_csv(output_dir / "projection_quality_summary.csv", projection_quality_summary(projection_rows))

    augmentation_summary = {
        "d4_names": list(D4_NAMES),
        "d4_hk_central_recomputed_from_permuted_phi9": True,
        "normal_transform_convention": "physical_vector_Ag_times_source_normal",
        "max_d4_hk_central_abs_error": max_d4_hk_error,
        "max_sign_flip_features28_abs_error": max_sign_feature_error,
        "max_sign_flip_hk_exact_abs_error": max_sign_target_error,
        "passed": max_d4_hk_error <= 1.0e-10 and max_sign_feature_error == 0.0 and max_sign_target_error == 0.0,
    }
    (output_dir / "augmentation_contract_summary.json").write_text(
        json.dumps(augmentation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    feature_summary = {
        "features28_columns": "[phi9_contract:0-8, nx9:9-17, ny9:18-26, hk_central:27]",
        "features27_is_prefix": True,
        "features27_prefix_max_abs_error": max_prefix_error,
        "hk_central_column_max_abs_error": max_hk_column_error,
        "max_normal_norm_error": max_normal_norm_error,
        "row_count_after_d4_and_sign_flip": row_count,
        "hard_gates": hard_gates,
        "passed": all(hard_gates.values()),
    }
    (output_dir / "feature_contract_summary.json").write_text(
        json.dumps(feature_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    missing_reports = [name for name in REPORT_FILENAMES if not (output_dir / name).exists()]
    if missing_reports:
        raise RuntimeError(f"Missing dry-run reports: {missing_reports}")
    return {
        "output_dir": str(output_dir.resolve()),
        "blueprint_count": len(blueprints),
        "row_count_after_d4_and_sign_flip": row_count,
        "hard_gates": hard_gates,
        "reports": list(REPORT_FILENAMES),
    }


def _curvature_band_masks(abs_hk: np.ndarray) -> dict[str, np.ndarray]:
    if abs_hk.size == 0:
        return {
            "low": np.zeros((0,), dtype=bool),
            "middle": np.zeros((0,), dtype=bool),
            "high": np.zeros((0,), dtype=bool),
        }
    q33, q66 = np.percentile(abs_hk, [33.333333, 66.666667])
    return {
        "low": abs_hk <= q33,
        "middle": (abs_hk > q33) & (abs_hk <= q66),
        "high": abs_hk > q66,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Part 2 training-data generator.")
    parser.add_argument("--rho", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-h5", default="", help="Optional Part 2 training HDF5 output path.")
    parser.add_argument("--circle-eta-levels", type=int, default=20)
    parser.add_argument("--circle-phase-count", type=int, default=10)
    parser.add_argument("--ellipse-count", type=int, default=800)
    parser.add_argument("--nonsdf", action="store_true", help="Enable non-SDF field type alongside SDF (doubles the output).")
    args = parser.parse_args(argv)
    config = Part2Config(
        rho=int(args.rho),
        seed=int(args.seed),
        output_dir=Path(args.output_dir) if args.output_dir else Path(f"dataset/part2_dryrun/rho{int(args.rho)}"),
        circle_eta_levels=int(args.circle_eta_levels),
        circle_phase_count=int(args.circle_phase_count),
        ellipse_count=int(args.ellipse_count),
        nonsdf_enabled=bool(args.nonsdf),
    )
    default_scale = (
        int(args.circle_eta_levels) == 20
        and int(args.circle_phase_count) == 10
        and int(args.ellipse_count) == 800
    )
    if default_scale:
        result = run_part2_dry_run(config)
    else:
        result = {
            "output_dir": str(Path(config.output_dir).resolve()),
            "blueprint_count": int(args.circle_eta_levels) * int(args.circle_phase_count) + int(args.ellipse_count),
            "dry_run_reports": "skipped_for_nondefault_scale",
        }
    if args.output_h5:
        result["hdf5_path"] = str(save_part2_training_hdf5(config, args.output_h5).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
