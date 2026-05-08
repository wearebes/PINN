from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import mpmath
import numpy as np
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from train_generate.config import DataConfig, GenerationConfig
    from train_generate.io import dataset_manifest_path, normalize_generation_config, save_training_dataset_hdf5
else:
    from .config import DataConfig, GenerationConfig
    from .io import dataset_manifest_path, normalize_generation_config, save_training_dataset_hdf5


SUPPORTED_INITIAL_FIELD_TYPES = {"sdf", "nonsdf"}
SUPPORTED_SHAPE_TYPES = {"circle", "ellipse"}
TWO_PI = float(2.0 * np.pi)
_PHI9_ROW_OFFSETS = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)
_PHI9_COL_OFFSETS = np.asarray([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int64)


class CircleGeometryGenerator:
    def __init__(self, resolution_rho: int, seed: int = 42, variations: int = 5):
        self.rho = int(resolution_rho)
        self.global_seed = int(seed)
        self.variations = int(variations)
        self.h = 1.0 / (self.rho - 1)
        self.r_min = 1.6 * self.h
        self.r_max = 0.5 - 2.0 * self.h
        self.num_radii = int(np.floor((self.rho - 8.2) / 2.0)) + 1
        if self.variations < 1:
            raise ValueError("variations must be >= 1")
        if self.num_radii < 1 or self.r_min >= self.r_max:
            raise ValueError(f"Resolution rho={self.rho} too small for valid circular interfaces.")
        self.radii_set = np.linspace(self.r_min, self.r_max, self.num_radii, dtype=np.float64)
        self.center_min = 0.5 - self.h / 2.0
        self.center_max = 0.5 + self.h / 2.0

    def _subseed(self, r_idx: int, v_idx: int) -> int:
        mask64 = (1 << 64) - 1
        x = int(self.global_seed) & mask64
        x ^= 1469598103934665603
        x &= mask64
        x ^= ((int(r_idx) + 1) * 1099511628211) & mask64
        x &= mask64
        x ^= ((int(v_idx) + 1) * 14029467366897019727) & mask64
        x &= mask64
        return int(x % (1 << 32))

    def generate_blueprints(self) -> list[dict[str, Any]]:
        blueprints: list[dict[str, Any]] = []
        for r_idx, radius in enumerate(self.radii_set):
            for v_idx in range(self.variations):
                sub_seed = self._subseed(r_idx, v_idx)
                rng = np.random.default_rng(sub_seed)
                blueprints.append(
                    {
                        "meta": {
                            "blueprint_id": f"circle_rho{self.rho}_r{r_idx:03d}_v{v_idx:02d}_s{sub_seed}",
                            "resolution": self.rho,
                            "shape_type": "circle",
                        },
                        "params": {
                            "h": float(self.h),
                            "radius": float(radius),
                            "center": (
                                float(rng.uniform(self.center_min, self.center_max)),
                                float(rng.uniform(self.center_min, self.center_max)),
                            ),
                        },
                    }
                )
        return blueprints


class EllipseGeometryGenerator:
    def __init__(self, resolution_rho: int, data_config: DataConfig):
        self.rho = int(resolution_rho)
        self.global_seed = int(data_config.geometry_seed)
        self.h = 1.0 / (self.rho - 1)
        self.num_a = int(data_config.ellipse_num_a)
        self.variations_per_a = int(data_config.ellipse_variations_per_a)
        self.axis_ratio_min = float(data_config.ellipse_axis_ratio_min)
        self.axis_ratio_max = float(data_config.ellipse_axis_ratio_max)
        self.rotation_min = float(data_config.ellipse_rotation_min)
        self.rotation_max = float(data_config.ellipse_rotation_max)
        self.a_min_factor = float(data_config.ellipse_a_min_factor)
        self.a_min = self.a_min_factor * self.h
        self.a_max = 0.5 - 2.0 * self.h
        self.center_min = 0.5 - self.h / 2.0
        self.center_max = 0.5 + self.h / 2.0

        if self.num_a < 1:
            raise ValueError("ellipse_num_a must be >= 1")
        if self.variations_per_a < 1:
            raise ValueError("ellipse_variations_per_a must be >= 1")
        if not (0.0 < self.axis_ratio_min <= self.axis_ratio_max <= 1.0):
            raise ValueError("Ellipse axis ratios must satisfy 0 < min <= max <= 1.")
        if self.a_min_factor <= 0.0:
            raise ValueError("ellipse_a_min_factor must be > 0.")
        if self.a_min >= self.a_max:
            raise ValueError(f"Resolution rho={self.rho} too small for valid elliptical interfaces.")
        self.a_set = np.linspace(self.a_min, self.a_max, self.num_a, dtype=np.float64)

    def _subseed(self, a_idx: int, v_idx: int) -> int:
        mask64 = (1 << 64) - 1
        x = int(self.global_seed) & mask64
        x ^= 7809847782465536322
        x &= mask64
        x ^= ((int(a_idx) + 1) * 1099511627791) & mask64
        x &= mask64
        x ^= ((int(v_idx) + 1) * 11400714819323198485) & mask64
        x &= mask64
        return int(x % (1 << 32))

    def generate_blueprints(self) -> list[dict[str, Any]]:
        blueprints: list[dict[str, Any]] = []
        for a_idx, a in enumerate(self.a_set):
            for v_idx in range(self.variations_per_a):
                sub_seed = self._subseed(a_idx, v_idx)
                rng = np.random.default_rng(sub_seed)
                axis_ratio = float(rng.uniform(self.axis_ratio_min, self.axis_ratio_max))
                b = float(a * axis_ratio)
                cx = float(rng.uniform(self.center_min, self.center_max))
                cy = float(rng.uniform(self.center_min, self.center_max))
                psi = float(rng.uniform(self.rotation_min, self.rotation_max))
                blueprints.append(
                    {
                        "meta": {
                            "blueprint_id": f"ellipse_rho{self.rho}_a{a_idx:03d}_v{v_idx:03d}_s{sub_seed}",
                            "resolution": self.rho,
                            "shape_type": "ellipse",
                        },
                        "params": {
                            "h": float(self.h),
                            "a": float(a),
                            "b": b,
                            "axis_ratio": axis_ratio,
                            "center": (cx, cy),
                            "psi": psi,
                        },
                    }
                )
        return blueprints


@lru_cache(maxsize=None)
def build_grid(rho: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, int(rho), dtype=np.float64)
    return np.meshgrid(x, x, indexing="ij")


def build_circle_sdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius


def build_circle_nonsdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return (X - cx) ** 2 + (Y - cy) ** 2 - radius**2


def ellipse_local_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cx: float,
    cy: float,
    psi: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))
    dx = X - cx
    dy = Y - cy
    u = dx * cos_psi + dy * sin_psi
    v = -dx * sin_psi + dy * cos_psi
    return u, v


def build_ellipse_nonsdf(u: np.ndarray, v: np.ndarray, *, a: float, b: float) -> np.ndarray:
    return (u**2) / (a**2) + (v**2) / (b**2) - 1.0


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
    rows = indices[:, 0]
    cols = indices[:, 1]
    row_idx = rows[:, None] + _PHI9_ROW_OFFSETS.reshape(1, 9)
    col_idx = cols[:, None] + _PHI9_COL_OFFSETS.reshape(1, 9)
    return np.asarray(phi[row_idx, col_idx], dtype=np.float32)


def _project_theta_scalar(u: float, v: float, a: float, b: float, theta0: float, *, max_iter: int, tol: float) -> float:
    theta = float(theta0 % TWO_PI)
    for _ in range(max_iter):
        st = float(np.sin(theta))
        ct = float(np.cos(theta))
        g = (b * b - a * a) * st * ct + a * u * st - b * v * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u * ct + b * v * st
        if abs(g) <= tol:
            return theta
        if abs(gp) < 1.0e-18:
            break
        step = g / gp
        theta = float((theta - step) % TWO_PI)
        if abs(step) <= tol:
            return theta

    coarse_thetas = np.linspace(0.0, TWO_PI, 720, endpoint=False, dtype=np.float64)
    coarse_ct = np.cos(coarse_thetas)
    coarse_st = np.sin(coarse_thetas)
    dist_sq = (a * coarse_ct - u) ** 2 + (b * coarse_st - v) ** 2
    theta = float(coarse_thetas[int(np.argmin(dist_sq))])
    for _ in range(max_iter * 4):
        st = float(np.sin(theta))
        ct = float(np.cos(theta))
        g = (b * b - a * a) * st * ct + a * u * st - b * v * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u * ct + b * v * st
        if abs(g) <= tol:
            break
        if abs(gp) < 1.0e-18:
            break
        step = g / gp
        theta = float((theta - step) % TWO_PI)
        if abs(step) <= tol:
            break
    return theta


def _hp_theta_tolerance(dps: int) -> mpmath.mpf:
    if int(dps) <= 20:
        raise ValueError("ellipse_hp_dps must be > 20 so the high-precision tolerance is meaningful.")
    return mpmath.power(10, -(int(dps) - 20))


def _project_theta_high_precision_scalar(
    u: float,
    v: float,
    a: float,
    b: float,
    theta0: float,
    *,
    dps: int,
    max_iter: int,
) -> float:
    tol = _hp_theta_tolerance(dps)

    def _refine(seed_theta: float) -> tuple[float, bool]:
        with mpmath.workdps(int(dps)):
            two_pi = mpmath.mpf(2) * mpmath.pi
            a_mp = mpmath.mpf(a)
            b_mp = mpmath.mpf(b)
            u_mp = mpmath.mpf(u)
            v_mp = mpmath.mpf(v)
            theta = mpmath.fmod(mpmath.mpf(seed_theta), two_pi)
            if theta < 0:
                theta += two_pi
            for _ in range(int(max_iter)):
                st = mpmath.sin(theta)
                ct = mpmath.cos(theta)
                g = (b_mp * b_mp - a_mp * a_mp) * st * ct + a_mp * u_mp * st - b_mp * v_mp * ct
                if mpmath.fabs(g) <= tol:
                    return float(theta), True
                gp = (b_mp * b_mp - a_mp * a_mp) * (ct * ct - st * st) + a_mp * u_mp * ct + b_mp * v_mp * st
                if mpmath.fabs(gp) <= mpmath.mpf("1e-40"):
                    break
                step = g / gp
                theta = mpmath.fmod(theta - step, two_pi)
                if theta < 0:
                    theta += two_pi
                if mpmath.fabs(step) <= tol:
                    return float(theta), True
            return float(theta), False

    refined_theta, converged = _refine(theta0)
    if converged:
        return refined_theta

    coarse_thetas = np.linspace(0.0, TWO_PI, 4096, endpoint=False, dtype=np.float64)
    coarse_ct = np.cos(coarse_thetas)
    coarse_st = np.sin(coarse_thetas)
    dist_sq = (float(a) * coarse_ct - float(u)) ** 2 + (float(b) * coarse_st - float(v)) ** 2
    coarse_seed = float(coarse_thetas[int(np.argmin(dist_sq))])
    refined_theta, converged = _refine(coarse_seed)
    if converged:
        return refined_theta
    raise RuntimeError(
        f"High-precision ellipse projection failed to converge for point (u={u}, v={v}) "
        f"with a={a}, b={b}, dps={dps}, max_iter={max_iter}."
    )


def project_theta_to_axis_aligned_ellipse_high_precision(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    dps: int,
    max_iter: int,
    initial_theta: np.ndarray | None = None,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f"u and v must have the same shape, got {u_arr.shape} and {v_arr.shape}.")
    if initial_theta is None:
        theta0 = project_theta_to_axis_aligned_ellipse(
            u_arr,
            v_arr,
            a=float(a),
            b=float(b),
            max_iter=20,
            tol=1.0e-12,
        )
    else:
        theta0 = np.asarray(initial_theta, dtype=np.float64)
        if theta0.shape != u_arr.shape:
            raise ValueError(f"initial_theta shape {theta0.shape} does not match input shape {u_arr.shape}.")

    theta_flat = np.empty(u_arr.size, dtype=np.float64)
    for idx, (u_item, v_item, theta_item) in enumerate(
        zip(u_arr.reshape(-1), v_arr.reshape(-1), theta0.reshape(-1), strict=True)
    ):
        theta_flat[idx] = _project_theta_high_precision_scalar(
            float(u_item),
            float(v_item),
            float(a),
            float(b),
            float(theta_item),
            dps=int(dps),
            max_iter=int(max_iter),
        )
    return theta_flat.reshape(u_arr.shape)


def project_theta_to_axis_aligned_ellipse(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    theta = np.mod(np.arctan2(a * v_arr, b * u_arr), TWO_PI)
    converged = np.zeros(theta.shape, dtype=bool)

    for _ in range(max_iter):
        st = np.sin(theta)
        ct = np.cos(theta)
        g = (b * b - a * a) * st * ct + a * u_arr * st - b * v_arr * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u_arr * ct + b * v_arr * st
        resolved = np.abs(g) <= tol
        safe = np.abs(gp) >= 1.0e-18
        step = np.zeros_like(theta)
        step[safe] = g[safe] / gp[safe]
        theta = np.mod(theta - step, TWO_PI)
        converged |= resolved | (safe & (np.abs(step) <= tol))
        if bool(np.all(converged)):
            break

    if not bool(np.all(converged)):
        unresolved = np.where(~converged)
        for flat_idx in zip(*unresolved, strict=True):
            theta[flat_idx] = _project_theta_scalar(
                float(u_arr[flat_idx]),
                float(v_arr[flat_idx]),
                float(a),
                float(b),
                float(theta[flat_idx]),
                max_iter=max_iter,
                tol=tol,
            )
    return theta.astype(np.float64, copy=False)


def build_ellipse_sdf(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    max_iter: int,
    tol: float,
    dps: int,
    hp_max_iter: int,
) -> np.ndarray:
    theta_seed = project_theta_to_axis_aligned_ellipse(u, v, a=a, b=b, max_iter=max_iter, tol=tol)
    theta = project_theta_to_axis_aligned_ellipse_high_precision(
        u,
        v,
        a=a,
        b=b,
        dps=int(dps),
        max_iter=int(hp_max_iter),
        initial_theta=theta_seed,
    )
    qx = a * np.cos(theta)
    qy = b * np.sin(theta)
    dist = np.sqrt((u - qx) ** 2 + (v - qy) ** 2)
    inside = build_ellipse_nonsdf(u, v, a=a, b=b) < 0.0
    return np.where(inside, -dist, dist).astype(np.float32, copy=False)


def ellipse_hkappa_from_theta(theta: np.ndarray, *, h: float, a: float, b: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    denom = (a * a * np.sin(theta) ** 2 + b * b * np.cos(theta) ** 2) ** 1.5
    hkappa = float(h) * (a * b) / denom
    return hkappa.astype(np.float64, copy=False)


def normalize_initial_field_types(initial_field_types: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in initial_field_types or ():
        field_type = str(item).strip().lower()
        if not field_type:
            continue
        if field_type not in SUPPORTED_INITIAL_FIELD_TYPES:
            raise ValueError(f"Unsupported initial_field_type={item!r}; expected one of {sorted(SUPPORTED_INITIAL_FIELD_TYPES)}.")
        if field_type not in normalized:
            normalized.append(field_type)
    if not normalized:
        raise ValueError("initial_field_types must contain at least one of: sdf, nonsdf.")
    return tuple(normalized)


def normalize_shape_types(shape_types: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in shape_types or ():
        shape_type = str(item).strip().lower()
        if not shape_type:
            continue
        if shape_type not in SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"Unsupported shape_type={item!r}; expected one of {sorted(SUPPORTED_SHAPE_TYPES)}.")
        if shape_type not in normalized:
            normalized.append(shape_type)
    if not normalized:
        raise ValueError("shape_types must contain at least one of: circle, ellipse.")
    return tuple(normalized)


def generate_blueprints(data_config: DataConfig) -> list[dict[str, Any]]:
    blueprints: list[dict[str, Any]] = []
    for rho in data_config.resolutions:
        if "circle" in data_config.shape_types:
            blueprints.extend(
                CircleGeometryGenerator(
                    resolution_rho=int(rho),
                    seed=data_config.geometry_seed,
                    variations=data_config.variations,
                ).generate_blueprints()
            )
        if "ellipse" in data_config.shape_types:
            blueprints.extend(EllipseGeometryGenerator(resolution_rho=int(rho), data_config=data_config).generate_blueprints())
    return blueprints


def _split_count(total: int, *, train_fraction: float, val_fraction: float) -> tuple[int, int]:
    if total < 3:
        raise ValueError("Each requested shape type needs at least 3 blueprints to create train/val/test splits.")
    train_end = max(1, int(round(total * train_fraction)))
    val_end = max(train_end + 1, int(round(total * (train_fraction + val_fraction))))
    val_end = min(val_end, total - 1)
    return train_end, val_end


def split_blueprint_indices(
    blueprints: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    total = len(blueprints)
    if total < 3:
        raise ValueError("Need at least 3 blueprints to create train/val/test splits.")
    if not (0.0 < train_fraction < 1.0) or not (0.0 <= val_fraction < 1.0):
        raise ValueError("Invalid split fractions.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    shape_types = sorted({str(blueprint["meta"]["shape_type"]) for blueprint in blueprints})
    for shape_offset, shape_type in enumerate(shape_types):
        shape_indices = np.asarray(
            [idx for idx, blueprint in enumerate(blueprints) if blueprint["meta"]["shape_type"] == shape_type],
            dtype=np.int64,
        )
        if shape_indices.size == 0:
            continue
        train_end, val_end = _split_count(int(shape_indices.size), train_fraction=train_fraction, val_fraction=val_fraction)
        order = shape_indices.copy()
        np.random.default_rng(seed + 7919 * (shape_offset + 1)).shuffle(order)
        split_indices["train"].extend(int(item) for item in order[:train_end])
        split_indices["val"].extend(int(item) for item in order[train_end:val_end])
        split_indices["test"].extend(int(item) for item in order[val_end:])
    return {name: sorted(indices) for name, indices in split_indices.items()}


def build_phi0_grid(
    blueprint: dict[str, Any],
    initial_field_type: str,
    *,
    data_config: DataConfig,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    shape_type = str(blueprint["meta"]["shape_type"])
    center = blueprint["params"]["center"]
    cx = float(center[0])
    cy = float(center[1])
    if shape_type == "circle":
        radius = float(blueprint["params"]["radius"])
        if initial_field_type == "sdf":
            phi0 = build_circle_sdf(X, Y, cx=cx, cy=cy, radius=radius)
        elif initial_field_type == "nonsdf":
            phi0 = build_circle_nonsdf(X, Y, cx=cx, cy=cy, radius=radius)
        else:
            raise ValueError(f"Unsupported initial_field_type={initial_field_type!r}.")
        return phi0.astype(np.float32, copy=False)

    if shape_type == "ellipse":
        a = float(blueprint["params"]["a"])
        b = float(blueprint["params"]["b"])
        psi = float(blueprint["params"]["psi"])
        u, v = ellipse_local_coordinates(X, Y, cx=cx, cy=cy, psi=psi)
        if initial_field_type == "nonsdf":
            phi0 = build_ellipse_nonsdf(u, v, a=a, b=b)
        elif initial_field_type == "sdf":
            phi0 = build_ellipse_sdf(
                u,
                v,
                a=a,
                b=b,
                max_iter=int(data_config.ellipse_sdf_newton_max_iter),
                tol=float(data_config.ellipse_sdf_newton_tol),
                dps=int(data_config.ellipse_hp_dps),
                hp_max_iter=int(data_config.ellipse_hp_newton_max_iter),
            )
        else:
            raise ValueError(f"Unsupported initial_field_type={initial_field_type!r}.")
        return np.asarray(phi0, dtype=np.float32)

    raise ValueError(f"Unsupported blueprint shape_type={shape_type!r}.")


def compute_hkappa_targets(
    blueprint: dict[str, Any],
    indices: np.ndarray,
    *,
    data_config: DataConfig,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    shape_type = str(blueprint["meta"]["shape_type"])
    h = float(blueprint["params"]["h"])
    if shape_type == "circle":
        radius = float(blueprint["params"]["radius"])
        return np.full((indices.shape[0], 1), h / radius, dtype=np.float64)

    center = blueprint["params"]["center"]
    cx = float(center[0])
    cy = float(center[1])
    a = float(blueprint["params"]["a"])
    b = float(blueprint["params"]["b"])
    psi = float(blueprint["params"]["psi"])
    rows = indices[:, 0]
    cols = indices[:, 1]
    u, v = ellipse_local_coordinates(X[rows, cols], Y[rows, cols], cx=cx, cy=cy, psi=psi)
    theta_seed = project_theta_to_axis_aligned_ellipse(
        u,
        v,
        a=a,
        b=b,
        max_iter=int(data_config.ellipse_sdf_newton_max_iter),
        tol=float(data_config.ellipse_sdf_newton_tol),
    )
    theta = project_theta_to_axis_aligned_ellipse_high_precision(
        u,
        v,
        a=a,
        b=b,
        dps=int(data_config.ellipse_hp_dps),
        max_iter=int(data_config.ellipse_hp_newton_max_iter),
        initial_theta=theta_seed,
    )
    hkappa = ellipse_hkappa_from_theta(theta, h=h, a=a, b=b)
    return hkappa.reshape(-1, 1).astype(np.float64, copy=False)


def sample_blueprint(
    blueprint: dict[str, Any],
    *,
    data_config: DataConfig,
) -> list[dict[str, np.ndarray]]:
    samples: list[dict[str, np.ndarray]] = []
    rho = int(blueprint["meta"]["resolution"])
    X, Y = build_grid(rho)
    for initial_field_type in normalize_initial_field_types(data_config.initial_field_types):
        phi0 = build_phi0_grid(blueprint, initial_field_type, data_config=data_config, X=X, Y=Y)
        indices = interface_indices(phi0)
        if indices.size == 0:
            raise RuntimeError(
                f"No interface nodes found for {blueprint['meta']['blueprint_id']} with initial_field_type={initial_field_type}."
            )
        samples.append(
            {
                "phi9": extract_phi9(phi0, indices),
                "hkappa_target": compute_hkappa_targets(blueprint, indices, data_config=data_config, X=X, Y=Y),
            }
        )
    return samples


def _sample_blueprint_task(task: tuple[int, str, dict[str, Any], DataConfig]) -> tuple[int, str, list[dict[str, np.ndarray]]]:
    blueprint_idx, split_name, blueprint, data_config = task
    try:
        samples = sample_blueprint(blueprint, data_config=data_config)
    except Exception as exc:  # pragma: no cover - exercised via subprocess error propagation
        blueprint_id = str(blueprint.get("meta", {}).get("blueprint_id", f"blueprint_{blueprint_idx}"))
        raise RuntimeError(
            f"Failed to generate samples for blueprint_id={blueprint_id} split={split_name}: {exc}"
        ) from exc
    return blueprint_idx, split_name, samples


def concat_split_samples(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not items:
        return {
            "phi9": np.zeros((0, 9), dtype=np.float32),
            "hkappa_target": np.zeros((0, 1), dtype=np.float32),
        }
    return {
        "phi9": np.concatenate([item["phi9"] for item in items], axis=0).astype(np.float32, copy=False),
        "hkappa_target": np.concatenate([item["hkappa_target"] for item in items], axis=0).astype(np.float32, copy=False),
    }


def _count_shapes_for_indices(blueprints: list[dict[str, Any]], indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        shape_type = str(blueprints[idx]["meta"]["shape_type"])
        counts[shape_type] = counts.get(shape_type, 0) + 1
    return counts


def generate_training_splits(
    data_config: DataConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    data_config = data_config or DataConfig()
    generation_config = normalize_generation_config(generation_config)
    data_config = DataConfig(
        resolutions=tuple(int(item) for item in data_config.resolutions),
        geometry_seed=int(data_config.geometry_seed),
        variations=int(data_config.variations),
        initial_field_types=normalize_initial_field_types(data_config.initial_field_types),
        train_fraction=float(data_config.train_fraction),
        val_fraction=float(data_config.val_fraction),
        shape_types=normalize_shape_types(data_config.shape_types),
        ellipse_num_a=int(data_config.ellipse_num_a),
        ellipse_variations_per_a=int(data_config.ellipse_variations_per_a),
        ellipse_axis_ratio_min=float(data_config.ellipse_axis_ratio_min),
        ellipse_axis_ratio_max=float(data_config.ellipse_axis_ratio_max),
        ellipse_rotation_min=float(data_config.ellipse_rotation_min),
        ellipse_rotation_max=float(data_config.ellipse_rotation_max),
        ellipse_a_min_factor=float(data_config.ellipse_a_min_factor),
        ellipse_sdf_newton_max_iter=int(data_config.ellipse_sdf_newton_max_iter),
        ellipse_sdf_newton_tol=float(data_config.ellipse_sdf_newton_tol),
        ellipse_hp_dps=int(data_config.ellipse_hp_dps),
        ellipse_hp_newton_max_iter=int(data_config.ellipse_hp_newton_max_iter),
    )
    blueprints = generate_blueprints(data_config)
    split_indices = split_blueprint_indices(
        blueprints,
        train_fraction=data_config.train_fraction,
        val_fraction=data_config.val_fraction,
        seed=data_config.geometry_seed,
    )

    split_samples: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    blueprint_to_split = {
        blueprint_idx: split_name
        for split_name, ids in split_indices.items()
        for blueprint_idx in ids
    }
    tasks = [
        (blueprint_idx, blueprint_to_split[blueprint_idx], blueprint, data_config)
        for blueprint_idx, blueprint in enumerate(blueprints)
    ]
    progress = tqdm(blueprints, desc="Generating stencil data", unit="blueprint")
    if int(generation_config.num_workers) <= 1:
        for task in tasks:
            blueprint_idx, split_name, samples = _sample_blueprint_task(task)
            split_samples[split_name].extend(samples)
            progress.update(1)
    else:
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(generation_config.num_workers), mp_context=mp_context) as executor:
            for blueprint_idx, split_name, samples in executor.map(_sample_blueprint_task, tasks):
                split_samples[split_name].extend(samples)
                progress.update(1)
    progress.close()

    splits = {name: concat_split_samples(items) for name, items in split_samples.items()}
    sizes = {name: int(split["phi9"].shape[0]) for name, split in splits.items()}
    split_shape_blueprint_counts = {
        name: _count_shapes_for_indices(blueprints, ids)
        for name, ids in split_indices.items()
    }

    return {
        "config": data_config,
        "generation_config": generation_config,
        "blueprints": blueprints,
        "split_blueprint_indices": split_indices,
        "split_blueprint_counts": {name: len(ids) for name, ids in split_indices.items()},
        "split_shape_blueprint_counts": split_shape_blueprint_counts,
        "splits": splits,
        "sizes": sizes,
    }


def describe_generated_splits(bundle: dict[str, Any]) -> None:
    data_cfg = bundle["config"]
    generation_cfg = bundle["generation_config"]
    print("Task: 3x3 phi stencil -> h*kappa")
    print(f"Number of blueprints: {len(bundle['blueprints'])}")
    print(f"Shape types: {data_cfg.shape_types}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field types: {data_cfg.initial_field_types}")
    print(f"Generation workers: {generation_cfg.num_workers}")
    print(
        f"Blueprint split fractions: train={data_cfg.train_fraction}, "
        f"val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}"
    )
    print(f"Config snapshot: {asdict(data_cfg)}")
    for split_name, split_size in bundle["sizes"].items():
        print(f"{split_name:>5s}: {split_size} samples, blueprint shapes={bundle['split_shape_blueprint_counts'][split_name]}")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    data_cfg = DataConfig()
    generation_cfg = GenerationConfig()
    parser = argparse.ArgumentParser(description="Generate 3x3 stencil training data and export it to HDF5.")
    parser.add_argument("--output", type=str, default="", help="Optional full output path override.")
    parser.add_argument("--output-dir", type=str, default=str(generation_cfg.output_dir))
    parser.add_argument("--dataset-name", type=str, default=generation_cfg.dataset_name)
    parser.add_argument("--resolutions", type=str, default=",".join(str(item) for item in data_cfg.resolutions))
    parser.add_argument("--geometry-seed", type=int, default=data_cfg.geometry_seed)
    parser.add_argument("--variations", type=int, default=data_cfg.variations)
    parser.add_argument("--initial-field-types", type=str, default=",".join(data_cfg.initial_field_types))
    parser.add_argument("--shape-types", type=str, default=",".join(data_cfg.shape_types))
    parser.add_argument("--train-fraction", type=float, default=data_cfg.train_fraction)
    parser.add_argument("--val-fraction", type=float, default=data_cfg.val_fraction)
    parser.add_argument("--ellipse-num-a", type=int, default=data_cfg.ellipse_num_a)
    parser.add_argument("--ellipse-variations-per-a", type=int, default=data_cfg.ellipse_variations_per_a)
    parser.add_argument("--ellipse-axis-ratio-min", type=float, default=data_cfg.ellipse_axis_ratio_min)
    parser.add_argument("--ellipse-axis-ratio-max", type=float, default=data_cfg.ellipse_axis_ratio_max)
    parser.add_argument("--ellipse-rotation-min", type=float, default=data_cfg.ellipse_rotation_min)
    parser.add_argument("--ellipse-rotation-max", type=float, default=data_cfg.ellipse_rotation_max)
    parser.add_argument("--ellipse-a-min-factor", type=float, default=data_cfg.ellipse_a_min_factor)
    parser.add_argument("--ellipse-sdf-newton-max-iter", type=int, default=data_cfg.ellipse_sdf_newton_max_iter)
    parser.add_argument("--ellipse-sdf-newton-tol", type=float, default=data_cfg.ellipse_sdf_newton_tol)
    parser.add_argument("--ellipse-hp-dps", type=int, default=data_cfg.ellipse_hp_dps)
    parser.add_argument("--ellipse-hp-newton-max-iter", type=int, default=data_cfg.ellipse_hp_newton_max_iter)
    parser.add_argument("--num-workers", type=int, default=generation_cfg.num_workers)
    parser.add_argument("--generation-batch-size", type=int, default=generation_cfg.generation_batch_size)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_config = DataConfig(
        resolutions=_parse_int_tuple(args.resolutions),
        geometry_seed=args.geometry_seed,
        variations=args.variations,
        initial_field_types=normalize_initial_field_types(_parse_str_tuple(str(args.initial_field_types))),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        shape_types=normalize_shape_types(_parse_str_tuple(str(args.shape_types))),
        ellipse_num_a=args.ellipse_num_a,
        ellipse_variations_per_a=args.ellipse_variations_per_a,
        ellipse_axis_ratio_min=args.ellipse_axis_ratio_min,
        ellipse_axis_ratio_max=args.ellipse_axis_ratio_max,
        ellipse_rotation_min=args.ellipse_rotation_min,
        ellipse_rotation_max=args.ellipse_rotation_max,
        ellipse_a_min_factor=args.ellipse_a_min_factor,
        ellipse_sdf_newton_max_iter=args.ellipse_sdf_newton_max_iter,
        ellipse_sdf_newton_tol=args.ellipse_sdf_newton_tol,
        ellipse_hp_dps=args.ellipse_hp_dps,
        ellipse_hp_newton_max_iter=args.ellipse_hp_newton_max_iter,
    )
    generation_config = normalize_generation_config(
        GenerationConfig(
            num_workers=args.num_workers,
            generation_batch_size=args.generation_batch_size,
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
        )
    )
    bundle = generate_training_splits(data_config=data_config, generation_config=generation_config)
    describe_generated_splits(bundle)
    output_path = save_training_dataset_hdf5(
        bundle,
        path=args.output or None,
    )
    print(f"Saved dataset to: {output_path.resolve()}")
    print(f"Saved manifest to: {dataset_manifest_path(output_path).resolve()}")


if __name__ == "__main__":
    main()
