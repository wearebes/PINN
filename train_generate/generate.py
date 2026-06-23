from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_generate.config import DataConfig, GenerationConfig
from train_generate.geometry_core import (
    STENCIL_OFFSETS,
    build_circle_nonsdf,
    build_circle_sdf,
    build_ellipse_nonsdf,
    build_ellipse_sdf,
    build_grid,
    ellipse_hkappa_from_theta,
    ellipse_local_coordinates,
    interface_indices,
    project_theta_to_axis_aligned_ellipse,
    project_theta_to_axis_aligned_ellipse_high_precision,
)
from train_generate.io import dataset_manifest_path, normalize_generation_config, save_training_dataset_hdf5


SUPPORTED_INITIAL_FIELD_TYPES = {"sdf", "nonsdf"}
SUPPORTED_SHAPE_TYPES = {"circle", "ellipse"}


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


def encode_patch_training_order(patch_2d: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch_2d, dtype=np.float64)
    return patch[:, ::-1].T.reshape(-1)


def extract_phi9(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 9), dtype=np.float32)
    rows = indices[:, 0]
    cols = indices[:, 1]
    row_idx = rows[:, None] + STENCIL_OFFSETS[:, 0].reshape(1, 9)
    col_idx = cols[:, None] + STENCIL_OFFSETS[:, 1].reshape(1, 9)
    return np.asarray(phi[row_idx, col_idx], dtype=np.float32)


def extract_grad9(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute normalised gradient direction at each of the 9 stencil positions.

    For every interface node and every stencil position k, uses central differences
    on the *full* phi field (not just the 3×3 patch) to compute (dx, dy) proportional
    to (∂φ/∂x, ∂φ/∂y) at that grid point, then normalises to unit length.

    Requires all interface nodes to sit at least 2 grid cells from the domain boundary
    (guaranteed by the 2-layer exclusion in ``interface_indices``).

    Returns
    -------
    np.ndarray, shape (N, 9, 2), dtype float32
        result[:, k, 0] = nx_k  (normalised x-component of gradient at stencil point k)
        result[:, k, 1] = ny_k  (normalised y-component of gradient at stencil point k)
    """
    if indices.size == 0:
        return np.zeros((0, 9, 2), dtype=np.float32)
    rows = indices[:, 0]
    cols = indices[:, 1]
    nrows, ncols = phi.shape
    grad9 = np.empty((len(rows), 9, 2), dtype=np.float32)
    for k in range(9):
        dr = int(STENCIL_OFFSETS[k, 0])
        dc = int(STENCIL_OFFSETS[k, 1])
        rk = rows + dr   # row of stencil point k for every node
        ck = cols + dc   # col of stencil point k for every node
        assert np.all(rk - 1 >= 0) and np.all(rk + 1 < nrows), (
            f"extract_grad9: row access out of bounds for stencil k={k} (dr={dr}). "
            "Interface nodes must be at least 2 rows from the domain boundary."
        )
        assert np.all(ck - 1 >= 0) and np.all(ck + 1 < ncols), (
            f"extract_grad9: col access out of bounds for stencil k={k} (dc={dc}). "
            "Interface nodes must be at least 2 cols from the domain boundary."
        )
        dx = phi[rk + 1, ck] - phi[rk - 1, ck]   # ∝ ∂φ/∂x at stencil position k
        dy = phi[rk, ck + 1] - phi[rk, ck - 1]   # ∝ ∂φ/∂y at stencil position k
        mag = np.sqrt(dx**2 + dy**2)
        safe_mag = np.where(mag > 0.0, mag, 1.0)  # protect against zero-gradient
        grad9[:, k, 0] = (dx / safe_mag).astype(np.float32)
        grad9[:, k, 1] = (dy / safe_mag).astype(np.float32)
    return grad9


def build_raw_features(
    phi: np.ndarray,
    indices: np.ndarray,
    *,
    scale_h: bool = False,
    h: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    phi9 = extract_phi9(phi, indices)
    if scale_h:
        if h is None or not (float(h) > 0.0):
            raise ValueError(f"scale_h=True requires positive h, got {h!r}.")
        features = (phi9 / np.float32(h)).astype(np.float32, copy=False)
    else:
        features = phi9.copy()
    return phi9, features


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


def normalize_scale_alpha(values: tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    if not values:
        return ()
    normalized = tuple(sorted({float(v) for v in values}))
    if any(v <= 0.0 for v in normalized):
        raise ValueError(f"augment_scale_alpha values must be positive, got {list(normalized)}.")
    if 1.0 not in normalized:
        raise ValueError(
            "augment_scale_alpha must include 1.0 because phi/h is the canonical deployment input."
        )
    return normalized


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
    h_blueprint = float(blueprint["params"]["h"])
    alpha_list = list(data_config.augment_scale_alpha)  # already normalized by generate_training_splits

    for initial_field_type in normalize_initial_field_types(data_config.initial_field_types):
        phi0 = build_phi0_grid(blueprint, initial_field_type, data_config=data_config, X=X, Y=Y)
        indices = interface_indices(phi0)
        if indices.size == 0:
            raise RuntimeError(
                f"No interface nodes found for {blueprint['meta']['blueprint_id']} "
                f"with initial_field_type={initial_field_type}."
            )

        phi9 = extract_phi9(phi0, indices)
        base_hkappa = compute_hkappa_targets(blueprint, indices, data_config=data_config, X=X, Y=Y)
        grad9 = extract_grad9(phi0, indices) if data_config.augment_gradient else None

        def _append(feats, hkappa, alpha_arr=None, *, _phi9=phi9):
            entry: dict[str, np.ndarray] = {"phi9": _phi9, "features": feats, "hkappa_target": hkappa}
            if alpha_arr is not None:
                entry["alpha_scale"] = alpha_arr
            samples.append(entry)
            if data_config.augment_sign_flip:
                flipped: dict[str, np.ndarray] = {"phi9": -_phi9, "features": -feats, "hkappa_target": -hkappa}
                if alpha_arr is not None:
                    flipped["alpha_scale"] = alpha_arr
                samples.append(flipped)

        if alpha_list:
            for alpha in alpha_list:
                sf = np.float32(float(alpha) * h_blueprint)
                feats = (phi9 / sf).astype(np.float32, copy=False)
                if grad9 is not None:
                    feats = np.concatenate(
                        [feats, grad9[:, :, 0], grad9[:, :, 1]], axis=1
                    ).astype(np.float32, copy=False)
                hkappa = (base_hkappa * float(alpha)).astype(np.float32, copy=False)
                alpha_arr = np.full(len(phi9), float(alpha), dtype=np.float32)
                _append(feats, hkappa, alpha_arr)
        else:
            _, feats = build_raw_features(phi0, indices, scale_h=bool(data_config.scale_h), h=h_blueprint)
            if grad9 is not None:
                feats = np.concatenate(
                    [feats, grad9[:, :, 0], grad9[:, :, 1]], axis=1
                ).astype(np.float32, copy=False)
            _append(feats, base_hkappa)

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


def concat_split_samples(items: list[dict[str, np.ndarray]], *, feature_dim: int = 9) -> dict[str, np.ndarray]:
    if not items:
        return {
            "phi9": np.zeros((0, 9), dtype=np.float32),
            "features": np.zeros((0, feature_dim), dtype=np.float32),
            "hkappa_target": np.zeros((0, 1), dtype=np.float32),
        }
    result: dict[str, np.ndarray] = {
        "phi9": np.concatenate([item["phi9"] for item in items], axis=0).astype(np.float32, copy=False),
        "features": np.concatenate([item["features"] for item in items], axis=0).astype(np.float32, copy=False),
        "hkappa_target": np.concatenate([item["hkappa_target"] for item in items], axis=0).astype(np.float32, copy=False),
    }
    if "alpha_scale" in items[0]:
        result["alpha_scale"] = np.concatenate(
            [item["alpha_scale"] for item in items], axis=0
        ).astype(np.float32, copy=False)
    return result


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
        augment_sign_flip=bool(data_config.augment_sign_flip),
        augment_gradient=bool(data_config.augment_gradient),
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
        scale_h=bool(data_config.scale_h),
        augment_scale_alpha=normalize_scale_alpha(data_config.augment_scale_alpha),
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

    feature_dim = 27 if data_config.augment_gradient else 9
    splits = {name: concat_split_samples(items, feature_dim=feature_dim) for name, items in split_samples.items()}
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


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


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
    parser.add_argument(
        "--augment-sign-flip",
        action=argparse.BooleanOptionalAction,
        default=data_cfg.augment_sign_flip,
        help="Emit both (phi9, hkappa_target) and (-phi9, -hkappa_target) for each generated sample.",
    )
    parser.add_argument(
        "--augment-gradient",
        action=argparse.BooleanOptionalAction,
        default=data_cfg.augment_gradient,
        help=(
            "Append per-node normalised gradient directions (nx9, ny9) to phi9 features, "
            "yielding 27D features with layout [phi9 | nx9 | ny9]. "
            "Gradient direction also negates under sign-flip augmentation."
        ),
    )
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
    parser.add_argument(
        "--scale-h",
        action="store_true",
        default=data_cfg.scale_h,
        help="If set, write features = phi9 / h (per blueprint). Default off; features == phi9.",
    )
    parser.add_argument(
        "--augment-alpha",
        type=str,
        default="",
        help=(
            "Comma-separated alpha values for training-only augmentation. "
            "Each alpha adds samples (phi/(alpha*h), alpha*h*kappa). "
            "Must include 1.0 (deployment input remains phi/h). "
            "Example: '0.5,1.0,2.0'."
        ),
    )
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
        augment_sign_flip=bool(args.augment_sign_flip),
        augment_gradient=bool(args.augment_gradient),
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
        scale_h=bool(args.scale_h),
        augment_scale_alpha=_parse_float_tuple(args.augment_alpha) if args.augment_alpha else (),
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
