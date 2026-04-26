from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from traingenerate.config import DataConfig, GenerationConfig, ReinitConfig, default_dataset_name
    from traingenerate.reinit import ReinitFieldPackBuilder
else:
    from .config import DataConfig, GenerationConfig, ReinitConfig, default_dataset_name
    from .reinit import ReinitFieldPackBuilder


TRAIN_LOADER_WORKERS = 12
DATASET_FORMAT_VERSION = 2
BASE_TRAJ_FIELDS = ("x", "y", "s", "phi_target", "cx", "cy", "radius", "h")
SUPPORTED_STORED_SAMPLE_FIELDS = ("phi", "phi_x", "phi_y")
DEFAULT_PREVIEW_SAMPLES = 5


def default_dataset_filename(cfl: float, setting: str = "setting1") -> str:
    return default_dataset_name(cfl=cfl, setting=setting)


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _normalize_stored_sample_fields(fields: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for field_name in fields or ():
        normalized = str(field_name).strip()
        if not normalized:
            continue
        if normalized not in SUPPORTED_STORED_SAMPLE_FIELDS:
            allowed = ", ".join(SUPPORTED_STORED_SAMPLE_FIELDS)
            raise ValueError(f"Unsupported stored sample field={normalized!r}; expected one of: {allowed}.")
        if normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return tuple(ordered)


def _normalize_generation_config(
    generation_config: GenerationConfig | None,
    *,
    reinit_config: ReinitConfig | None = None,
) -> GenerationConfig:
    cfg = generation_config or GenerationConfig()
    stored_sample_fields = _normalize_stored_sample_fields(cfg.stored_sample_fields)
    if int(cfg.num_workers) < 1:
        raise ValueError("GenerationConfig.num_workers must be >= 1.")
    if int(cfg.generation_batch_size) < 1:
        raise ValueError("GenerationConfig.generation_batch_size must be >= 1.")
    if str(cfg.dataset_name).strip():
        return GenerationConfig(
            num_workers=int(cfg.num_workers),
            generation_batch_size=int(cfg.generation_batch_size),
            output_dir=Path(cfg.output_dir),
            dataset_name=str(cfg.dataset_name),
            stored_sample_fields=stored_sample_fields,
        )

    fallback_name = default_dataset_name((reinit_config or ReinitConfig()).cfl)
    return GenerationConfig(
        num_workers=int(cfg.num_workers),
        generation_batch_size=int(cfg.generation_batch_size),
        output_dir=Path(cfg.output_dir),
        dataset_name=fallback_name,
        stored_sample_fields=stored_sample_fields,
    )


def _resolve_output_path(
    *,
    generation_config: GenerationConfig,
    reinit_config: ReinitConfig,
    path: str | Path | None = None,
) -> Path:
    if path is not None:
        return Path(path)

    normalized_config = _normalize_generation_config(generation_config, reinit_config=reinit_config)
    output_dir = Path(normalized_config.output_dir)
    dataset_name = normalized_config.dataset_name or default_dataset_filename(reinit_config.cfl)
    return output_dir / dataset_name


class CircleGeometryGenerator:
    def __init__(self, resolution_rho: int, seed: int = 42, variations: int = 5):
        self.rho = int(resolution_rho)
        self.global_seed = int(seed)
        self.variations = int(variations)

        if self.variations < 1:
            raise ValueError("variations must be >= 1")

        self.h = 1.0 / (self.rho - 1)
        self.r_min = 1.6 * self.h
        self.r_max = 0.5 - 2.0 * self.h
        self.num_radii = int(np.floor((self.rho - 8.2) / 2.0)) + 1

        if self.num_radii < 1 or self.r_min >= self.r_max:
            raise ValueError(f"Resolution rho={self.rho} too small for valid circular interfaces.")

        self.radii_set = np.linspace(self.r_min, self.r_max, self.num_radii, dtype=float)
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
            analytic_kappa = 1.0 / radius
            for v_idx in range(self.variations):
                sub_seed = self._subseed(r_idx, v_idx)
                rng = np.random.default_rng(sub_seed)
                cx = rng.uniform(self.center_min, self.center_max)
                cy = rng.uniform(self.center_min, self.center_max)
                blueprints.append(
                    {
                        "meta": {
                            "blueprint_id": f"rho{self.rho}_r{r_idx:03d}_v{v_idx:02d}_s{sub_seed}",
                            "blueprint_idx": int(self.rho * 100000 + r_idx * self.variations + v_idx),
                            "geometry_type": "circle",
                            "resolution": self.rho,
                            "radius_idx": int(r_idx),
                            "variation_idx": int(v_idx),
                            "global_seed": self.global_seed,
                            "sub_seed": sub_seed,
                        },
                        "params": {
                            "h": float(self.h),
                            "radius": float(radius),
                            "center": (float(cx), float(cy)),
                        },
                        "label": {
                            "source": "analytic_circle",
                            "kappa": float(analytic_kappa),
                        },
                    }
                )
        return blueprints


class LevelSetFieldBuilder:
    def __init__(self, dtype=np.float64) -> None:
        self.dtype = dtype

    def _build_grid(self, rho: int):
        h = 1.0 / (int(rho) - 1)
        x = np.linspace(0.0, 1.0, int(rho), dtype=self.dtype)
        X, Y = np.meshgrid(x, x, indexing="ij")
        return x, X, Y, float(h)

    @staticmethod
    def _parse_blueprint(blueprint: dict[str, Any]) -> tuple[int, float, float, float]:
        return (
            int(blueprint["meta"]["resolution"]),
            float(blueprint["params"]["radius"]),
            float(blueprint["params"]["center"][0]),
            float(blueprint["params"]["center"][1]),
        )

    def _pack(self, blueprint, phi, phi_type, x, X, Y, h, return_grid):
        packed = {
            "meta": {**copy.deepcopy(blueprint["meta"]), "stage": "level_set_field"},
            "params": {**copy.deepcopy(blueprint["params"])},
            "label": {**copy.deepcopy(blueprint["label"])},
            "field": {"phi_type": phi_type, "indexing": "ij", "phi": phi.astype(self.dtype, copy=False)},
        }
        if return_grid:
            packed["grid"] = {"x": x, "X": X, "Y": Y, "h": float(h)}
        return packed

    def build_circle_sdf(self, blueprint: dict[str, Any], *, return_grid: bool = True):
        rho, radius, cx, cy = self._parse_blueprint(blueprint)
        x, X, Y, h = self._build_grid(rho)
        phi = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius
        return self._pack(blueprint, phi, "circle_sdf", x, X, Y, h, return_grid)

    def build_circle_nonsdf(self, blueprint: dict[str, Any], *, return_grid: bool = True):
        rho, radius, cx, cy = self._parse_blueprint(blueprint)
        x, X, Y, h = self._build_grid(rho)
        phi = (X - cx) ** 2 + (Y - cy) ** 2 - radius ** 2
        return self._pack(blueprint, phi, "circle_nonsdf", x, X, Y, h, return_grid)


def bilinear_interpolate_field(field: np.ndarray, h: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = field.shape[0]

    gx = np.clip(x, 0.0, 1.0 - 1.0e-12) / h
    gy = np.clip(y, 0.0, 1.0 - 1.0e-12) / h

    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, rho - 1)
    j1 = np.clip(j0 + 1, 0, rho - 1)

    wx = gx - i0
    wy = gy - j0

    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    ).astype(np.float32)


def circle_sdf(x: np.ndarray, y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius


def circle_nonsdf(x: np.ndarray, y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return (x - cx) ** 2 + (y - cy) ** 2 - radius ** 2


def sample_points_from_reference_sdf_band(
    *,
    cx: float,
    cy: float,
    radius: float,
    h: float,
    n_points: int,
    band_half_width_cells: float,
    proposal_half_width_cells: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    band_half_width = float(band_half_width_cells) * float(h)
    proposal_half_width = max(float(proposal_half_width_cells) * float(h), band_half_width)
    accepted_x: list[np.ndarray] = []
    accepted_y: list[np.ndarray] = []
    accepted = 0
    attempts = 0

    while accepted < n_points:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Failed to sample enough points from the reference SDF band.")

        batch_n = max(4 * (n_points - accepted), 256)
        theta = rng.uniform(0.0, 2.0 * np.pi, batch_n)
        rho = rng.uniform(max(radius - proposal_half_width, 0.0), radius + proposal_half_width, batch_n)
        x = cx + rho * np.cos(theta)
        y = cy + rho * np.sin(theta)

        inside = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
        if not np.any(inside):
            continue

        x = x[inside]
        y = y[inside]
        phi = circle_sdf(x, y, cx=cx, cy=cy, radius=radius)
        mask = np.abs(phi) <= band_half_width
        if not np.any(mask):
            continue

        x_keep = x[mask]
        y_keep = y[mask]
        take = min(n_points - accepted, x_keep.shape[0])
        accepted_x.append(x_keep[:take].astype(np.float32))
        accepted_y.append(y_keep[:take].astype(np.float32))
        accepted += take

    return np.concatenate(accepted_x), np.concatenate(accepted_y)


def _to_tensor_column(values: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(values, axis=0)).view(-1, 1)


def _split_blueprint_indices(
    n_items: int,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, set[int]]:
    if n_items < 1:
        raise ValueError("Need at least one blueprint to split.")
    if train_fraction <= 0.0 or val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("Expected 0 < train_fraction and train_fraction + val_fraction < 1.")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_items)
    train_n = int(train_fraction * n_items)
    val_n = int(val_fraction * n_items)
    train_ids = perm[:train_n]
    val_ids = perm[train_n:train_n + val_n]
    test_ids = perm[train_n + val_n:]
    return {
        "train": set(int(idx) for idx in train_ids),
        "val": set(int(idx) for idx in val_ids),
        "test": set(int(idx) for idx in test_ids),
    }


def _new_sample_store(generation_config: GenerationConfig | None = None) -> dict[str, dict[str, list[np.ndarray]]]:
    return {
        "traj": {key: [] for key in _traj_field_names(generation_config)},
    }


def _extend_sample_store(target: dict[str, dict[str, list[np.ndarray]]], result: dict[str, Any]) -> None:
    for family in ("traj",):
        for key, chunks in result[family].items():
            target[family][key].extend(chunks)


def _tensorize_sample_store(store: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        family: {key: _to_tensor_column(value) for key, value in family_store.items()}
        for family, family_store in store.items()
    }


def _make_loader(data: dict[str, torch.Tensor], *, batch_size: int, shuffle: bool) -> DataLoader:
    ordered_keys = tuple(data.keys())
    dataset = TensorDataset(*(data[key] for key in ordered_keys))
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=TRAIN_LOADER_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=TRAIN_LOADER_WORKERS > 0,
    )
    loader._field_names = ordered_keys  # type: ignore[attr-defined]
    return loader


def _blueprints_for_split(bundle: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    indices = bundle.get("split_blueprint_indices", {}).get(split_name)
    if indices is None:
        raise KeyError(f"Bundle does not contain split_blueprint_indices[{split_name!r}]. Regenerate the dataset.")
    blueprints = bundle["blueprints"]
    return [blueprints[int(idx)] for idx in indices]


def _dynamic_store() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ("x", "y", "s", "cx", "cy", "radius", "h")}


def _traj_field_names(generation_config: GenerationConfig | None) -> tuple[str, ...]:
    extras = ()
    if generation_config is not None:
        extras = _normalize_stored_sample_fields(generation_config.stored_sample_fields)
    return (*BASE_TRAJ_FIELDS, *extras)


def _grid_gradient_center(field: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    grad_x, grad_y = np.gradient(np.asarray(field, dtype=np.float64), float(h), float(h), edge_order=2)
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def _analytic_phi_gradients(
    *,
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    initial_field_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if initial_field_type == "nonsdf":
        return (
            (2.0 * (x - float(cx))).astype(np.float32),
            (2.0 * (y - float(cy))).astype(np.float32),
        )

    dx = x - float(cx)
    dy = y - float(cy)
    denom = np.sqrt(dx ** 2 + dy ** 2)
    safe = np.where(denom > 1.0e-12, denom, 1.0)
    phi_x = np.where(denom > 1.0e-12, dx / safe, 0.0)
    phi_y = np.where(denom > 1.0e-12, dy / safe, 0.0)
    return phi_x.astype(np.float32), phi_y.astype(np.float32)


def _append_optional_traj_fields(
    traj_data: dict[str, list[np.ndarray]],
    *,
    stored_sample_fields: tuple[str, ...],
    field_grid: np.ndarray,
    h: float,
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    initial_field_type: str,
    use_analytic_gradients: bool,
) -> np.ndarray:
    phi_values = bilinear_interpolate_field(field_grid, h, x, y)
    if "phi" in stored_sample_fields:
        traj_data["phi"].append(phi_values)

    if "phi_x" in stored_sample_fields or "phi_y" in stored_sample_fields:
        if use_analytic_gradients:
            phi_x_values, phi_y_values = _analytic_phi_gradients(
                x=x,
                y=y,
                cx=cx,
                cy=cy,
                radius=radius,
                initial_field_type=initial_field_type,
            )
        else:
            grad_x_grid, grad_y_grid = _grid_gradient_center(field_grid, h)
            phi_x_values = bilinear_interpolate_field(grad_x_grid, h, x, y)
            phi_y_values = bilinear_interpolate_field(grad_y_grid, h, x, y)

        if "phi_x" in stored_sample_fields:
            traj_data["phi_x"].append(phi_x_values)
        if "phi_y" in stored_sample_fields:
            traj_data["phi_y"].append(phi_y_values)

    return phi_values


def _append_circle_columns(
    store: dict[str, list[np.ndarray]],
    *,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    h: float,
) -> None:
    store["x"].append(x.astype(np.float32, copy=False))
    store["y"].append(y.astype(np.float32, copy=False))
    store["s"].append(s.astype(np.float32, copy=False))
    store["cx"].append(np.full_like(x, cx, dtype=np.float32))
    store["cy"].append(np.full_like(x, cy, dtype=np.float32))
    store["radius"].append(np.full_like(x, radius, dtype=np.float32))
    store["h"].append(np.full_like(x, h, dtype=np.float32))


def _stored_s_values(data_config: DataConfig, reinit_config: ReinitConfig) -> np.ndarray:
    return np.asarray(
        [0.0, *(float(step) * reinit_config.cfl for step in data_config.reinit_steps)],
        dtype=np.float32,
    )


class DynamicCircleLoader:
    _field_names = ("x", "y", "s", "cx", "cy", "radius", "h")

    def __init__(
        self,
        bundle: dict[str, Any],
        split_name: str,
        *,
        family: str,
        seed: int,
        batch_size: int | None = None,
        shuffle: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        if family not in ("pde", "interface"):
            raise ValueError(f"Unsupported dynamic loader family={family!r}.")
        self.bundle = bundle
        self.split_name = split_name
        self.family = family
        self.seed = int(seed)
        generation_config = bundle.get("generation_config") or GenerationConfig()
        self.batch_size = int(batch_size or generation_config.generation_batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device) if device is not None else None
        self.blueprints = _blueprints_for_split(bundle, split_name)

        data_config = bundle["config"]
        reinit_config = bundle["reinit_config"]
        if family == "pde":
            self.total_samples = len(self.blueprints) * int(data_config.n_pde_per_circle)
        else:
            self.total_samples = (
                len(self.blueprints)
                * len(_stored_s_values(data_config, reinit_config))
                * int(data_config.n_interface_per_time)
            )

    def __len__(self) -> int:
        if self.device is not None and self.device.type == "cuda":
            data_config = self.bundle["config"]
            reinit_config = self.bundle["reinit_config"]
            if self.family == "pde":
                return len(self.blueprints) * int(np.ceil(int(data_config.n_pde_per_circle) / self.batch_size))
            return (
                len(self.blueprints)
                * len(_stored_s_values(data_config, reinit_config))
                * int(np.ceil(int(data_config.n_interface_per_time) / self.batch_size))
            )
        return int(np.ceil(self.total_samples / self.batch_size))

    @staticmethod
    def _empty_store() -> dict[str, list[np.ndarray]]:
        return _dynamic_store()

    @staticmethod
    def _store_count(store: dict[str, list[np.ndarray]]) -> int:
        return int(sum(chunk.shape[0] for chunk in store["x"]))

    @staticmethod
    def _emit(store: dict[str, list[np.ndarray]]) -> dict[str, torch.Tensor]:
        tensors = {key: _to_tensor_column(chunks) for key, chunks in store.items()}
        if torch.cuda.is_available():
            tensors = {key: tensor.pin_memory() for key, tensor in tensors.items()}
        return tensors

    @staticmethod
    def _torch_columns(
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        cx: float,
        cy: float,
        radius: float,
        h: float,
    ) -> dict[str, torch.Tensor]:
        return {
            "x": x.view(-1, 1),
            "y": y.view(-1, 1),
            "s": s.view(-1, 1),
            "cx": torch.full_like(x.view(-1, 1), float(cx)),
            "cy": torch.full_like(y.view(-1, 1), float(cy)),
            "radius": torch.full_like(x.view(-1, 1), float(radius)),
            "h": torch.full_like(x.view(-1, 1), float(h)),
        }

    @staticmethod
    def _sample_band_torch(
        *,
        cx: float,
        cy: float,
        radius: float,
        h: float,
        n_points: int,
        band_half_width_cells: float,
        proposal_half_width_cells: float,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        band_half_width = float(band_half_width_cells) * float(h)
        proposal_half_width = max(float(proposal_half_width_cells) * float(h), band_half_width)
        low = max(float(radius) - proposal_half_width, 0.0)
        high = float(radius) + proposal_half_width

        theta = torch.rand(n_points, device=device, generator=generator) * (2.0 * np.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        eps = 1.0e-12
        inf = torch.full_like(theta, float("inf"))

        x_bound = torch.where(
            cos_theta > eps,
            (1.0 - float(cx)) / cos_theta.clamp_min(eps),
            torch.where(cos_theta < -eps, -float(cx) / cos_theta.clamp_max(-eps), inf),
        )
        y_bound = torch.where(
            sin_theta > eps,
            (1.0 - float(cy)) / sin_theta.clamp_min(eps),
            torch.where(sin_theta < -eps, -float(cy) / sin_theta.clamp_max(-eps), inf),
        )
        domain_bound = torch.minimum(x_bound, y_bound)
        lower = torch.full_like(theta, low)
        upper = torch.minimum(torch.full_like(theta, high), domain_bound)
        upper = torch.maximum(upper, lower + 1.0e-7)
        rho = lower + torch.rand(n_points, device=device, generator=generator) * (upper - lower)

        x = float(cx) + rho * cos_theta
        y = float(cy) + rho * sin_theta
        return x.view(-1, 1), y.view(-1, 1)

    def _iter_torch(self):
        if self.device is None:
            raise RuntimeError("Device sampling requires a torch device.")

        data_config = self.bundle["config"]
        reinit_config = self.bundle["reinit_config"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(self.blueprints))
        if self.shuffle:
            rng.shuffle(order)

        for blueprint_idx in order:
            blueprint = self.blueprints[int(blueprint_idx)]
            h = float(blueprint["params"]["h"])
            radius = float(blueprint["params"]["radius"])
            cx = float(blueprint["params"]["center"][0])
            cy = float(blueprint["params"]["center"][1])

            if self.family == "pde":
                remaining = int(data_config.n_pde_per_circle)
                s_max = max(data_config.reinit_steps) * reinit_config.cfl
                while remaining > 0:
                    count = min(remaining, self.batch_size)
                    x, y = self._sample_band_torch(
                        cx=cx,
                        cy=cy,
                        radius=radius,
                        h=h,
                        n_points=count,
                        band_half_width_cells=data_config.band_half_width_cells,
                        proposal_half_width_cells=data_config.proposal_half_width_cells,
                        generator=generator,
                        device=self.device,
                    )
                    s = torch.rand(count, 1, device=self.device, generator=generator) * float(s_max)
                    yield self._torch_columns(x=x, y=y, s=s, cx=cx, cy=cy, radius=radius, h=h)
                    remaining -= count
            else:
                s_values = _stored_s_values(data_config, reinit_config)
                if self.shuffle:
                    s_values = rng.permutation(s_values)
                for s_value in s_values:
                    remaining = int(data_config.n_interface_per_time)
                    while remaining > 0:
                        count = min(remaining, self.batch_size)
                        theta = torch.rand(count, 1, device=self.device, generator=generator) * (2.0 * np.pi)
                        x = float(cx) + float(radius) * torch.cos(theta)
                        y = float(cy) + float(radius) * torch.sin(theta)
                        s = torch.full((count, 1), float(s_value), dtype=torch.float32, device=self.device)
                        yield self._torch_columns(x=x, y=y, s=s, cx=cx, cy=cy, radius=radius, h=h)
                        remaining -= count

    def __iter__(self):
        if self.device is not None and self.device.type == "cuda":
            yield from self._iter_torch()
            return

        data_config = self.bundle["config"]
        reinit_config = self.bundle["reinit_config"]
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(self.blueprints))
        if self.shuffle:
            rng.shuffle(order)

        store = self._empty_store()

        def append_and_yield_if_full(
            *,
            x: np.ndarray,
            y: np.ndarray,
            s: np.ndarray,
            cx: float,
            cy: float,
            radius: float,
            h: float,
        ):
            nonlocal store
            start = 0
            while start < x.shape[0]:
                room = self.batch_size - self._store_count(store)
                take = min(room, x.shape[0] - start)
                _append_circle_columns(
                    store,
                    x=x[start:start + take],
                    y=y[start:start + take],
                    s=s[start:start + take],
                    cx=cx,
                    cy=cy,
                    radius=radius,
                    h=h,
                )
                start += take
                if self._store_count(store) >= self.batch_size:
                    emitted = self._emit(store)
                    store = self._empty_store()
                    yield emitted

        for blueprint_idx in order:
            blueprint = self.blueprints[int(blueprint_idx)]
            h = float(blueprint["params"]["h"])
            radius = float(blueprint["params"]["radius"])
            cx = float(blueprint["params"]["center"][0])
            cy = float(blueprint["params"]["center"][1])

            if self.family == "pde":
                remaining = int(data_config.n_pde_per_circle)
                s_max = max(data_config.reinit_steps) * reinit_config.cfl
                while remaining > 0:
                    count = min(remaining, self.batch_size)
                    x, y = sample_points_from_reference_sdf_band(
                        cx=cx,
                        cy=cy,
                        radius=radius,
                        h=h,
                        n_points=count,
                        band_half_width_cells=data_config.band_half_width_cells,
                        proposal_half_width_cells=data_config.proposal_half_width_cells,
                        rng=rng,
                    )
                    s = rng.uniform(0.0, s_max, size=x.shape[0]).astype(np.float32)
                    yield from append_and_yield_if_full(x=x, y=y, s=s, cx=cx, cy=cy, radius=radius, h=h)
                    remaining -= count
            else:
                s_values = _stored_s_values(data_config, reinit_config)
                if self.shuffle:
                    s_values = rng.permutation(s_values)
                for s_value in s_values:
                    remaining = int(data_config.n_interface_per_time)
                    while remaining > 0:
                        count = min(remaining, self.batch_size)
                        theta = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
                        x = (cx + radius * np.cos(theta)).astype(np.float32)
                        y = (cy + radius * np.sin(theta)).astype(np.float32)
                        s = np.full_like(x, float(s_value), dtype=np.float32)
                        yield from append_and_yield_if_full(x=x, y=y, s=s, cx=cx, cy=cy, radius=radius, h=h)
                        remaining -= count

        if self._store_count(store) > 0:
            yield self._emit(store)


def sample_pde_loader(
    bundle: dict[str, Any],
    split_name: str,
    *,
    seed: int,
    batch_size: int | None = None,
    shuffle: bool = True,
    device: torch.device | str | None = None,
) -> DynamicCircleLoader:
    return DynamicCircleLoader(
        bundle,
        split_name,
        family="pde",
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
    )


def sample_interface_loader(
    bundle: dict[str, Any],
    split_name: str,
    *,
    seed: int,
    batch_size: int | None = None,
    shuffle: bool = True,
    device: torch.device | str | None = None,
) -> DynamicCircleLoader:
    return DynamicCircleLoader(
        bundle,
        split_name,
        family="interface",
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
    )


def _build_field_family(
    blueprint: dict[str, Any],
    *,
    reinit_builder: ReinitFieldPackBuilder,
    field_builder: LevelSetFieldBuilder,
    initial_field_type: str,
    reinit_steps: tuple[int, ...],
) -> dict[str, Any]:
    pack_sdf = field_builder.build_circle_sdf(blueprint, return_grid=False)
    if initial_field_type == "sdf":
        pack_phi0 = copy.deepcopy(pack_sdf)
        snapshots = {
            step: copy.deepcopy(pack_sdf["field"]["phi"]).astype(np.float32)
            for step in reinit_steps
        }
    elif initial_field_type == "nonsdf":
        pack_phi0 = field_builder.build_circle_nonsdf(blueprint, return_grid=False)
        reinit_results = reinit_builder.build(pack_phi0, steps_list=list(reinit_steps))
        snapshots = {step: reinit_results[str(step)]["field"]["phi"].astype(np.float32) for step in reinit_steps}
    else:
        raise ValueError(f"Unsupported initial_field_type={initial_field_type!r}; expected 'sdf' or 'nonsdf'.")
    return {
        "sdf": pack_sdf["field"]["phi"].astype(np.float32),
        "phi0": pack_phi0["field"]["phi"].astype(np.float32),
        "snapshots": snapshots,
    }


def _generate_all_blueprints(data_config: DataConfig) -> list[dict[str, Any]]:
    blueprints: list[dict[str, Any]] = []
    for rho in data_config.resolutions:
        generator = CircleGeometryGenerator(
            resolution_rho=int(rho),
            seed=data_config.geometry_seed,
            variations=data_config.variations,
        )
        blueprints.extend(generator.generate_blueprints())
    return blueprints


def _build_validation_fields(
    blueprints: list[dict[str, Any]],
    *,
    data_config: DataConfig,
    reinit_config: ReinitConfig,
) -> list[dict[str, Any]]:
    if not blueprints or data_config.validation_field_limit <= 0:
        return []

    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(
        cfl=reinit_config.cfl,
        eps_weno=reinit_config.eps_weno,
        eps_sign_factor=reinit_config.eps_sign_factor,
        sign_mode=reinit_config.sign_mode,
        time_order=reinit_config.time_order,
        space_order=reinit_config.space_order,
    )
    validation_stride = max(1, len(blueprints) // max(1, data_config.validation_field_limit))
    validation_fields: list[dict[str, Any]] = []

    for blueprint_idx, blueprint in enumerate(blueprints):
        if blueprint_idx % validation_stride != 0:
            continue
        field_family = _build_field_family(
            blueprint,
            reinit_builder=reinit_builder,
            field_builder=field_builder,
            initial_field_type=data_config.initial_field_type,
            reinit_steps=data_config.reinit_steps,
        )
        validation_fields.append(
            {
                "blueprint": copy.deepcopy(blueprint),
                "sdf": field_family["sdf"],
                "phi0": field_family["phi0"],
                "snapshots": field_family["snapshots"],
            }
        )
        if len(validation_fields) >= data_config.validation_field_limit:
            break

    return validation_fields


def _blueprint_sample_seed(base_seed: int, blueprint_idx: int) -> int:
    return int((int(base_seed) * 1000003 + int(blueprint_idx) * 9176 + 12345) % (1 << 32))


def _generate_blueprint_samples(
    blueprint_idx: int,
    blueprint: dict[str, Any],
    *,
    data_config_dict: dict[str, Any],
    reinit_config_dict: dict[str, Any],
    generation_config_dict: dict[str, Any],
    keep_validation_field: bool,
) -> dict[str, Any]:
    data_config = DataConfig(**data_config_dict)
    reinit_config = ReinitConfig(**reinit_config_dict)
    generation_config = _normalize_generation_config(GenerationConfig(**generation_config_dict), reinit_config=reinit_config)
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(
        cfl=reinit_config.cfl,
        eps_weno=reinit_config.eps_weno,
        eps_sign_factor=reinit_config.eps_sign_factor,
        sign_mode=reinit_config.sign_mode,
        time_order=reinit_config.time_order,
        space_order=reinit_config.space_order,
    )

    h = float(blueprint["params"]["h"])
    radius = float(blueprint["params"]["radius"])
    cx = float(blueprint["params"]["center"][0])
    cy = float(blueprint["params"]["center"][1])
    rng = np.random.default_rng(_blueprint_sample_seed(data_config.geometry_seed, blueprint_idx))
    field_family = _build_field_family(
        blueprint,
        reinit_builder=reinit_builder,
        field_builder=field_builder,
        initial_field_type=data_config.initial_field_type,
        reinit_steps=data_config.reinit_steps,
    )
    phi0_grid = field_family["phi0"]
    stored_sample_fields = _normalize_stored_sample_fields(generation_config.stored_sample_fields)

    traj_data: dict[str, list[np.ndarray]] = {
        key: [] for key in _traj_field_names(generation_config)
    }

    x0, y0 = sample_points_from_reference_sdf_band(
        cx=cx,
        cy=cy,
        radius=radius,
        h=h,
        n_points=data_config.n_traj_per_time,
        band_half_width_cells=data_config.band_half_width_cells,
        proposal_half_width_cells=data_config.proposal_half_width_cells,
        rng=rng,
    )
    traj_data["x"].append(x0)
    traj_data["y"].append(y0)
    traj_data["s"].append(np.zeros_like(x0, dtype=np.float32))
    traj_data["phi_target"].append(
        _append_optional_traj_fields(
            traj_data,
            stored_sample_fields=stored_sample_fields,
            field_grid=phi0_grid,
            h=h,
            x=x0,
            y=y0,
            cx=cx,
            cy=cy,
            radius=radius,
            initial_field_type=data_config.initial_field_type,
            use_analytic_gradients=True,
        )
    )
    traj_data["cx"].append(np.full_like(x0, cx, dtype=np.float32))
    traj_data["cy"].append(np.full_like(x0, cy, dtype=np.float32))
    traj_data["radius"].append(np.full_like(x0, radius, dtype=np.float32))
    traj_data["h"].append(np.full_like(x0, h, dtype=np.float32))

    for step in data_config.reinit_steps:
        phi_step = field_family["snapshots"][step]
        s_step = step * reinit_config.cfl

        x_traj, y_traj = sample_points_from_reference_sdf_band(
            cx=cx,
            cy=cy,
            radius=radius,
            h=h,
            n_points=data_config.n_traj_per_time,
            band_half_width_cells=data_config.band_half_width_cells,
            proposal_half_width_cells=data_config.proposal_half_width_cells,
            rng=rng,
        )
        traj_data["x"].append(x_traj)
        traj_data["y"].append(y_traj)
        traj_data["s"].append(np.full_like(x_traj, s_step, dtype=np.float32))
        traj_data["phi_target"].append(
            _append_optional_traj_fields(
                traj_data,
                stored_sample_fields=stored_sample_fields,
                field_grid=phi_step,
                h=h,
                x=x_traj,
                y=y_traj,
                cx=cx,
                cy=cy,
                radius=radius,
                initial_field_type=data_config.initial_field_type,
                use_analytic_gradients=False,
            )
        )
        traj_data["cx"].append(np.full_like(x_traj, cx, dtype=np.float32))
        traj_data["cy"].append(np.full_like(x_traj, cy, dtype=np.float32))
        traj_data["radius"].append(np.full_like(x_traj, radius, dtype=np.float32))
        traj_data["h"].append(np.full_like(x_traj, h, dtype=np.float32))

    result = {
        "blueprint_idx": blueprint_idx,
        "traj": traj_data,
        "validation_record": None,
    }
    if keep_validation_field:
        result["validation_record"] = {
            "blueprint": copy.deepcopy(blueprint),
            "sdf": field_family["sdf"],
            "phi0": phi0_grid,
            "snapshots": field_family["snapshots"],
        }
    return result


def generate_training_bundle(
    data_config: DataConfig | None = None,
    reinit_config: ReinitConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    data_config = data_config or DataConfig()
    reinit_config = reinit_config or ReinitConfig()
    generation_config = _normalize_generation_config(generation_config, reinit_config=reinit_config)

    all_blueprints = _generate_all_blueprints(data_config)

    split_ids = _split_blueprint_indices(
        len(all_blueprints),
        train_fraction=data_config.train_fraction,
        val_fraction=data_config.val_fraction,
        seed=data_config.geometry_seed,
    )
    split_stores = {split_name: _new_sample_store(generation_config) for split_name in ("train", "val", "test")}
    sdf_fields: list[dict[str, Any]] = []

    validation_stride = max(1, len(all_blueprints) // max(1, data_config.validation_field_limit))
    worker_count = int(generation_config.num_workers)
    task_args = [
        (
            blueprint_idx,
            blueprint,
            {
                "data_config_dict": asdict(data_config),
                "reinit_config_dict": asdict(reinit_config),
                "generation_config_dict": asdict(generation_config),
                "keep_validation_field": blueprint_idx % validation_stride == 0,
            },
        )
        for blueprint_idx, blueprint in enumerate(all_blueprints)
    ]

    progress = tqdm(total=len(task_args), desc="Generating circle data", unit="blueprint")
    results: list[dict[str, Any]] = []

    if worker_count == 1:
        for blueprint_idx, blueprint, extra in task_args:
            progress.set_postfix(
                rho=int(blueprint["meta"]["resolution"]),
                workers=worker_count,
            )
            result = _generate_blueprint_samples(
                blueprint_idx,
                blueprint,
                data_config_dict=extra["data_config_dict"],
                reinit_config_dict=extra["reinit_config_dict"],
                generation_config_dict=extra["generation_config_dict"],
                keep_validation_field=extra["keep_validation_field"],
            )
            results.append(result)
            progress.update(1)
    else:
        max_workers = min(worker_count, len(task_args), os.cpu_count() or worker_count)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _generate_blueprint_samples,
                    blueprint_idx,
                    blueprint,
                    data_config_dict=extra["data_config_dict"],
                    reinit_config_dict=extra["reinit_config_dict"],
                    generation_config_dict=extra["generation_config_dict"],
                    keep_validation_field=extra["keep_validation_field"],
                )
                for blueprint_idx, blueprint, extra in task_args
            ]
            for future in futures:
                results.append(future.result())
                progress.set_postfix(workers=max_workers)
                progress.update(1)
    progress.close()

    results.sort(key=lambda item: item["blueprint_idx"])
    for result in results:
        split_name = next(name for name, ids in split_ids.items() if result["blueprint_idx"] in ids)
        _extend_sample_store(split_stores[split_name], result)
        if result["validation_record"] is not None and len(sdf_fields) < data_config.validation_field_limit:
            sdf_fields.append(result["validation_record"])

    split_tensors = {
        split_name: _tensorize_sample_store(store)
        for split_name, store in split_stores.items()
    }

    return {
        "config": data_config,
        "reinit_config": reinit_config,
        "generation_config": generation_config,
        "blueprints": all_blueprints,
        "split_blueprint_indices": {name: sorted(ids) for name, ids in split_ids.items()},
        "split_blueprint_counts": {name: len(ids) for name, ids in split_ids.items()},
        "sdf_validation_fields": sdf_fields,
        "loaders": {
            f"{split_name}_traj": _make_loader(
                split_tensors[split_name]["traj"],
                batch_size=generation_config.generation_batch_size,
                shuffle=split_name == "train",
            )
            for split_name in ("train", "val", "test")
        },
        "sizes": {
            f"{split_name}_traj": split_tensors[split_name]["traj"]["x"].shape[0]
            for split_name in ("train", "val", "test")
        },
    }


def dataset_manifest_path(dataset_path: str | Path) -> Path:
    return Path(dataset_path).with_suffix(".json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _preview_records_from_loader(loader: DataLoader, *, limit: int = DEFAULT_PREVIEW_SAMPLES) -> list[dict[str, float]]:
    field_names = tuple(getattr(loader, "_field_names"))
    tensors = loader.dataset.tensors
    if not tensors:
        return []

    n_items = min(int(tensors[0].shape[0]), int(limit))
    preview: list[dict[str, float]] = []
    for row_idx in range(n_items):
        preview.append(
            {
                name: float(tensor[row_idx].reshape(-1)[0].item())
                for name, tensor in zip(field_names, tensors, strict=True)
            }
        )
    return preview


def build_dataset_manifest(bundle: dict[str, Any], dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    data_cfg = bundle["config"]
    reinit_cfg = bundle["reinit_config"]
    generation_cfg = bundle.get("generation_config") or GenerationConfig()
    generation_cfg = _normalize_generation_config(generation_cfg, reinit_config=reinit_cfg)

    splits: dict[str, Any] = {}
    for split_name, loader in bundle["loaders"].items():
        field_names = tuple(getattr(loader, "_field_names"))
        tensors = loader.dataset.tensors
        fields = [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in zip(field_names, tensors, strict=True)
        ]
        splits[split_name] = {
            "size": int(tensors[0].shape[0]) if tensors else 0,
            "fields": fields,
            "sample_preview": _preview_records_from_loader(loader, limit=DEFAULT_PREVIEW_SAMPLES),
        }

    stored_sample_fields = _normalize_stored_sample_fields(generation_cfg.stored_sample_fields)
    manifest = {
        "dataset_file": {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "manifest_path": str(dataset_manifest_path(dataset_path)),
        },
        "configs": {
            "data": _json_ready(asdict(data_cfg)),
            "generation": _json_ready(asdict(generation_cfg)),
            "reinit": _json_ready(asdict(reinit_cfg)),
        },
        "split_blueprint_counts": _json_ready(bundle.get("split_blueprint_counts", {})),
        "stored_sample_fields": list(stored_sample_fields),
        "contains": {
            field_name: field_name in stored_sample_fields
            for field_name in SUPPORTED_STORED_SAMPLE_FIELDS
        },
        "splits": splits,
    }
    return manifest


def save_dataset_manifest(bundle: dict[str, Any], dataset_path: str | Path) -> Path:
    manifest_path = dataset_manifest_path(dataset_path)
    manifest = build_dataset_manifest(bundle, dataset_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def read_dataset_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    manifest_path = dataset_manifest_path(dataset_path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_dataset_summary_from_hdf5(dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    manifest = {
        "dataset_file": {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "manifest_path": str(dataset_manifest_path(dataset_path)),
        },
        "configs": {},
        "split_blueprint_counts": {},
        "stored_sample_fields": [],
        "contains": {
            field_name: False
            for field_name in SUPPORTED_STORED_SAMPLE_FIELDS
        },
        "splits": {},
    }

    with h5py.File(dataset_path, "r") as handle:
        stored_sample_fields_raw = handle.attrs.get("stored_sample_fields_json", "[]")
        if isinstance(stored_sample_fields_raw, bytes):
            stored_sample_fields_raw = stored_sample_fields_raw.decode("utf-8")
        stored_sample_fields = list(_normalize_stored_sample_fields(json.loads(str(stored_sample_fields_raw))))
        manifest["stored_sample_fields"] = stored_sample_fields
        manifest["contains"] = {
            field_name: field_name in stored_sample_fields
            for field_name in SUPPORTED_STORED_SAMPLE_FIELDS
        }
        manifest["configs"] = {
            "data": {
                "resolutions": [int(item) for item in handle["resolutions"][:]] if "resolutions" in handle else [],
                "reinit_steps": [int(item) for item in handle["reinit_steps"][:]] if "reinit_steps" in handle else [],
                "geometry_seed": int(handle.attrs.get("geometry_seed", 0)),
                "variations": int(handle.attrs.get("variations", 0)),
                "initial_field_type": str(handle.attrs.get("initial_field_type", "")),
                "n_traj_per_time": int(handle.attrs.get("n_traj_per_time", 0)),
                "n_pde_per_circle": int(handle.attrs.get("n_pde_per_circle", 0)),
                "n_interface_per_time": int(handle.attrs.get("n_interface_per_time", 0)),
                "band_half_width_cells": float(handle.attrs.get("band_half_width_cells", 0.0)),
                "proposal_half_width_cells": float(handle.attrs.get("proposal_half_width_cells", 0.0)),
                "train_fraction": float(handle.attrs.get("train_fraction", 0.0)),
                "val_fraction": float(handle.attrs.get("val_fraction", 0.0)),
                "validation_field_limit": int(handle.attrs.get("validation_field_limit", 0)),
            },
            "generation": {
                "num_workers": int(handle.attrs.get("num_workers", 0)),
                "generation_batch_size": int(handle.attrs.get("generation_batch_size", handle.attrs.get("batch_size", 0))),
                "output_dir": str(handle.attrs.get("output_dir", dataset_path.parent)),
                "dataset_name": str(handle.attrs.get("dataset_name", dataset_path.name)),
                "stored_sample_fields": stored_sample_fields,
            },
            "reinit": {
                "cfl": float(handle.attrs.get("cfl", 0.0)),
                "eps_weno": float(handle.attrs.get("eps_weno", 0.0)),
                "eps_sign_factor": float(handle.attrs.get("eps_sign_factor", 0.0)),
                "sign_mode": str(handle.attrs.get("sign_mode", "")),
                "time_order": int(handle.attrs.get("time_order", 0)),
                "space_order": int(handle.attrs.get("space_order", 0)),
            },
        }
        manifest["split_blueprint_counts"] = {
            split_name: int(handle.attrs[f"{split_name}_blueprints"])
            for split_name in ("train", "val", "test")
            if f"{split_name}_blueprints" in handle.attrs
        }

        for split_name in ("train_traj", "val_traj", "test_traj"):
            if split_name not in handle:
                continue
            group = handle[split_name]
            field_names = list(group.keys())
            fields = [
                {
                    "name": field_name,
                    "shape": list(group[field_name].shape),
                    "dtype": str(group[field_name].dtype),
                }
                for field_name in field_names
            ]
            preview_size = min(int(group[field_names[0]].shape[0]) if field_names else 0, DEFAULT_PREVIEW_SAMPLES)
            preview: list[dict[str, float]] = []
            for row_idx in range(preview_size):
                preview.append(
                    {
                        field_name: float(np.asarray(group[field_name][row_idx]).reshape(-1)[0].item())
                        for field_name in field_names
                    }
                )
            manifest["splits"][split_name] = {
                "size": int(group[field_names[0]].shape[0]) if field_names else 0,
                "fields": fields,
                "sample_preview": preview,
            }

    return manifest


def save_training_bundle_hdf5(bundle: dict[str, Any], path: str | Path | None = None) -> Path:
    reinit_cfg = bundle["reinit_config"]
    generation_cfg = _normalize_generation_config(bundle.get("generation_config"), reinit_config=reinit_cfg)
    output_path = _resolve_output_path(
        generation_config=generation_cfg,
        reinit_config=reinit_cfg,
        path=path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_cfg = bundle["config"]
    loaders = bundle["loaders"]

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_format_version"] = DATASET_FORMAT_VERSION
        handle.attrs["geometry_seed"] = int(data_cfg.geometry_seed)
        handle.attrs["variations"] = int(data_cfg.variations)
        handle.attrs["initial_field_type"] = str(data_cfg.initial_field_type)
        handle.attrs["generation_batch_size"] = int(generation_cfg.generation_batch_size)
        handle.attrs["num_workers"] = int(generation_cfg.num_workers)
        handle.attrs["output_dir"] = str(Path(generation_cfg.output_dir))
        handle.attrs["dataset_name"] = str(generation_cfg.dataset_name)
        handle.attrs["stored_sample_fields_json"] = json.dumps(list(generation_cfg.stored_sample_fields))
        handle.attrs["n_traj_per_time"] = int(data_cfg.n_traj_per_time)
        handle.attrs["n_pde_per_circle"] = int(data_cfg.n_pde_per_circle)
        handle.attrs["n_interface_per_time"] = int(data_cfg.n_interface_per_time)
        handle.attrs["band_half_width_cells"] = float(data_cfg.band_half_width_cells)
        handle.attrs["proposal_half_width_cells"] = float(data_cfg.proposal_half_width_cells)
        handle.attrs["train_fraction"] = float(data_cfg.train_fraction)
        handle.attrs["val_fraction"] = float(data_cfg.val_fraction)
        handle.attrs["validation_field_limit"] = int(data_cfg.validation_field_limit)
        handle.attrs["cfl"] = float(reinit_cfg.cfl)
        handle.attrs["eps_weno"] = float(reinit_cfg.eps_weno)
        handle.attrs["eps_sign_factor"] = float(reinit_cfg.eps_sign_factor)
        handle.attrs["sign_mode"] = str(reinit_cfg.sign_mode)
        handle.attrs["time_order"] = int(reinit_cfg.time_order)
        handle.attrs["space_order"] = int(reinit_cfg.space_order)
        handle.attrs["time_coordinate"] = "s=tau/h"
        for split_name, count in bundle.get("split_blueprint_counts", {}).items():
            handle.attrs[f"{split_name}_blueprints"] = int(count)
        handle.create_dataset("resolutions", data=np.asarray(data_cfg.resolutions, dtype=np.int32))
        handle.create_dataset("reinit_steps", data=np.asarray(data_cfg.reinit_steps, dtype=np.int32))
        handle.create_dataset("blueprints_json", data=json.dumps(bundle.get("blueprints", [])).encode("utf-8"))
        split_group = handle.create_group("split_blueprint_indices")
        for split_name, indices in bundle.get("split_blueprint_indices", {}).items():
            split_group.create_dataset(split_name, data=np.asarray(indices, dtype=np.int32))

        validation_group = handle.create_group("sdf_validation_fields")
        for field_idx, record in enumerate(bundle.get("sdf_validation_fields", [])):
            field_group = validation_group.create_group(str(field_idx))
            field_group.attrs["blueprint_json"] = json.dumps(record["blueprint"])
            field_group.create_dataset("sdf", data=np.asarray(record["sdf"], dtype=np.float32), compression="gzip")
            field_group.create_dataset("phi0", data=np.asarray(record["phi0"], dtype=np.float32), compression="gzip")
            snapshots_group = field_group.create_group("snapshots")
            for step, phi in record["snapshots"].items():
                snapshots_group.create_dataset(str(int(step)), data=np.asarray(phi, dtype=np.float32), compression="gzip")

        for split_name, loader in loaders.items():
            group = handle.create_group(split_name)
            field_names = loader._field_names  # type: ignore[attr-defined]
            for name, tensor in zip(field_names, loader.dataset.tensors, strict=True):
                group.create_dataset(name, data=tensor.cpu().numpy(), compression="gzip")

    save_dataset_manifest(bundle, output_path)
    return output_path


def load_training_bundle_from_hdf5(path: str | Path, *, batch_size: int | None = None) -> dict[str, Any]:
    """Load training bundle from existing HDF5 file for training."""
    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path.resolve()}")
    if batch_size is not None and int(batch_size) < 1:
        raise ValueError("batch_size override must be >= 1.")

    with h5py.File(h5_path, "r") as handle:
        dataset_format_version = int(handle.attrs.get("dataset_format_version", 1))
        if dataset_format_version != DATASET_FORMAT_VERSION:
            raise ValueError(
                f"Dataset format version {dataset_format_version} is incompatible with the current "
                f"trajectory/PDE/interface pipeline. Regenerate the HDF5 dataset so it has "
                f"dataset_format_version={DATASET_FORMAT_VERSION}."
            )
        default_data_config = DataConfig()
        default_generation_config = GenerationConfig()
        geometry_seed = int(handle.attrs["geometry_seed"])
        variations = int(handle.attrs.get("variations", default_data_config.variations))
        initial_field_type = str(handle.attrs.get("initial_field_type", default_data_config.initial_field_type))
        stored_generation_batch_size = int(
            handle.attrs.get(
                "generation_batch_size",
                handle.attrs.get("batch_size", default_generation_config.generation_batch_size),
            )
        )
        raw_stored_sample_fields = handle.attrs.get("stored_sample_fields_json", "[]")
        if isinstance(raw_stored_sample_fields, bytes):
            raw_stored_sample_fields = raw_stored_sample_fields.decode("utf-8")
        stored_sample_fields = tuple(json.loads(str(raw_stored_sample_fields)))
        num_workers = int(handle.attrs.get("num_workers", default_generation_config.num_workers))
        output_dir = Path(str(handle.attrs.get("output_dir", h5_path.parent)))
        dataset_name = str(handle.attrs.get("dataset_name", h5_path.name))
        n_traj_per_time = int(handle.attrs.get("n_traj_per_time", default_data_config.n_traj_per_time))
        n_pde_per_circle = int(handle.attrs.get("n_pde_per_circle", default_data_config.n_pde_per_circle))
        n_interface_per_time = int(handle.attrs.get("n_interface_per_time", default_data_config.n_interface_per_time))
        band_half_width_cells = float(handle.attrs.get("band_half_width_cells", default_data_config.band_half_width_cells))
        proposal_half_width_cells = float(handle.attrs.get("proposal_half_width_cells", default_data_config.proposal_half_width_cells))
        train_fraction = float(handle.attrs.get("train_fraction", default_data_config.train_fraction))
        val_fraction = float(handle.attrs.get("val_fraction", default_data_config.val_fraction))
        validation_field_limit = int(handle.attrs.get("validation_field_limit", default_data_config.validation_field_limit))
        split_blueprint_counts = {
            name: int(handle.attrs[f"{name}_blueprints"])
            for name in ("train", "val", "test")
            if f"{name}_blueprints" in handle.attrs
        }
        cfl = float(handle.attrs["cfl"])
        eps_weno = float(handle.attrs["eps_weno"])
        eps_sign_factor = float(handle.attrs["eps_sign_factor"])
        sign_mode = str(handle.attrs["sign_mode"])
        time_order = int(handle.attrs["time_order"])
        space_order = int(handle.attrs["space_order"])
        resolutions = tuple(handle["resolutions"][:])
        reinit_steps = tuple(handle["reinit_steps"][:])
        if "blueprints_json" in handle:
            raw_blueprints = handle["blueprints_json"][()]
            if isinstance(raw_blueprints, bytes):
                raw_blueprints = raw_blueprints.decode("utf-8")
            blueprints = json.loads(str(raw_blueprints))
        else:
            blueprints = []
        split_blueprint_indices = {
            name: [int(item) for item in handle["split_blueprint_indices"][name][:]]
            for name in ("train", "val", "test")
            if "split_blueprint_indices" in handle and name in handle["split_blueprint_indices"]
        }

    def _read_traj_split(group_name: str) -> dict[str, torch.Tensor]:
        with h5py.File(h5_path, "r") as handle:
            group = handle[group_name]
            keys = list(group.keys())
            arrays = {key: group[key][:] for key in keys}

        if "s" not in arrays and "t" in arrays:
            if "h" in arrays:
                arrays["s"] = arrays["t"] / arrays["h"]
            else:
                arrays["s"] = arrays["t"]
            del arrays["t"]

        field_order = ("x", "y", "s", "phi_target", "cx", "cy", "radius", "h")
        keys = [key for key in field_order if key in arrays]
        missing = [key for key in field_order if key not in arrays]
        if missing:
            raise ValueError(f"Dataset group {group_name!r} is missing required v2 columns: {missing}.")
        return {key: torch.from_numpy(arrays[key]) for key in keys}

    traj_splits = {
        "train_traj": _read_traj_split("train_traj"),
        "val_traj": _read_traj_split("val_traj"),
        "test_traj": _read_traj_split("test_traj"),
    }

    data_config = DataConfig(
        resolutions=resolutions,
        geometry_seed=geometry_seed,
        variations=variations,
        initial_field_type=initial_field_type,
        reinit_steps=reinit_steps,
        n_traj_per_time=n_traj_per_time,
        n_pde_per_circle=n_pde_per_circle,
        n_interface_per_time=n_interface_per_time,
        band_half_width_cells=band_half_width_cells,
        proposal_half_width_cells=proposal_half_width_cells,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        validation_field_limit=validation_field_limit,
    )
    reinit_config = ReinitConfig(
        cfl=cfl,
        eps_weno=eps_weno,
        eps_sign_factor=eps_sign_factor,
        sign_mode=sign_mode,
        time_order=time_order,
        space_order=space_order,
    )
    generation_config = _normalize_generation_config(
        GenerationConfig(
            num_workers=num_workers,
            generation_batch_size=stored_generation_batch_size,
            output_dir=output_dir,
            dataset_name=dataset_name,
            stored_sample_fields=stored_sample_fields,
        ),
        reinit_config=reinit_config,
    )

    if not blueprints:
        blueprints = _generate_all_blueprints(data_config)

    sdf_validation_fields: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        if "sdf_validation_fields" in handle:
            validation_group = handle["sdf_validation_fields"]
            for key in sorted(validation_group.keys(), key=lambda item: int(item)):
                field_group = validation_group[key]
                snapshots_group = field_group["snapshots"]
                sdf_validation_fields.append(
                    {
                        "blueprint": json.loads(str(field_group.attrs["blueprint_json"])),
                        "sdf": field_group["sdf"][:],
                        "phi0": field_group["phi0"][:],
                        "snapshots": {int(step): snapshots_group[step][:] for step in snapshots_group.keys()},
                    }
                )

    if not sdf_validation_fields:
        sdf_validation_fields = _build_validation_fields(
            blueprints,
            data_config=data_config,
            reinit_config=reinit_config,
        )

    bundle = {
        "config": data_config,
        "reinit_config": reinit_config,
        "generation_config": generation_config,
        "split_blueprint_indices": split_blueprint_indices,
        "split_blueprint_counts": split_blueprint_counts,
        "traj_splits": traj_splits,
        "sizes": {},
        "blueprints": blueprints,
        "sdf_validation_fields": sdf_validation_fields,
    }
    
    # Compute sizes
    for split_name, split in bundle["traj_splits"].items():
        first_tensor = next(iter(split.values()), None)
        bundle["sizes"][split_name] = int(first_tensor.shape[0]) if first_tensor is not None else 0
    
    return bundle


def describe_bundle(bundle: dict[str, Any]) -> None:
    reinit_cfg = bundle["reinit_config"]
    data_cfg = bundle["config"]
    generation_cfg = bundle.get("generation_config")
    print(f"Number of blueprints: {len(bundle['blueprints'])}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field type: {data_cfg.initial_field_type}")
    print(f"Reinit steps: {data_cfg.reinit_steps}")
    if generation_cfg is not None:
        print(f"Generation batch size: {generation_cfg.generation_batch_size}")
        print(f"Generation workers: {generation_cfg.num_workers}")
        print(f"Configured output path: {generation_cfg.output_path()}")
        print(f"Stored sample fields: {generation_cfg.stored_sample_fields or '()'}")
    print(f"Trajectory samples per stored time per circle: {data_cfg.n_traj_per_time}")
    print(f"PDE samples per circle per epoch: {data_cfg.n_pde_per_circle}")
    print(f"Interface samples per stored time per circle per epoch: {data_cfg.n_interface_per_time}")
    print(f"Interface samples per circle per epoch: {data_cfg.n_interface_per_time * (1 + len(data_cfg.reinit_steps))}")
    print(f"Blueprint split fractions: train={data_cfg.train_fraction}, val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}")
    if bundle.get("split_blueprint_counts"):
        print(f"Blueprint split counts: {bundle['split_blueprint_counts']}")
    print(f"CFL: {reinit_cfg.cfl}")
    print(f"eps_weno: {reinit_cfg.eps_weno}")
    print(f"eps_sign_factor: {reinit_cfg.eps_sign_factor}")
    for split_name, split_size in bundle["sizes"].items():
        print(f"{split_name:>14s}: {split_size}")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    data_cfg = DataConfig()
    generation_cfg = GenerationConfig()
    reinit_cfg = ReinitConfig()
    parser = argparse.ArgumentParser(description="Generate PINN training data and export it to HDF5.")

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional full output path override. Defaults to GenerationConfig.output_dir / dataset_name.",
    )
    output_group.add_argument("--output-dir", type=str, default=str(generation_cfg.output_dir))
    output_group.add_argument("--dataset-name", type=str, default=generation_cfg.dataset_name)

    data_group = parser.add_argument_group("data content")
    data_group.add_argument("--resolutions", type=str, default=",".join(str(item) for item in data_cfg.resolutions))
    data_group.add_argument("--reinit-steps", type=str, default=",".join(str(item) for item in data_cfg.reinit_steps))
    data_group.add_argument("--geometry-seed", type=int, default=data_cfg.geometry_seed)
    data_group.add_argument("--variations", type=int, default=data_cfg.variations)
    data_group.add_argument("--initial-field-type", type=str, choices=("sdf", "nonsdf"), default=data_cfg.initial_field_type)
    data_group.add_argument("--n-traj-per-time", type=int, default=data_cfg.n_traj_per_time)
    data_group.add_argument("--n-pde-per-circle", type=int, default=data_cfg.n_pde_per_circle)
    data_group.add_argument("--n-interface-per-time", type=int, default=data_cfg.n_interface_per_time)
    data_group.add_argument("--band-half-width-cells", type=float, default=data_cfg.band_half_width_cells)
    data_group.add_argument("--proposal-half-width-cells", type=float, default=data_cfg.proposal_half_width_cells)
    data_group.add_argument("--train-fraction", type=float, default=data_cfg.train_fraction)
    data_group.add_argument("--val-fraction", type=float, default=data_cfg.val_fraction)
    data_group.add_argument("--validation-field-limit", type=int, default=data_cfg.validation_field_limit)

    generation_group = parser.add_argument_group("generation runtime")
    generation_group.add_argument("--num-workers", type=int, default=generation_cfg.num_workers)
    generation_group.add_argument("--generation-batch-size", type=int, default=generation_cfg.generation_batch_size)
    generation_group.add_argument(
        "--stored-sample-fields",
        type=str,
        default=",".join(generation_cfg.stored_sample_fields),
        help="Optional comma-separated extra per-sample fields to store. Supported: phi, phi_x, phi_y.",
    )

    reinit_group = parser.add_argument_group("reinitialization")
    reinit_group.add_argument("--cfl", type=float, default=reinit_cfg.cfl)
    reinit_group.add_argument("--eps-weno", type=float, default=reinit_cfg.eps_weno)
    reinit_group.add_argument("--eps-sign-factor", type=float, default=reinit_cfg.eps_sign_factor)
    reinit_group.add_argument("--sign-mode", type=str, default=reinit_cfg.sign_mode)
    reinit_group.add_argument("--time-order", type=int, default=reinit_cfg.time_order)
    reinit_group.add_argument("--space-order", type=int, default=reinit_cfg.space_order)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_config = DataConfig(
        resolutions=_parse_int_tuple(args.resolutions),
        geometry_seed=args.geometry_seed,
        variations=args.variations,
        initial_field_type=args.initial_field_type,
        reinit_steps=_parse_int_tuple(args.reinit_steps),
        n_traj_per_time=args.n_traj_per_time,
        n_pde_per_circle=args.n_pde_per_circle,
        n_interface_per_time=args.n_interface_per_time,
        band_half_width_cells=args.band_half_width_cells,
        proposal_half_width_cells=args.proposal_half_width_cells,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        validation_field_limit=args.validation_field_limit,
    )
    reinit_config = ReinitConfig(
        cfl=args.cfl,
        eps_weno=args.eps_weno,
        eps_sign_factor=args.eps_sign_factor,
        sign_mode=args.sign_mode,
        time_order=args.time_order,
        space_order=args.space_order,
    )
    generation_config = _normalize_generation_config(
        GenerationConfig(
            num_workers=args.num_workers,
            generation_batch_size=args.generation_batch_size,
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
            stored_sample_fields=_parse_str_tuple(args.stored_sample_fields),
        ),
        reinit_config=reinit_config,
    )

    bundle = generate_training_bundle(
        data_config=data_config,
        reinit_config=reinit_config,
        generation_config=generation_config,
    )
    output_path = save_training_bundle_hdf5(bundle, args.output or None)
    print(f"HDF5 file: {output_path.resolve()}")
    print(f"Dataset manifest: {dataset_manifest_path(output_path).resolve()}")
    describe_bundle(bundle)


if __name__ == "__main__":
    main()
