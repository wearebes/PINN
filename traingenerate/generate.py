from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from traingenerate.config import DataConfig, GenerationConfig
    from traingenerate.io import dataset_manifest_path, normalize_generation_config, save_training_dataset_hdf5
else:
    from .config import DataConfig, GenerationConfig
    from .io import dataset_manifest_path, normalize_generation_config, save_training_dataset_hdf5


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
            for v_idx in range(self.variations):
                sub_seed = self._subseed(r_idx, v_idx)
                rng = np.random.default_rng(sub_seed)
                blueprints.append(
                    {
                        "meta": {
                            "blueprint_id": f"rho{self.rho}_r{r_idx:03d}_v{v_idx:02d}_s{sub_seed}",
                            "resolution": self.rho,
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


def build_grid(rho: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, int(rho), dtype=np.float64)
    return np.meshgrid(x, x, indexing="ij")


def build_circle_sdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius


def build_circle_nonsdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return (X - cx) ** 2 + (Y - cy) ** 2 - radius**2


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


def build_phi0_grid(blueprint: dict[str, Any], initial_field_type: str) -> np.ndarray:
    rho = int(blueprint["meta"]["resolution"])
    radius = float(blueprint["params"]["radius"])
    cx = float(blueprint["params"]["center"][0])
    cy = float(blueprint["params"]["center"][1])
    X, Y = build_grid(rho)
    if initial_field_type == "sdf":
        phi0 = build_circle_sdf(X, Y, cx=cx, cy=cy, radius=radius)
    elif initial_field_type == "nonsdf":
        phi0 = build_circle_nonsdf(X, Y, cx=cx, cy=cy, radius=radius)
    else:
        raise ValueError(f"Unsupported initial_field_type={initial_field_type!r}; expected 'sdf' or 'nonsdf'.")
    return phi0.astype(np.float32, copy=False)


def normalize_initial_field_types(initial_field_types: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in initial_field_types or ():
        field_type = str(item).strip().lower()
        if not field_type:
            continue
        if field_type not in {"sdf", "nonsdf"}:
            raise ValueError(f"Unsupported initial_field_type={item!r}; expected 'sdf' or 'nonsdf'.")
        if field_type not in normalized:
            normalized.append(field_type)
    if not normalized:
        raise ValueError("initial_field_types must contain at least one of: sdf, nonsdf.")
    return tuple(normalized)


def generate_blueprints(data_config: DataConfig) -> list[dict[str, Any]]:
    blueprints: list[dict[str, Any]] = []
    for rho in data_config.resolutions:
        blueprints.extend(
            CircleGeometryGenerator(
                resolution_rho=int(rho),
                seed=data_config.geometry_seed,
                variations=data_config.variations,
            ).generate_blueprints()
        )
    return blueprints


def split_blueprint_indices(
    total: int,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if total < 3:
        raise ValueError("Need at least 3 blueprints to create train/val/test splits.")
    if not (0.0 < train_fraction < 1.0) or not (0.0 <= val_fraction < 1.0):
        raise ValueError("Invalid split fractions.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    order = np.arange(total, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    train_end = max(1, int(round(total * train_fraction)))
    val_end = max(train_end + 1, int(round(total * (train_fraction + val_fraction))))
    val_end = min(val_end, total - 1)
    return {
        "train": sorted(int(item) for item in order[:train_end]),
        "val": sorted(int(item) for item in order[train_end:val_end]),
        "test": sorted(int(item) for item in order[val_end:]),
    }


def sample_blueprint(
    blueprint: dict[str, Any],
    *,
    data_config: DataConfig,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    samples: list[dict[str, np.ndarray]] = []
    h = float(blueprint["params"]["h"])
    radius = float(blueprint["params"]["radius"])
    for initial_field_type in normalize_initial_field_types(data_config.initial_field_types):
        phi0 = build_phi0_grid(blueprint, initial_field_type)
        indices = interface_indices(phi0)
        if indices.size == 0:
            raise RuntimeError(
                f"No interface nodes found for {blueprint['meta']['blueprint_id']} with initial_field_type={initial_field_type}."
            )

        if indices.shape[0] > int(data_config.n_samples_per_circle):
            chosen = rng.choice(indices.shape[0], size=int(data_config.n_samples_per_circle), replace=False)
            indices = indices[chosen]

        samples.append(
            {
                "phi9": extract_phi9(phi0, indices),
                "hkappa_target": np.full((indices.shape[0], 1), h / radius, dtype=np.float32),
            }
        )
    return samples


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


def generate_training_splits(data_config: DataConfig | None = None) -> dict[str, Any]:
    data_config = data_config or DataConfig()
    data_config = DataConfig(
        resolutions=tuple(int(item) for item in data_config.resolutions),
        geometry_seed=int(data_config.geometry_seed),
        variations=int(data_config.variations),
        initial_field_types=normalize_initial_field_types(data_config.initial_field_types),
        n_samples_per_circle=int(data_config.n_samples_per_circle),
        train_fraction=float(data_config.train_fraction),
        val_fraction=float(data_config.val_fraction),
    )
    blueprints = generate_blueprints(data_config)
    split_indices = split_blueprint_indices(
        len(blueprints),
        train_fraction=data_config.train_fraction,
        val_fraction=data_config.val_fraction,
        seed=data_config.geometry_seed,
    )

    split_samples: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    progress = tqdm(blueprints, desc="Generating stencil data", unit="blueprint")
    for blueprint_idx, blueprint in enumerate(progress):
        samples = sample_blueprint(
            blueprint,
            data_config=data_config,
            rng=np.random.default_rng(data_config.geometry_seed * 100003 + blueprint_idx),
        )
        split_name = next(name for name, ids in split_indices.items() if blueprint_idx in ids)
        split_samples[split_name].extend(samples)

    splits = {name: concat_split_samples(items) for name, items in split_samples.items()}
    sizes = {name: int(split["phi9"].shape[0]) for name, split in splits.items()}

    return {
        "config": data_config,
        "blueprints": blueprints,
        "split_blueprint_indices": split_indices,
        "split_blueprint_counts": {name: len(ids) for name, ids in split_indices.items()},
        "splits": splits,
        "sizes": sizes,
    }


def describe_generated_splits(bundle: dict[str, Any]) -> None:
    data_cfg = bundle["config"]
    print("Task: 3x3 phi stencil -> h*kappa")
    print(f"Number of blueprints: {len(bundle['blueprints'])}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field types: {data_cfg.initial_field_types}")
    print(f"Samples per circle per field type: {data_cfg.n_samples_per_circle}")
    print(
        f"Blueprint split fractions: train={data_cfg.train_fraction}, "
        f"val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}"
    )
    print(f"Config snapshot: {asdict(data_cfg)}")
    for split_name, split_size in bundle["sizes"].items():
        print(f"{split_name:>5s}: {split_size}")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


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
    parser.add_argument("--n-samples-per-circle", type=int, default=data_cfg.n_samples_per_circle)
    parser.add_argument("--train-fraction", type=float, default=data_cfg.train_fraction)
    parser.add_argument("--val-fraction", type=float, default=data_cfg.val_fraction)
    parser.add_argument("--num-workers", type=int, default=generation_cfg.num_workers)
    parser.add_argument("--generation-batch-size", type=int, default=generation_cfg.generation_batch_size)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_config = DataConfig(
        resolutions=_parse_int_tuple(args.resolutions),
        geometry_seed=args.geometry_seed,
        variations=args.variations,
        initial_field_types=normalize_initial_field_types(tuple(part.strip() for part in str(args.initial_field_types).split(","))),
        n_samples_per_circle=args.n_samples_per_circle,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    generation_config = normalize_generation_config(
        GenerationConfig(
            num_workers=args.num_workers,
            generation_batch_size=args.generation_batch_size,
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
        )
    )
    bundle = generate_training_splits(data_config=data_config)
    describe_generated_splits(bundle)
    output_path = save_training_dataset_hdf5(
        {
            **bundle,
            "generation_config": generation_config,
        },
        path=args.output or None,
    )
    print(f"Saved dataset to: {output_path.resolve()}")
    print(f"Saved manifest to: {dataset_manifest_path(output_path).resolve()}")


if __name__ == "__main__":
    main()
