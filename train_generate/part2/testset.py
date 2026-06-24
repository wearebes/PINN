from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from train_generate.geometry_core import build_grid
from train_generate.part2.generate import (
    build_fields,
    build_part2_blueprints,
    features28_for_field,
    projection_diagnostics,
    validate_field_consistency,
)
from train_generate.part2.io import (
    append_part2_rows,
    create_part2_hdf5_datasets,
    write_blueprint_inventory,
)
from train_generate.part2.config import Part2Config


def write_blueprint_inventory_csv(path: str | Path, blueprints: list[dict]) -> Path:
    path = Path(path)
    write_blueprint_inventory(path, blueprints)
    return path


def _geometry_signature(row: dict) -> tuple:
    shape_type = str(row["shape_type"])
    if shape_type == "circle":
        return (
            shape_type,
            int(row["rho"]),
            _round_float(row["r"]),
            _round_float(row["c_x"]),
            _round_float(row["c_y"]),
        )
    return (
        shape_type,
        int(row["rho"]),
        _round_float(row["a"]),
        _round_float(row["b"]),
        _round_float(row["q"]),
        _round_float(row["psi"]),
        _round_float(row["c_x"]),
        _round_float(row["c_y"]),
    )


def _round_float(value: object) -> float:
    return round(float(value), 15)


def _read_inventory_signatures(paths: Iterable[str | Path]) -> set[tuple]:
    signatures: set[tuple] = set()
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("shape_type"):
                    signatures.add(_geometry_signature(row))
    return signatures


def _select_non_overlapping_blueprints(
    blueprints: list[dict],
    *,
    used_signatures: set[tuple],
    n_circle: int,
    n_ellipse: int,
) -> list[dict]:
    selected: list[dict] = []
    for shape_type, needed in (("circle", int(n_circle)), ("ellipse", int(n_ellipse))):
        candidates = [
            item
            for item in blueprints
            if item["shape_type"] == shape_type and _geometry_signature(item) not in used_signatures
        ]
        if len(candidates) < needed:
            raise RuntimeError(f"Need {needed} non-overlapping {shape_type} blueprints, got {len(candidates)}.")
        selected.extend(candidates[:needed])
    for geometry_id, item in enumerate(selected):
        item["geometry_id"] = geometry_id
        item["split"] = "test"
    return selected


def build_part2_sdf_testset(
    *,
    rho: int,
    seed: int,
    n_circle: int,
    n_ellipse: int,
    train_blueprint_inventories: list[str | Path],
    output_path: str | Path,
    candidate_circle_eta_levels: int = 20,
    candidate_circle_phase_count: int = 10,
    candidate_ellipse_count: int = 800,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_config = Part2Config(
        rho=int(rho),
        seed=int(seed),
        output_dir=output_path.parent,
        circle_eta_levels=int(candidate_circle_eta_levels),
        circle_phase_count=int(candidate_circle_phase_count),
        ellipse_count=int(candidate_ellipse_count),
        nonsdf_enabled=False,
    )
    used_signatures = _read_inventory_signatures(train_blueprint_inventories)
    blueprints = _select_non_overlapping_blueprints(
        build_part2_blueprints(candidate_config),
        used_signatures=used_signatures,
        n_circle=int(n_circle),
        n_ellipse=int(n_ellipse),
    )
    X, Y = build_grid(int(rho))

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_family"] = "part2_sdf_testset"
        handle.attrs["part2_schema_version"] = 1
        handle.attrs["rho"] = int(rho)
        handle.attrs["h"] = float(candidate_config.h)
        handle.attrs["seed"] = int(seed)
        handle.attrs["part2_test_base_geometry_count"] = int(len(blueprints))
        handle.attrs["initial_field_types_json"] = json.dumps(["sdf"])
        handle.attrs["features28_columns"] = "[phi9_contract:0-8, nx9:9-17, ny9:18-26, hk_central:27]"
        handle.attrs["target_name"] = "hk_exact"
        handle.create_dataset("blueprints_json", data=json.dumps(blueprints, sort_keys=True).encode("utf-8"))
        datasets = create_part2_hdf5_datasets(handle)

        for blueprint in blueprints:
            fields = build_fields(blueprint, config=candidate_config, X=X, Y=Y)
            indices, _node_row = validate_field_consistency(fields, blueprint=blueprint, config=candidate_config)
            hk_exact, _projection_diag = projection_diagnostics(blueprint, indices, config=candidate_config, X=X, Y=Y)
            features28, phi9 = features28_for_field(fields["sdf"], indices, h=float(candidate_config.h))
            hk_central = features28[:, 27:28]
            n_rows = int(features28.shape[0])
            append_part2_rows(
                datasets,
                {
                    "features28": features28.astype(np.float32),
                    "features27": features28[:, :27].astype(np.float32),
                    "phi9": phi9.astype(np.float32),
                    "hk_exact": hk_exact.astype(np.float32),
                    "hk_central": hk_central.astype(np.float32),
                    "hk_residual": (hk_exact - hk_central).astype(np.float32),
                    "geometry_id": np.full(n_rows, int(blueprint["geometry_id"]), dtype=np.int32),
                    "blueprint_id": [str(blueprint["blueprint_id"])] * n_rows,
                    "split": ["test"] * n_rows,
                    "shape_type": [str(blueprint["shape_type"])] * n_rows,
                    "field_type": ["sdf"] * n_rows,
                    "d4_id": np.zeros(n_rows, dtype=np.int8),
                    "d4_name": ["e"] * n_rows,
                    "sign_flip": np.zeros(n_rows, dtype=np.int8),
                    "rho": np.full(n_rows, int(rho), dtype=np.int32),
                    "h": np.full(n_rows, float(candidate_config.h), dtype=np.float64),
                    "grid_i": indices[:, 0].astype(np.int32),
                    "grid_j": indices[:, 1].astype(np.int32),
                },
            )
        handle.attrs["row_count"] = int(datasets["features28"].shape[0])
    return output_path
