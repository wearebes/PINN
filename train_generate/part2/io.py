from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def quantile_row(group: str, values: np.ndarray, *, quantiles: tuple[float, ...]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    row: dict[str, Any] = {"group": group, "count": int(arr.size)}
    if arr.size == 0:
        for q in quantiles:
            row[f"p{q:g}"] = ""
        row["min"] = ""
        row["max"] = ""
        return row
    for q in quantiles:
        row[f"p{q:g}"] = float(np.percentile(arr, q))
    row["min"] = float(np.min(arr))
    row["max"] = float(np.max(arr))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _string_dataset(handle: h5py.File, name: str, shape: tuple[int, ...]) -> h5py.Dataset:
    return handle.create_dataset(name, shape=shape, maxshape=(None,), dtype=h5py.string_dtype("utf-8"), chunks=True)


def _numeric_dataset(handle: h5py.File, name: str, shape: tuple[int, ...], dtype: str | np.dtype) -> h5py.Dataset:
    chunks = (min(8192, max(1, int(shape[0]) if shape[0] else 8192)),) + tuple(shape[1:])
    return handle.create_dataset(name, shape=shape, maxshape=(None,) + tuple(shape[1:]), dtype=dtype, chunks=chunks)


def create_part2_hdf5_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "features28": _numeric_dataset(handle, "features28", (0, 28), np.float32),
        "features27": _numeric_dataset(handle, "features27", (0, 27), np.float32),
        "phi9": _numeric_dataset(handle, "phi9", (0, 9), np.float32),
        "hk_exact": _numeric_dataset(handle, "hk_exact", (0, 1), np.float32),
        "hk_central": _numeric_dataset(handle, "hk_central", (0, 1), np.float32),
        "hk_residual": _numeric_dataset(handle, "hk_residual", (0, 1), np.float32),
        "geometry_id": _numeric_dataset(handle, "geometry_id", (0,), np.int32),
        "blueprint_id": _string_dataset(handle, "blueprint_id", (0,)),
        "split": _string_dataset(handle, "split", (0,)),
        "shape_type": _string_dataset(handle, "shape_type", (0,)),
        "field_type": _string_dataset(handle, "field_type", (0,)),
        "d4_id": _numeric_dataset(handle, "d4_id", (0,), np.int8),
        "d4_name": _string_dataset(handle, "d4_name", (0,)),
        "sign_flip": _numeric_dataset(handle, "sign_flip", (0,), np.int8),
        "rho": _numeric_dataset(handle, "rho", (0,), np.int32),
        "h": _numeric_dataset(handle, "h", (0,), np.float64),
        "grid_i": _numeric_dataset(handle, "grid_i", (0,), np.int32),
        "grid_j": _numeric_dataset(handle, "grid_j", (0,), np.int32),
    }


def append_part2_rows(datasets: dict[str, h5py.Dataset], rows: dict[str, np.ndarray | list[str]]) -> None:
    n_rows = int(np.asarray(rows["features28"]).shape[0])
    if n_rows == 0:
        return
    start = int(datasets["features28"].shape[0])
    end = start + n_rows
    for name, dataset in datasets.items():
        dataset.resize((end,) + dataset.shape[1:])
        dataset[start:end] = rows[name]


def write_blueprint_inventory(path: Path, blueprints: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in blueprints:
        rows.append(
            {
                "geometry_id": item["geometry_id"],
                "blueprint_id": item["blueprint_id"],
                "shape_type": item["shape_type"],
                "rho": item["rho"],
                "h": item["h"],
                "split": item["split"],
                "eta": item.get("eta", ""),
                "r": item.get("r", ""),
                "c_x": item["c_x"],
                "c_y": item["c_y"],
                "a": item.get("a", ""),
                "b": item.get("b", ""),
                "q": item.get("q", ""),
                "psi": item.get("psi", ""),
                "hk_exact_min": item["hk_exact_min"],
                "hk_exact_max": item["hk_exact_max"],
            }
        )
    write_csv(path, rows)


def node_count_summary(node_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(node_rows)
    for key in ("all", "shape_type=circle", "shape_type=ellipse", "split=train", "split=val", "split=test"):
        if key == "all":
            selected = node_rows
        elif key.startswith("shape_type="):
            value = key.split("=", 1)[1]
            selected = [row for row in node_rows if row["shape_type"] == value]
        else:
            value = key.split("=", 1)[1]
            selected = [row for row in node_rows if row["split"] == value]
        counts = np.asarray([row["interface_nodes"] for row in selected], dtype=np.float64)
        near_zero = np.asarray([row["near_zero_nonsdf_points"] for row in selected], dtype=np.float64)
        summary = quantile_row(f"summary|{key}|interface_nodes", counts, quantiles=(0, 25, 50, 75, 100))
        summary["near_zero_total"] = int(np.sum(near_zero)) if near_zero.size else 0
        rows.append(summary)
    return rows


def projection_quality_summary(projection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not projection_rows:
        return []
    result: list[dict[str, Any]] = []
    for metric in (
        "projection_iterations",
        "projection_residual",
        "newton_internal_residual",
        "distance_to_interface",
        "distance_minus_coarse_global_min",
    ):
        if metric == "distance_minus_coarse_global_min":
            values = np.asarray(
                [row["distance_to_interface"] - row["coarse_global_min_distance"] for row in projection_rows],
                dtype=np.float64,
            )
        else:
            values = np.asarray([row[metric] for row in projection_rows], dtype=np.float64)
        result.append(quantile_row(metric, values, quantiles=(50, 90, 95, 99, 99.9)))
    result.append(
        {
            "group": "projection_converged",
            "count": len(projection_rows),
            "p50": int(all(bool(row["projection_converged"]) for row in projection_rows)),
            "p90": "",
            "p95": "",
            "p99": "",
            "p99.9": "",
            "min": "",
            "max": "",
        }
    )
    return result
