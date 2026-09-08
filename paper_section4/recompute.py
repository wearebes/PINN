from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import qmc

from evaluate.flower import evaluate_flower
from evaluate.shared import (
    apply_feature_transform,
    central_difference_hkappa_from_phi9,
    decode_phi9_to_patch,
    load_feature_transform,
    load_model_from_checkpoint,
    predict_hkappa_full_batch,
    resolve_feature_transform,
)
from paper_section4.core import (
    PACKAGE_DIR,
    RESOLUTIONS,
    RESULTS_DIR,
    MetricAccumulator,
    atomic_csv,
    atomic_gzip_csv,
    atomic_json,
    model_path,
    normalization_path,
    read_csv,
    sha256_file,
    write_provenance,
)
from testdata_generate.config import TestDataConfig, make_scenarios_for_rho_model
from testdata_generate.generate import generate_test_data
from train_generate.config import DataConfig
from train_generate.generate import (
    build_phi0_grid,
    build_raw_features,
    compute_hkappa_targets,
    extract_grad9,
    generate_blueprints,
    sample_blueprint,
    split_blueprint_indices,
)
from train_generate.geometry_core import (
    build_ellipse_sdf,
    build_grid as build_training_grid,
    ellipse_local_coordinates,
    interface_indices,
)


FIGURES = ("fig01_cross_resolution", "fig02_circle_r_over_h", "fig03_flower_sensitivity", "fig04_flower_error_map")
FLOWER_H5 = RESULTS_DIR / "flower_rho256_hgradient_steps0-30.h5"


def _read_partial(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    return None


def _write_partial(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_json(path, rows)


def _load_models(device: torch.device) -> dict[int, tuple[torch.nn.Module, dict[str, Any]]]:
    loaded: dict[int, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for rho in RESOLUTIONS:
        checkpoint = model_path(rho)
        sidecar = normalization_path(rho)
        model, metadata = load_model_from_checkpoint(checkpoint, device=device)
        embedded, _ = resolve_feature_transform(
            model_path=checkpoint, explicit_path=None, checkpoint_meta=metadata
        )
        explicit = load_feature_transform(sidecar)
        for key in ("mean", "std"):
            if not np.array_equal(embedded[key], explicit[key]):
                raise ValueError(f"rho={rho}: checkpoint and normalization CSV disagree for {key}")
        if int(explicit["raw_feature_dim"]) != 27 or explicit["feature_order"] != "phi9+nx9+ny9":
            raise ValueError(f"rho={rho}: expected the 27D hgradient feature contract")
        loaded[rho] = (model, explicit)
    return loaded


def _training_config(test_rho: int) -> DataConfig:
    return DataConfig(
        resolutions=(int(test_rho),),
        geometry_seed=42,
        variations=12,
        initial_field_types=("sdf", "nonsdf"),
        augment_sign_flip=True,
        augment_gradient=True,
        train_fraction=0.70,
        val_fraction=0.15,
        shape_types=("circle", "ellipse"),
        scale_h=True,
    )


def _sample_ellipse_blueprint_narrow_band(
    blueprint: dict[str, Any], config: DataConfig
) -> list[dict[str, np.ndarray]]:
    """Evaluate the exact generator contract without full-grid HP SDF work.

    The ellipse SDF and implicit field have exactly the same zero-set/sign
    pattern.  Interface selection can therefore use the cheap implicit field,
    while the high-precision SDF is evaluated only at the 5x5 neighbourhood
    read by phi9 and the centred-gradient features.  Values returned at every
    accessed node are computed by the same geometry routine as
    ``sample_blueprint``.
    """

    rho = int(blueprint["meta"]["resolution"])
    X, Y = build_training_grid(rho)
    h = float(blueprint["params"]["h"])
    nonsdf = build_phi0_grid(blueprint, "nonsdf", data_config=config, X=X, Y=Y)
    indices = interface_indices(nonsdf)
    if indices.size == 0:
        raise RuntimeError(f"No interface nodes for {blueprint['meta']['blueprint_id']}")

    base_hkappa = compute_hkappa_targets(
        blueprint, indices, data_config=config, X=X, Y=Y
    )

    sparse_sdf = np.asarray(nonsdf, dtype=np.float64).copy()
    offsets = np.asarray(
        [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)], dtype=np.int64
    )
    required = np.unique(
        (indices[:, None, :] + offsets[None, :, :]).reshape(-1, 2), axis=0
    )
    rows, cols = required[:, 0], required[:, 1]
    center = blueprint["params"]["center"]
    u, v = ellipse_local_coordinates(
        X[rows, cols],
        Y[rows, cols],
        cx=float(center[0]),
        cy=float(center[1]),
        psi=float(blueprint["params"]["psi"]),
    )
    sparse_sdf[rows, cols] = build_ellipse_sdf(
        u,
        v,
        a=float(blueprint["params"]["a"]),
        b=float(blueprint["params"]["b"]),
        max_iter=int(config.ellipse_sdf_newton_max_iter),
        tol=float(config.ellipse_sdf_newton_tol),
        dps=int(config.ellipse_hp_dps),
        hp_max_iter=int(config.ellipse_hp_newton_max_iter),
    )
    sparse_sdf = sparse_sdf.astype(np.float32, copy=False)

    samples: list[dict[str, np.ndarray]] = []
    for phi in (sparse_sdf, nonsdf):
        phi9, features = build_raw_features(phi, indices, scale_h=True, h=h)
        grad9 = extract_grad9(phi, indices)
        features = np.concatenate(
            [features, grad9[:, :, 0], grad9[:, :, 1]], axis=1
        ).astype(np.float32, copy=False)
        samples.append(
            {"phi9": phi9, "features": features, "hkappa_target": base_hkappa}
        )
        samples.append(
            {"phi9": -phi9, "features": -features, "hkappa_target": -base_hkappa}
        )
    return samples


def _sample_blueprint_for_section4(
    blueprint: dict[str, Any], config: DataConfig
) -> list[dict[str, np.ndarray]]:
    if str(blueprint["meta"]["shape_type"]) == "ellipse":
        return _sample_ellipse_blueprint_narrow_band(blueprint, config)
    return sample_blueprint(blueprint, data_config=config)


def _sample_worker(payload: tuple[dict[str, Any], DataConfig]):
    blueprint, config = payload
    return _sample_blueprint_for_section4(blueprint, config)


def _evaluate_test_resolution(
    test_rho: int,
    models: dict[int, tuple[torch.nn.Module, dict[str, Any]]],
    *,
    device: torch.device,
    workers: int,
) -> list[dict[str, Any]]:
    config = _training_config(test_rho)
    blueprints = generate_blueprints(config)
    split = split_blueprint_indices(
        blueprints,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.geometry_seed,
    )
    test_blueprints = [blueprints[index] for index in split["test"]]
    nn_metrics = {rho: MetricAccumulator() for rho in RESOLUTIONS}
    fd_metric = MetricAccumulator()

    # Geometry generation naturally arrives one blueprint at a time, but five
    # tiny model forwards per blueprint are needlessly expensive on CPU.  Keep
    # generation streaming while coalescing rows into bounded inference batches.
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    buffered_rows = 0
    inference_batch_rows = 262_144

    def flush_inference_batch() -> None:
        nonlocal buffered_rows
        if not feature_parts:
            return
        features = np.concatenate(feature_parts, axis=0)
        target = np.concatenate(target_parts, axis=0)
        for train_rho, (model, transform) in models.items():
            prediction = predict_hkappa_full_batch(
                model, features, transform=transform, device=device
            )
            nn_metrics[train_rho].update(prediction, target)
        feature_parts.clear()
        target_parts.clear()
        buffered_rows = 0

    payloads = ((blueprint, config) for blueprint in test_blueprints)
    if workers <= 1:
        generated = map(_sample_worker, payloads)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        generated = pool.map(_sample_worker, payloads, chunksize=4)
    try:
        for blueprint_index, blueprint_samples in enumerate(generated, start=1):
            features = np.concatenate([sample["features"] for sample in blueprint_samples], axis=0)
            phi9 = np.concatenate([sample["phi9"] for sample in blueprint_samples], axis=0)
            target = np.concatenate([sample["hkappa_target"] for sample in blueprint_samples], axis=0).reshape(-1)
            fd_metric.update(central_difference_hkappa_from_phi9(phi9), target)
            feature_parts.append(features)
            target_parts.append(target)
            buffered_rows += target.size
            if buffered_rows >= inference_batch_rows:
                flush_inference_batch()
            if blueprint_index % 100 == 0 or blueprint_index == len(test_blueprints):
                print(
                    f"[fig01] rho={test_rho}: {blueprint_index}/{len(test_blueprints)} blueprints",
                    flush=True,
                )
        flush_inference_batch()
    finally:
        if pool is not None:
            pool.shutdown()

    rows: list[dict[str, Any]] = []
    for train_rho in RESOLUTIONS:
        rows.append(
            {
                "method": "NN",
                "train_rho": train_rho,
                "test_rho": test_rho,
                **nn_metrics[train_rho].row(),
            }
        )
    rows.append(
        {
            "method": "FD",
            "train_rho": "",
            "test_rho": test_rho,
            **fd_metric.row(),
        }
    )
    return rows


def recompute_fig01(
    models: dict[int, tuple[torch.nn.Module, dict[str, Any]]],
    *,
    device: torch.device,
    workers: int,
    resume: bool,
) -> Path:
    partial_dir = RESULTS_DIR / ".partial" / "fig01"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for test_rho in RESOLUTIONS:
        partial = partial_dir / f"rho{test_rho}.json"
        reused = _read_partial(partial) if resume else None
        if reused is not None:
            rows.extend(reused)
            print(f"[fig01] reuse test rho={test_rho}")
            continue
        print(f"[fig01] evaluate test rho={test_rho}")
        current = _evaluate_test_resolution(
            test_rho, models, device=device, workers=workers
        )
        _write_partial(partial, current)
        rows.extend(current)
    rows.sort(key=lambda row: (int(row["test_rho"]), row["method"] == "FD", int(row["train_rho"] or 0)))
    output = atomic_csv(
        RESULTS_DIR / "fig01_cross_resolution.source.csv",
        rows,
        ["method", "train_rho", "test_rho", "sample_count", "mse", "mae", "maxae"],
    )
    write_provenance(
        "fig01_cross_resolution",
        output,
        protocol={
            "test_split": "geometry-level 70/15/15 split, seed 42",
            "feature_order": "phi9+nx9+ny9",
            "scale_h": True,
            "nn_pairs": 25,
            "fd_rows": 5,
            "streaming": True,
            "ellipse_sdf_evaluation": "exact high-precision 5x5 interface band; array-equivalent to full-grid generator at all accessed nodes",
        },
    )
    return output


def _circle_interface_nodes(q: float, offset: np.ndarray) -> np.ndarray:
    cx, cy = (float(offset[0]), float(offset[1]))
    nodes: set[tuple[int, int]] = set()
    for j in range(int(np.ceil(cy - q)), int(np.floor(cy + q)) + 1):
        remaining = q * q - (j - cy) ** 2
        if remaining < 0.0:
            continue
        root = np.sqrt(max(remaining, 0.0))
        for x_crossing in (cx - root, cx + root):
            lo = int(np.floor(x_crossing))
            nodes.add((lo, j))
            nodes.add((lo + 1, j))
    for i in range(int(np.ceil(cx - q)), int(np.floor(cx + q)) + 1):
        remaining = q * q - (i - cx) ** 2
        if remaining < 0.0:
            continue
        root = np.sqrt(max(remaining, 0.0))
        for y_crossing in (cy - root, cy + root):
            lo = int(np.floor(y_crossing))
            nodes.add((i, lo))
            nodes.add((i, lo + 1))
    return np.asarray(sorted(nodes), dtype=np.int64)


def _circle_features(q: float, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = _circle_interface_nodes(q, offset)
    local = np.empty((len(nodes), 5, 5), dtype=np.float64)
    for a, dr in enumerate(range(-2, 3)):
        for b, dc in enumerate(range(-2, 3)):
            local[:, a, b] = np.sqrt(
                (nodes[:, 0] + dr - float(offset[0])) ** 2
                + (nodes[:, 1] + dc - float(offset[1])) ** 2
            ) - float(q)
    offsets = np.asarray(
        [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)],
        dtype=np.int64,
    )
    phi9 = np.column_stack([local[:, 2 + dr, 2 + dc] for dr, dc in offsets]).astype(np.float32)
    nx = np.empty_like(phi9)
    ny = np.empty_like(phi9)
    for k, (dr, dc) in enumerate(offsets):
        dx = local[:, 2 + dr + 1, 2 + dc] - local[:, 2 + dr - 1, 2 + dc]
        dy = local[:, 2 + dr, 2 + dc + 1] - local[:, 2 + dr, 2 + dc - 1]
        magnitude = np.sqrt(dx * dx + dy * dy)
        nx[:, k] = dx / magnitude
        ny[:, k] = dy / magnitude
    features = np.concatenate([phi9, nx, ny], axis=1).astype(np.float32)
    return phi9, features


def recompute_fig02(
    models: dict[int, tuple[torch.nn.Module, dict[str, Any]]],
    *,
    device: torch.device,
    resume: bool,
) -> Path:
    q_values = np.unique(
        np.r_[np.geomspace(1.6, 253.5, 33), [13.5, 29.5, 61.5, 125.5, 253.5]]
    )
    if len(q_values) != 37:
        raise AssertionError(f"Expected 37 R/h values, got {len(q_values)}")
    offsets = qmc.Sobol(d=2, scramble=True, seed=42).random_base2(8)
    partial_dir = RESULTS_DIR / ".partial" / "fig02"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, q_value in enumerate(q_values):
        partial = partial_dir / f"q{index:02d}.json"
        reused = _read_partial(partial) if resume else None
        if reused is not None:
            rows.extend(reused)
            continue
        print(f"[fig02] R/h={q_value:.8g} ({index + 1}/37)")
        fd_metric = MetricAccumulator()
        nn_metrics = {rho: MetricAccumulator() for rho in RESOLUTIONS}
        for offset in offsets:
            phi9, features = _circle_features(float(q_value), offset)
            target = np.full(len(phi9), 1.0 / float(q_value), dtype=np.float64)
            fd_metric.update(central_difference_hkappa_from_phi9(phi9), target)
            for rho, (model, transform) in models.items():
                prediction = predict_hkappa_full_batch(
                    model, features, transform=transform, device=device
                )
                nn_metrics[rho].update(prediction, target)
        current = [
            {
                "method": "NN",
                "train_rho": rho,
                "r_over_h": float(q_value),
                "sobol_positions": 256,
                **nn_metrics[rho].row(),
            }
            for rho in RESOLUTIONS
        ]
        current.append(
            {
                "method": "FD",
                "train_rho": "",
                "r_over_h": float(q_value),
                "sobol_positions": 256,
                **fd_metric.row(),
            }
        )
        _write_partial(partial, current)
        rows.extend(current)
    rows.sort(key=lambda row: (float(row["r_over_h"]), row["method"] == "FD", int(row["train_rho"] or 0)))
    output = atomic_csv(
        RESULTS_DIR / "fig02_circle_r_over_h.source.csv",
        rows,
        ["method", "train_rho", "r_over_h", "sobol_positions", "sample_count", "mse", "mae", "maxae"],
    )
    write_provenance(
        "fig02_circle_r_over_h",
        output,
        protocol={
            "geometry": "exact circle SDF with R=1 and h=1/(R/h)",
            "r_over_h": [float(value) for value in q_values],
            "sobol": {"scrambled": True, "seed": 42, "positions_per_ratio": 256},
            "feature_order": "phi9+nx9+ny9",
        },
    )
    return output


def _flower_config() -> TestDataConfig:
    return TestDataConfig(
        rho_model=256,
        requested_rho_model=256,
        test_iters=tuple(range(31)),
        scale_h=True,
        augment_gradient=True,
    )


def _flower_dataset_matches(path: Path) -> bool:
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        return (
            handle["features"].shape == (35681, 27)
            and bool(handle.attrs.get("scale_h", False))
            and str(handle.attrs.get("feature_order", "")) == "phi9+nx9+ny9"
            and sorted(set(int(value) for value in handle["iter"][:])) == list(range(31))
        )


def ensure_flower_dataset(*, resume: bool) -> Path:
    if resume and _flower_dataset_matches(FLOWER_H5):
        print(f"[flower] reuse {FLOWER_H5}")
        return FLOWER_H5
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return generate_test_data(_flower_config(), output=FLOWER_H5)


def _sdf_summary_rows(
    *,
    family: str,
    step: int,
    selection: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"No SDF defect samples for {family} step {step} selection={selection}")
    common = {
        "family": family,
        "step": int(step),
        "method": "SDF",
        "selection": selection,
        "sample_count": int(values.size),
    }
    return [
        {**common, "metric": "sdf_mean", "value": float(np.mean(values))},
        {**common, "metric": "sdf_max", "value": float(np.max(values))},
    ]


def _interface_sdf_defect_rows() -> list[dict[str, Any]]:
    """Reproduce the historical flower diagnostic on current-interface nodes.

    The flower HDF5 contains one raw 3x3 ``phi9`` stencil per selected
    interface-adjacent node.  The mean/maximum summaries follow the historical
    flower SDF-error diagnostic, while the population is restricted to the
    same current interface nodes used by the curvature evaluation.
    """
    with h5py.File(FLOWER_H5, "r") as handle:
        phi9 = np.asarray(handle["phi9"][:], dtype=np.float64)
        h_values = np.asarray(handle["h"][:], dtype=np.float64).reshape(-1)
        case_ids = np.asarray(handle["case_id"][:]).reshape(-1)
        steps = np.asarray(handle["iter"][:]).reshape(-1)

    patch = decode_phi9_to_patch(phi9).astype(np.float64, copy=False)
    dphi_dy = 0.5 * (patch[:, 2, 1] - patch[:, 0, 1]) / h_values
    dphi_dx = 0.5 * (patch[:, 1, 2] - patch[:, 1, 0]) / h_values
    defect = np.abs(np.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy) - 1.0)

    rows: list[dict[str, Any]] = []
    scenarios = make_scenarios_for_rho_model(256)
    for case_id, scenario in enumerate(scenarios):
        family = str(scenario.experiment_type)
        for step in range(31):
            mask = (case_ids == case_id) & (steps == step)
            rows.extend(
                _sdf_summary_rows(
                    family=family,
                    step=step,
                    selection="interface_nodes",
                    values=defect[mask],
                )
            )
    return rows


def _evaluate_flower(device: torch.device) -> dict[str, Any]:
    return evaluate_flower(
        dataset_path=FLOWER_H5,
        model_path=model_path(256),
        normalization_csv_path=normalization_path(256),
        device=device,
        angle_bin_deg=30.0,
    )


def recompute_flower(*, device: torch.device, resume: bool) -> tuple[Path, Path]:
    ensure_flower_dataset(resume=resume)
    result = _evaluate_flower(device)
    if result["failed_case_count"]:
        raise ValueError(f"Flower evaluation has {result['failed_case_count']} failed slices")

    fig03_rows: list[dict[str, Any]] = []
    for case in result["cases"]:
        family = str(case["case_label"]).rsplit("_", 1)[0]
        step = int(case["iter"])
        count = int(case["sample_count"])
        fig03_rows.extend(
            [
                {"family": family, "step": step, "metric": "hkappa_mse", "method": "NN", "selection": "interface_nodes", "value": float(case["model_vs_analytic"]["mse"]), "sample_count": count},
                {"family": family, "step": step, "metric": "hkappa_mse", "method": "FD", "selection": "interface_nodes", "value": float(case["numeric_vs_analytic"]["mse"]), "sample_count": count},
            ]
        )
    fig03_rows.extend(_interface_sdf_defect_rows())
    fig03_rows.sort(key=lambda row: (row["family"], int(row["step"]), row["metric"], row["selection"], row["method"]))
    fig03 = atomic_csv(
        RESULTS_DIR / "fig03_flower_sensitivity.source.csv",
        fig03_rows,
        ["family", "step", "metric", "method", "selection", "value", "sample_count"],
    )

    selected_steps = {0, 10, 20, 30}
    fig04_rows: list[dict[str, Any]] = []
    for case in result["cases"]:
        step = int(case["iter"])
        if step not in selected_steps:
            continue
        family = str(case["case_label"]).rsplit("_", 1)[0]
        for theta, xy, pred, truth, fd in zip(
            case["theta"], case["xy"], case["pred_hkappa"], case["true_hkappa"], case["numeric_hkappa"]
        ):
            fig04_rows.append(
                {
                    "family": family,
                    "step": step,
                    "theta_rad": float(theta),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "nn_pred_hkappa": float(pred),
                    "analytic_hkappa": float(truth),
                    "fd_hkappa": float(fd),
                    "nn_abs_error": abs(float(pred) - float(truth)),
                    "fd_abs_error": abs(float(fd) - float(truth)),
                }
            )
    fig04_rows.sort(key=lambda row: (row["family"], int(row["step"]), float(row["theta_rad"])))
    fig04 = atomic_gzip_csv(
        RESULTS_DIR / "fig04_flower_error_map.source.csv.gz",
        fig04_rows,
        ["family", "step", "theta_rad", "x", "y", "nn_pred_hkappa", "analytic_hkappa", "fd_hkappa", "nn_abs_error", "fd_abs_error"],
    )
    common = {
        "shared_hdf5": FLOWER_H5.name,
        "shared_hdf5_sha256": sha256_file(FLOWER_H5),
        "rho_model": 256,
        "steps": list(range(31)),
        "feature_order": "phi9+nx9+ny9",
        "scale_h": True,
        "sign_mode": "dynamic_phi",
        "reinitialization": {"space": "WENO5", "time": "RK3", "cfl": 0.5, "eps_weno": 1e-6, "eps_sign_factor": 2.5},
    }
    write_provenance(
        "fig03_flower_sensitivity",
        fig03,
        protocol={
            **common,
            "curvature_sampling": "current interface-adjacent nodes",
            "sdf_sampling": "the same current interface-adjacent nodes stored in the flower HDF5",
            "sdf_statistics": ["mean", "max"],
            "sdf_gradient": "second-order centered difference",
        },
        extra_inputs=[FLOWER_H5],
    )
    write_provenance(
        "fig04_flower_error_map",
        fig04,
        protocol={**common, "selected_steps": [0, 10, 20, 30], "spatial_render": "interface nodes only, no full-field interpolation"},
        extra_inputs=[FLOWER_H5],
    )
    return fig03, fig04


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute reproducible Section 4 source data")
    parser.add_argument("--figure", choices=("all", "fig01", "fig02", "fig03", "fig04"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models = _load_models(device)
    if args.figure in {"all", "fig01"}:
        recompute_fig01(models, device=device, workers=max(1, args.workers), resume=args.resume)
    if args.figure in {"all", "fig02"}:
        recompute_fig02(models, device=device, resume=args.resume)
    if args.figure in {"all", "fig03", "fig04"}:
        recompute_flower(device=device, resume=args.resume)
    print(f"Section 4 source data ready: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
