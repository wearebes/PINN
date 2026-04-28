from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from model.model import EPS, ReinitPINN, compute_model_curvature, count_parameters
from testdata_generate.generate import interface_indices
from traingenerate.generate import load_training_bundle_from_hdf5
from traingenerate.reinit import LevelSetReinitializer

from .config import EvalConfig


REQUIRED_DATASETS = ("phi9", "xy", "iter", "rho_model", "h", "case_id", "hkappa_target")


class DisabledSwanLabRun:
    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0

    def update(self, target: np.ndarray, pred: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        error = pred - target
        abs_error = np.abs(error)
        self.count += int(target.shape[0])
        self.sum_abs += float(abs_error.sum())
        self.sum_sq += float((error**2).sum())
        self.max_abs = max(self.max_abs, float(abs_error.max()) if abs_error.size else 0.0)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"N_samples": 0, "MAE_hk": 0.0, "MSE_hk": 0.0, "MaxAE_hk": 0.0}
        return {
            "N_samples": self.count,
            "MAE_hk": self.sum_abs / self.count,
            "MSE_hk": self.sum_sq / self.count,
            "MaxAE_hk": self.max_abs,
        }


def decode_phi9(phi9: np.ndarray) -> np.ndarray:
    arr = np.asarray(phi9, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"phi9 must have shape (N, 9), got {arr.shape}.")
    return arr.reshape(-1, 3, 3).transpose(0, 2, 1)[:, :, ::-1]


def compute_numeric_hkappa(phi9: np.ndarray, h: np.ndarray) -> np.ndarray:
    patch = decode_phi9(phi9)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if patch.shape[0] != h.shape[0]:
        raise ValueError(f"phi9 length {patch.shape[0]} does not match h length {h.shape[0]}.")

    phi_x = (patch[:, 1, 2] - patch[:, 1, 0]) / (2.0 * h)
    phi_y = (patch[:, 2, 1] - patch[:, 0, 1]) / (2.0 * h)
    phi_xx = (patch[:, 1, 2] - 2.0 * patch[:, 1, 1] + patch[:, 1, 0]) / (h**2)
    phi_yy = (patch[:, 2, 1] - 2.0 * patch[:, 1, 1] + patch[:, 0, 1]) / (h**2)
    phi_xy = (patch[:, 2, 2] - patch[:, 2, 0] - patch[:, 0, 2] + patch[:, 0, 0]) / (4.0 * h**2)
    grad_sq = phi_x**2 + phi_y**2 + EPS
    kappa = (phi_xx * phi_y**2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / (
        grad_sq * np.sqrt(grad_sq)
    )
    return (h * kappa).astype(np.float32, copy=False)


def _csv_to_tuple(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _json_attr(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _torch_flower_phi0(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    theta = torch.atan2(y, x)
    radius = torch.sqrt(x.square() + y.square())
    return radius - a * torch.cos(p * theta) - b


def compute_autograd_hkappa(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    h: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    phi0 = _torch_flower_phi0(x, y, a, b, p)
    phi = model(x, y, phi0, s)

    phi_x, phi_y = torch.autograd.grad(
        phi,
        (x, y),
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )
    phi_xx = torch.autograd.grad(
        phi_x,
        x,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=False,
        retain_graph=True,
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y,
        y,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=False,
        retain_graph=True,
    )[0]
    phi_xy = torch.autograd.grad(
        phi_x,
        y,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=False,
        retain_graph=False,
    )[0]
    grad_sq = phi_x.square() + phi_y.square() + EPS
    kappa = (phi_xx * phi_y.square() - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x.square()) / (
        grad_sq * torch.sqrt(grad_sq)
    )
    return h * kappa


def load_model(config: EvalConfig, device: torch.device) -> ReinitPINN:
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path.resolve()}")
    model = ReinitPINN(hidden_units=config.hidden_units, activation=config.activation)
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a PyTorch state dict in {model_path}, got {type(checkpoint)!r}.")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def inspect_test_data(test_data_path: Path) -> dict[str, Any]:
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path.resolve()}")
    with h5py.File(test_data_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise KeyError(f"Missing required HDF5 datasets in {test_data_path}: {missing}")
        scenarios = _json_attr(handle.attrs.get("scenarios_json"), [])
        test_iters = _json_attr(handle.attrs.get("test_iters_json"), [])
        dataset_shapes = {
            name: {
                "shape": tuple(int(item) for item in dataset.shape),
                "dtype": str(dataset.dtype),
            }
            for name, dataset in handle.items()
        }
        attrs = {
            "schema_version": int(handle.attrs.get("schema_version", 0)),
            "method_code": str(handle.attrs.get("method_code", "")),
            "mode": str(handle.attrs.get("mode", "")),
            "sign_mode": str(handle.attrs.get("sign_mode", "")),
            "cfl": float(handle.attrs.get("cfl", 0.0)),
            "eps_sign_factor": float(handle.attrs.get("eps_sign_factor", 0.0)),
            "time_order": int(handle.attrs.get("time_order", 0)),
            "space_order": int(handle.attrs.get("space_order", 0)),
            "sampling_rule": str(handle.attrs.get("sampling_rule", "")),
            "stencil_encoding": str(handle.attrs.get("stencil_encoding", "")),
            "target_rule": str(handle.attrs.get("target_rule", "")),
            "test_iters": [int(item) for item in test_iters],
            "scenarios": scenarios,
            "dataset_shapes": dataset_shapes,
            "sample_count": int(handle["hkappa_target"].shape[0]),
        }
    return attrs


def inspect_circle_test_data(config: EvalConfig) -> dict[str, Any]:
    dataset_path = Path(config.circle_dataset_path)
    bundle = load_training_bundle_from_hdf5(dataset_path, batch_size=config.batch_size)
    split_indices = bundle.get("split_blueprint_indices", {}).get("test", [])
    blueprints = bundle.get("blueprints", [])
    test_blueprints = [blueprints[int(idx)] for idx in split_indices]
    data_cfg = bundle["config"]
    reinit_cfg = bundle["reinit_config"]
    resolutions = sorted({int(item["meta"]["resolution"]) for item in test_blueprints})
    test_iters = list(range(1, int(max(data_cfg.reinit_steps)) + 1))
    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_name": dataset_path.name,
        "sample_geometry": "circle",
        "split_name": "test",
        "sampling_mode": str(config.circle_sampling_mode),
        "band_width_cells": float(config.circle_band_width_cells),
        "field_count": int(len(test_blueprints)),
        "resolutions": resolutions,
        "final_step": int(max(data_cfg.reinit_steps)),
        "test_iters": test_iters,
        "cfl": float(reinit_cfg.cfl),
        "time_order": int(reinit_cfg.time_order),
        "space_order": int(reinit_cfg.space_order),
        "sign_mode": str(reinit_cfg.sign_mode),
    }


def build_swanlab_config(
    *,
    config: EvalConfig,
    metadata: dict[str, Any],
    device: torch.device,
    parameter_count: int,
) -> dict[str, Any]:
    if config.eval_geometry == "circle":
        swan_config: dict[str, Any] = {
            "eval/geometry": "circle",
            "circle_data/path": metadata["dataset_path"],
            "circle_data/name": metadata["dataset_name"],
            "circle_data/split_name": metadata["split_name"],
            "circle_data/sampling_mode": metadata["sampling_mode"],
            "circle_data/band_width_cells": metadata["band_width_cells"],
            "circle_data/field_count": metadata["field_count"],
            "circle_data/resolutions": metadata["resolutions"],
            "circle_data/final_step": metadata["final_step"],
            "circle_data/test_iters": metadata["test_iters"],
            "circle_data/cfl": metadata["cfl"],
            "circle_data/sign_mode": metadata["sign_mode"],
            "circle_data/time_order": metadata["time_order"],
            "circle_data/space_order": metadata["space_order"],
            "model/path": str(Path(config.model_path).resolve()),
            "model/name": Path(config.model_path).name,
            "model/hidden_units": int(config.hidden_units),
            "model/activation": config.activation,
            "model/parameter_count": int(parameter_count),
            "eval/batch_size": int(config.batch_size),
            "eval/device": str(device),
            "eval/curvature_quantity": "h*kappa",
            "eval/target_geometry": "circle",
            "eval/swanlab_step": "final_step",
        }
        if config.max_samples is not None:
            swan_config["eval/max_samples"] = int(config.max_samples)
        return swan_config

    swan_config: dict[str, Any] = {
        "eval/geometry": "flower",
        "test_data/path": str(Path(config.test_data_path).resolve()),
        "test_data/name": Path(config.test_data_path).name,
        "test_data/schema_version": metadata["schema_version"],
        "test_data/method_code": metadata["method_code"],
        "test_data/mode": metadata["mode"],
        "test_data/sign_mode": metadata["sign_mode"],
        "test_data/cfl": metadata["cfl"],
        "test_data/eps_sign_factor": metadata["eps_sign_factor"],
        "test_data/time_order": metadata["time_order"],
        "test_data/space_order": metadata["space_order"],
        "test_data/sampling_rule": metadata["sampling_rule"],
        "test_data/stencil_encoding": metadata["stencil_encoding"],
        "test_data/target_rule": metadata["target_rule"],
        "test_data/test_iters": metadata["test_iters"],
        "test_data/sample_count": metadata["sample_count"],
        "model/path": str(Path(config.model_path).resolve()),
        "model/name": Path(config.model_path).name,
        "model/hidden_units": int(config.hidden_units),
        "model/activation": config.activation,
        "model/parameter_count": int(parameter_count),
        "eval/batch_size": int(config.batch_size),
        "eval/device": str(device),
        "eval/curvature_quantity": "h*kappa",
        "eval/model_s_input": "iter*cfl",
        "eval/numeric_curvature_source": "phi9_central_difference",
        "eval/swanlab_step": "iter",
    }
    if config.max_samples is not None:
        swan_config["eval/max_samples"] = int(config.max_samples)
    for name, spec in metadata["dataset_shapes"].items():
        swan_config[f"test_data/datasets/{name}/shape"] = spec["shape"]
        swan_config[f"test_data/datasets/{name}/dtype"] = spec["dtype"]
    for scenario in metadata["scenarios"]:
        exp_id = str(scenario["exp_id"])
        for key in ("experiment_type", "rho_model", "N", "h", "a", "b", "p"):
            swan_config[f"test_data/scenario/{exp_id}/{key}"] = scenario[key]
    return swan_config


def init_swanlab_run(
    *,
    config: EvalConfig,
    metadata: dict[str, Any],
    device: torch.device,
    parameter_count: int,
):
    if config.swanlab_mode == "disabled":
        return DisabledSwanLabRun()

    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging was requested, but the `swanlab` package is not installed in the current environment."
        ) from exc

    logdir = config.swanlab_logdir or "swanlog"
    cache_dir = Path(logdir) / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))

    init_kwargs: dict[str, Any] = {
        "project": config.swanlab_project,
        "experiment_name": config.swanlab_experiment_name,
        "description": config.swanlab_description,
        "tags": list(config.swanlab_tags),
        "group": config.swanlab_group or None,
        "workspace": config.swanlab_workspace or None,
        "logdir": logdir,
        "mode": config.swanlab_mode,
        "config": build_swanlab_config(
            config=config,
            metadata=metadata,
            device=device,
            parameter_count=parameter_count,
        ),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return swanlab.init(**init_kwargs)


def _numeric_hkappa_from_circle_field(phi: np.ndarray, h: float, mask: np.ndarray) -> np.ndarray:
    i, j = np.where(mask)
    phi_x = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * h)
    phi_y = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * h)
    phi_xx = (phi[i + 1, j] - 2.0 * phi[i, j] + phi[i - 1, j]) / (h**2)
    phi_yy = (phi[i, j + 1] - 2.0 * phi[i, j] + phi[i, j - 1]) / (h**2)
    phi_xy = (phi[i + 1, j + 1] - phi[i + 1, j - 1] - phi[i - 1, j + 1] + phi[i - 1, j - 1]) / (4.0 * h**2)
    grad_sq = phi_x**2 + phi_y**2 + EPS
    kappa = (phi_xx * phi_y**2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / (
        grad_sq * np.sqrt(grad_sq)
    )
    return (h * kappa).astype(np.float32, copy=False)


def _circle_sampling_mask(
    *,
    sampling_mode: str,
    sdf_true: np.ndarray,
    phi_current: np.ndarray,
    h: float,
    band_width_cells: float,
) -> np.ndarray:
    if sampling_mode == "moving_interface":
        indices = interface_indices(phi_current)
        if indices.size == 0:
            return np.zeros_like(phi_current, dtype=bool)
        mask = np.zeros_like(phi_current, dtype=bool)
        mask[indices[:, 0], indices[:, 1]] = True
        return mask

    if sampling_mode == "true_interface":
        indices = interface_indices(sdf_true)
        if indices.size == 0:
            return np.zeros_like(phi_current, dtype=bool)
        mask = np.zeros_like(phi_current, dtype=bool)
        mask[indices[:, 0], indices[:, 1]] = True
        return mask

    if sampling_mode == "reference_band":
        mask = np.abs(sdf_true) <= (float(band_width_cells) * float(h))
        mask = np.asarray(mask, dtype=bool)
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        return mask

    raise ValueError(
        f"Unsupported circle_sampling_mode={sampling_mode!r}; expected one of: "
        "moving_interface, true_interface, reference_band."
    )


def evaluate_circle_curvature(
    config: EvalConfig,
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    device: torch.device,
    swanlab_run: Any | None = None,
) -> dict[str, Any]:
    bundle = load_training_bundle_from_hdf5(Path(config.circle_dataset_path), batch_size=config.batch_size)
    blueprints = bundle.get("blueprints", [])
    split_indices = bundle.get("split_blueprint_indices", {}).get("test", [])
    selected_blueprints = [blueprints[int(idx)] for idx in split_indices]
    if config.max_samples is not None:
        selected_blueprints = selected_blueprints[: int(config.max_samples)]

    data_cfg = bundle["config"]
    reinit_cfg = bundle["reinit_config"]
    test_iters = list(range(1, int(max(data_cfg.reinit_steps)) + 1))
    sampling_mode = str(config.circle_sampling_mode)
    band_width_cells = float(config.circle_band_width_cells)
    reinitializer = LevelSetReinitializer(
        indexing="ij",
        cfl=reinit_cfg.cfl,
        eps_weno=reinit_cfg.eps_weno,
        eps_sign_factor=reinit_cfg.eps_sign_factor,
        sign_mode=reinit_cfg.sign_mode,
        time_order=reinit_cfg.time_order,
        space_order=reinit_cfg.space_order,
    )

    metric_groups = {
        "numeric": {
            "by_rho_iter": defaultdict(MetricAccumulator),
            "by_case_iter": defaultdict(MetricAccumulator),
            "overall": MetricAccumulator(),
        },
        "model": {
            "by_rho_iter": defaultdict(MetricAccumulator),
            "by_case_iter": defaultdict(MetricAccumulator),
            "overall": MetricAccumulator(),
        },
    }

    model.eval()
    for field_idx, blueprint in enumerate(selected_blueprints, start=1):
        rho = int(blueprint["meta"]["resolution"])
        cx = float(blueprint["params"]["center"][0])
        cy = float(blueprint["params"]["center"][1])
        radius = float(blueprint["params"]["radius"])
        h = float(blueprint["params"]["h"])
        case_id = str(blueprint["meta"]["blueprint_id"])

        x = np.linspace(0.0, 1.0, rho, dtype=np.float32)
        X, Y = np.meshgrid(x, x, indexing="ij")
        sdf = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius
        if data_cfg.initial_field_type == "nonsdf":
            phi0 = (X - cx) ** 2 + (Y - cy) ** 2 - radius ** 2
        else:
            phi0 = sdf.copy()
        phi_num = phi0.astype(np.float64, copy=True)
        batch_size = int(config.batch_size)
        reference_mask = None
        if sampling_mode in {"true_interface", "reference_band"}:
            reference_mask = _circle_sampling_mask(
                sampling_mode=sampling_mode,
                sdf_true=sdf,
                phi_current=phi_num,
                h=h,
                band_width_cells=band_width_cells,
            )
            if not np.any(reference_mask):
                raise RuntimeError(f"No fixed reference nodes for circle case={case_id} with mode={sampling_mode}.")

        for iteration in test_iters:
            phi_num = reinitializer.reinitialize(phi_num, h, 1)
            if reference_mask is None:
                mask = _circle_sampling_mask(
                    sampling_mode=sampling_mode,
                    sdf_true=sdf,
                    phi_current=phi_num,
                    h=h,
                    band_width_cells=band_width_cells,
                )
            else:
                mask = reference_mask
            rows, cols = np.where(mask)
            if rows.size == 0:
                raise RuntimeError(f"No circle samples for case={case_id}, iter={iteration}, mode={sampling_mode}.")
            target_np = np.full(rows.shape[0], h / max(radius, EPS), dtype=np.float32)
            numeric_pred_np = _numeric_hkappa_from_circle_field(phi_num, h, mask)

            x_nodes = X[rows, cols].reshape(-1, 1).astype(np.float32)
            y_nodes = Y[rows, cols].reshape(-1, 1).astype(np.float32)
            s_nodes = np.full_like(x_nodes, float(iteration) * float(reinit_cfg.cfl), dtype=np.float32)
            model_preds: list[np.ndarray] = []
            for start in range(0, x_nodes.shape[0], batch_size):
                end = min(start + batch_size, x_nodes.shape[0])
                xt = torch.from_numpy(x_nodes[start:end]).to(device).requires_grad_(True)
                yt = torch.from_numpy(y_nodes[start:end]).to(device).requires_grad_(True)
                st = torch.from_numpy(s_nodes[start:end]).to(device).requires_grad_(True)
                cxt = torch.full_like(xt, cx)
                cyt = torch.full_like(yt, cy)
                rt = torch.full_like(xt, radius)
                ht = torch.full_like(xt, h)
                model_pred = compute_model_curvature(model, xt, yt, cxt, cyt, rt, st)
                model_preds.append((ht * model_pred).detach().cpu().numpy().reshape(-1))
            model_pred_np = np.concatenate(model_preds, axis=0)

            predictions = {
                "numeric": numeric_pred_np,
                "model": model_pred_np,
            }
            for metric_name, pred_np in predictions.items():
                metric_groups[metric_name]["overall"].update(target_np, pred_np)
                metric_groups[metric_name]["by_rho_iter"][(rho, iteration)].update(target_np, pred_np)
                metric_groups[metric_name]["by_case_iter"][(case_id, iteration)].update(target_np, pred_np)

        print(f"Processed fields {field_idx}/{len(selected_blueprints)}")

    results = {
        metric_name: {
            "rho_iter": {
                key: accumulator.as_dict()
                for key, accumulator in sorted(metric_group["by_rho_iter"].items())
            },
            "case_iter": {
                key: accumulator.as_dict()
                for key, accumulator in sorted(metric_group["by_case_iter"].items())
            },
            "overall": metric_group["overall"].as_dict(),
        }
        for metric_name, metric_group in metric_groups.items()
    }

    if swanlab_run is not None:
        for metric_name, metric_group in results.items():
            for (rho_value, metric_iter), metrics in metric_group["rho_iter"].items():
                prefix = f"eval/{metric_name}/rho_{rho_value}"
                swanlab_run.log(
                    {
                        f"{prefix}/MAE_hk": float(metrics["MAE_hk"]),
                        f"{prefix}/MSE_hk": float(metrics["MSE_hk"]),
                        f"{prefix}/MaxAE_hk": float(metrics["MaxAE_hk"]),
                        f"{prefix}/N_samples": int(metrics["N_samples"]),
                    },
                    step=int(metric_iter),
                )
        swanlab_run.log(
            {
                f"eval/{metric_name}/all/MAE_hk": float(metric_group["overall"]["MAE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/MSE_hk": float(metric_group["overall"]["MSE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/MaxAE_hk": float(metric_group["overall"]["MaxAE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/N_samples": int(metric_group["overall"]["N_samples"])
                for metric_name, metric_group in results.items()
            },
            step=max(test_iters) if test_iters else 0,
        )

    return results


def evaluate_curvature(
    config: EvalConfig,
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    device: torch.device,
    swanlab_run: Any | None = None,
) -> dict[str, Any]:
    cfl = float(metadata["cfl"])
    scenario_by_case = {
        int(case_id): scenario for case_id, scenario in enumerate(metadata["scenarios"])
    }
    if not scenario_by_case:
        raise ValueError("Test data metadata does not contain scenarios_json.")

    metric_groups = {
        "numeric": {
            "by_rho_iter": defaultdict(MetricAccumulator),
            "by_case_iter": defaultdict(MetricAccumulator),
            "overall": MetricAccumulator(),
        },
        "model": {
            "by_rho_iter": defaultdict(MetricAccumulator),
            "by_case_iter": defaultdict(MetricAccumulator),
            "overall": MetricAccumulator(),
        },
    }

    with h5py.File(config.test_data_path, "r") as handle:
        sample_count = int(handle["hkappa_target"].shape[0])
        limit = sample_count if config.max_samples is None else min(int(config.max_samples), sample_count)
        batch_size = int(config.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            phi9_np = np.asarray(handle["phi9"][start:end], dtype=np.float32)
            xy_np = np.asarray(handle["xy"][start:end], dtype=np.float32)
            iter_np = np.asarray(handle["iter"][start:end], dtype=np.float32)
            rho_np = np.asarray(handle["rho_model"][start:end], dtype=np.int32)
            h_np = np.asarray(handle["h"][start:end], dtype=np.float32)
            case_np = np.asarray(handle["case_id"][start:end], dtype=np.int32)
            target_np = np.asarray(handle["hkappa_target"][start:end], dtype=np.float32)

            a_np = np.empty_like(h_np, dtype=np.float32)
            b_np = np.empty_like(h_np, dtype=np.float32)
            p_np = np.empty_like(h_np, dtype=np.float32)
            exp_ids = []
            for case_id in np.unique(case_np):
                if int(case_id) not in scenario_by_case:
                    raise KeyError(f"case_id={int(case_id)} is not present in scenarios_json.")
                scenario = scenario_by_case[int(case_id)]
                mask = case_np == int(case_id)
                a_np[mask] = float(scenario["a"])
                b_np[mask] = float(scenario["b"])
                p_np[mask] = float(scenario["p"])
            for case_id in case_np:
                exp_ids.append(str(scenario_by_case[int(case_id)]["exp_id"]))

            x = torch.from_numpy(xy_np[:, 0:1]).to(device)
            y = torch.from_numpy(xy_np[:, 1:2]).to(device)
            s = torch.from_numpy((iter_np * cfl).reshape(-1, 1).astype(np.float32)).to(device)
            h = torch.from_numpy(h_np.reshape(-1, 1)).to(device)
            a = torch.from_numpy(a_np.reshape(-1, 1)).to(device)
            b = torch.from_numpy(b_np.reshape(-1, 1)).to(device)
            p = torch.from_numpy(p_np.reshape(-1, 1)).to(device)

            model_pred = compute_autograd_hkappa(model, x=x, y=y, s=s, h=h, a=a, b=b, p=p)
            model_pred_np = model_pred.detach().cpu().numpy().reshape(-1)
            numeric_pred_np = compute_numeric_hkappa(phi9_np, h_np).reshape(-1)

            predictions = {
                "numeric": numeric_pred_np,
                "model": model_pred_np,
            }
            for metric_name, pred_np in predictions.items():
                metric_groups[metric_name]["overall"].update(target_np, pred_np)
            for rho in np.unique(rho_np):
                rho_mask = rho_np == int(rho)
                for iteration in np.unique(iter_np[rho_mask].astype(np.int32)):
                    mask = rho_mask & (iter_np.astype(np.int32) == int(iteration))
                    for metric_name, pred_np in predictions.items():
                        metric_groups[metric_name]["by_rho_iter"][(int(rho), int(iteration))].update(
                            target_np[mask], pred_np[mask]
                        )
            for exp_id in sorted(set(exp_ids)):
                exp_mask = np.asarray([item == exp_id for item in exp_ids], dtype=bool)
                for iteration in np.unique(iter_np[exp_mask].astype(np.int32)):
                    mask = exp_mask & (iter_np.astype(np.int32) == int(iteration))
                    for metric_name, pred_np in predictions.items():
                        metric_groups[metric_name]["by_case_iter"][(exp_id, int(iteration))].update(
                            target_np[mask], pred_np[mask]
                        )

            print(f"Processed samples {end}/{limit}")

    results = {
        metric_name: {
            "rho_iter": {
                key: accumulator.as_dict()
                for key, accumulator in sorted(metric_group["by_rho_iter"].items())
            },
            "case_iter": {
                key: accumulator.as_dict()
                for key, accumulator in sorted(metric_group["by_case_iter"].items())
            },
            "overall": metric_group["overall"].as_dict(),
        }
        for metric_name, metric_group in metric_groups.items()
    }

    if swanlab_run is not None:
        all_iterations = sorted(
            {
                iteration
                for metric_group in results.values()
                for _, iteration in metric_group["rho_iter"]
            }
        )
        for iteration in all_iterations:
            payload: dict[str, float | int] = {}
            for metric_name, metric_group in results.items():
                for (rho, metric_iter), metrics in metric_group["rho_iter"].items():
                    if metric_iter != iteration:
                        continue
                    prefix = f"eval/{metric_name}/rho_{rho}"
                    payload[f"{prefix}/MAE_hk"] = float(metrics["MAE_hk"])
                    payload[f"{prefix}/MSE_hk"] = float(metrics["MSE_hk"])
                    payload[f"{prefix}/MaxAE_hk"] = float(metrics["MaxAE_hk"])
                    payload[f"{prefix}/N_samples"] = int(metrics["N_samples"])
                for (exp_id, metric_iter), metrics in metric_group["case_iter"].items():
                    if metric_iter != iteration:
                        continue
                    prefix = f"eval/{metric_name}/case_{exp_id}"
                    payload[f"{prefix}/MAE_hk"] = float(metrics["MAE_hk"])
                    payload[f"{prefix}/MSE_hk"] = float(metrics["MSE_hk"])
                    payload[f"{prefix}/MaxAE_hk"] = float(metrics["MaxAE_hk"])
                    payload[f"{prefix}/N_samples"] = int(metrics["N_samples"])
            if payload:
                swanlab_run.log(payload, step=int(iteration))
        swanlab_run.log(
            {
                f"eval/{metric_name}/all/MAE_hk": float(metric_group["overall"]["MAE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/MSE_hk": float(metric_group["overall"]["MSE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/MaxAE_hk": float(metric_group["overall"]["MaxAE_hk"])
                for metric_name, metric_group in results.items()
            }
            | {
                f"eval/{metric_name}/all/N_samples": int(metric_group["overall"]["N_samples"])
                for metric_name, metric_group in results.items()
            },
            step=max(metadata["test_iters"]) if metadata["test_iters"] else 0,
        )

    return results


def print_summary(results: dict[str, Any]) -> None:
    print("\nCurvature evaluation summary (prediction/target: h*kappa vs projected analytic target)")
    for metric_name in ("numeric", "model"):
        metric_group = results[metric_name]
        print(f"\n[{metric_name}]")

        print("Case summary (smooth/acute kept separate)")
        print("case_id     iter N_samples MAE_hk MSE_hk MaxAE_hk")

        def case_sort_key(item: tuple[tuple[str, int], dict[str, float | int]]) -> tuple[int, int, int]:
            (exp_id, iteration), _ = item
            if "_" in exp_id:
                family, rho_text = exp_id.split("_", 1)
            else:
                family, rho_text = exp_id, "0"
            family_rank = 0 if family == "smooth" else 1 if family == "acute" else 2
            try:
                rho_value = int(rho_text)
            except ValueError:
                rho_value = 0
            return family_rank, rho_value, int(iteration)

        for (exp_id, iteration), metrics in sorted(metric_group["case_iter"].items(), key=case_sort_key):
            print(
                f"{exp_id:<11s} {iteration:>4d} {int(metrics['N_samples']):>9d} "
                f"{float(metrics['MAE_hk']):.6e} {float(metrics['MSE_hk']):.6e} "
                f"{float(metrics['MaxAE_hk']):.6e}"
            )

        print("\nResolution summary (rho_model mixes smooth + acute at the same resolution)")
        print("rho_model    iter N_samples MAE_hk MSE_hk MaxAE_hk")
        for (rho, iteration), metrics in metric_group["rho_iter"].items():
            print(
                f"{rho:>9d} {iteration:>4d} {int(metrics['N_samples']):>9d} "
                f"{float(metrics['MAE_hk']):.6e} {float(metrics['MSE_hk']):.6e} "
                f"{float(metrics['MaxAE_hk']):.6e}"
            )

        overall = metric_group["overall"]
        print("\nOverall")
        print(
            f"N_samples={int(overall['N_samples'])} "
            f"MAE_hk={float(overall['MAE_hk']):.6e} "
            f"MSE_hk={float(overall['MSE_hk']):.6e} "
            f"MaxAE_hk={float(overall['MaxAE_hk']):.6e}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = EvalConfig()
    parser = argparse.ArgumentParser(description="Evaluate autograd h*kappa curvature on flower or circle test data.")
    parser.add_argument("--geometry", type=str, choices=("flower", "circle"), default=defaults.eval_geometry)
    parser.add_argument("--test-data", type=str, default=str(defaults.test_data_path))
    parser.add_argument("--circle-dataset", type=str, default=str(defaults.circle_dataset_path))
    parser.add_argument(
        "--circle-sampling-mode",
        type=str,
        choices=("moving_interface", "true_interface", "reference_band"),
        default=defaults.circle_sampling_mode,
    )
    parser.add_argument("--circle-band-width-cells", type=float, default=defaults.circle_band_width_cells)
    parser.add_argument("--model", type=str, default=str(defaults.model_path))
    parser.add_argument("--hidden-units", type=int, default=defaults.hidden_units)
    parser.add_argument("--activation", type=str, choices=("tanh", "silu", "relu"), default=defaults.activation)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--num-threads", type=int, default=defaults.num_threads)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test sample limit.")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default=defaults.swanlab_project)
    parser.add_argument("--swanlab-experiment-name", type=str, default=defaults.swanlab_experiment_name)
    parser.add_argument("--swanlab-description", type=str, default=defaults.swanlab_description)
    parser.add_argument("--swanlab-tags", type=str, default=",".join(defaults.swanlab_tags))
    parser.add_argument("--swanlab-group", type=str, default=defaults.swanlab_group)
    parser.add_argument("--swanlab-workspace", type=str, default=defaults.swanlab_workspace)
    parser.add_argument("--swanlab-logdir", type=str, default=defaults.swanlab_logdir)
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        choices=("cloud", "local", "offline", "disabled"),
        default=defaults.swanlab_mode,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        eval_geometry=args.geometry,
        test_data_path=Path(args.test_data),
        circle_dataset_path=Path(args.circle_dataset),
        circle_sampling_mode=args.circle_sampling_mode,
        circle_band_width_cells=args.circle_band_width_cells,
        model_path=Path(args.model),
        hidden_units=args.hidden_units,
        activation=args.activation,
        batch_size=args.batch_size,
        device=args.device,
        num_threads=args.num_threads,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        use_swanlab=bool(args.use_swanlab),
        swanlab_project=args.swanlab_project,
        swanlab_experiment_name=args.swanlab_experiment_name,
        swanlab_description=args.swanlab_description,
        swanlab_tags=_csv_to_tuple(args.swanlab_tags),
        swanlab_group=args.swanlab_group,
        swanlab_workspace=args.swanlab_workspace,
        swanlab_logdir=args.swanlab_logdir,
        swanlab_mode=args.swanlab_mode,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.num_threads < 1:
        raise ValueError("num_threads must be >= 1.")

    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Pass --device cpu to run on CPU.")
    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(config.num_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(config.num_threads))))
        except RuntimeError:
            pass
    if config.eval_geometry == "circle":
        metadata = inspect_circle_test_data(config)
    else:
        metadata = inspect_test_data(Path(config.test_data_path))
    model = load_model(config, device)
    parameter_count = count_parameters(model)

    print(f"Geometry: {config.eval_geometry}")
    if config.eval_geometry == "circle":
        print(f"Circle dataset: {Path(config.circle_dataset_path).resolve()}")
        print(f"Circle sampling mode: {config.circle_sampling_mode}")
        if config.circle_sampling_mode == "reference_band":
            print(f"Band width (cells): {config.circle_band_width_cells}")
        print(f"Fields: {metadata['field_count'] if config.max_samples is None else min(config.max_samples, metadata['field_count'])}")
        print(f"Final step: {metadata['final_step']}")
        print(f"Test iters: {metadata['test_iters']}")
    else:
        print(f"Test data: {Path(config.test_data_path).resolve()}")
        print(f"Samples: {metadata['sample_count'] if config.max_samples is None else min(config.max_samples, metadata['sample_count'])}")
        print(f"CFL: {metadata['cfl']}")
        print(f"Test iters: {metadata['test_iters']}")
    print(f"Model: {Path(config.model_path).resolve()}")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    if device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Parameter count: {parameter_count}")

    swanlab_run = None
    if config.use_swanlab:
        swanlab_run = init_swanlab_run(
            config=config,
            metadata=metadata,
            device=device,
            parameter_count=parameter_count,
        )

    try:
        if config.eval_geometry == "circle":
            results = evaluate_circle_curvature(
                config,
                model=model,
                metadata=metadata,
                device=device,
                swanlab_run=swanlab_run,
            )
        else:
            results = evaluate_curvature(
                config,
                model=model,
                metadata=metadata,
                device=device,
                swanlab_run=swanlab_run,
            )
        print_summary(results)
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    main()
