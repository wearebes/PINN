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

from model.model import EPS, ReinitPINN, count_parameters

from .config import EvalConfig


REQUIRED_DATASETS = ("xy", "iter", "rho_model", "h", "case_id", "hkappa_target")


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


def build_swanlab_config(
    *,
    config: EvalConfig,
    metadata: dict[str, Any],
    device: torch.device,
    parameter_count: int,
) -> dict[str, Any]:
    swan_config: dict[str, Any] = {
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

    by_rho_iter: dict[tuple[int, int], MetricAccumulator] = defaultdict(MetricAccumulator)
    by_case_iter: dict[tuple[str, int], MetricAccumulator] = defaultdict(MetricAccumulator)
    overall = MetricAccumulator()

    with h5py.File(config.test_data_path, "r") as handle:
        sample_count = int(handle["hkappa_target"].shape[0])
        limit = sample_count if config.max_samples is None else min(int(config.max_samples), sample_count)
        batch_size = int(config.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
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

            pred = compute_autograd_hkappa(model, x=x, y=y, s=s, h=h, a=a, b=b, p=p)
            pred_np = pred.detach().cpu().numpy().reshape(-1)

            overall.update(target_np, pred_np)
            for rho in np.unique(rho_np):
                rho_mask = rho_np == int(rho)
                for iteration in np.unique(iter_np[rho_mask].astype(np.int32)):
                    mask = rho_mask & (iter_np.astype(np.int32) == int(iteration))
                    by_rho_iter[(int(rho), int(iteration))].update(target_np[mask], pred_np[mask])
            for exp_id in sorted(set(exp_ids)):
                exp_mask = np.asarray([item == exp_id for item in exp_ids], dtype=bool)
                for iteration in np.unique(iter_np[exp_mask].astype(np.int32)):
                    mask = exp_mask & (iter_np.astype(np.int32) == int(iteration))
                    by_case_iter[(exp_id, int(iteration))].update(target_np[mask], pred_np[mask])

            print(f"Processed samples {end}/{limit}")

    rho_metrics = {key: accumulator.as_dict() for key, accumulator in sorted(by_rho_iter.items())}
    case_metrics = {key: accumulator.as_dict() for key, accumulator in sorted(by_case_iter.items())}
    overall_metrics = overall.as_dict()

    if swanlab_run is not None:
        for iteration in sorted({iteration for _, iteration in rho_metrics}):
            payload: dict[str, float | int] = {}
            for (rho, metric_iter), metrics in rho_metrics.items():
                if metric_iter != iteration:
                    continue
                prefix = f"eval/rho_{rho}"
                payload[f"{prefix}/MAE_hk"] = float(metrics["MAE_hk"])
                payload[f"{prefix}/MSE_hk"] = float(metrics["MSE_hk"])
                payload[f"{prefix}/MaxAE_hk"] = float(metrics["MaxAE_hk"])
                payload[f"{prefix}/N_samples"] = int(metrics["N_samples"])
            for (exp_id, metric_iter), metrics in case_metrics.items():
                if metric_iter != iteration:
                    continue
                prefix = f"eval/case_{exp_id}"
                payload[f"{prefix}/MAE_hk"] = float(metrics["MAE_hk"])
                payload[f"{prefix}/MSE_hk"] = float(metrics["MSE_hk"])
                payload[f"{prefix}/MaxAE_hk"] = float(metrics["MaxAE_hk"])
                payload[f"{prefix}/N_samples"] = int(metrics["N_samples"])
            if payload:
                swanlab_run.log(payload, step=int(iteration))
        swanlab_run.log(
            {
                "eval/all/MAE_hk": float(overall_metrics["MAE_hk"]),
                "eval/all/MSE_hk": float(overall_metrics["MSE_hk"]),
                "eval/all/MaxAE_hk": float(overall_metrics["MaxAE_hk"]),
                "eval/all/N_samples": int(overall_metrics["N_samples"]),
            },
            step=max(metadata["test_iters"]) if metadata["test_iters"] else 0,
        )

    return {
        "overall": overall_metrics,
        "rho_iter": rho_metrics,
        "case_iter": case_metrics,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("\nCurvature evaluation summary (target/prediction: h*kappa)")
    print("rho_model iter N_samples MAE_hk MSE_hk MaxAE_hk")
    for (rho, iteration), metrics in results["rho_iter"].items():
        print(
            f"{rho:>9d} {iteration:>4d} {int(metrics['N_samples']):>9d} "
            f"{float(metrics['MAE_hk']):.6e} {float(metrics['MSE_hk']):.6e} "
            f"{float(metrics['MaxAE_hk']):.6e}"
        )
    overall = results["overall"]
    print("\nOverall")
    print(
        f"N_samples={int(overall['N_samples'])} "
        f"MAE_hk={float(overall['MAE_hk']):.6e} "
        f"MSE_hk={float(overall['MSE_hk']):.6e} "
        f"MaxAE_hk={float(overall['MaxAE_hk']):.6e}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = EvalConfig()
    parser = argparse.ArgumentParser(description="Evaluate autograd h*kappa curvature on flower test data.")
    parser.add_argument("--test-data", type=str, default=str(defaults.test_data_path))
    parser.add_argument("--model", type=str, default=str(defaults.model_path))
    parser.add_argument("--hidden-units", type=int, default=defaults.hidden_units)
    parser.add_argument("--activation", type=str, choices=("tanh", "silu", "relu"), default=defaults.activation)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)
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
        test_data_path=Path(args.test_data),
        model_path=Path(args.model),
        hidden_units=args.hidden_units,
        activation=args.activation,
        batch_size=args.batch_size,
        device=args.device,
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

    metadata = inspect_test_data(Path(config.test_data_path))
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Pass --device cpu to run on CPU.")
    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    parameter_count = count_parameters(model)

    print(f"Test data: {Path(config.test_data_path).resolve()}")
    print(f"Model: {Path(config.model_path).resolve()}")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Samples: {metadata['sample_count'] if config.max_samples is None else min(config.max_samples, metadata['sample_count'])}")
    print(f"CFL: {metadata['cfl']}")
    print(f"Test iters: {metadata['test_iters']}")
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
