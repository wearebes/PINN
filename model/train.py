from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.config import TrainConfig, default_dataset_path, default_output_model_path
    from model.model import EPS, ReinitPINN, compute_losses, compute_model_curvature, compute_phi0_from_circle, count_parameters
    from traingenerate.generate import describe_bundle, load_training_bundle_from_hdf5, sample_interface_loader, sample_pde_loader
else:
    from .config import TrainConfig, default_dataset_path, default_output_model_path
    from .model import EPS, ReinitPINN, compute_losses, compute_model_curvature, compute_phi0_from_circle, count_parameters
    from traingenerate.generate import describe_bundle, load_training_bundle_from_hdf5, sample_interface_loader, sample_pde_loader


def loader_batch_to_dict(loader, batch) -> dict[str, torch.Tensor]:
    field_names = getattr(loader, "_field_names")
    return {name: tensor for name, tensor in zip(field_names, batch, strict=True)}


def cycle_loader(loader):
    while True:
        for batch in loader:
            if isinstance(batch, dict):
                yield batch
            else:
                yield loader_batch_to_dict(loader, batch)


def evaluate_epoch(
    model: nn.Module,
    *,
    traj_loader,
    pde_loader,
    interface_loader,
    device: torch.device,
    criterion: nn.Module,
    eps_sign_factor: float,
    weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    traj_iter = cycle_loader(traj_loader)
    pde_iter = cycle_loader(pde_loader)
    interface_iter = cycle_loader(interface_loader)
    num_steps = max(len(traj_loader), len(pde_loader), len(interface_loader))
    sums = {key: 0.0 for key in ("traj", "pde", "interface", "total")}

    for _ in range(num_steps):
        with torch.enable_grad():
            losses = compute_losses(
                model,
                traj_batch=next(traj_iter),
                pde_batch=next(pde_iter),
                interface_batch=next(interface_iter),
                device=device,
                criterion=criterion,
                eps_sign_factor=eps_sign_factor,
                weights=weights,
            )
        for key in sums:
            sums[key] += float(losses[key].detach().cpu().item())

    return {key: value / num_steps for key, value in sums.items()}


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state_dict, handle)
            handle.flush()
            os.fsync(handle.fileno())

        loaded_state = torch.load(tmp_path, map_location="cpu")
        if not isinstance(loaded_state, dict) or not loaded_state:
            raise RuntimeError(f"Checkpoint verification failed for temporary file: {tmp_path}")

        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint was not persisted correctly: {path}")
    return path


def create_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=train_config.lr)


def _csv_to_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_swanlab_config(
    *,
    train_config: TrainConfig,
    bundle: dict[str, Any],
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    parameter_count: int,
) -> dict[str, Any]:
    data_cfg = bundle["config"]
    reinit_cfg = bundle["reinit_config"]
    generation_cfg = bundle.get("generation_config")
    split_counts = bundle.get("split_blueprint_counts", {})
    sizes = bundle.get("sizes", {})

    config: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_name": Path(dataset_path).name,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "parameter_count": int(parameter_count),
    }
    config.update(
        {
            f"train/{key}": value
            for key, value in asdict(train_config).items()
            if key not in {"dataset_path", "output_model_path"}
        }
    )
    config["train/dataset_path"] = str(Path(train_config.dataset_path).resolve())
    config["train/output_model_path"] = str(Path(train_config.output_model_path).resolve())
    config.update({f"data/{key}": value for key, value in asdict(data_cfg).items()})
    if generation_cfg is not None:
        config.update(
            {
                f"generation/{key}": str(value) if isinstance(value, Path) else value
                for key, value in asdict(generation_cfg).items()
            }
        )
    config.update({f"reinit/{key}": value for key, value in asdict(reinit_cfg).items()})
    config.update({f"split_blueprints/{key}": int(value) for key, value in split_counts.items()})
    config.update({f"split_sizes/{key}": int(value) for key, value in sizes.items()})
    return config


def init_swanlab_run(
    *,
    train_config: TrainConfig,
    bundle: dict[str, Any],
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    parameter_count: int,
    project: str | None,
    experiment_name: str | None,
    description: str | None,
    tags: list[str] | None,
    group: str | None,
    workspace: str | None,
    logdir: str | None,
    mode: str | None,
):
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging was requested, but the `swanlab` package is not installed in the current environment."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": project,
        "workspace": workspace,
        "experiment_name": experiment_name,
        "description": description,
        "tags": tags,
        "group": group,
        "logdir": logdir,
        "mode": mode,
        "config": build_swanlab_config(
            train_config=train_config,
            bundle=bundle,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            device=device,
            parameter_count=parameter_count,
        ),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    run = swanlab.init(**init_kwargs)
    run.log(
        {
            "system/parameter_count": int(parameter_count),
        },
        step=0,
    )
    return run


def aggregate_validation_metrics(all_validation_metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not all_validation_metrics:
        return {
            "validation/field_count": 0.0,
            "validation/mean_sdf_abs_mae": float("nan"),
            "validation/mean_eikonal_band_mae": float("nan"),
            "validation/mean_interface_drift_mae": float("nan"),
            "validation/mean_curvature_mae": float("nan"),
        }
    return {
        "validation/field_count": float(len(all_validation_metrics)),
        "validation/mean_sdf_abs_mae": float(np.mean([item["sdf_abs_mae"] for item in all_validation_metrics])),
        "validation/mean_eikonal_band_mae": float(np.mean([item["eikonal_band_mae"] for item in all_validation_metrics])),
        "validation/mean_interface_drift_mae": float(np.mean([item["interface_drift_mae"] for item in all_validation_metrics])),
        "validation/mean_curvature_mae": float(np.mean([item["curvature_mae"] for item in all_validation_metrics])),
    }


def get_epoch_weights(train_config: TrainConfig, epoch: int) -> dict[str, float]:
    return {
        "traj": train_config.lambda_traj,
        "pde": train_config.lambda_pde,
        "interface": train_config.lambda_interface,
    }


def train_model(
    model: nn.Module,
    *,
    bundle: dict[str, Any],
    train_config: TrainConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    output_model_path: str | Path | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    swanlab_run: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    train_config = train_config or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path("best_reinit_pinn.pt")
    checkpoint_path = checkpoint_path.resolve()
    output_model_path = Path(output_model_path) if output_model_path is not None else Path(train_config.output_model_path)
    output_model_path = output_model_path.resolve()

    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    criterion = nn.MSELoss()
    model = model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=train_config.lr)

    loaders = bundle["loaders"]
    reinit_cfg = bundle["reinit_config"]
    selection_weights = {
        "traj": train_config.lambda_traj,
        "pde": train_config.lambda_pde,
        "interface": train_config.lambda_interface,
    }
    history: dict[str, Any] = {
        "train_traj": [],
        "train_pde": [],
        "train_interface": [],
        "train_total": [],
        "val_traj": [],
        "val_pde": [],
        "val_interface": [],
        "val_total": [],
        "val_total_monitor": [],
        "weight_traj": [],
        "weight_pde": [],
        "weight_interface": [],
        "best_epoch": -1,
        "best_checkpoint_path": str(checkpoint_path),
        "output_model_path": str(output_model_path),
    }

    best_val_total = float("inf")
    best_state = None
    wait = 0
    train_traj_iter = cycle_loader(loaders["train_traj"])

    for epoch in range(train_config.max_epochs):
        model.train()
        weights = get_epoch_weights(train_config, epoch)
        train_pde_loader = sample_pde_loader(
            bundle,
            "train",
            seed=train_config.seed + epoch,
            batch_size=train_config.batch_size,
            shuffle=True,
            device=device,
        )
        train_interface_loader = sample_interface_loader(
            bundle,
            "train",
            seed=train_config.seed + 100_000 + epoch,
            batch_size=train_config.batch_size,
            shuffle=True,
            device=device,
        )
        train_pde_iter = cycle_loader(train_pde_loader)
        train_interface_iter = cycle_loader(train_interface_loader)
        train_steps = max(len(loaders["train_traj"]), len(train_pde_loader), len(train_interface_loader))
        train_sums = {key: 0.0 for key in ("traj", "pde", "interface", "total")}

        for _ in range(train_steps):
            optimizer.zero_grad()
            losses = compute_losses(
                model,
                traj_batch=next(train_traj_iter),
                pde_batch=next(train_pde_iter),
                interface_batch=next(train_interface_iter),
                device=device,
                criterion=criterion,
                eps_sign_factor=reinit_cfg.eps_sign_factor,
                weights=weights,
            )
            losses["total"].backward()
            optimizer.step()
            for key in train_sums:
                train_sums[key] += float(losses[key].detach().cpu().item())

        train_means = {key: value / train_steps for key, value in train_sums.items()}
        val_pde_loader = sample_pde_loader(
            bundle,
            "val",
            seed=train_config.seed + 200_000 + epoch,
            batch_size=train_config.batch_size,
            shuffle=False,
            device=device,
        )
        val_interface_loader = sample_interface_loader(
            bundle,
            "val",
            seed=train_config.seed + 300_000 + epoch,
            batch_size=train_config.batch_size,
            shuffle=False,
            device=device,
        )
        val_means = evaluate_epoch(
            model,
            traj_loader=loaders["val_traj"],
            pde_loader=val_pde_loader,
            interface_loader=val_interface_loader,
            device=device,
            criterion=criterion,
            eps_sign_factor=reinit_cfg.eps_sign_factor,
            weights=weights,
        )
        val_monitor_means = val_means

        for key in ("traj", "pde", "interface", "total"):
            history[f"train_{key}"].append(train_means[key])  # type: ignore[index]
            history[f"val_{key}"].append(val_means[key])  # type: ignore[index]
        history["val_total_monitor"].append(val_monitor_means["total"])
        history["weight_traj"].append(weights["traj"])
        history["weight_pde"].append(weights["pde"])
        history["weight_interface"].append(weights["interface"])

        print(
            f"Epoch {epoch:03d} | "
            f"Weights(Traj/PDE/Interface): {weights['traj']:.3f}/{weights['pde']:.3f}/{weights['interface']:.3f} | "
            f"Train Traj: {train_means['traj']:.6e} | "
            f"Train PDE: {train_means['pde']:.6e} | "
            f"Train Interface: {train_means['interface']:.6e} | "
            f"Train Total: {train_means['total']:.6e}"
        )
        print(
            f"           Val Traj: {val_means['traj']:.6e} | "
            f"Val PDE: {val_means['pde']:.6e} | "
            f"Val Interface: {val_means['interface']:.6e} | "
            f"Val Total: {val_means['total']:.6e} | "
            f"Monitor Total: {val_monitor_means['total']:.6e}"
        )
        if swanlab_run is not None:
            swanlab_run.log(
                {
                    "train/traj_loss": train_means["traj"],
                    "train/pde_loss": train_means["pde"],
                    "train/interface_loss": train_means["interface"],
                    "train/total_loss": train_means["total"],
                    "val/traj_loss": val_means["traj"],
                    "val/pde_loss": val_means["pde"],
                    "val/interface_loss": val_means["interface"],
                    "val/total_loss": val_means["total"],
                    "val/monitor_total_loss": val_monitor_means["total"],
                    "weights/traj": weights["traj"],
                    "weights/pde": weights["pde"],
                    "weights/interface": weights["interface"],
                    "monitor/best_val_total": min(best_val_total, val_monitor_means["total"]),
                },
                step=epoch + 1,
            )

        if val_monitor_means["total"] < best_val_total:
            best_val_total = val_monitor_means["total"]
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            wait = 0
            save_checkpoint(model, checkpoint_path)
        else:
            wait += 1
            if wait >= train_config.patience:
                print(f"Early stopping at epoch {epoch} with patience={train_config.patience}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(model, checkpoint_path)
    save_checkpoint(model, output_model_path)
    return model, history


def plot_loss_history(history: dict[str, list[float] | int]) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(epochs, history["train_traj"], label="Train Traj")
    axes[0].plot(epochs, history["train_pde"], label="Train PDE")
    axes[0].plot(epochs, history["train_interface"], label="Train Interface")
    axes[0].plot(epochs, history["train_total"], label="Train Total", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Training Losses")
    axes[0].legend()

    axes[1].plot(epochs, history["val_traj"], label="Val Traj")
    axes[1].plot(epochs, history["val_pde"], label="Val PDE")
    axes[1].plot(epochs, history["val_interface"], label="Val Interface")
    axes[1].plot(epochs, history["val_total"], label="Val Total", linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation Losses")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_weight_schedule(history: dict[str, list[float] | int], train_config: TrainConfig | None = None) -> None:
    epochs = np.arange(1, len(history["weight_pde"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["weight_traj"], label="Traj Weight")
    plt.plot(epochs, history["weight_pde"], label="PDE Weight")
    plt.plot(epochs, history["weight_interface"], label="Interface Weight")
    plt.xlabel("Epoch")
    plt.ylabel("Effective Weight")
    plt.title("Loss Weight Schedule")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_sdf_metrics(
    model: nn.Module,
    field_record: dict[str, Any],
    *,
    final_step: int = 20,
    cfl: float = 0.01,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blueprint = field_record["blueprint"]
    h = float(blueprint["params"]["h"])
    radius = float(blueprint["params"]["radius"])
    rho = int(blueprint["meta"]["resolution"])

    x = np.linspace(0.0, 1.0, rho, dtype=np.float32)
    X, Y = np.meshgrid(x, x, indexing="ij")
    s_final = np.full_like(X, final_step * cfl, dtype=np.float32)

    xt = torch.from_numpy(X.reshape(-1, 1)).to(device).requires_grad_(True)
    yt = torch.from_numpy(Y.reshape(-1, 1)).to(device).requires_grad_(True)
    st = torch.from_numpy(s_final.reshape(-1, 1)).to(device).requires_grad_(True)
    cx_t = torch.full_like(xt, float(blueprint["params"]["center"][0]))
    cy_t = torch.full_like(yt, float(blueprint["params"]["center"][1]))
    radius_t = torch.full_like(xt, radius)
    model = model.to(device)
    model.eval()

    with torch.enable_grad():
        phi0t = compute_phi0_from_circle(xt, yt, cx_t, cy_t, radius_t)
        phi = model(xt, yt, phi0t, st)
        phi_x = torch.autograd.grad(phi, xt, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
        phi_y = torch.autograd.grad(phi, yt, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
        grad_norm = torch.sqrt(phi_x.square() + phi_y.square() + EPS)

    pred = phi.detach().cpu().numpy().reshape(X.shape)
    grad = grad_norm.detach().cpu().numpy().reshape(X.shape)
    sdf = np.asarray(field_record["sdf"], dtype=np.float32)
    sdf_band = np.abs(sdf) <= (3.0 * h)
    if not np.any(sdf_band):
        sdf_band = np.ones_like(sdf, dtype=bool)

    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False, dtype=np.float32)
    xi = float(blueprint["params"]["center"][0]) + radius * np.cos(theta)
    yi = float(blueprint["params"]["center"][1]) + radius * np.sin(theta)
    si = np.full_like(xi, final_step * cfl)

    xi_t = torch.from_numpy(xi.reshape(-1, 1)).to(device).requires_grad_(True)
    yi_t = torch.from_numpy(yi.reshape(-1, 1)).to(device).requires_grad_(True)
    si_t = torch.from_numpy(si.reshape(-1, 1)).to(device).requires_grad_(True)
    cxi_t = torch.full_like(xi_t, float(blueprint["params"]["center"][0]))
    cyi_t = torch.full_like(yi_t, float(blueprint["params"]["center"][1]))
    ri_t = torch.full_like(xi_t, radius)

    with torch.enable_grad():
        phi0_i_t = compute_phi0_from_circle(xi_t, yi_t, cxi_t, cyi_t, ri_t)
        phi_i = model(xi_t, yi_t, phi0_i_t, si_t)
        kappa_i = compute_model_curvature(model, xi_t, yi_t, cxi_t, cyi_t, ri_t, si_t)

    interface_phi = phi_i.detach().cpu().numpy().reshape(-1)
    kappa_pred = kappa_i.detach().cpu().numpy().reshape(-1)

    return {
        "blueprint": blueprint,
        "grid_x": X,
        "grid_y": Y,
        "pred_final": pred,
        "exact_sdf": sdf,
        "sdf_abs_mae": float(np.mean(np.abs(pred - sdf))),
        "eikonal_band_mae": float(np.mean(np.abs(grad[sdf_band] - 1.0))),
        "interface_drift_mae": float(np.mean(np.abs(interface_phi))),
        "curvature_mae": float(np.mean(np.abs(kappa_pred - (1.0 / radius)))),
    }


def plot_validation_snapshot(
    model: nn.Module,
    bundle: dict[str, Any],
    *,
    data_config,
    device: torch.device | None = None,
) -> dict[str, Any]:
    validation_record = bundle["sdf_validation_fields"][len(bundle["sdf_validation_fields"]) // 2]
    validation_metrics = evaluate_sdf_metrics(
        model,
        validation_record,
        final_step=max(data_config.reinit_steps),
        cfl=float(bundle["reinit_config"].cfl),
        device=device,
    )

    pred_final = validation_metrics["pred_final"]
    exact_sdf = validation_metrics["exact_sdf"]
    abs_error = np.abs(pred_final - exact_sdf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(pred_final.T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    axes[0].set_title("Predicted Final Field")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(exact_sdf.T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    axes[1].set_title("Exact Circle SDF")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(abs_error.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    axes[2].set_title("Absolute Error")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    print(f"SDF abs MAE:      {validation_metrics['sdf_abs_mae']:.6e}")
    print(f"Eikonal band MAE: {validation_metrics['eikonal_band_mae']:.6e}")
    print(f"Interface drift:  {validation_metrics['interface_drift_mae']:.6e}")
    print(f"Curvature MAE:    {validation_metrics['curvature_mae']:.6e}")
    return validation_metrics


def summarize_validation_metrics(
    model: nn.Module,
    bundle: dict[str, Any],
    *,
    data_config,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    all_validation_metrics = [
        evaluate_sdf_metrics(
            model,
            record,
            final_step=max(data_config.reinit_steps),
            cfl=float(bundle["reinit_config"].cfl),
            device=device,
        )
        for record in bundle["sdf_validation_fields"]
    ]
    summary = aggregate_validation_metrics(all_validation_metrics)

    print(f"Validation fields:         {len(all_validation_metrics)}")
    print(f"Mean SDF abs MAE:          {summary['validation/mean_sdf_abs_mae']:.6e}")
    print(f"Mean Eikonal band MAE:     {summary['validation/mean_eikonal_band_mae']:.6e}")
    print(f"Mean interface drift MAE:  {summary['validation/mean_interface_drift_mae']:.6e}")
    print(f"Mean curvature MAE:        {summary['validation/mean_curvature_mae']:.6e}")
    return all_validation_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    train_cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Run the PINN training experiment from the command line.")
    parser.add_argument("--dataset-output", type=str, default=str(default_dataset_path()))
    parser.add_argument("--checkpoint", type=str, default="best_reinit_pinn.pt")
    parser.add_argument("--output-model", type=str, default=str(default_output_model_path()))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--hidden-units", type=int, default=train_cfg.hidden_units)
    parser.add_argument("--activation", type=str, choices=("tanh", "silu", "relu"), default=train_cfg.activation)
    parser.add_argument("--lr", type=float, default=train_cfg.lr)
    parser.add_argument("--max-epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--lambda-traj", type=float, default=train_cfg.lambda_traj)
    parser.add_argument("--lambda-pde", type=float, default=train_cfg.lambda_pde)
    parser.add_argument("--lambda-interface", type=float, default=train_cfg.lambda_interface)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_cfg.batch_size,
        help="Training batch size. Defaults to TrainConfig.batch_size in model/config.py.",
    )
    parser.add_argument("--seed", type=int, default=train_cfg.seed)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="PINN")
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-description", type=str, default="")
    parser.add_argument("--swanlab-tags", type=str, default="")
    parser.add_argument("--swanlab-group", type=str, default="")
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument("--swanlab-mode", type=str, choices=("cloud", "local", "offline", "disabled"), default="cloud")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset_path = Path(args.dataset_output)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path.resolve()}\n"
            "Generate the dataset first with `python -m traingenerate.generate`."
        )

    train_config = TrainConfig(
        hidden_units=args.hidden_units,
        activation=args.activation,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lambda_traj=args.lambda_traj,
        lambda_pde=args.lambda_pde,
        lambda_interface=args.lambda_interface,
        batch_size=args.batch_size,
        seed=args.seed,
        dataset_path=Path(args.dataset_output),
        output_model_path=Path(args.output_model),
    )
    if train_config.batch_size < 1:
        raise ValueError("batch_size must be >= 1. Edit TrainConfig.batch_size or pass --batch-size with a positive integer.")

    print(f"Loading existing dataset from: {dataset_path.resolve()}")
    bundle = load_training_bundle_from_hdf5(dataset_path, batch_size=train_config.batch_size)
    data_config = bundle["config"]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReinitPINN(hidden_units=train_config.hidden_units, activation=train_config.activation).to(device)
    optimizer = create_optimizer(model, train_config)

    print(f"Device: {device}")
    print(f"Dataset file: {dataset_path.resolve()}")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Activation: {train_config.activation}")
    print(
        f"Loss weights: traj={train_config.lambda_traj}, "
        f"pde={train_config.lambda_pde}, interface={train_config.lambda_interface}"
    )
    print(f"Training batch size: {train_config.batch_size}")
    if bundle.get("generation_config") is not None:
        print(f"Generation batch size: {bundle['generation_config'].generation_batch_size}")
    describe_bundle(bundle)

    swanlab_run = None
    if args.use_swanlab:
        swanlab_run = init_swanlab_run(
            train_config=train_config,
            bundle=bundle,
            dataset_path=dataset_path,
            checkpoint_path=args.checkpoint,
            device=device,
            parameter_count=count_parameters(model),
            project=args.swanlab_project or None,
            experiment_name=args.swanlab_experiment_name or None,
            description=args.swanlab_description or None,
            tags=_csv_to_list(args.swanlab_tags),
            group=args.swanlab_group or None,
            workspace=args.swanlab_workspace or None,
            logdir=args.swanlab_logdir or None,
            mode=args.swanlab_mode or None,
    )

    model, history = train_model(
        model,
        bundle=bundle,
        train_config=train_config,
        device=device,
        checkpoint_path=args.checkpoint,
        output_model_path=train_config.output_model_path,
        optimizer=optimizer,
        swanlab_run=swanlab_run,
    )

    print(f"Best checkpoint: {history['best_checkpoint_path']}")
    print(f"Output model: {history['output_model_path']}")
    print(f"Best epoch: {history['best_epoch']}")
    final_validation_metrics = summarize_validation_metrics(model, bundle, data_config=data_config, device=device)
    final_summary = aggregate_validation_metrics(final_validation_metrics)
    if swanlab_run is not None:
        best_epoch = int(history["best_epoch"])
        swanlab_run.log(
            {
                "summary/best_epoch": best_epoch,
                "summary/best_val_total_monitor": float(history["val_total_monitor"][best_epoch]) if best_epoch >= 0 else float("nan"),
                **final_summary,
            },
            step=len(history["train_total"]),
        )
        swanlab_run.finish()

    if args.plot:
        plot_loss_history(history)
        plot_weight_schedule(history, train_config)
        plot_validation_snapshot(model, bundle, data_config=data_config, device=device)


if __name__ == "__main__":
    main()
