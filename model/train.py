from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.config import TrainConfig, default_dataset_path, default_output_model_path
    from model.model import HKappaStencilNet, count_parameters
    from traingenerate.io import FIELD_ORDER, load_training_arrays_from_hdf5
else:
    from .config import TrainConfig, default_dataset_path, default_output_model_path
    from .model import HKappaStencilNet, count_parameters
    from traingenerate.io import FIELD_ORDER, load_training_arrays_from_hdf5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=train_config.lr)


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=f"{path.suffix}.tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state_dict, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def normalization_csv_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_phi9_normalization.csv")


def compute_phi9_normalization(train_phi9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_phi9, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_phi9, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)
    return mean, std


def apply_phi9_normalization(
    dataset_bundle: dict[str, Any],
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    normalized_splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split in dataset_bundle["splits"].items():
        normalized_splits[split_name] = {
            "phi9": ((split["phi9"] - mean) / std).astype(np.float32, copy=False),
            "hkappa_target": split["hkappa_target"].astype(np.float32, copy=False),
        }
    return {
        **dataset_bundle,
        "splits": normalized_splits,
    }


def save_phi9_normalization_csv(path: str | Path, *, mean: np.ndarray, std: np.ndarray) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi_index", "mean", "std", "variance"])
        for idx in range(mean.shape[0]):
            writer.writerow([idx + 1, float(mean[idx]), float(std[idx]), float(std[idx] ** 2)])
    return path


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
    config: dict[str, Any] = {
        "task": "3x3 phi stencil -> h*kappa",
        "dataset_path": str(Path(dataset_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "parameter_count": int(parameter_count),
    }
    config.update({f"train/{key}": str(value) if isinstance(value, Path) else value for key, value in asdict(train_config).items()})
    config.update({f"data/{key}": value for key, value in asdict(bundle["config"]).items()})
    config.update(
        {
            f"generation/{key}": str(value) if isinstance(value, Path) else value
            for key, value in asdict(bundle["generation_config"]).items()
        }
    )
    config.update({f"split_blueprints/{key}": int(value) for key, value in bundle.get("split_blueprint_counts", {}).items()})
    config.update({f"split_sizes/{key}": int(value) for key, value in bundle.get("sizes", {}).items()})
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
        raise ImportError("SwanLab logging was requested, but `swanlab` is not installed.") from exc

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
    return swanlab.init(**init_kwargs)


def loader_batch_to_dict(loader, batch) -> dict[str, torch.Tensor]:
    field_names = getattr(loader, "_field_names")
    return {name: tensor for name, tensor in zip(field_names, batch, strict=True)}


def make_loader(split: dict[str, torch.Tensor], *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(*(split[name] for name in FIELD_ORDER))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loader._field_names = FIELD_ORDER  # type: ignore[attr-defined]
    return loader


def build_torch_splits(dataset_bundle: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        split_name: {
            field_name: torch.from_numpy(split[field_name])
            for field_name in FIELD_ORDER
        }
        for split_name, split in dataset_bundle["splits"].items()
    }


def describe_dataset_bundle(dataset_bundle: dict[str, Any], *, batch_size: int) -> None:
    data_cfg = dataset_bundle["config"]
    generation_cfg = dataset_bundle["generation_config"]
    print("Task: 3x3 phi stencil -> h*kappa")
    print(f"Number of blueprints: {len(dataset_bundle['blueprints'])}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field types: {data_cfg.initial_field_types}")
    print(f"Samples per circle per field type: {data_cfg.n_samples_per_circle}")
    print(f"Dataset generation batch size: {generation_cfg.generation_batch_size}")
    print(f"Training batch size: {batch_size}")
    print(
        f"Blueprint split fractions: train={data_cfg.train_fraction}, "
        f"val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}"
    )
    for split_name, split_size in dataset_bundle["sizes"].items():
        print(f"{split_name:>5s}: {split_size}")


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def run_epoch(
    model: HKappaStencilNet,
    loader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_count = 0
    for raw_batch in loader:
        batch = _move_batch_to_device(loader_batch_to_dict(loader, raw_batch), device)
        _, loss = model.predict_and_loss(batch["phi9"], batch["hkappa_target"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(batch["phi9"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        raise ValueError("Encountered an empty split during training.")
    return total_loss / total_count


def train_model(
    model: HKappaStencilNet,
    *,
    bundle: dict[str, Any],
    train_config: TrainConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    swanlab_run: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    train_config = train_config or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path(train_config.output_model_path)

    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = model.to(device)
    optimizer = optimizer or create_optimizer(model, train_config)
    train_loader = make_loader(bundle["splits"]["train"], batch_size=train_config.batch_size, shuffle=True)
    val_loader = make_loader(bundle["splits"]["val"], batch_size=train_config.batch_size, shuffle=False)
    history: dict[str, Any] = {
        "train_hk": [],
        "val_hk": [],
        "best_epoch": -1,
        "best_checkpoint_path": str(checkpoint_path.resolve()),
    }
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(train_config.max_epochs):
        train_loss = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device=device, optimizer=None)
        history["train_hk"].append(train_loss)
        history["val_hk"].append(val_loss)

        print(
            f"Epoch {epoch:03d} | Train hk MSE: {train_loss:.6e} | "
            f"Val hk MSE: {val_loss:.6e}"
        )
        if swanlab_run is not None:
            swanlab_run.log(
                {
                    "train/hk_mse": train_loss,
                    "val/hk_mse": val_loss,
                    "monitor/best_val_hk_mse": min(best_val, val_loss),
                },
                step=epoch + 1,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
    return model, history


def build_arg_parser() -> argparse.ArgumentParser:
    train_cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the 3x3 stencil h*kappa predictor.")
    parser.add_argument("--dataset-output", type=str, default=str(default_dataset_path()))
    parser.add_argument("--output-model", type=str, default=str(default_output_model_path()))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--hidden-units", type=int, default=train_cfg.hidden_units)
    parser.add_argument("--lr", type=float, default=train_cfg.lr)
    parser.add_argument("--max-epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--batch-size", type=int, default=train_cfg.batch_size)
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
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        dataset_path=dataset_path,
        output_model_path=Path(args.output_model),
    )
    if train_config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    print(f"Loading existing dataset from: {dataset_path.resolve()}")
    dataset_bundle_raw = load_training_arrays_from_hdf5(dataset_path)
    phi9_mean, phi9_std = compute_phi9_normalization(dataset_bundle_raw["splits"]["train"]["phi9"])
    stats_csv_path = save_phi9_normalization_csv(
        normalization_csv_path(train_config.output_model_path),
        mean=phi9_mean,
        std=phi9_std,
    )
    dataset_bundle = apply_phi9_normalization(dataset_bundle_raw, mean=phi9_mean, std=phi9_std)
    bundle = {
        **dataset_bundle,
        "splits": build_torch_splits(dataset_bundle),
    }
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HKappaStencilNet(hidden_units=train_config.hidden_units)
    optimizer = create_optimizer(model, train_config)

    print(f"Device: {device}")
    print(f"Dataset file: {dataset_path.resolve()}")
    print(f"Parameter count: {count_parameters(model)}")
    describe_dataset_bundle(dataset_bundle, batch_size=train_config.batch_size)
    print(f"Phi9 normalization CSV: {stats_csv_path.resolve()}")

    swanlab_run = None
    if args.use_swanlab:
        swanlab_run = init_swanlab_run(
            train_config=train_config,
            bundle=bundle,
            dataset_path=dataset_path,
            checkpoint_path=args.output_model,
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
        checkpoint_path=args.output_model,
        optimizer=optimizer,
        swanlab_run=swanlab_run,
    )

    if swanlab_run is not None:
        swanlab_run.log(
            {
                "summary/best_epoch": int(history["best_epoch"]),
                "summary/best_checkpoint_path": history["best_checkpoint_path"],
                "summary/final_val_hk_mse": float(history["val_hk"][-1]),
            }
        )
        swanlab_run.finish()

    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best checkpoint: {Path(history['best_checkpoint_path']).resolve()}")


if __name__ == "__main__":
    main()
