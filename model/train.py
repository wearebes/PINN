from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate.shared import csv_to_list, init_swanlab_run, normalization_csv_path, save_checkpoint_bundle
    from model.config import CNN_TrainConfig, MLP_TrainConfig, ModelType, create_train_config, default_dataset_path, default_output_model_path
    from model.model import create_model, count_parameters
    from train_generate.io import FIELD_ORDER, load_training_arrays_from_hdf5
else:
    from evaluate.shared import csv_to_list, init_swanlab_run, normalization_csv_path, save_checkpoint_bundle
    from .config import CNN_TrainConfig, MLP_TrainConfig, ModelType, create_train_config, default_dataset_path, default_output_model_path
    from .model import create_model, count_parameters
    from train_generate.io import FIELD_ORDER, load_training_arrays_from_hdf5


TRAIN_SPLIT_NAMES = ("train", "val")

# 设置optimizer 为 Adam
def create_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def emit_phase_event(path: str | Path | None, event: str, **payload: Any) -> None:
    if not path:
        return
    event_path = Path(path).resolve()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_ns": time.time_ns(),
        "event": event,
        **payload,
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "_orig_mod"):
        unwrapped = getattr(unwrapped, "_orig_mod")
    return unwrapped


def compute_phi9_normalization(train_phi9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_phi9 = np.asarray(train_phi9, dtype=np.float32)
    if train_phi9.ndim != 2 or train_phi9.shape[1] != 9:
        raise ValueError(f"Training phi9 must have shape (N, 9), got {train_phi9.shape}.")
    if train_phi9.shape[0] == 0:
        raise ValueError("Cannot compute phi9 normalization from an empty training split.")
    if np.any(~np.isfinite(train_phi9)):
        raise ValueError("Training phi9 contains non-finite values; cannot compute normalization.")
    mean = np.mean(train_phi9, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_phi9, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)
    return mean, std


def save_phi9_normalization_csv(path: str | Path, *, mean: np.ndarray, std: np.ndarray, dataset_path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path).resolve()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi_index", "mean", "std", "variance", "source_split", "feature_count", "dataset_path"])
        for idx in range(9):
            writer.writerow([idx + 1, float(mean[idx]), float(std[idx]), float(std[idx] ** 2), "train", 9, str(dataset_path)])
    return path


def apply_phi9_normalization(dataset_bundle: dict[str, Any], *, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split in dataset_bundle["splits"].items():
        splits[split_name] = {
            "phi9": ((np.asarray(split["phi9"], dtype=np.float32) - mean) / std).astype(np.float32, copy=False),
            "hkappa_target": np.asarray(split["hkappa_target"], dtype=np.float32),
        }
    return {**dataset_bundle, "splits": splits}


def _validate_split(split_name: str, split: dict[str, Any]) -> None:
    missing = [field_name for field_name in FIELD_ORDER if field_name not in split]
    if missing:
        raise ValueError(f"Split {split_name!r} is missing required fields: {missing}.")
    sample_count = None
    for field_name in FIELD_ORDER:
        values = split[field_name]
        shape = tuple(int(dim) for dim in values.shape)
        if len(shape) < 1:
            raise ValueError(f"Split {split_name!r} field {field_name!r} must have a batch dimension, got {shape}.")
        if shape[0] == 0:
            raise ValueError(f"Split {split_name!r} is empty and cannot be used for training.")
        if sample_count is None:
            sample_count = shape[0]
        elif shape[0] != sample_count:
            raise ValueError(f"Split {split_name!r} fields do not share the same first dimension.")


def validate_training_splits(dataset_bundle: dict[str, Any]) -> None:
    for split_name in TRAIN_SPLIT_NAMES:
        if split_name not in dataset_bundle["splits"]:
            raise ValueError(f"Training requires split {split_name!r}.")
        _validate_split(split_name, dataset_bundle["splits"][split_name])

def build_torch_splits(dataset_bundle: dict[str, Any], *, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    return {
        split_name: {field_name: torch.from_numpy(split[field_name]).to(device) for field_name in FIELD_ORDER}
        for split_name, split in dataset_bundle["splits"].items()
    }


def split_size(split: dict[str, torch.Tensor]) -> int:
    return int(split[FIELD_ORDER[0]].shape[0])


def iter_split_batches(
    split: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = split_size(split)
    if shuffle:
        indices = torch.randperm(size, device=split[FIELD_ORDER[0]].device)
        for start in range(0, size, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield split["phi9"].index_select(0, batch_indices), split["hkappa_target"].index_select(0, batch_indices)
    else:
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            yield split["phi9"][start:end], split["hkappa_target"][start:end]


def make_autocast_context(*, device: torch.device, amp_enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
    return nullcontext()


def maybe_warmup_compiled_training(
    model: nn.Module,
    split: dict[str, torch.Tensor],
    *,
    train_config: Any,
    device: torch.device,
    amp_enabled: bool,
    profile_cuda_timing: bool = False,
    phase_log_path: str | Path | None = None,
) -> None:
    if device.type != "cuda":
        return
    warmup_count = min(int(train_config.batch_size), split_size(split))
    if warmup_count <= 0:
        return
    emit_phase_event(phase_log_path, "compile_warmup_begin", warmup_samples=warmup_count)
    checkpoint_model = unwrap_model(model)
    saved_state = {key: value.detach().cpu().clone() for key, value in checkpoint_model.state_dict().items()}
    warmup_optimizer = create_optimizer(checkpoint_model, train_config.lr)
    warmup_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    phi9_batch = split["phi9"][:warmup_count]
    target_batch = split["hkappa_target"][:warmup_count]
    warmup_optimizer.zero_grad(set_to_none=True)
    with make_autocast_context(device=device, amp_enabled=amp_enabled):
        _, loss = model.predict_and_loss(phi9_batch, target_batch)
    if warmup_scaler.is_enabled():
        warmup_scaler.scale(loss).backward()
        warmup_scaler.step(warmup_optimizer)
        warmup_scaler.update()
    else:
        loss.backward()
        warmup_optimizer.step()
    warmup_optimizer.zero_grad(set_to_none=True)
    checkpoint_model.load_state_dict(saved_state)
    if profile_cuda_timing and device.type == "cuda":
        torch.cuda.synchronize(device)
    emit_phase_event(phase_log_path, "compile_warmup_end", warmup_samples=warmup_count)


def run_epoch(
    model: nn.Module,
    split: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    grad_accum_steps: int = 1,
    profile_cuda_timing: bool = False,
    phase_log_path: str | Path | None = None,
    phase_name: str | None = None,
    epoch: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_count = 0
    phase_started = False
    accum_count = 0
    for phi9_batch, target_batch in iter_split_batches(split, batch_size=batch_size, shuffle=shuffle):
        if not phase_started and phase_name is not None and epoch is not None:
            emit_phase_event(phase_log_path, f"{phase_name}_epoch_begin", epoch=int(epoch))
            phase_started = True
        current_batch_size = int(phi9_batch.shape[0])
        if training:
            if accum_count == 0:
                optimizer.zero_grad(set_to_none=True)
        with make_autocast_context(device=device, amp_enabled=amp_enabled):
            _, batch_loss = model.predict_and_loss(phi9_batch, target_batch)
        if training:
            scaled_loss = batch_loss / grad_accum_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        total_loss = total_loss + batch_loss.detach().to(dtype=torch.float32) * current_batch_size
        total_count += current_batch_size
        if training:
            accum_count += 1
            if accum_count >= grad_accum_steps:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                accum_count = 0
    if training and accum_count > 0:
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    if total_count == 0:
        raise ValueError("Encountered an empty split during training.")
    if phase_started and phase_name is not None and epoch is not None:
        if profile_cuda_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        emit_phase_event(
            phase_log_path,
            f"{phase_name}_epoch_end",
            epoch=int(epoch),
            sample_count=int(total_count),
        )
    return float((total_loss / total_count).item())


def train_model(
    model: nn.Module,
    *,
    bundle: dict[str, Any],
    train_config: Any,
    model_type: str,
    checkpoint_path: str | Path,
    checkpoint_model: nn.Module,
    device: torch.device,
    swanlab_run: Any | None = None,
    amp_enabled: bool = False,
    phase_log_path: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    validate_training_splits(bundle)
    optimizer = create_optimizer(checkpoint_model, train_config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    wait = 0
    history = {"train_hk": [], "val_hk": [], "best_epoch": -1, "best_checkpoint_path": str(Path(checkpoint_path).resolve())}
    for epoch in range(train_config.max_epochs):
        train_loss = run_epoch(
            model,
            bundle["splits"]["train"],
            batch_size=train_config.batch_size,
            shuffle=True,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_accum_steps=train_config.grad_accum_steps,
            profile_cuda_timing=train_config.profile_cuda_timing,
            phase_log_path=phase_log_path,
            phase_name="train",
            epoch=epoch,
        )
        val_loss = run_epoch(
            model,
            bundle["splits"]["val"],
            batch_size=train_config.batch_size,
            shuffle=False,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            grad_accum_steps=1,
            profile_cuda_timing=train_config.profile_cuda_timing,
            phase_log_path=phase_log_path,
            phase_name="val",
            epoch=epoch,
        )
        history["train_hk"].append(train_loss)
        history["val_hk"].append(val_loss)
        print(f"Epoch {epoch:03d} | Train hk MSE: {train_loss:.6e} | Val hk MSE: {val_loss:.6e}")
        if swanlab_run is not None:
            swanlab_run.log({
                "train/hk_mse": train_loss,
                "val/hk_mse": val_loss,
                "monitor/best_val_hk_mse": min(best_val, val_loss),
            }, step=epoch + 1)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in checkpoint_model.state_dict().items()}
            save_checkpoint_bundle(checkpoint_model, model_type=model_type, model_config=train_config, path=checkpoint_path)
            emit_phase_event(
                phase_log_path,
                "checkpoint_saved",
                epoch=int(epoch),
                checkpoint_path=str(Path(checkpoint_path).resolve()),
                kind="best_so_far",
            )
        else:
            wait += 1
            if wait >= train_config.patience:
                print(f"Early stopping at epoch {epoch} with patience={train_config.patience}.")
                emit_phase_event(phase_log_path, "early_stopping", epoch=int(epoch), patience=int(train_config.patience))
                break
    if best_state is not None:
        checkpoint_model.load_state_dict(best_state)
        save_checkpoint_bundle(checkpoint_model, model_type=model_type, model_config=train_config, path=checkpoint_path)
        emit_phase_event(
            phase_log_path,
            "checkpoint_saved",
            epoch=int(best_epoch),
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            kind="final_best",
        )
    history["best_epoch"] = best_epoch
    return model, history


def describe_dataset_bundle(dataset_bundle: dict[str, Any], *, batch_size: int) -> None:
    data_cfg = dataset_bundle["config"]
    generation_cfg = dataset_bundle["generation_config"]
    print("Task: 3x3 phi stencil -> h*kappa")
    print(f"Number of blueprints: {len(dataset_bundle['blueprints'])}")
    print(f"Shape types: {getattr(data_cfg, 'shape_types', ('circle',))}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field types: {data_cfg.initial_field_types}")
    print(f"Dataset generation batch size: {generation_cfg.generation_batch_size}")
    print(f"Training batch size: {batch_size}")
    print(f"Blueprint split fractions: train={data_cfg.train_fraction}, val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}")
    for split_name, split_size in dataset_bundle["sizes"].items():
        print(f"{split_name:>5s}: {split_size}")


def build_swanlab_config(*, train_config: Any, dataset_bundle: dict[str, Any], dataset_path: str | Path, checkpoint_path: str | Path, device: torch.device, parameter_count: int, model_type: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task": "3x3 phi stencil -> h*kappa",
        "model_type": model_type,
        "dataset_path": str(Path(dataset_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "parameter_count": int(parameter_count),
    }
    config.update({f"train/{key}": str(value) if isinstance(value, Path) else value for key, value in asdict(train_config).items()})
    config.update({f"data/{key}": value for key, value in asdict(dataset_bundle["config"]).items()})
    config.update({f"generation/{key}": str(value) if isinstance(value, Path) else value for key, value in asdict(dataset_bundle["generation_config"]).items()})
    config.update({f"split_sizes/{key}": int(value) for key, value in dataset_bundle.get("sizes", {}).items()})
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    mlp_defaults = MLP_TrainConfig()
    cnn_defaults = CNN_TrainConfig()
    parser = argparse.ArgumentParser(description="Train the 3x3 stencil h*kappa predictor.")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--dataset-output", type=str, default=str(default_dataset_path()))
    parser.add_argument("--output-model", type=str, default=str(default_output_model_path()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-units", type=int, default=mlp_defaults.hidden_units)
    parser.add_argument("--kernel-size", type=int, default=cnn_defaults.kernel_size)
    parser.add_argument("--padding", type=int, default=cnn_defaults.padding)
    parser.add_argument("--activation", type=str, default=mlp_defaults.activation, choices=["relu", "tanh", "silu"])
    parser.add_argument("--lr", type=float, default=mlp_defaults.lr)
    parser.add_argument("--max-epochs", type=int, default=mlp_defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=mlp_defaults.patience)
    parser.add_argument("--batch-size", type=int, default=mlp_defaults.batch_size)
    parser.add_argument("--seed", type=int, default=mlp_defaults.seed)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="PINN")
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-description", type=str, default="")
    parser.add_argument("--swanlab-tags", type=str, default="")
    parser.add_argument("--swanlab-group", type=str, default="")
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument("--swanlab-mode", type=str, choices=("cloud", "local", "offline", "disabled"), default="cloud")
    parser.add_argument("--phase-log-path", type=str, default="")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="none",
                        choices=["none", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--profile-cuda-timing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = Path(args.dataset_output)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path.resolve()}\n"
            "Generate the dataset first with `python -m train_generate.generate`."
        )
    model_type: ModelType = args.model_type  # type: ignore[assignment]
    overrides = {
        "activation": args.activation,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset_path": dataset_path,
        "output_model_path": Path(args.output_model),
        "compile_mode": args.compile_mode,
        "grad_accum_steps": args.grad_accum_steps,
        "profile_cuda_timing": args.profile_cuda_timing,
    }
    if model_type == "mlp":
        overrides["hidden_units"] = args.hidden_units
    else:
        overrides["kernel_size"] = args.kernel_size
        overrides["padding"] = args.padding
    train_config = create_train_config(model_type, **overrides)
    raw_bundle = load_training_arrays_from_hdf5(dataset_path)
    validate_training_splits(raw_bundle)
    mean, std = compute_phi9_normalization(raw_bundle["splits"]["train"]["phi9"])
    stats_csv_path = save_phi9_normalization_csv(normalization_csv_path(train_config.output_model_path), mean=mean, std=std, dataset_path=dataset_path)
    normalized_bundle = apply_phi9_normalization(raw_bundle, mean=mean, std=std)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_log_path = Path(args.phase_log_path).resolve() if args.phase_log_path else None
    if phase_log_path is not None:
        phase_log_path.parent.mkdir(parents=True, exist_ok=True)
        phase_log_path.write_text("", encoding="utf-8")
        emit_phase_event(
            phase_log_path,
            "run_initialized",
            dataset_path=str(dataset_path.resolve()),
            output_model=str(Path(args.output_model).resolve()),
            model_type=str(model_type),
        )
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    compile_enabled = device.type == "cuda" and not args.disable_compile and hasattr(torch, "compile")
    compile_mode = args.compile_mode if not args.disable_compile else "none"
    bundle = {**normalized_bundle, "splits": build_torch_splits(normalized_bundle, device=device)}
    checkpoint_model = create_model(train_config).to(device)
    model = checkpoint_model
    if compile_enabled and compile_mode != "none":
        model = torch.compile(checkpoint_model, mode=compile_mode)
    emit_phase_event(
        phase_log_path,
        "data_ready",
        device=str(device),
        train_size=int(normalized_bundle["sizes"]["train"]),
        val_size=int(normalized_bundle["sizes"]["val"]),
        test_size=int(normalized_bundle["sizes"].get("test", 0)),
        batch_size=int(train_config.batch_size),
        amp_enabled=bool(amp_enabled),
        compile_enabled=bool(compile_enabled),
    )
    if compile_enabled and compile_mode != "none":
        maybe_warmup_compiled_training(
            model,
            bundle["splits"]["train"],
            train_config=train_config,
            device=device,
            amp_enabled=amp_enabled,
            profile_cuda_timing=train_config.profile_cuda_timing,
            phase_log_path=phase_log_path,
        )
    print(f"Loading existing dataset from: {dataset_path.resolve()}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Compile enabled: {compile_enabled}" + (f" (mode: {compile_mode})" if compile_enabled and compile_mode != "none" else f" (mode: {compile_mode})"))
    print(f"Gradient accumulation steps: {train_config.grad_accum_steps}")
    print(f"Profile CUDA timing: {train_config.profile_cuda_timing}")
    print(f"Dataset file: {dataset_path.resolve()}")
    print(f"Parameter count: {count_parameters(checkpoint_model)}")
    describe_dataset_bundle(normalized_bundle, batch_size=train_config.batch_size)
    print("Phi9 normalization source split: train")
    print(f"Phi9 normalization CSV: {stats_csv_path.resolve()}")
    swanlab_run = None
    if args.use_swanlab:
        swanlab_run = init_swanlab_run(
            project=args.swanlab_project or None,
            experiment_name=args.swanlab_experiment_name or None,
            description=args.swanlab_description or None,
            tags=csv_to_list(args.swanlab_tags),
            group=args.swanlab_group or None,
            workspace=args.swanlab_workspace or None,
            logdir=args.swanlab_logdir or None,
            mode=args.swanlab_mode or None,
            config=build_swanlab_config(
                train_config=train_config,
                dataset_bundle=normalized_bundle,
                dataset_path=dataset_path,
                checkpoint_path=args.output_model,
                device=device,
                parameter_count=count_parameters(checkpoint_model),
                model_type=model_type,
            ),
        )
    emit_phase_event(phase_log_path, "training_begin")
    _, history = train_model(
        model,
        bundle=bundle,
        train_config=train_config,
        model_type=model_type,
        checkpoint_path=args.output_model,
        checkpoint_model=checkpoint_model,
        device=device,
        swanlab_run=swanlab_run,
        amp_enabled=amp_enabled,
        phase_log_path=phase_log_path,
    )
    emit_phase_event(phase_log_path, "training_end", best_epoch=int(history["best_epoch"]))
    if swanlab_run is not None:
        swanlab_run.log({
            "summary/best_epoch": int(history["best_epoch"]),
            "summary/best_val_hk_mse": float(min(history["val_hk"])),
            "summary/final_val_hk_mse": float(history["val_hk"][-1]),
        })
        swanlab_run.finish()
    emit_phase_event(phase_log_path, "run_complete", best_epoch=int(history["best_epoch"]))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best checkpoint: {Path(history['best_checkpoint_path']).resolve()}")


if __name__ == "__main__":
    main()
