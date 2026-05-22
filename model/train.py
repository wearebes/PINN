from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    from evaluate.shared import (
        apply_feature_transform,
        csv_to_list,
        fit_feature_transform,
        init_swanlab_run,
        normalization_csv_path,
        save_checkpoint_bundle,
        save_feature_stats,
    )
    from model.config import CNN_TrainConfig, MLP_TrainConfig, ModelType, create_train_config, default_dataset_path, default_output_model_path
    from model.model import count_parameters, create_model
    from train_generate.io import load_training_arrays_from_hdf5
else:
    from evaluate.shared import (
        apply_feature_transform,
        csv_to_list,
        fit_feature_transform,
        init_swanlab_run,
        normalization_csv_path,
        save_checkpoint_bundle,
        save_feature_stats,
    )
    from .config import CNN_TrainConfig, MLP_TrainConfig, ModelType, create_train_config, default_dataset_path, default_output_model_path
    from .model import count_parameters, create_model
    from train_generate.io import load_training_arrays_from_hdf5


TRAIN_SPLIT_NAMES = ("train", "val")
TRAINING_FIELD_NAMES = ("features", "hkappa_target")


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


def _validate_split(split_name: str, split: dict[str, Any], *, require_phi9: bool = True) -> None:
    required_fields = ("features", "hkappa_target") if not require_phi9 else ("phi9", "features", "hkappa_target")
    for field_name in required_fields:
        if field_name not in split:
            raise ValueError(f"Split {split_name!r} is missing required field {field_name!r}.")
    if require_phi9:
        if split["phi9"].ndim != 2 or split["phi9"].shape[1] != 9:
            raise ValueError(f"Split {split_name!r} phi9 must have shape (N, 9), got {split['phi9'].shape}.")
    if split["features"].ndim != 2 or split["features"].shape[1] != 9:
        raise ValueError(f"Split {split_name!r} features must have shape (N, 9), got {split['features'].shape}.")
    sample_count = int(split["features"].shape[0])
    if sample_count == 0:
        raise ValueError(f"Split {split_name!r} is empty and cannot be used for training.")
    if int(split["hkappa_target"].shape[0]) != sample_count:
        raise ValueError(f"Split {split_name!r} fields do not share the same first dimension.")
    if require_phi9 and int(split["phi9"].shape[0]) != sample_count:
        raise ValueError(f"Split {split_name!r} fields do not share the same first dimension.")


def validate_training_splits(dataset_bundle: dict[str, Any], *, require_phi9: bool = True) -> None:
    for split_name in TRAIN_SPLIT_NAMES:
        if split_name not in dataset_bundle["splits"]:
            raise ValueError(f"Training requires split {split_name!r}.")
        _validate_split(split_name, dataset_bundle["splits"][split_name], require_phi9=require_phi9)


def transform_dataset_bundle(dataset_bundle: dict[str, Any], *, feature_transform: dict[str, Any]) -> dict[str, Any]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split in dataset_bundle["splits"].items():
        splits[split_name] = {
            "phi9": np.asarray(split["phi9"], dtype=np.float32),
            "features": apply_feature_transform(split["features"], feature_transform),
            "hkappa_target": np.asarray(split["hkappa_target"], dtype=np.float32),
        }
    return {**dataset_bundle, "splits": splits, "sizes": {name: int(split['features'].shape[0]) for name, split in splits.items()}}


def build_torch_splits(dataset_bundle: dict[str, Any], *, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, split in dataset_bundle["splits"].items():
        tensors: dict[str, torch.Tensor] = {}
        for field_name in TRAINING_FIELD_NAMES:
            arr = np.asarray(split[field_name], dtype=np.float32)
            t = torch.from_numpy(arr)
            if device.type == "cuda":
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                t = t.to(device, non_blocking=True)
            else:
                t = t.to(device)
            tensors[field_name] = t
        splits[split_name] = tensors
    return splits

def split_size(split: dict[str, torch.Tensor]) -> int:
    return int(split["features"].shape[0])


def iter_split_batches(
    split: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
):
    size = split_size(split)
    if shuffle:
        indices = torch.randperm(size, device=split["features"].device)
        for start in range(0, size, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield split["features"].index_select(0, batch_indices), split["hkappa_target"].index_select(0, batch_indices)
    else:
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            yield split["features"][start:end], split["hkappa_target"][start:end]


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
    feature_batch = split["features"][:warmup_count]
    target_batch = split["hkappa_target"][:warmup_count]
    warmup_optimizer.zero_grad(set_to_none=True)
    with make_autocast_context(device=device, amp_enabled=amp_enabled):
        _, loss = model.predict_and_loss(feature_batch, target_batch)
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
    for feature_batch, target_batch in iter_split_batches(split, batch_size=batch_size, shuffle=shuffle):
        if not phase_started and phase_name is not None and epoch is not None:
            emit_phase_event(phase_log_path, f"{phase_name}_epoch_begin", epoch=int(epoch))
            phase_started = True
        current_batch_size = int(feature_batch.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with make_autocast_context(device=device, amp_enabled=amp_enabled):
            _, batch_loss = model.predict_and_loss(feature_batch, target_batch)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
        total_loss = total_loss + batch_loss.detach().to(dtype=torch.float32) * current_batch_size
        total_count += current_batch_size
        if training:
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
    feature_transform: dict[str, Any],
    device: torch.device,
    swanlab_run: Any | None = None,
    amp_enabled: bool = False,
    phase_log_path: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    validate_training_splits(bundle, require_phi9=False)
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
            save_checkpoint_bundle(
                checkpoint_model,
                model_type=model_type,
                model_config=train_config,
                path=checkpoint_path,
                feature_transform=feature_transform,
            )
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
        save_checkpoint_bundle(
            checkpoint_model,
            model_type=model_type,
            model_config=train_config,
            path=checkpoint_path,
            feature_transform=feature_transform,
        )
        emit_phase_event(
            phase_log_path,
            "checkpoint_saved",
            epoch=int(best_epoch),
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            kind="final_best",
        )
    history["best_epoch"] = best_epoch
    return model, history


def describe_dataset_bundle(dataset_bundle: dict[str, Any], *, batch_size: int, feature_transform: dict[str, Any]) -> None:
    data_cfg = dataset_bundle["config"]
    generation_cfg = dataset_bundle["generation_config"]
    print("Task: 3x3 stencil features -> h*kappa")
    print(f"Number of blueprints: {len(dataset_bundle['blueprints'])}")
    print(f"Shape types: {getattr(data_cfg, 'shape_types', ('circle',))}")
    print(f"Resolutions: {data_cfg.resolutions}")
    print(f"Initial field types: {data_cfg.initial_field_types}")
    print("Feature order: phi9")
    print(f"Raw feature dim: {int(feature_transform['raw_feature_dim'])}")
    print(f"Model input dim: {int(feature_transform['output_dim'])}")
    print(f"Transform kind: {feature_transform['transform_kind']}")
    print(f"Dataset generation batch size: {generation_cfg.generation_batch_size}")
    print(f"Training batch size: {batch_size}")
    print(f"Blueprint split fractions: train={data_cfg.train_fraction}, val={data_cfg.val_fraction}, test={1.0 - data_cfg.train_fraction - data_cfg.val_fraction}")
    for split_name, split_size in dataset_bundle["sizes"].items():
        print(f"{split_name:>5s}: {split_size}")


def build_swanlab_config(*, train_config: Any, dataset_bundle: dict[str, Any], dataset_path: str | Path, checkpoint_path: str | Path, normalization_output: str | Path, feature_transform: dict[str, Any], device: torch.device, parameter_count: int, model_type: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task": "3x3 stencil features -> h*kappa",
        "model_type": model_type,
        "dataset_path": str(Path(dataset_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "normalization_csv_path": str(Path(normalization_output).resolve()),
        "device": str(device),
        "parameter_count": int(parameter_count),
        "transform/kind": feature_transform["transform_kind"],
        "transform/raw_feature_dim": int(feature_transform["raw_feature_dim"]),
        "transform/output_dim": int(feature_transform["output_dim"]),
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
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--hidden-units", type=int, default=mlp_defaults.hidden_units)
    parser.add_argument("--kernel-size", type=int, default=cnn_defaults.kernel_size)
    parser.add_argument("--padding", type=int, default=cnn_defaults.padding)
    parser.add_argument("--normalization-csv", type=str, default="")
    parser.add_argument("--lr", type=float, default=mlp_defaults.lr)
    parser.add_argument("--max-epochs", type=int, default=mlp_defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=mlp_defaults.patience)
    parser.add_argument("--batch-size", type=int, default=None)
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
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["none", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--profile-cuda-timing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()   #传递参数
    dataset_path = Path(args.dataset_output)
    mlp_defaults = MLP_TrainConfig()
    cnn_defaults = CNN_TrainConfig()
    
    model_type: ModelType = args.model_type  # type: ignore[assignment]
    raw_bundle = load_training_arrays_from_hdf5(dataset_path)
    validate_training_splits(raw_bundle)
    raw_feature_dim = int(raw_bundle["splits"]["train"]["features"].shape[1])
    if raw_feature_dim != 9:
        raise ValueError(
            f"Training requires a V1 phi9 dataset with 9D features, got raw_feature_dim={raw_feature_dim}."
        )
    feature_transform = fit_feature_transform(
        raw_bundle["splits"]["train"]["features"],
        dataset_path=dataset_path,
    )
    normalization_output = Path(args.normalization_csv) if args.normalization_csv else normalization_csv_path(args.output_model)
    saved_normalization_path = save_feature_stats(normalization_output, transform=feature_transform, dataset_path=dataset_path)
    transformed_bundle = transform_dataset_bundle(raw_bundle, feature_transform=feature_transform)

    default_batch_size = mlp_defaults.batch_size if model_type == "mlp" else cnn_defaults.batch_size
    overrides = {
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size if args.batch_size is not None else default_batch_size,
        "seed": args.seed,
        "dataset_path": dataset_path,
        "output_model_path": Path(args.output_model),
        "compile_mode": args.compile_mode,
        "profile_cuda_timing": args.profile_cuda_timing,
        "input_dim": int(feature_transform["output_dim"]),
        "raw_feature_dim": raw_feature_dim,
    }
    if model_type == "mlp":
        overrides["hidden_units"] = args.hidden_units
    else:
        overrides["kernel_size"] = args.kernel_size
        overrides["padding"] = args.padding
    train_config = create_train_config(model_type, **overrides)

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
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    amp_enabled = device.type == "cuda" and not args.disable_amp
    compile_enabled = device.type == "cuda" and not args.disable_compile and hasattr(torch, "compile")
    compile_mode = args.compile_mode if not args.disable_compile else "none"
    bundle = {**transformed_bundle, "splits": build_torch_splits(transformed_bundle, device=device)}
    checkpoint_model = create_model(train_config).to(device)
    model = checkpoint_model
    if compile_enabled and compile_mode != "none":
        model = torch.compile(checkpoint_model, mode=compile_mode)
    emit_phase_event(
        phase_log_path,
        "data_ready",
        device=str(device),
        train_size=int(transformed_bundle["sizes"]["train"]),
        val_size=int(transformed_bundle["sizes"]["val"]),
        test_size=int(transformed_bundle["sizes"].get("test", 0)),
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
    print(f"Compile enabled: {compile_enabled} (mode: {compile_mode})")
    print(f"Profile CUDA timing: {train_config.profile_cuda_timing}")
    print(f"Dataset file: {dataset_path.resolve()}")
    print(f"Normalization CSV: {saved_normalization_path.resolve()}")
    print(f"Parameter count: {count_parameters(checkpoint_model)}")
    describe_dataset_bundle(transformed_bundle, batch_size=train_config.batch_size, feature_transform=feature_transform)
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
                dataset_bundle=transformed_bundle,
                dataset_path=dataset_path,
                checkpoint_path=args.output_model,
                normalization_output=saved_normalization_path,
                feature_transform=feature_transform,
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
        feature_transform=feature_transform,
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
