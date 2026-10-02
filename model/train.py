from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.shared import (
    add_swanlab_args,
    apply_feature_transform,
    csv_to_list,
    fit_feature_transform,
    init_swanlab_run,
    normalization_csv_path,
    save_checkpoint_bundle,
    save_feature_stats,
)
from model.config import (
    CNN_TrainConfig,
    MLP_TrainConfig,
    ModelType,
    TrainConfig,
    create_train_config,
)
from model.model import HKappaCNN, HKappaStencilNet
from train_generate.io import load_training_arrays_from_hdf5

TRAIN_SPLIT_NAMES = ("train", "val")

# ── Phase / event logging

def emit_phase_event(path: Path | None, event: str, **payload: Any) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_ns": time.time_ns(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")

# ── Data validation

def _validate_split(split_name: str, split: dict[str, Any], *, require_phi9: bool = True) -> None:
    required = ("phi9", "features", "hkappa_target") if require_phi9 else ("features", "hkappa_target")
    for field in required:
        if field not in split:
            raise ValueError(f"Split {split_name!r} is missing required field {field!r}.")
    if require_phi9 and (split["phi9"].ndim != 2 or split["phi9"].shape[1] != 9):
        raise ValueError(f"Split {split_name!r} phi9 must have shape (N, 9), got {split['phi9'].shape}.")
    feat_dim = int(split["features"].shape[1]) if split["features"].ndim == 2 else -1
    if split["features"].ndim != 2 or feat_dim not in (9, 11, 15, 18, 19, 27):
        raise ValueError(
            f"Split {split_name!r} features must have shape (N, 9), (N, 11), (N, 15), (N, 18), (N, 19), or (N, 27), got {split['features'].shape}."
        )
    n = int(split["features"].shape[0])
    if n == 0:
        raise ValueError(f"Split {split_name!r} is empty.")
    for key in ("hkappa_target", *(("phi9",) if require_phi9 else ())):
        if int(split[key].shape[0]) != n:
            raise ValueError(f"Split {split_name!r}: field {key!r} length mismatch.")

def validate_training_splits(bundle: dict[str, Any], *, require_phi9: bool = True) -> None:
    for name in TRAIN_SPLIT_NAMES:
        if name not in bundle["splits"]:
            raise ValueError(f"Training requires split {name!r}.")
        _validate_split(name, bundle["splits"][name], require_phi9=require_phi9)

# ── Feature-transform application

def transform_dataset_bundle(bundle: dict[str, Any], *, feature_transform: dict[str, Any]) -> dict[str, Any]:
    splits = {
        name: {
            "phi9":          np.asarray(s["phi9"],          dtype=np.float32),
            "features":      apply_feature_transform(s["features"], feature_transform),
            "hkappa_target": np.asarray(s["hkappa_target"], dtype=np.float32),
        }
        for name, s in bundle["splits"].items()
    }
    sizes = {name: int(s["features"].shape[0]) for name, s in splits.items()}
    return {**bundle, "splits": splits, "sizes": sizes}

def train_loop(dataloader, model, criterion, optimizer, device, scaler):
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    amp_enabled = scaler.is_enabled()

    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions = model(features)
            loss = criterion(predictions, targets)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Weight by sample count so a short final batch is counted correctly.
        total_loss += loss.detach().float() * len(features)

    return (total_loss / len(dataloader.dataset)).item()


def val_loop(dataloader, model, criterion, device, amp_enabled):
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)

    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(features)
                loss = criterion(predictions, targets)
            total_loss += loss.float() * len(features)

    return (total_loss / len(dataloader.dataset)).item()


# ── Human-readable run summary

def describe_dataset_bundle(
    bundle: dict[str, Any], *, batch_size: int, feature_transform: dict[str, Any],
) -> None:
    dc  = bundle["config"]
    gc  = bundle["generation_config"]
    frac = 1.0 - dc.train_fraction - dc.val_fraction
    rows: list[tuple[str, Any]] = [
        ("Task",              "3x3 stencil features -> h*kappa"),
        ("Blueprints",        len(bundle["blueprints"])),
        ("Shape types",       getattr(dc, "shape_types", ("circle",))),
        ("Resolutions",       dc.resolutions),
        ("Field types",       dc.initial_field_types),
        ("Feature order",     feature_transform.get("feature_order", "phi9")),
        ("Raw / model dim",   f"{feature_transform['raw_feature_dim']} / {feature_transform['output_dim']}"),
        ("Transform",         feature_transform["transform_kind"]),
        ("Gen batch size",    gc.generation_batch_size),
        ("Train batch size",  batch_size),
        ("Split fractions",   f"train={dc.train_fraction}  val={dc.val_fraction}  test={frac:.3f}"),
        *((f"  {name}", size) for name, size in bundle["sizes"].items()),
    ]
    for label, value in rows:
        print(f"  {label:<24s} {value}")

# ── SwanLab config builder

def _path_str(v: Any) -> Any:
    return str(v) if isinstance(v, Path) else v

def build_swanlab_config(
    *,
    train_config: TrainConfig,
    bundle: dict[str, Any],
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    normalization_output: str | Path,
    feature_transform: dict[str, Any],
    device: torch.device,
    parameter_count: int,
    model_type: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "task":                     "3x3 stencil features -> h*kappa",
        "model_type":               model_type,
        "dataset_path":             str(Path(dataset_path).resolve()),
        "checkpoint_path":          str(Path(checkpoint_path).resolve()),
        "normalization_csv_path":   str(Path(normalization_output).resolve()),
        "device":                   str(device),
        "parameter_count":          int(parameter_count),
        "transform/kind":           feature_transform["transform_kind"],
        "transform/raw_feature_dim": int(feature_transform["raw_feature_dim"]),
        "transform/output_dim":     int(feature_transform["output_dim"]),
    }
    cfg.update({f"train/{k}": _path_str(v) for k, v in asdict(train_config).items()})
    cfg.update({f"data/{k}":  v            for k, v in asdict(bundle["config"]).items()})
    cfg.update({f"gen/{k}":   _path_str(v) for k, v in asdict(bundle["generation_config"]).items()})
    cfg.update({f"size/{k}":  int(v)       for k, v in bundle.get("sizes", {}).items()})
    return cfg

# ── main() helpers

def setup_device(args: argparse.Namespace) -> torch.device:
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device

def load_and_transform_dataset(
    args: argparse.Namespace, *, dataset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if args.use_pca:
        from train_generate.pca_dataset import ensure_pca_dataset

        # The PCA sidecar is already projected; do not transform it again.
        pca_path, transform_path, feature_transform = ensure_pca_dataset(dataset_path)
        bundle = load_training_arrays_from_hdf5(pca_path)
        validate_training_splits(bundle)
        return bundle, feature_transform, Path(transform_path)
    raw_bundle = load_training_arrays_from_hdf5(dataset_path)
    validate_training_splits(raw_bundle)
    feature_transform = fit_feature_transform(
        raw_bundle["splits"]["train"]["features"], dataset_path=dataset_path,
    )
    explicit_output = args.feature_transform or args.normalization_csv
    norm_output = Path(explicit_output) if explicit_output else normalization_csv_path(args.output_model_path)
    saved_norm_path = save_feature_stats(norm_output, transform=feature_transform, dataset_path=dataset_path)
    return transform_dataset_bundle(raw_bundle, feature_transform=feature_transform), feature_transform, saved_norm_path


# ── Argument parser

def build_arg_parser() -> argparse.ArgumentParser:
    mlp = MLP_TrainConfig()
    cnn = CNN_TrainConfig()
    p   = argparse.ArgumentParser(description="Train the 3x3 stencil h*kappa predictor.")
    # paths
    p.add_argument("--model-type",       default="mlp",                  choices=["mlp", "cnn"])
    p.add_argument("--dataset-output", dest="dataset_path", type=Path, default=mlp.dataset_path)
    p.add_argument("--output-model", dest="output_model_path", type=Path, default=mlp.output_model_path)
    p.add_argument("--feature-transform", default="",
                   help="Explicit feature-transform sidecar output (.csv for non-PCA). Supersedes --normalization-csv.")
    p.add_argument("--normalization-csv", default="", help="Deprecated alias of --feature-transform (non-PCA CSV only).")
    p.add_argument("--use-pca", action="store_true",
                   help="MLP only: train on the offline PCA-18 (V3) projection of a V2 27D dataset.")
    p.add_argument("--device",           default="")
    # architecture
    p.add_argument("--hidden-units",  type=int, default=mlp.hidden_units)
    p.add_argument("--kernel-size",   type=int, default=cnn.kernel_size)
    p.add_argument("--padding",       type=int, default=cnn.padding)
    # optimizer / loss  (defaults come from config — single source of truth)
    p.add_argument("--optimizer-type", default=mlp.optimizer_type, choices=["adamw", "adam", "sgd"])
    p.add_argument("--loss-fn",        default=mlp.loss_fn,        choices=["mse", "mae", "huber"])
    # training hyperparameters
    p.add_argument("--lr",          type=float, default=mlp.lr)
    p.add_argument("--l2-reg",      type=float, default=mlp.l2_reg)
    p.add_argument("--max-epochs",  type=int,   default=mlp.max_epochs)
    p.add_argument("--patience",    type=int,   default=mlp.patience)
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=mlp.seed)
    # infrastructure
    p.add_argument("--phase-log-path",    default="")
    p.add_argument("--disable-amp",       action="store_true")
    p.add_argument("--disable-compile",   action="store_true")
    p.add_argument("--compile-mode",      default=mlp.compile_mode,
                   choices=["none", "default", "reduce-overhead", "max-autotune"],
                   help="CUDA torch.compile mode; the first epoch includes compilation. Use none to disable.")
    p.add_argument("--profile-cuda-timing", action="store_true")
    add_swanlab_args(p)
    return p

# ── Entry point

def main() -> None:
    args       = build_arg_parser().parse_args()
    model_type: ModelType = args.model_type
    dataset_path = Path(args.dataset_path)

    if model_type == "cnn" and args.use_pca:
        raise ValueError("CNN does not support --use-pca; PCA-18 is an MLP-only feature path.")

    device = setup_device(args)

    transformed_bundle, feature_transform, saved_norm_path = load_and_transform_dataset(
        args, dataset_path=dataset_path,
    )
    model_input_dim = int(transformed_bundle["splits"]["train"]["features"].shape[1])
    if model_type == "cnn" and model_input_dim != 9:
        raise ValueError(
            f"CNN requires 9D features, got model input dim={model_input_dim}. "
            "Use --model-type mlp for 15D (V6), 19D (V5), 11D (V4), 27D (V2), or PCA-18 (V3) datasets."
        )

    overrides = {key: value for key, value in vars(args).items()
                 if key != "model_type" and value is not None}
    train_config = create_train_config(
        model_type, **overrides,
        input_dim=int(feature_transform["output_dim"]),
        raw_feature_dim=int(feature_transform["raw_feature_dim"]),
    )
    torch.manual_seed(train_config.seed)

    if model_type == "mlp":
        model = HKappaStencilNet(train_config).to(device)
    else:
        model = HKappaCNN(train_config).to(device)
    checkpoint_model = model
    checkpoint_path = train_config.output_model_path.resolve()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    amp_enabled      = device.type == "cuda" and not args.disable_amp
    compile_mode     = args.compile_mode if not args.disable_compile else "none"
    compile_enabled = device.type == "cuda" and compile_mode != "none"
    model = (
        torch.compile(checkpoint_model, mode=compile_mode)
        if compile_enabled
        else checkpoint_model
    )

    # Optional phase log
    phase_log_path: Path | None = None
    if args.phase_log_path:
        phase_log_path = Path(args.phase_log_path).resolve()
        phase_log_path.parent.mkdir(parents=True, exist_ok=True)
        phase_log_path.write_text("", encoding="utf-8")
        emit_phase_event(phase_log_path, "run_initialized",
                         dataset_path=str(dataset_path.resolve()),
                         output_model=str(Path(args.output_model_path).resolve()),
                         model_type=str(model_type))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_split = transformed_bundle["splits"]["train"]
    val_split = transformed_bundle["splits"]["val"]
    train_dataset = TensorDataset(
        torch.as_tensor(train_split["features"], dtype=torch.float32),
        torch.as_tensor(train_split["hkappa_target"], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.as_tensor(val_split["features"], dtype=torch.float32),
        torch.as_tensor(val_split["hkappa_target"], dtype=torch.float32),
    )
    batch_size = train_config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if train_config.loss_fn == "mse":
        criterion = nn.MSELoss()
    elif train_config.loss_fn == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss()

    lr = train_config.lr
    l2_reg = train_config.l2_reg
    if train_config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif train_config.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg, momentum=0.9)

    print(f"Dataset:    {dataset_path.resolve()}")
    print(f"Model:      {model_type}  params={parameter_count}")
    print(f"Device:     {device}  AMP={amp_enabled}  compile={compile_enabled}({compile_mode})")
    print(f"Optimizer:  {train_config.optimizer_type}  lr={train_config.lr}  wd={train_config.l2_reg}")
    print(f"Loss fn:    {train_config.loss_fn}")
    print(f"Norm CSV:   {saved_norm_path.resolve()}")
    describe_dataset_bundle(transformed_bundle, batch_size=train_config.batch_size,
                            feature_transform=feature_transform)

    emit_phase_event(phase_log_path, "data_ready",
                     device=str(device), train_size=int(transformed_bundle["sizes"]["train"]),
                     val_size=int(transformed_bundle["sizes"]["val"]),
                     test_size=int(transformed_bundle["sizes"].get("test", 0)),
                     batch_size=int(train_config.batch_size),
                     amp_enabled=bool(amp_enabled), compile_enabled=bool(compile_enabled))
    # SwanLab  (optional)
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
                train_config=train_config, bundle=transformed_bundle,
                dataset_path=dataset_path, checkpoint_path=args.output_model_path,
                normalization_output=saved_norm_path, feature_transform=feature_transform,
                device=device, parameter_count=parameter_count,
                model_type=model_type,
            ),
        )

    emit_phase_event(phase_log_path, "training_begin")
    best_val   = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    wait       = 0
    history: dict[str, Any] = {
        "train_hk": [], "val_hk": [], "best_epoch": -1,
        "best_checkpoint_path": str(checkpoint_path.resolve()),
    }
    for epoch in range(train_config.max_epochs):
        emit_phase_event(phase_log_path, "train_epoch_begin", epoch=epoch)
        train_loss = train_loop(train_loader, model, criterion, optimizer, device, scaler)
        if train_config.profile_cuda_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        emit_phase_event(phase_log_path, "train_epoch_end", epoch=epoch, sample_count=len(train_dataset))

        emit_phase_event(phase_log_path, "val_epoch_begin", epoch=epoch)
        val_loss = val_loop(val_loader, model, criterion, device, amp_enabled)
        if train_config.profile_cuda_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        emit_phase_event(phase_log_path, "val_epoch_end", epoch=epoch, sample_count=len(val_dataset))

        history["train_hk"].append(train_loss)
        history["val_hk"].append(val_loss)
        print(f"Epoch {epoch:03d} | Train hk loss: {train_loss:.8e} | Val hk loss: {val_loss:.8e}")

        if swanlab_run is not None:
            swanlab_run.log({
                "train/hk_loss":            train_loss,
                "val/hk_loss":              val_loss,
                "monitor/best_val_hk_loss": min(best_val, val_loss),
            }, step=epoch + 1)

        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in checkpoint_model.state_dict().items()}
            save_checkpoint_bundle(checkpoint_model, model_type=model_type,
                                   model_config=train_config, path=checkpoint_path,
                                   feature_transform=feature_transform)
            emit_phase_event(phase_log_path, "checkpoint_saved", epoch=int(epoch),
                             checkpoint_path=str(checkpoint_path.resolve()), kind="best_so_far")
        else:
            wait += 1
            if wait >= train_config.patience:
                print(f"Early stopping at epoch {epoch} with patience={train_config.patience}.")
                emit_phase_event(phase_log_path, "early_stopping",
                                 epoch=int(epoch), patience=int(train_config.patience))
                break

    if best_state is not None:
        checkpoint_model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    emit_phase_event(phase_log_path, "training_end", best_epoch=int(history["best_epoch"]))

    if swanlab_run is not None:
        swanlab_run.log({
            "summary/best_epoch":        int(history["best_epoch"]),
            "summary/best_val_hk_loss":  float(min(history["val_hk"])),
            "summary/final_val_hk_loss": float(history["val_hk"][-1]),
        })
        swanlab_run.finish()
    emit_phase_event(phase_log_path, "run_complete", best_epoch=int(history["best_epoch"]))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best checkpoint: {Path(history['best_checkpoint_path']).resolve()}")


if __name__ == "__main__":
    main()
