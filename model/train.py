from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

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
    default_dataset_path,
    default_output_model_path,
)
from model.model import count_parameters, create_loss_fn, create_model, create_optimizer
from train_generate.io import load_training_arrays_from_hdf5

TRAIN_SPLIT_NAMES = ("train", "val")

# argparse attribute names that differ from their TrainConfig field names
_CLI_RENAMES: dict[str, str] = {
    "dataset_output": "dataset_path",
    "output_model":   "output_model_path",
}

# ── Runtime context

@dataclass
class TrainContext:
    device:              torch.device
    amp_enabled:         bool
    scaler:              torch.amp.GradScaler
    compile_enabled:     bool
    compile_mode:        str
    profile_cuda_timing: bool
    phase_log_path:      Path | None = None

# ── Phase / event logging

def emit_phase_event(path: Path | None, event: str, **payload: Any) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_ns": time.time_ns(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")

# ── Model unwrapping  (compiled models have _orig_mod)

def unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "_orig_mod"):
        unwrapped = getattr(unwrapped, "_orig_mod")
    return unwrapped

# ── Data validation

def _validate_split(split_name: str, split: dict[str, Any], *, require_phi9: bool = True) -> None:
    required = ("phi9", "features", "hkappa_target") if require_phi9 else ("features", "hkappa_target")
    for field in required:
        if field not in split:
            raise ValueError(f"Split {split_name!r} is missing required field {field!r}.")
    if require_phi9 and (split["phi9"].ndim != 2 or split["phi9"].shape[1] != 9):
        raise ValueError(f"Split {split_name!r} phi9 must have shape (N, 9), got {split['phi9'].shape}.")
    feat_dim = int(split["features"].shape[1]) if split["features"].ndim == 2 else -1
    if split["features"].ndim != 2 or feat_dim not in (9, 18, 27):
        raise ValueError(
            f"Split {split_name!r} features must have shape (N, 9), (N, 18), or (N, 27), got {split['features'].shape}."
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

# ── DataLoader construction  (data pre-loaded onto device)

def build_split_loaders(
    bundle: dict[str, Any], *, device: torch.device, batch_size: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, split in bundle["splits"].items():
        feat_t = torch.from_numpy(np.asarray(split["features"],      dtype=np.float32)).to(device)
        tgt_t  = torch.from_numpy(np.asarray(split["hkappa_target"], dtype=np.float32)).to(device)
        loaders[name] = DataLoader(
            TensorDataset(feat_t, tgt_t),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )
    return loaders

# ── AMP autocast helper

def make_autocast_context(*, device: torch.device, amp_enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
    return nullcontext()

# ── Compiled-model warmup

def maybe_warmup_compiled_training(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    train_config: TrainConfig,
    ctx: TrainContext,
    loss_fn: Callable,
) -> None:
    if ctx.device.type != "cuda":
        return
    dataset = train_loader.dataset
    assert isinstance(dataset, TensorDataset)
    warmup_count = min(int(train_config.batch_size), len(dataset))
    if warmup_count <= 0:
        return

    emit_phase_event(ctx.phase_log_path, "compile_warmup_begin", warmup_samples=warmup_count)
    checkpoint_model = unwrap_model(model)
    saved_state    = {k: v.detach().cpu().clone() for k, v in checkpoint_model.state_dict().items()}
    warmup_opt     = create_optimizer(checkpoint_model, train_config)
    warmup_scaler  = torch.amp.GradScaler("cuda", enabled=ctx.amp_enabled)

    feat, tgt = dataset.tensors[0][:warmup_count], dataset.tensors[1][:warmup_count]
    warmup_opt.zero_grad(set_to_none=True)
    with make_autocast_context(device=ctx.device, amp_enabled=ctx.amp_enabled):
        loss = loss_fn(model(feat), tgt)
    warmup_scaler.scale(loss).backward()
    warmup_scaler.step(warmup_opt)
    warmup_scaler.update()
    warmup_opt.zero_grad(set_to_none=True)

    checkpoint_model.load_state_dict(saved_state)
    if ctx.profile_cuda_timing:
        torch.cuda.synchronize(ctx.device)
    emit_phase_event(ctx.phase_log_path, "compile_warmup_end", warmup_samples=warmup_count)

# ── Single epoch runner

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: Callable,
    ctx: TrainContext,
    phase_name: str | None = None,
    epoch: int | None = None,
) -> float:
    if training and optimizer is None:
        raise ValueError("optimizer must be provided when training=True.")

    model.train(mode=training)
    total_loss  = torch.zeros((), device=ctx.device, dtype=torch.float32)
    total_count = 0
    phase_started = False

    with nullcontext() if training else torch.no_grad():
        for feat_batch, tgt_batch in loader:
            if not phase_started and phase_name is not None and epoch is not None:
                emit_phase_event(ctx.phase_log_path, f"{phase_name}_epoch_begin", epoch=int(epoch))
                phase_started = True

            n = int(feat_batch.shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)
            with make_autocast_context(device=ctx.device, amp_enabled=ctx.amp_enabled):
                batch_loss = loss_fn(model(feat_batch), tgt_batch)
            if training:
                ctx.scaler.scale(batch_loss).backward()
                ctx.scaler.step(optimizer)
                ctx.scaler.update()

            total_loss  = total_loss + batch_loss.detach().to(torch.float32) * n
            total_count += n

    if total_count == 0:
        raise ValueError("Encountered an empty split during training.")
    if phase_started and phase_name is not None and epoch is not None:
        if ctx.profile_cuda_timing and ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        emit_phase_event(ctx.phase_log_path, f"{phase_name}_epoch_end",
                         epoch=int(epoch), sample_count=int(total_count))

    return float((total_loss / total_count).item())

# ── Core training loop

def train_model(
    model: nn.Module,
    *,
    loaders: dict[str, DataLoader],
    train_config: TrainConfig,
    model_type: str,
    checkpoint_path: str | Path,
    checkpoint_model: nn.Module,
    feature_transform: dict[str, Any],
    ctx: TrainContext,
    loss_fn: Callable,
    swanlab_run: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer  = create_optimizer(checkpoint_model, train_config)
    best_val   = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    wait       = 0
    history: dict[str, Any] = {
        "train_hk": [], "val_hk": [], "best_epoch": -1,
        "best_checkpoint_path": str(Path(checkpoint_path).resolve()),
    }
    epoch_kw = dict(loss_fn=loss_fn, ctx=ctx)

    for epoch in range(train_config.max_epochs):
        train_loss = run_epoch(model, loaders["train"], training=True,
                               optimizer=optimizer, phase_name="train", epoch=epoch, **epoch_kw)
        val_loss   = run_epoch(model, loaders["val"],   training=False,
                               optimizer=None,      phase_name="val",   epoch=epoch, **epoch_kw)

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
            emit_phase_event(ctx.phase_log_path, "checkpoint_saved", epoch=int(epoch),
                             checkpoint_path=str(Path(checkpoint_path).resolve()), kind="best_so_far")
        else:
            wait += 1
            if wait >= train_config.patience:
                print(f"Early stopping at epoch {epoch} with patience={train_config.patience}.")
                emit_phase_event(ctx.phase_log_path, "early_stopping",
                                 epoch=int(epoch), patience=int(train_config.patience))
                break

    if best_state is not None:
        checkpoint_model.load_state_dict(best_state)
        save_checkpoint_bundle(checkpoint_model, model_type=model_type,
                               model_config=train_config, path=checkpoint_path,
                               feature_transform=feature_transform)
        emit_phase_event(ctx.phase_log_path, "checkpoint_saved", epoch=int(best_epoch),
                         checkpoint_path=str(Path(checkpoint_path).resolve()), kind="final_best")

    history["best_epoch"] = best_epoch
    return model, history

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
        try:
            torch.backends.cudnn.benchmark        = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
        except Exception:
            pass
    return device

def load_and_transform_dataset(
    args: argparse.Namespace, *, dataset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if getattr(args, "use_pca", False):
        return _load_pca_dataset(args, dataset_path=dataset_path)
    raw_bundle = load_training_arrays_from_hdf5(dataset_path)
    validate_training_splits(raw_bundle)
    feature_transform = fit_feature_transform(
        raw_bundle["splits"]["train"]["features"], dataset_path=dataset_path,
    )
    explicit_output = getattr(args, "feature_transform", "") or args.normalization_csv
    norm_output = Path(explicit_output) if explicit_output else normalization_csv_path(args.output_model)
    saved_norm_path = save_feature_stats(norm_output, transform=feature_transform, dataset_path=dataset_path)
    return transform_dataset_bundle(raw_bundle, feature_transform=feature_transform), feature_transform, saved_norm_path


def _load_pca_dataset(
    args: argparse.Namespace, *, dataset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    # Offline PCA-18: features are already projected to 18D in the *_pca18.h5
    # dataset, so we skip transform_dataset_bundle entirely. The fitted V3
    # transform is returned for embedding into the checkpoint, and the NPZ
    # sidecar path takes the "normalization output" slot.
    from train_generate.pca_dataset import ensure_pca_dataset

    pca_h5_path, npz_path, feature_transform = ensure_pca_dataset(dataset_path)
    raw_bundle = load_training_arrays_from_hdf5(pca_h5_path)
    validate_training_splits(raw_bundle)
    return raw_bundle, feature_transform, Path(npz_path)

def build_train_config_from_args(
    args: argparse.Namespace,
    *,
    raw_feature_dim: int,
    model_type: ModelType,
    feature_transform: dict[str, Any],
) -> TrainConfig:
    # Auto-map: rename the 2 CLI args whose names differ from their config fields,
    # then let create_train_config ignore unrecognised keys (e.g. disable_amp).
    overrides = {_CLI_RENAMES.get(k, k): v for k, v in vars(args).items()}
    overrides["dataset_path"]      = Path(overrides["dataset_path"])
    overrides["output_model_path"] = Path(overrides["output_model_path"])
    overrides["input_dim"]         = int(feature_transform["output_dim"])
    overrides["raw_feature_dim"]   = raw_feature_dim
    if overrides.get("batch_size") is None:        # keep config default when not passed
        overrides.pop("batch_size", None)
    overrides.pop("model_type", None)              # already passed as positional arg
    return create_train_config(model_type, **overrides)

# ── Argument parser

def build_arg_parser() -> argparse.ArgumentParser:
    mlp = MLP_TrainConfig()
    cnn = CNN_TrainConfig()
    p   = argparse.ArgumentParser(description="Train the 3x3 stencil h*kappa predictor.")
    # paths
    p.add_argument("--model-type",       default="mlp",                  choices=["mlp", "cnn"])
    p.add_argument("--dataset-output",   default=str(default_dataset_path()))
    p.add_argument("--output-model",     default=str(default_output_model_path()))
    p.add_argument("--feature-transform", default="",
                   help="Explicit feature-transform sidecar output (.csv for V1/V2). Supersedes --normalization-csv.")
    p.add_argument("--normalization-csv", default="", help="Deprecated alias of --feature-transform (V1/V2 CSV only).")
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
                   choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--profile-cuda-timing", action="store_true")
    add_swanlab_args(p)
    return p

# ── Entry point

def main() -> None:
    args       = build_arg_parser().parse_args()
    model_type: ModelType = args.model_type   # type: ignore[assignment]
    dataset_path = Path(args.dataset_output)

    if model_type == "cnn" and getattr(args, "use_pca", False):
        raise ValueError("CNN does not support --use-pca; PCA-18 is an MLP-only feature path.")

    # 1. device
    device = setup_device(args)

    # 2. data
    transformed_bundle, feature_transform, saved_norm_path = load_and_transform_dataset(
        args, dataset_path=dataset_path,
    )
    model_input_dim = int(transformed_bundle["splits"]["train"]["features"].shape[1])
    if model_type == "cnn" and model_input_dim != 9:
        raise ValueError(
            f"CNN requires 9D features, got model input dim={model_input_dim}. "
            "Use --model-type mlp for 27D (V2) or PCA-18 (V3) datasets."
        )

    # 3. config + seed
    train_config = build_train_config_from_args(
        args, raw_feature_dim=int(feature_transform["raw_feature_dim"]), model_type=model_type,
        feature_transform=feature_transform,
    )
    torch.manual_seed(train_config.seed)

    # 4. model
    checkpoint_model = create_model(train_config).to(device)
    amp_enabled      = device.type == "cuda" and not args.disable_amp
    compile_mode     = args.compile_mode if not args.disable_compile else "none"
    compile_enabled  = device.type == "cuda" and not args.disable_compile and hasattr(torch, "compile")
    model = (
        torch.compile(checkpoint_model, mode=compile_mode)
        if compile_enabled and compile_mode != "none"
        else checkpoint_model
    )

    # 5. TrainContext  (bundles all hardware/logging state)
    phase_log_path: Path | None = None
    if args.phase_log_path:
        phase_log_path = Path(args.phase_log_path).resolve()
        phase_log_path.parent.mkdir(parents=True, exist_ok=True)
        phase_log_path.write_text("", encoding="utf-8")
        emit_phase_event(phase_log_path, "run_initialized",
                         dataset_path=str(dataset_path.resolve()),
                         output_model=str(Path(args.output_model).resolve()),
                         model_type=str(model_type))
    ctx = TrainContext(
        device=device,
        amp_enabled=amp_enabled,
        scaler=torch.amp.GradScaler("cuda", enabled=amp_enabled),
        compile_enabled=compile_enabled,
        compile_mode=compile_mode,
        profile_cuda_timing=train_config.profile_cuda_timing,
        phase_log_path=phase_log_path,
    )

    # 6. loaders + loss
    loaders = build_split_loaders(transformed_bundle, device=device, batch_size=train_config.batch_size)
    loss_fn = create_loss_fn(train_config)

    # 7. print summary
    print(f"Dataset:    {dataset_path.resolve()}")
    print(f"Model:      {model_type}  params={count_parameters(checkpoint_model)}")
    print(f"Device:     {device}  AMP={amp_enabled}  compile={compile_enabled}({compile_mode})")
    print(f"Optimizer:  {train_config.optimizer_type}  lr={train_config.lr}  wd={train_config.l2_reg}")
    print(f"Loss fn:    {train_config.loss_fn}")
    print(f"Norm CSV:   {saved_norm_path.resolve()}")
    describe_dataset_bundle(transformed_bundle, batch_size=train_config.batch_size,
                            feature_transform=feature_transform)

    # 8. compile warmup
    emit_phase_event(ctx.phase_log_path, "data_ready",
                     device=str(device), train_size=int(transformed_bundle["sizes"]["train"]),
                     val_size=int(transformed_bundle["sizes"]["val"]),
                     test_size=int(transformed_bundle["sizes"].get("test", 0)),
                     batch_size=int(train_config.batch_size),
                     amp_enabled=bool(amp_enabled), compile_enabled=bool(compile_enabled))
    if ctx.compile_enabled and ctx.compile_mode != "none":
        maybe_warmup_compiled_training(model, loaders["train"],
                                       train_config=train_config, ctx=ctx, loss_fn=loss_fn)

    # 9. SwanLab  (optional)
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
                dataset_path=dataset_path, checkpoint_path=args.output_model,
                normalization_output=saved_norm_path, feature_transform=feature_transform,
                device=device, parameter_count=count_parameters(checkpoint_model),
                model_type=model_type,
            ),
        )

    # 10. train
    emit_phase_event(ctx.phase_log_path, "training_begin")
    _, history = train_model(
        model,
        loaders=loaders, train_config=train_config, model_type=model_type,
        checkpoint_path=args.output_model, checkpoint_model=checkpoint_model,
        feature_transform=feature_transform, ctx=ctx,
        loss_fn=loss_fn, swanlab_run=swanlab_run,
    )
    emit_phase_event(ctx.phase_log_path, "training_end", best_epoch=int(history["best_epoch"]))

    # 11. finish
    if swanlab_run is not None:
        swanlab_run.log({
            "summary/best_epoch":        int(history["best_epoch"]),
            "summary/best_val_hk_loss":  float(min(history["val_hk"])),
            "summary/final_val_hk_loss": float(history["val_hk"][-1]),
        })
        swanlab_run.finish()
    emit_phase_event(ctx.phase_log_path, "run_complete", best_epoch=int(history["best_epoch"]))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best checkpoint: {Path(history['best_checkpoint_path']).resolve()}")


if __name__ == "__main__":
    main()
