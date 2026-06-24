"""Test 1 (part2): evaluate the trained DCTS h*kappa models.

Reports two test variants against every checkpoint in --models-dir:
  - "analytic (in-distribution)": dataset/part2_dcts/main/processed/test.h5,
    same nx9/ny9 convention the models were trained on. Reference only.
  - "central-diff (Test 1)": the same test geometries with nx9/ny9 replaced by
    a central-difference estimate (evaluate.part2.fd_testset) -- this is the
    realistic, solver-like feature convention the user asked to test against.

Note: hk_central (the no-model central-difference baseline) only depends on
phi9, not nx9/ny9, so its numbers are identical across both variants by
construction; only the model's own predictions can differ between them.

Usage:
    python -m evaluate.part2.evaluate_fd_testset
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate.part2.fd_testset import assert_matches_existing_test_h5, build_fd_test_split, save_fd_test_h5
from evaluate.shared import (
    compute_metrics,
    group_metric_rows,
    load_model_from_checkpoint,
    predict_hkappa_full_batch,
    resolve_feature_transform,
    write_csv_rows,
)
from train_generate.part2.dcts.config import DctsConfig

DEFAULT_ANALYTIC_TEST_H5 = Path("dataset/part2_dcts/main/processed/test.h5")
DEFAULT_FD_TEST_H5 = Path("dataset/part2_dcts/main/processed/test_fd.h5")
DEFAULT_MODELS_DIR = Path("out/7367")


def load_test_h5(path: Path) -> dict:
    """Load either test.h5 (analytic) or test_fd.h5 (central-diff) -- same schema."""
    with h5py.File(path, "r") as handle:
        return {
            "features27": np.asarray(handle["features27"][:], dtype=np.float32),
            "target_hk": np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1),
            "hk_central": np.asarray(handle["hk_central"][:], dtype=np.float64).reshape(-1),
            "shape": [v.decode() if isinstance(v, bytes) else str(v) for v in handle["shape"][:]],
            "coarse_regime": np.asarray(handle["coarse_regime"][:], dtype=np.int64),
        }


def load_or_build_fd_test(path: Path, *, existing_test_h5: Path, rebuild: bool) -> dict:
    if path.exists() and not rebuild:
        return load_test_h5(path)
    config = DctsConfig.main()
    print(f"[evaluate_fd_testset] building {path} ...")
    merged_fd = build_fd_test_split(config, split="test")
    if existing_test_h5.exists():
        assert_matches_existing_test_h5(merged_fd, existing_test_h5)
        print(f"[gate] PASS: regenerated split matches {existing_test_h5}.")
    save_fd_test_h5(merged_fd, path, config=config, split="test")
    return load_test_h5(path)


def discover_checkpoints(models_dir: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted(models_dir.glob("*.pt"))


def evaluate_one(model_path: Path, dataset: dict, *, device: torch.device) -> dict:
    model, checkpoint_meta = load_model_from_checkpoint(model_path, device=device)
    transform, transform_source = resolve_feature_transform(
        model_path=model_path, explicit_path=None, checkpoint_meta=checkpoint_meta,
    )
    features = dataset["features27"]
    if int(transform["raw_feature_dim"]) != int(features.shape[1]):
        raise ValueError(
            f"{model_path}: checkpoint expects {transform['raw_feature_dim']}D features, "
            f"dataset has {features.shape[1]}D."
        )
    predictions = predict_hkappa_full_batch(model, features, transform=transform, device=device)
    target = dataset["target_hk"]
    by_shape = group_metric_rows(
        labels=np.asarray(dataset["shape"]), prediction=predictions, target=target,
        numeric=dataset["hk_central"], label_name="shape",
    )
    by_band = group_metric_rows(
        labels=dataset["coarse_regime"], prediction=predictions, target=target,
        numeric=dataset["hk_central"], label_name="coarse_regime",
    )
    return {
        "model_path": model_path,
        "transform_source": transform_source,
        "n": int(features.shape[0]),
        "model_overall": compute_metrics(predictions, target),
        "numeric_overall": compute_metrics(dataset["hk_central"], target),
        "by_shape": by_shape,
        "by_band": by_band,
    }


def _print_summary(label: str, result: dict) -> None:
    m, b = result["model_overall"], result["numeric_overall"]
    print(
        f"{label:<26s} {result['model_path'].stem:<20s} N={result['n']:6d}  "
        f"model MSE={m['mse']:.4e} MAE={m['mae']:.4e} MaxAE={m['maxae']:.4e}  |  "
        f"central-diff-baseline MSE={b['mse']:.4e} MAE={b['mae']:.4e}"
    )


def _flatten_rows(label: str, result: dict) -> list[dict]:
    rows = []
    for group_kind, group_rows in (("shape", result["by_shape"]), ("coarse_regime", result["by_band"])):
        for row in group_rows:
            m = row["model_vs_analytic"]
            b = row["numeric_vs_analytic"]
            rows.append({
                "test_variant": label,
                "model": result["model_path"].stem,
                "group_kind": group_kind,
                "group_value": row["display"],
                "sample_count": row["sample_count"],
                "model_mse": m["mse"], "model_mae": m["mae"], "model_maxae": m["maxae"],
                "central_diff_mse": b["mse"], "central_diff_mae": b["mae"], "central_diff_maxae": b["maxae"],
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate part2 DCTS h*kappa models: analytic-normal test vs central-difference-normal Test 1."
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--analytic-test-h5", default=str(DEFAULT_ANALYTIC_TEST_H5))
    parser.add_argument("--fd-test-h5", default=str(DEFAULT_FD_TEST_H5))
    parser.add_argument("--rebuild-fd-test", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--output-csv", default=str(DEFAULT_MODELS_DIR / "test1_fd_normals_metrics.csv"))
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analytic_path = Path(args.analytic_test_h5)
    fd_path = Path(args.fd_test_h5)
    datasets: dict[str, dict] = {}
    if analytic_path.exists():
        datasets["analytic (in-distribution)"] = load_test_h5(analytic_path)
    else:
        print(f"[warn] {analytic_path} not found; skipping the in-distribution reference.")
    datasets["central-diff (Test 1)"] = load_or_build_fd_test(
        fd_path, existing_test_h5=analytic_path, rebuild=args.rebuild_fd_test,
    )

    checkpoints = discover_checkpoints(Path(args.models_dir), args.models)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {args.models_dir} (or pass --models explicitly).")

    all_rows: list[dict] = []
    for label, dataset in datasets.items():
        for model_path in checkpoints:
            result = evaluate_one(model_path, dataset, device=device)
            _print_summary(label, result)
            all_rows.extend(_flatten_rows(label, result))
        print()

    output_csv = Path(args.output_csv)
    write_csv_rows(output_csv, all_rows, fieldnames=(
        "test_variant", "model", "group_kind", "group_value", "sample_count",
        "model_mse", "model_mae", "model_maxae", "central_diff_mse", "central_diff_mae", "central_diff_maxae",
    ))
    print(f"Wrote per-group metrics -> {output_csv}")


if __name__ == "__main__":
    main()
