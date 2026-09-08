from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from paper_section4.core import (
    RESOLUTIONS,
    RESULTS_DIR,
    atomic_json,
    read_csv,
    sha256_file,
)
from paper_section4.recompute import (
    FLOWER_H5,
    _sample_blueprint_for_section4,
    _training_config,
)
from train_generate.generate import (
    generate_blueprints,
    sample_blueprint,
    split_blueprint_indices,
)


EXPECTED = {
    "fig01_cross_resolution": ("fig01_cross_resolution.source.csv", 30),
    "fig02_circle_r_over_h": ("fig02_circle_r_over_h.source.csv", 222),
    "fig03_flower_sensitivity": ("fig03_flower_sensitivity.source.csv", 248),
    "fig04_flower_error_map": ("fig04_flower_error_map.source.csv.gz", 4604),
}


def _finite(rows: list[dict[str, str]], fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: non-finite values in {field}")


def _validate_provenance(figure: str, source_path: Path) -> dict[str, Any]:
    path = RESULTS_DIR / f"{figure}.provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["source_data"] != source_path.name:
        raise ValueError(f"{figure}: provenance source filename mismatch")
    if payload["source_data_sha256"] != sha256_file(source_path):
        raise ValueError(f"{figure}: provenance source hash mismatch")
    return payload


def _validate_narrow_band_equivalence() -> dict[str, Any]:
    checked: list[int] = []
    for rho in (32, 64, 128, 256):
        config = _training_config(rho)
        blueprints = generate_blueprints(config)
        split = split_blueprint_indices(
            blueprints,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            seed=config.geometry_seed,
        )
        blueprint = next(
            blueprints[index]
            for index in split["test"]
            if blueprints[index]["meta"]["shape_type"] == "ellipse"
        )
        reference = sample_blueprint(blueprint, data_config=config)
        narrow_band = _sample_blueprint_for_section4(blueprint, config)
        if len(reference) != len(narrow_band):
            raise ValueError(f"rho={rho}: narrow-band sample group count mismatch")
        for sample_index, (expected, actual) in enumerate(zip(reference, narrow_band, strict=True)):
            if expected.keys() != actual.keys():
                raise ValueError(f"rho={rho} sample={sample_index}: key mismatch")
            for key in expected:
                if not np.array_equal(expected[key], actual[key]):
                    raise ValueError(
                        f"rho={rho} sample={sample_index} key={key}: narrow-band array mismatch"
                    )
        checked.append(rho)
    return {
        "passed": True,
        "resolutions": checked,
        "comparison": "np.array_equal for every phi9, 27D feature, and target array",
    }


def validate_package(*, report_path: Path | None = None) -> dict[str, Any]:
    generator_equivalence = _validate_narrow_band_equivalence()
    source_rows: dict[str, list[dict[str, str]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for figure, (filename, expected_rows) in EXPECTED.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_csv(path)
        if len(rows) != expected_rows:
            raise ValueError(f"{figure}: expected {expected_rows} rows, got {len(rows)}")
        source_rows[figure] = rows
        provenance[figure] = _validate_provenance(figure, path)

    fig01 = source_rows["fig01_cross_resolution"]
    _finite(fig01, ("mse", "mae", "maxae"), "fig01")
    nn_pairs = {(int(row["train_rho"]), int(row["test_rho"])) for row in fig01 if row["method"] == "NN"}
    fd_tests = {int(row["test_rho"]) for row in fig01 if row["method"] == "FD"}
    expected_pairs = {(train, test) for train in RESOLUTIONS for test in RESOLUTIONS}
    if nn_pairs != expected_pairs or fd_tests != set(RESOLUTIONS):
        raise ValueError("fig01: incomplete 25 NN + 5 FD matrix")

    fig02 = source_rows["fig02_circle_r_over_h"]
    _finite(fig02, ("r_over_h", "mse", "mae", "maxae"), "fig02")
    q_values = sorted({float(row["r_over_h"]) for row in fig02})
    if len(q_values) != 37 or any(int(row["sobol_positions"]) != 256 for row in fig02):
        raise ValueError("fig02: expected 37 ratios and 256 Sobol positions per row")
    for q_value in q_values:
        rows = [row for row in fig02 if float(row["r_over_h"]) == q_value]
        if len(rows) != 6:
            raise ValueError(f"fig02: R/h={q_value} does not have five NN rows plus FD")

    if not FLOWER_H5.exists():
        raise FileNotFoundError(FLOWER_H5)
    with h5py.File(FLOWER_H5, "r") as handle:
        if handle["features"].shape != (35681, 27):
            raise ValueError(f"flower HDF5 feature shape is {handle['features'].shape}")
        if not bool(handle.attrs["scale_h"]) or str(handle.attrs["feature_order"]) != "phi9+nx9+ny9":
            raise ValueError("flower HDF5 feature contract mismatch")
        counts = Counter((int(case), int(step)) for case, step in zip(handle["case_id"][:], handle["iter"][:]))
        if set(step for _, step in counts) != set(range(31)):
            raise ValueError("flower HDF5 does not contain steps 0..30")
        if any(counts[(0, step)] != 527 or counts[(1, step)] != 624 for step in range(31)):
            raise ValueError("flower HDF5 node counts are not 527/624 at every step")
    flower_hash = sha256_file(FLOWER_H5)
    for figure in ("fig03_flower_sensitivity", "fig04_flower_error_map"):
        if provenance[figure]["protocol"]["shared_hdf5_sha256"] != flower_hash:
            raise ValueError(f"{figure}: shared flower HDF5 hash mismatch")

    fig03 = source_rows["fig03_flower_sensitivity"]
    _finite(fig03, ("value",), "fig03")
    structure = Counter(
        (row["family"], int(row["step"]), row["metric"], row["method"], row["selection"])
        for row in fig03
    )
    for family in ("smooth", "acute"):
        interface_count = 527 if family == "smooth" else 624
        for step in range(31):
            for method in ("NN", "FD"):
                if structure[(family, step, "hkappa_mse", method, "interface_nodes")] != 1:
                    raise ValueError(f"fig03: missing {method} interface-node MSE for {family} step {step}")
            for metric in ("sdf_mean", "sdf_max"):
                if structure[(family, step, metric, "SDF", "interface_nodes")] != 1:
                    raise ValueError(f"fig03: missing interface-node {metric} for {family} step {step}")
            current = [
                row for row in fig03
                if row["family"] == family and int(row["step"]) == step
            ]
            if any(row["selection"] != "interface_nodes" for row in current):
                raise ValueError(f"fig03: non-interface selection for {family} step {step}")
            if any(int(row["sample_count"]) != interface_count for row in current):
                raise ValueError(f"fig03: bad interface sample count for {family} step {step}")
            summaries = {
                row["metric"]: float(row["value"])
                for row in current
                if row["method"] == "SDF"
            }
            if summaries["sdf_max"] < summaries["sdf_mean"]:
                raise ValueError(f"fig03: maximum below mean for {family} step {step}")

    current_summary: dict[str, Any] = {}
    documented = {
        "smooth": {"nn_min_step": 11, "nn_step30_rise_pct": 4.6, "fd_min_step": 2, "fd_step30_ratio": 2.02},
        "acute": {"nn_min_step": 6, "nn_step30_rise_pct": 15.8, "fd_min_step": 2, "fd_step30_ratio": 1.83},
    }
    for family in ("smooth", "acute"):
        by_method: dict[str, dict[int, float]] = defaultdict(dict)
        for row in fig03:
            if row["family"] == family and row["metric"] == "hkappa_mse":
                by_method[row["method"]][int(row["step"])] = float(row["value"])
        nn_min_step = min(by_method["NN"], key=by_method["NN"].get)
        fd_min_step = min(by_method["FD"], key=by_method["FD"].get)
        current_summary[family] = {
            "nn_min_step": nn_min_step,
            "nn_step30_rise_pct": (by_method["NN"][30] / by_method["NN"][nn_min_step] - 1.0) * 100.0,
            "fd_min_step": fd_min_step,
            "fd_step30_ratio": by_method["FD"][30] / by_method["FD"][fd_min_step],
        }

    fig04 = source_rows["fig04_flower_error_map"]
    _finite(fig04, ("theta_rad", "x", "y", "nn_pred_hkappa", "analytic_hkappa", "fd_hkappa", "nn_abs_error", "fd_abs_error"), "fig04")
    nn_recomputed = np.abs(
        np.asarray([float(row["nn_pred_hkappa"]) for row in fig04])
        - np.asarray([float(row["analytic_hkappa"]) for row in fig04])
    )
    fd_recomputed = np.abs(
        np.asarray([float(row["fd_hkappa"]) for row in fig04])
        - np.asarray([float(row["analytic_hkappa"]) for row in fig04])
    )
    nn_stored = np.asarray([float(row["nn_abs_error"]) for row in fig04])
    fd_stored = np.asarray([float(row["fd_abs_error"]) for row in fig04])
    if not np.allclose(nn_stored, nn_recomputed, rtol=0.0, atol=1.0e-14):
        raise ValueError("fig04: stored NN absolute errors do not match prediction/truth")
    if not np.allclose(fd_stored, fd_recomputed, rtol=0.0, atol=1.0e-14):
        raise ValueError("fig04: stored FD absolute errors do not match prediction/truth")
    counts = Counter((row["family"], int(row["step"])) for row in fig04)
    for family, expected_count in (("smooth", 527), ("acute", 624)):
        for step in (0, 10, 20, 30):
            if counts[(family, step)] != expected_count:
                raise ValueError(f"fig04: {family} step {step} has {counts[(family, step)]} rows")
    errors = nn_stored.astype(np.float64, copy=False)
    positive = errors[errors > 0.0]
    p95 = float(np.percentile(errors, 95.0))
    color_limits = {
        "p05_positive": float(np.percentile(positive, 5.0)),
        "p95_all": p95,
        "raw_min_positive": float(np.min(positive)),
        "raw_max": float(np.max(errors)),
        "nodes_above_p95": int(np.count_nonzero(errors > p95)),
    }

    report = {
        "passed": True,
        "source_rows": {figure: len(rows) for figure, rows in source_rows.items()},
        "flower_hdf5": {"sha256": flower_hash, "samples": 35681, "feature_dim": 27},
        "current_model_flower_summary": current_summary,
        "documented_original_flower_summary": documented,
        "original_result_match": current_summary == documented,
        "fig04_color_limits": color_limits,
        "ellipse_narrow_band_equivalence": generator_equivalence,
        "note": "Current checkpoints are authoritative; differences from the documented original are expected and must not be presented as an exact reproduction.",
    }
    destination = report_path or RESULTS_DIR / "validation_report.json"
    atomic_json(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the reproducible Section 4 source-data package")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate_package(report_path=args.report)
    print(f"PASS: {args.report or RESULTS_DIR / 'validation_report.json'}")
    print(json.dumps(report["current_model_flower_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
