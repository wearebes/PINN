"""Test 1 (part2): rebuild the DCTS test split with central-difference normals.

DCTS trains on *analytic* nx9/ny9 -- the outward normal at the Newton-projected
foot point, computed in closed form. A real solver never has that: it only has
a level-set field and estimates the gradient by central differences. This
module regenerates the exact same test-split geometries (same DctsConfig, same
seed) via train_generate.part2.dcts.select, then replaces nx9/ny9 with the
central-difference estimate taken from the 5x5 phi patch that select.py already
computes. So the only thing that changes relative to
dataset/part2_dcts/main/processed/test.h5 is the normal-vector convention:
  - circle:  analytic SDF,        central-difference nx9/ny9
  - ellipse: Newton-projected SDF, central-difference nx9/ny9

Usage:
    python -m evaluate.part2.fd_testset
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from train_generate.geometry_core import STENCIL_OFFSETS
from train_generate.part2.dcts import bins, select
from train_generate.part2.dcts.config import DctsConfig
from train_generate.part2.generate import central_difference_hkappa_from_phi9_float64

FEATURES27_COLUMNS = "[phi9:0-8, nx9_fd:9-17, ny9_fd:18-26]"
DEFAULT_EXISTING_TEST_H5 = Path("dataset/part2_dcts/main/processed/test.h5")
DEFAULT_OUTPUT_H5 = Path("dataset/part2_dcts/main/processed/test_fd.h5")


def fd_grad_components_inner(phi5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference (gx, gy) at the 9 inner stencil nodes (h=1).

    Same +/-1 footprint as train_generate.part2.dcts.patch.fd_grad_norm_inner,
    but keeps the signed components instead of collapsing to the norm.
    """
    phi5 = np.asarray(phi5, dtype=np.float64)
    n = phi5.shape[0]
    gx9 = np.empty((n, 9), dtype=np.float64)
    gy9 = np.empty((n, 9), dtype=np.float64)
    for k, (di, dj) in enumerate(STENCIL_OFFSETS):
        ix = int(di) + 2
        iy = int(dj) + 2
        gx9[:, k] = 0.5 * (phi5[:, ix + 1, iy] - phi5[:, ix - 1, iy])
        gy9[:, k] = 0.5 * (phi5[:, ix, iy + 1] - phi5[:, ix, iy - 1])
    return gx9, gy9


def central_diff_normals(phi5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx9, gy9 = fd_grad_components_inner(phi5)
    mag = np.sqrt(gx9 * gx9 + gy9 * gy9)
    bad = int(np.count_nonzero((~np.isfinite(mag)) | (mag <= 0.0)))
    if bad:
        raise ValueError(f"central-difference normal magnitude has {bad} non-positive or non-finite values.")
    return gx9 / mag, gy9 / mag


def build_fd_test_split(config: DctsConfig, *, split: str = "test") -> dict:
    """Regenerate `split` with `config`, then swap in central-difference nx9/ny9."""
    edges = bins.fine_bin_edges(config.eta_min, config.eta_max, config.n_fine_bins)
    centers = bins.fine_bin_centers(edges)
    merged, records = select.generate_split(config, split=split, edges=edges, centers=centers)

    phi5 = np.asarray(merged["phi5"], dtype=np.float64)
    phi9 = np.asarray(merged["phi9"], dtype=np.float64)
    nx9_fd, ny9_fd = central_diff_normals(phi5)
    hk_central_fd = central_difference_hkappa_from_phi9_float64(phi9)
    features27_fd = np.concatenate([phi9, nx9_fd, ny9_fd], axis=1).astype(np.float32)

    out = dict(merged)
    out["nx9_fd"] = nx9_fd.astype(np.float32)
    out["ny9_fd"] = ny9_fd.astype(np.float32)
    out["features27_fd"] = features27_fd
    out["hk_central_fd"] = hk_central_fd.astype(np.float32)
    out["records"] = records
    return out


def assert_matches_existing_test_h5(
    merged: dict, existing_path: str | Path, *, rtol: float = 1.0e-4, atol: float = 1.0e-5,
) -> None:
    """Gate: the regenerated canonical split must reproduce test.h5 row-for-row.

    Confirms the only change between this dataset and the analytic-normal
    test.h5 is the nx9/ny9 convention, not a drift in geometry sampling.
    """
    existing_path = Path(existing_path)
    with h5py.File(existing_path, "r") as handle:
        existing_target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
        existing_eta = np.asarray(handle["eta"][:], dtype=np.float64).reshape(-1)
        existing_geometry_id = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["geometry_id"][:]]
        existing_phi9 = np.asarray(handle["features27"][:, :9], dtype=np.float64)
        existing_nx9 = np.asarray(handle["features27"][:, 9:18], dtype=np.float64)
        existing_ny9 = np.asarray(handle["features27"][:, 18:27], dtype=np.float64)

    hk_exact = np.asarray(merged["hk_exact"], dtype=np.float64).reshape(-1)
    eta = np.asarray(merged["eta"], dtype=np.float64).reshape(-1)
    geometry_id = list(merged["geometry_id"])
    phi9 = np.asarray(merged["phi9"], dtype=np.float64)
    nx9 = np.asarray(merged["nx9"], dtype=np.float64)
    ny9 = np.asarray(merged["ny9"], dtype=np.float64)

    if geometry_id != existing_geometry_id:
        raise RuntimeError(
            f"Regenerated geometry_id sequence does not match {existing_path}; "
            "DCTS sampling must have drifted since the dataset was generated."
        )
    for name, got, want in (
        ("target_hk", hk_exact, existing_target_hk),
        ("eta", eta, existing_eta),
        ("phi9", phi9, existing_phi9),
        ("nx9 (analytic)", nx9, existing_nx9),
        ("ny9 (analytic)", ny9, existing_ny9),
    ):
        if not np.allclose(got, want, rtol=rtol, atol=atol):
            max_err = float(np.max(np.abs(got - want)))
            raise RuntimeError(f"Regenerated {name} does not match {existing_path} (max abs err={max_err:.3e}).")


def save_fd_test_h5(merged_fd: dict, path: str | Path, *, config: DctsConfig, split: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(merged_fd["features27_fd"].shape[0])
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_family"] = "part2_dcts_stage0_fd_normals"
        handle.attrs["schema_version"] = 1
        handle.attrs["split"] = split
        handle.attrs["tag"] = config.tag
        handle.attrs["h"] = float(config.h)
        handle.attrs["eta_min"] = float(config.eta_min)
        handle.attrs["eta_max"] = float(config.eta_max)
        handle.attrs["n_fine_bins"] = int(config.n_fine_bins)
        handle.attrs["sdf_mode"] = "sdf"
        handle.attrs["normal_source"] = "central_diff"
        handle.attrs["features27_columns"] = FEATURES27_COLUMNS
        handle.attrs["row_count"] = n
        handle.create_dataset("features27", data=merged_fd["features27_fd"])
        handle.create_dataset("target_hk", data=np.asarray(merged_fd["hk_exact"], dtype=np.float32).reshape(-1, 1))
        handle.create_dataset("hk_central", data=np.asarray(merged_fd["hk_central_fd"], dtype=np.float32).reshape(-1, 1))
        handle.create_dataset("eta", data=np.asarray(merged_fd["eta"], dtype=np.float32).reshape(-1, 1))
        handle.create_dataset("fine_bin", data=np.asarray(merged_fd["fine_bin"], dtype=np.int32))
        handle.create_dataset("coarse_regime", data=np.asarray(merged_fd["coarse_regime"], dtype=np.int32))
        handle.create_dataset("geometry_id", data=list(merged_fd["geometry_id"]), dtype=h5py.string_dtype(encoding="utf-8"))
        handle.create_dataset("pack_id", data=list(merged_fd["pack_id"]), dtype=h5py.string_dtype(encoding="utf-8"))
        handle.create_dataset("shape", data=list(merged_fd["shape"]), dtype=h5py.string_dtype(encoding="utf-8"))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the part2 Test-1 dataset: DCTS test geometries with central-difference nx9/ny9."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--existing-test-h5", default=str(DEFAULT_EXISTING_TEST_H5))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_H5))
    parser.add_argument("--skip-gate", action="store_true", help="Skip the regeneration-matches-test.h5 gate.")
    args = parser.parse_args(argv)

    config = DctsConfig.main()
    print(f"[fd_testset] regenerating split={args.split!r} from {config.output_dir} (base_seed={config.base_seed}) ...")
    merged_fd = build_fd_test_split(config, split=args.split)
    n = int(merged_fd["features27_fd"].shape[0])
    print(f"[fd_testset] regenerated {n} rows.")

    existing = Path(args.existing_test_h5)
    if not args.skip_gate:
        if not existing.exists():
            raise FileNotFoundError(f"--existing-test-h5 not found: {existing}. Pass --skip-gate to bypass.")
        assert_matches_existing_test_h5(merged_fd, existing)
        print(f"[gate] PASS: regenerated split matches {existing} (geometry/phi9/analytic-normals identical).")

    saved = save_fd_test_h5(merged_fd, args.out, config=config, split=args.split)
    print(f"Wrote {n} rows -> {saved}")


if __name__ == "__main__":
    main()
