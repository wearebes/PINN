"""Test 2 (part2): evaluate across real grid resolutions {32,64,128,256,512,1024}.

Test 1 (evaluate/part2/fd_testset.py) used DCTS's dimensionless local-patch
generator: h=1 always, the patch centre sits at a free *continuous* offset d0
from the interface. This project already measured that representation to be
resolution-independent by construction (phi/h matches to ~1e-15 across
rho=64 vs rho=512 -- see memory part2-training-data-contract.md), so a
literal resolution sweep built the same way would just re-confirm that fact.

What a real solver grid adds that the continuous patch cannot capture is
QUANTIZATION: the model never gets to sit its stencil exactly on the
interface-relative offset it would like -- it only has nodes at integer
multiples of h = 1/(rho-1). So here we:
  1. place a circle/ellipse at a random sub-pixel phase,
  2. compute the true interface point + a continuous d0 offset along the
     normal (same d0 convention as DCTS/Test 1),
  3. SNAP that point to the nearest real grid node at spacing h,
  4. evaluate phi at the 25 real, absolute-coordinate stencil nodes around it.

circle:  analytic SDF,                  central-difference nx9/ny9 (Test 1 convention)
ellipse: float64 Newton-projected SDF,  central-difference nx9/ny9 (NO mpmath --
         DCTS's own v2.1 plan locked float64-only Newton after measuring mpmath
         was both unnecessary and, over a full grid, computationally
         infeasible at rho>~64; this module never builds a full grid at all,
         only the 25 stencil points per sample, so it stays cheap even at
         rho=1024).

Usage:
    python -m evaluate.part2.resolution_sweep
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
from evaluate.part2.evaluate_fd_testset import discover_checkpoints, evaluate_one
from evaluate.part2.fd_testset import central_diff_normals
from evaluate.shared import compute_metrics, write_csv_rows
from train_generate.geometry_core import TWO_PI, ellipse_hkappa_from_theta
from train_generate.part2.dcts import bins, patch
from train_generate.part2.dcts.config import DctsConfig
from train_generate.part2.generate import central_difference_hkappa_from_phi9_float64

RESOLUTIONS = (32, 64, 128, 256, 512, 1024)
N_PER_RESOLUTION = 20_000
CIRCLE_FRACTION = 0.30
N_FINE_BINS = 100
N_COARSE_REGIMES = 8
STENCIL_DI = np.array([-2, -1, 0, 1, 2], dtype=np.float64)

DEFAULT_MODELS_DIR = Path("out/7367")
DEFAULT_CACHE_H5 = Path("dataset/part2_dcts/main/processed/resolution_sweep.h5")
DEFAULT_OUTPUT_CSV = DEFAULT_MODELS_DIR / "test2_resolution_sweep_metrics.csv"
DEFAULT_OUTPUT_PLOT = DEFAULT_MODELS_DIR / "test2_resolution_sweep.png"


def _project_theta_batched(
    u: np.ndarray, v: np.ndarray, *, a: np.ndarray, b: np.ndarray, max_iter: int, tol: float,
) -> np.ndarray:
    """Vectorized float64 Newton ellipse projection with PER-ELEMENT a/b.

    train_generate.geometry_core.project_theta_to_axis_aligned_ellipse's main
    loop already supports array a/b via elementwise broadcasting, but its
    slow-point fallback assumes scalar a/b (DCTS only ever calls it with one
    ellipse, hence one scalar a/b, at a time). This evaluation script batches
    many distinct ellipses together, so we keep our own copy of just the
    vectorized loop -- no fallback -- rather than changing that shared,
    gated training-pipeline function. 30 float64 Newton iterations on a
    smooth, well-conditioned target converges quadratically; any residual
    left after `max_iter` is far below what matters for an MSE diagnostic.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    theta = np.mod(np.arctan2(a * v, b * u), TWO_PI)
    for _ in range(int(max_iter)):
        st = np.sin(theta)
        ct = np.cos(theta)
        g = (b * b - a * a) * st * ct + a * u * st - b * v * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u * ct + b * v * st
        safe = np.abs(gp) >= 1.0e-18
        step = np.zeros_like(theta)
        step[safe] = g[safe] / gp[safe]
        theta = np.mod(theta - step, TWO_PI)
        if bool(np.all(np.abs(g) <= tol)):
            break
    return theta


def _grid_snap_stencil(x0: np.ndarray, y0: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Snap each continuous (x0,y0) to the nearest real grid node at spacing h.

    Returns the 25 absolute stencil coordinates (n,5,5), axis0=ix axis1=iy,
    matching train_generate.part2.dcts.patch's [ix,iy] convention.
    """
    i0 = np.round(x0 / h)
    j0 = np.round(y0 / h)
    xs = (i0[:, None] + STENCIL_DI[None, :]) * h  # (n,5)
    ys = (j0[:, None] + STENCIL_DI[None, :]) * h  # (n,5)
    n = x0.shape[0]
    x25 = np.broadcast_to(xs[:, :, None], (n, 5, 5))
    y25 = np.broadcast_to(ys[:, None, :], (n, 5, 5))
    return x25, y25


def _finish_patches(phi25: np.ndarray, *, hk_exact: np.ndarray, h: float, shape: str) -> dict:
    n = int(phi25.shape[0])
    phi9 = patch.inner_phi9(phi25)
    nx9, ny9 = central_diff_normals(phi25)
    hk_central = central_difference_hkappa_from_phi9_float64(phi9).reshape(-1)
    features27 = np.concatenate([phi9 / h, nx9, ny9], axis=1).astype(np.float32)
    return {
        "features27": features27,
        "target_hk": np.asarray(hk_exact, dtype=np.float64).reshape(-1),
        "hk_central": hk_central.astype(np.float64),
        "shape": [shape] * n,
    }


def sample_circle_patches(rho: int, n: int, *, rng: np.random.Generator, eta_min: float, eta_max: float) -> dict:
    if n <= 0:
        return _finish_patches(np.zeros((0, 5, 5)), hk_exact=np.zeros((0,)), h=1.0 / (rho - 1), shape="circle")
    h = 1.0 / (rho - 1)
    eta = np.exp(rng.uniform(np.log(eta_min), np.log(eta_max), size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    d0 = rng.uniform(-0.5, 0.5, size=n) * h
    cx = rng.uniform(0.0, h, size=n)
    cy = rng.uniform(0.0, h, size=n)
    r = h / eta

    px = cx + r * np.cos(theta)
    py = cy + r * np.sin(theta)
    x0 = px + d0 * np.cos(theta)
    y0 = py + d0 * np.sin(theta)
    x25, y25 = _grid_snap_stencil(x0, y0, h)

    phi25 = np.sqrt((x25 - cx[:, None, None]) ** 2 + (y25 - cy[:, None, None]) ** 2) - r[:, None, None]
    return _finish_patches(phi25, hk_exact=eta, h=h, shape="circle")


def sample_ellipse_patches(
    rho: int, n: int, *, rng: np.random.Generator, eta_min: float, eta_max: float,
    q_values: tuple[float, ...], psi_values: tuple[float, ...],
) -> dict:
    if n <= 0:
        return _finish_patches(np.zeros((0, 5, 5)), hk_exact=np.zeros((0,)), h=1.0 / (rho - 1), shape="ellipse")
    h = 1.0 / (rho - 1)
    q = rng.choice(np.asarray(q_values, dtype=np.float64), size=n)
    psi = rng.choice(np.asarray(psi_values, dtype=np.float64), size=n)
    eta_max_local = np.exp(rng.uniform(np.log(eta_min), np.log(eta_max), size=n))
    a = h / (eta_max_local * q * q)
    b = q * a
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    d0 = rng.uniform(-0.5, 0.5, size=n) * h
    cx = rng.uniform(0.0, h, size=n)
    cy = rng.uniform(0.0, h, size=n)

    pu = a * np.cos(t)
    pv = b * np.sin(t)
    nlu = b * np.cos(t)
    nlv = a * np.sin(t)
    nmag = np.sqrt(nlu * nlu + nlv * nlv)
    nlu, nlv = nlu / nmag, nlv / nmag
    x0_u = pu + d0 * nlu
    x0_v = pv + d0 * nlv
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    x0 = cx + x0_u * cos_psi - x0_v * sin_psi
    y0 = cy + x0_u * sin_psi + x0_v * cos_psi
    x25, y25 = _grid_snap_stencil(x0, y0, h)

    cos_psi5 = cos_psi[:, None, None]
    sin_psi5 = sin_psi[:, None, None]
    dx = x25 - cx[:, None, None]
    dy = y25 - cy[:, None, None]
    u25 = dx * cos_psi5 + dy * sin_psi5
    v25 = -dx * sin_psi5 + dy * cos_psi5

    a25 = np.broadcast_to(a[:, None, None], (n, 5, 5))
    b25 = np.broadcast_to(b[:, None, None], (n, 5, 5))
    theta25 = _project_theta_batched(
        u25.reshape(-1), v25.reshape(-1), a=a25.reshape(-1), b=b25.reshape(-1), max_iter=30, tol=1.0e-12,
    ).reshape(n, 5, 5)
    qu = a25 * np.cos(theta25)
    qv = b25 * np.sin(theta25)
    dist = np.sqrt((u25 - qu) ** 2 + (v25 - qv) ** 2)
    inside = (u25 ** 2) / (a25 ** 2) + (v25 ** 2) / (b25 ** 2) < 1.0
    phi25 = np.where(inside, -dist, dist)

    hk_exact = ellipse_hkappa_from_theta(t, h=h, a=a, b=b)
    return _finish_patches(phi25, hk_exact=hk_exact, h=h, shape="ellipse")


def build_resolution_dataset(
    rho: int, *, seed: int, eta_min: float, eta_max: float,
    q_values: tuple[float, ...], psi_values: tuple[float, ...],
    n_total: int = N_PER_RESOLUTION, circle_fraction: float = CIRCLE_FRACTION,
) -> dict:
    rng = np.random.default_rng(seed + int(rho))
    n_circle = int(round(n_total * circle_fraction))
    n_ellipse = n_total - n_circle
    circle = sample_circle_patches(rho, n_circle, rng=rng, eta_min=eta_min, eta_max=eta_max)
    ellipse = sample_ellipse_patches(
        rho, n_ellipse, rng=rng, eta_min=eta_min, eta_max=eta_max, q_values=q_values, psi_values=psi_values,
    )
    merged = {
        key: np.concatenate([circle[key], ellipse[key]], axis=0) for key in ("features27", "target_hk", "hk_central")
    }
    merged["shape"] = circle["shape"] + ellipse["shape"]

    edges = bins.fine_bin_edges(eta_min, eta_max, N_FINE_BINS)
    fine_bin = bins.fine_bin_index(merged["target_hk"], edges)
    merged["fine_bin"] = np.asarray(fine_bin, dtype=np.int64)
    merged["coarse_regime"] = np.asarray(
        bins.coarse_regime(fine_bin, N_FINE_BINS, N_COARSE_REGIMES), dtype=np.int64,
    )
    merged["rho"] = np.full(merged["target_hk"].shape[0], int(rho), dtype=np.int32)
    merged["h"] = np.full(merged["target_hk"].shape[0], 1.0 / (rho - 1), dtype=np.float64)
    return merged


def save_resolution_sweep_h5(datasets_by_rho: dict[int, dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [datasets_by_rho[rho] for rho in sorted(datasets_by_rho)]
    merged = {
        key: np.concatenate([d[key] for d in ordered], axis=0)
        for key in ("features27", "target_hk", "hk_central", "fine_bin", "coarse_regime", "rho", "h")
    }
    shape = [s for d in ordered for s in d["shape"]]
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_family"] = "part2_resolution_sweep"
        handle.attrs["resolutions"] = list(sorted(datasets_by_rho))
        handle.attrs["normal_source"] = "central_diff_grid_snapped"
        handle.attrs["sdf_mode"] = "sdf"
        handle.attrs["row_count"] = int(merged["features27"].shape[0])
        handle.create_dataset("features27", data=merged["features27"])
        handle.create_dataset("target_hk", data=merged["target_hk"].astype(np.float32).reshape(-1, 1))
        handle.create_dataset("hk_central", data=merged["hk_central"].astype(np.float32).reshape(-1, 1))
        handle.create_dataset("fine_bin", data=merged["fine_bin"].astype(np.int32))
        handle.create_dataset("coarse_regime", data=merged["coarse_regime"].astype(np.int32))
        handle.create_dataset("rho", data=merged["rho"].astype(np.int32))
        handle.create_dataset("h", data=merged["h"].astype(np.float64))
        handle.create_dataset("shape", data=shape, dtype=h5py.string_dtype(encoding="utf-8"))
    return path


def load_resolution_sweep_h5(path: str | Path) -> dict[int, dict]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        rho_col = np.asarray(handle["rho"][:], dtype=np.int32)
        features27 = np.asarray(handle["features27"][:], dtype=np.float32)
        target_hk = np.asarray(handle["target_hk"][:], dtype=np.float64).reshape(-1)
        hk_central = np.asarray(handle["hk_central"][:], dtype=np.float64).reshape(-1)
        coarse_regime = np.asarray(handle["coarse_regime"][:], dtype=np.int64)
        shape = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["shape"][:]]
    out: dict[int, dict] = {}
    for rho in sorted(set(int(v) for v in rho_col.tolist())):
        mask = rho_col == rho
        out[rho] = {
            "features27": features27[mask],
            "target_hk": target_hk[mask],
            "hk_central": hk_central[mask],
            "coarse_regime": coarse_regime[mask],
            "shape": [s for s, keep in zip(shape, mask.tolist()) if keep],
        }
    return out


def _make_plot(plot_data: dict[str, dict[int, float]], numeric_baseline: dict[int, float], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(
        RESOLUTIONS, [numeric_baseline[r] for r in RESOLUTIONS],
        color="#7f7f7f", linestyle="--", marker="x", linewidth=1.5, markersize=7,
        label="Central diff (no model)", zorder=1,
    )
    for stem, per_rho in plot_data.items():
        ax.plot(
            RESOLUTIONS, [per_rho[r] for r in RESOLUTIONS],
            marker="o", linewidth=2, markersize=6, label=stem, zorder=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(RESOLUTIONS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Grid resolution (rho)", fontsize=12)
    ax.set_ylabel("MSE (vs analytic h*kappa)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Test 2: part2 model MSE vs grid resolution (central-diff, grid-snapped)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test 2 (part2): evaluate the DCTS h*kappa model(s) across real grid resolutions."
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--n-per-resolution", type=int, default=N_PER_RESOLUTION)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--cache-h5", default=str(DEFAULT_CACHE_H5))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-plot", default=str(DEFAULT_OUTPUT_PLOT))
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DctsConfig.main()

    cache_path = Path(args.cache_h5)
    if cache_path.exists() and not args.rebuild:
        print(f"[resolution_sweep] loading cached dataset <- {cache_path}")
        datasets_by_rho = load_resolution_sweep_h5(cache_path)
    else:
        datasets_by_rho = {}
        for rho in RESOLUTIONS:
            print(f"[resolution_sweep] building rho={rho} (h={1.0/(rho-1):.6e}) ...")
            datasets_by_rho[rho] = build_resolution_dataset(
                rho, seed=args.seed, eta_min=config.eta_min, eta_max=config.eta_max,
                q_values=config.q_values, psi_values=config.psi_values, n_total=args.n_per_resolution,
            )
        save_resolution_sweep_h5(datasets_by_rho, cache_path)
        print(f"[resolution_sweep] wrote cache -> {cache_path}")

    checkpoints = discover_checkpoints(Path(args.models_dir), args.models)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {args.models_dir} (or pass --models explicitly).")

    rows: list[dict] = []
    plot_data: dict[str, dict[int, float]] = {m.stem: {} for m in checkpoints}
    numeric_baseline: dict[int, float] = {}
    for rho in RESOLUTIONS:
        dataset = datasets_by_rho[rho]
        numeric_baseline[rho] = compute_metrics(dataset["hk_central"], dataset["target_hk"])["mse"]
        for model_path in checkpoints:
            result = evaluate_one(model_path, dataset, device=device)
            plot_data[model_path.stem][rho] = result["model_overall"]["mse"]
            m, b = result["model_overall"], result["numeric_overall"]
            print(
                f"rho={rho:5d} {model_path.stem:<20s} N={result['n']:6d}  "
                f"model MSE={m['mse']:.4e} MAE={m['mae']:.4e} MaxAE={m['maxae']:.4e}  |  "
                f"central-diff MSE={b['mse']:.4e}"
            )
            for group_kind, group_rows in (("shape", result["by_shape"]), ("coarse_regime", result["by_band"])):
                for row in group_rows:
                    gm, gb = row["model_vs_analytic"], row["numeric_vs_analytic"]
                    rows.append({
                        "resolution": rho, "model": model_path.stem, "group_kind": group_kind,
                        "group_value": row["display"], "sample_count": row["sample_count"],
                        "model_mse": gm["mse"], "model_mae": gm["mae"], "model_maxae": gm["maxae"],
                        "central_diff_mse": gb["mse"], "central_diff_mae": gb["mae"], "central_diff_maxae": gb["maxae"],
                    })
        print()

    output_csv = Path(args.output_csv)
    write_csv_rows(output_csv, rows, fieldnames=(
        "resolution", "model", "group_kind", "group_value", "sample_count",
        "model_mse", "model_mae", "model_maxae", "central_diff_mse", "central_diff_mae", "central_diff_maxae",
    ))
    print(f"Wrote per-group metrics -> {output_csv}")

    plot_path = _make_plot(plot_data, numeric_baseline, Path(args.output_plot))
    print(f"Wrote plot -> {plot_path}")


if __name__ == "__main__":
    main()
