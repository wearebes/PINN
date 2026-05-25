from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .shared import compute_metrics


ANGLE_BIN_FIELDNAMES = (
    "rho_model",
    "case_id",
    "case_label",
    "iter",
    "case_key",
    "angle_bin_start_deg",
    "angle_bin_end_deg",
    "sample_count",
    "numeric_mse",
    "numeric_mae",
    "numeric_maxae",
    "model_mse",
    "model_mae",
    "model_maxae",
)


def sanitize_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw))


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv_rows(
    output_path: str | Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0].keys()) if rows else tuple()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_swanlab_image(swanlab_module: Any, image_path: str | Path) -> Any:
    image_path = Path(image_path).resolve()
    last_error: Exception | None = None
    constructor_candidates: list[tuple[Any, str]] = []
    if hasattr(swanlab_module, "Image"):
        constructor_candidates.append((getattr(swanlab_module, "Image"), str(image_path)))
    media_namespace = getattr(swanlab_module, "media", None)
    if media_namespace is not None and hasattr(media_namespace, "Image"):
        constructor_candidates.append((getattr(media_namespace, "Image"), str(image_path)))
    data_namespace = getattr(swanlab_module, "data", None)
    if data_namespace is not None and hasattr(data_namespace, "Image"):
        constructor_candidates.append((getattr(data_namespace, "Image"), str(image_path)))
    for constructor, value in constructor_candidates:
        try:
            return constructor(value)
        except Exception as exc:  # pragma: no cover - defensive SDK fallback
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Unable to construct a SwanLab image object for {image_path}: {last_error}") from last_error
    raise RuntimeError("The installed SwanLab SDK does not expose an Image constructor.")


def build_swanlab_table_payload(
    swanlab_module: Any,
    rows: list[dict[str, Any]] | list[list[Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> Any:
    if rows and isinstance(rows[0], dict):
        dict_rows = [dict(item) for item in rows]  # shallow copy for deterministic order
        headers = list(fieldnames or dict_rows[0].keys())
        table_rows: list[list[Any]] = [headers]
        for row in dict_rows:
            table_rows.append([row.get(name, "") for name in headers])
    else:
        table_rows = list(rows)
    echarts_namespace = getattr(swanlab_module, "echarts", None)
    if echarts_namespace is not None and hasattr(echarts_namespace, "table"):
        return echarts_namespace.table(table_rows)
    text_type = getattr(swanlab_module, "Text", None)
    if text_type is not None:
        return text_type("\n".join(" | ".join(str(item) for item in row) for row in table_rows))
    return table_rows


def metric_summary(metric: dict[str, float]) -> dict[str, float]:
    mse = float(metric["mse"])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(metric["mae"]),
        "max_abs_err": float(metric["maxae"]),
    }


def build_case_summary_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        case_rows,
        key=lambda row: (float(row["summary"]["mae"]), str(row["case_key"]), int(row["iter"]), int(row["case_id"])),
    )
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        summary = row["summary"]
        numeric_summary = row["numeric_summary"]
        rows.append(
            {
                "rank": int(rank),
                "rho_model": int(row.get("rho_model", -1)),
                "case_key": str(row["case_key"]),
                "case_id": int(row["case_id"]),
                "case_label": str(row["case_label"]),
                "iter": int(row["iter"]),
                "sample_count": int(row["sample_count"]),
                "rmse": float(summary["rmse"]),
                "mae": float(summary["mae"]),
                "max_abs_err": float(summary["max_abs_err"]),
                "numeric_rmse": float(numeric_summary["rmse"]),
                "numeric_mae": float(numeric_summary["mae"]),
                "numeric_max_abs_err": float(numeric_summary["max_abs_err"]),
            }
        )
    return rows


def _empty_metric_row() -> dict[str, float]:
    return {"mse": float("nan"), "mae": float("nan"), "maxae": float("nan")}


def validate_angle_bin_deg(bin_deg: float) -> int:
    bin_deg_float = float(bin_deg)
    if not np.isfinite(bin_deg_float) or bin_deg_float <= 0.0:
        raise ValueError(f"bin_deg must be a positive finite value, got {bin_deg!r}.")
    bin_deg_int = int(round(bin_deg_float))
    if not np.isclose(bin_deg_float, float(bin_deg_int)):
        raise ValueError(f"bin_deg must be an integer number of degrees, got {bin_deg!r}.")
    if 360 % bin_deg_int != 0:
        raise ValueError(f"bin_deg must divide 360 exactly, got {bin_deg_int}.")
    return bin_deg_int


def build_angle_bin_rows(case_rows: list[dict[str, Any]], *, bin_deg: float) -> list[dict[str, Any]]:
    bin_deg_int = validate_angle_bin_deg(bin_deg)
    n_bins = 360 // bin_deg_int
    edges_deg = np.arange(n_bins + 1, dtype=np.float64) * float(bin_deg_int)
    rows: list[dict[str, Any]] = []

    for case_row in case_rows:
        theta = np.asarray(case_row["theta"], dtype=np.float64)
        pred_hkappa = np.asarray(case_row["pred_hkappa"], dtype=np.float64)
        true_hkappa = np.asarray(case_row["true_hkappa"], dtype=np.float64)
        numeric_hkappa = np.asarray(case_row["numeric_hkappa"], dtype=np.float64)
        theta_deg = np.mod(np.degrees(theta), 360.0)
        bin_idx = np.minimum(np.floor(theta_deg / float(bin_deg_int)).astype(np.int64), n_bins - 1)
        for idx in range(n_bins):
            mask = bin_idx == idx
            if int(np.count_nonzero(mask)) > 0:
                numeric_metric = compute_metrics(numeric_hkappa[mask], true_hkappa[mask])
                model_metric = compute_metrics(pred_hkappa[mask], true_hkappa[mask])
            else:
                numeric_metric = _empty_metric_row()
                model_metric = _empty_metric_row()
            rows.append(
                {
                    "rho_model": int(case_row.get("rho_model", -1)),
                    "case_id": int(case_row["case_id"]),
                    "case_label": str(case_row["case_label"]),
                    "iter": int(case_row["iter"]),
                    "case_key": str(case_row["case_key"]),
                    "angle_bin_start_deg": int(edges_deg[idx]),
                    "angle_bin_end_deg": int(edges_deg[idx + 1]),
                    "sample_count": int(np.count_nonzero(mask)),
                    "numeric_mse": float(numeric_metric["mse"]),
                    "numeric_mae": float(numeric_metric["mae"]),
                    "numeric_maxae": float(numeric_metric["maxae"]),
                    "model_mse": float(model_metric["mse"]),
                    "model_mae": float(model_metric["mae"]),
                    "model_maxae": float(model_metric["maxae"]),
                }
            )
    return rows


def _sorted_case_arrays(case_entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(case_entry["theta"], dtype=np.float64).reshape(-1)
    xy = np.asarray(case_entry["xy"], dtype=np.float64)
    analytic = np.asarray(case_entry["true_hkappa"], dtype=np.float64).reshape(-1)
    numeric = np.asarray(case_entry["numeric_hkappa"], dtype=np.float64).reshape(-1)
    prediction = np.asarray(case_entry["pred_hkappa"], dtype=np.float64).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy with shape (N, 2), got {xy.shape}.")
    if not (theta.shape[0] == xy.shape[0] == analytic.shape[0] == numeric.shape[0] == prediction.shape[0]):
        raise ValueError("Case arrays do not have matching lengths.")
    order = np.argsort(theta, kind="mergesort")
    return (
        theta[order],
        xy[order],
        analytic[order],
        numeric[order],
        prediction[order],
    )


def _closed_xy(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return xy
    return np.vstack([xy, xy[:1]])


def render_curvature_overview(
    case_entries: list[dict[str, Any]],
    output_path: str | Path,
    *,
    suptitle: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not case_entries:
        raise ValueError("At least one case entry is required to render a curvature overview.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(case_entries)
    fig, axes = plt.subplots(3, ncols, figsize=(5.5 * ncols, 11.0), squeeze=False, constrained_layout=True)

    for col_idx, case_entry in enumerate(case_entries):
        theta, xy, analytic, numeric, prediction = _sorted_case_arrays(case_entry)
        label = str(case_entry.get("title") or case_entry["case_key"])

        ax_shape = axes[0, col_idx]
        curve_xy = _closed_xy(xy)
        ax_shape.plot(curve_xy[:, 0], curve_xy[:, 1], color="0.80", linewidth=1.0)
        scatter = ax_shape.scatter(xy[:, 0], xy[:, 1], c=analytic, cmap="coolwarm", s=10, linewidths=0.0)
        fig.colorbar(scatter, ax=ax_shape, fraction=0.046, pad=0.04)
        ax_shape.set_aspect("equal")
        ax_shape.set_xticks([])
        ax_shape.set_yticks([])
        ax_shape.set_title(f"{label}\nBoundary curvature", fontsize=10)

        ax_curve = axes[1, col_idx]
        ax_curve.plot(theta, analytic, label="analytic", linewidth=1.6)
        ax_curve.plot(theta, numeric, label="numeric", linewidth=1.4, linestyle="--")
        ax_curve.plot(theta, prediction, label="model", linewidth=1.4)
        ax_curve.set_title("h*kappa(theta)", fontsize=10)
        ax_curve.set_xlabel("theta")
        ax_curve.set_ylabel("h*kappa")
        ax_curve.grid(True, alpha=0.25)
        if col_idx == 0:
            ax_curve.legend(loc="best")

        ax_err = axes[2, col_idx]
        ax_err.plot(theta, np.abs(numeric - analytic), label="|numeric-analytic|", linewidth=1.4, linestyle="--")
        ax_err.plot(theta, np.abs(prediction - analytic), label="|model-analytic|", linewidth=1.4)
        ax_err.set_title("Absolute error", fontsize=10)
        ax_err.set_xlabel("theta")
        ax_err.set_ylabel("|error|")
        ax_err.grid(True, alpha=0.25)
        if col_idx == 0:
            ax_err.legend(loc="best")

    if suptitle:
        fig.suptitle(str(suptitle), fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
