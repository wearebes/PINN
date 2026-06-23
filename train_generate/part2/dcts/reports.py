"""Diagnostic reports for DCTS Stage 0 (CSV outputs under reports/).

  - occupancy.csv                  : per (split, shape, fine_bin) quota/selected/deficiency
  - ellipse_diversity.csv          : distinct (q, psi, eta_max) per (split, fine_bin)
  - eta_distribution.csv           : per (split, fine_bin) canonical pack counts (log-uniformity)
  - gate11_normal_degeneration.csv : per (split, shape, fine_bin) FD ||grad phi|| degeneration
                                     + medial-axis / curvature-centre proximity (Gate 11, mandatory)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from train_generate.part2.io import write_csv
from train_generate.part2.dcts.config import DctsConfig


def write_occupancy(path: Path, records_by_split: dict[str, list[dict]]) -> None:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        for rec in records_by_split.get(split, []):
            rows.append(dict(rec))
    write_csv(path, rows)


def write_ellipse_diversity(path: Path, merged_by_split: dict[str, dict], config: DctsConfig) -> None:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        m = merged_by_split[split]
        shape = np.asarray(m["shape"])
        fb = np.asarray(m["fine_bin"])
        q = np.asarray(m["q"]); psi = np.asarray(m["psi"]); em = np.asarray(m["eta_max"])
        is_e = shape == "ellipse"
        for j in range(config.n_fine_bins):
            sel = is_e & (fb == j)
            if not np.any(sel):
                rows.append({"split": split, "fine_bin": j, "ellipse_packs": 0, "distinct_geometries": 0})
                continue
            tuples = {(round(float(a), 9), round(float(b), 9), round(float(c), 12))
                      for a, b, c in zip(q[sel], psi[sel], em[sel])}
            rows.append({"split": split, "fine_bin": j, "ellipse_packs": int(np.count_nonzero(sel)),
                         "distinct_geometries": len(tuples)})
    write_csv(path, rows)


def write_eta_distribution(path: Path, merged_by_split: dict[str, dict], config: DctsConfig) -> None:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        m = merged_by_split[split]
        fb = np.asarray(m["fine_bin"])
        shape = np.asarray(m["shape"])
        target = int(config.per_bin[split])
        for j in range(config.n_fine_bins):
            total = int(np.count_nonzero(fb == j))
            nc = int(np.count_nonzero((fb == j) & (shape == "circle")))
            ne = int(np.count_nonzero((fb == j) & (shape == "ellipse")))
            rows.append({"split": split, "fine_bin": j, "target": target, "total": total,
                         "circle": nc, "ellipse": ne, "deficit": int(max(0, target - total))})
    write_csv(path, rows)


def write_gate11(path: Path, merged_by_split: dict[str, dict], config: DctsConfig) -> dict:
    """Per (split, shape, fine_bin) FD degeneration + medial proximity. Returns summary."""
    thresholds = config.gate11_thresholds
    rows: list[dict] = []
    worst = {"min_fd_grad": np.inf, "min_medial": np.inf}

    def group_rows(split_label: str, shape_label: str, min_grad: np.ndarray, medial: np.ndarray, fb: np.ndarray):
        for j in range(config.n_fine_bins):
            sel = fb == j
            n = int(np.count_nonzero(sel))
            if n == 0:
                continue
            mg = min_grad[sel]
            md = medial[sel]
            row = {"split": split_label, "shape": shape_label, "fine_bin": j, "count": n}
            for t in thresholds:
                row[f"frac_fd_grad_lt_{t:g}"] = float(np.count_nonzero(mg < t) / n)
            row["min_fd_grad"] = float(mg.min())
            row["p1_fd_grad"] = float(np.percentile(mg, 1))
            row["medial_p1"] = float(np.percentile(md, 1))
            row["medial_p50"] = float(np.percentile(md, 50))
            row["medial_min"] = float(md.min())
            row["frac_medial_lt_1.0"] = float(np.count_nonzero(md < 1.0) / n)
            row["frac_medial_lt_0.5"] = float(np.count_nonzero(md < 0.5) / n)
            rows.append(row)
            worst["min_fd_grad"] = min(worst["min_fd_grad"], float(mg.min()))
            worst["min_medial"] = min(worst["min_medial"], float(md.min()))

    # Combine all splits' canonical packs per shape (geometry population view).
    pooled: dict[str, dict[str, list]] = {"circle": {}, "ellipse": {}}
    for split in ("train", "val", "test"):
        m = merged_by_split[split]
        shape = np.asarray(m["shape"])
        fb = np.asarray(m["fine_bin"])
        min_grad = np.min(np.asarray(m["fd_grad_norm9"]), axis=1)
        medial = np.asarray(m["medial_min_dist"])
        for shape_label in ("circle", "ellipse"):
            s = shape == shape_label
            group_rows(split, shape_label, min_grad[s], medial[s], fb[s])
            pooled[shape_label].setdefault("mg", []).append(min_grad[s])
            pooled[shape_label].setdefault("md", []).append(medial[s])
            pooled[shape_label].setdefault("fb", []).append(fb[s])
    for shape_label in ("circle", "ellipse"):
        if pooled[shape_label]:
            group_rows("all", shape_label,
                       np.concatenate(pooled[shape_label]["mg"]),
                       np.concatenate(pooled[shape_label]["md"]),
                       np.concatenate(pooled[shape_label]["fb"]))
    write_csv(path, rows)
    return worst
