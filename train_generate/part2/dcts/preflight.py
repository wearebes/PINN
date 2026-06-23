"""Main-scale capacity preflight (no replacement, no oversampling, no projection).

For every (split, shape, fine_bin) it checks whether the non-replacement candidate
pool can meet the quota. Circles are direct-by-quota (always exact). Ellipses draw
from the round-robin pool built by select.build_ellipse_candidates; capacity is the
sum over geometries of min(cap, in-band arc-length samples). A bin is deficient iff
total_available < quota.

Cheap: builds candidate t-lists only (eta tables + arc-length sampling), never
projects the 5x5 patches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from train_generate.part2.io import write_csv
from train_generate.part2.dcts import bins, select
from train_generate.part2.dcts.config import DctsConfig

SPLITS = ("train", "val", "test")


def ellipse_capacity(config: DctsConfig, *, split: str, fine_bin: int, edges: np.ndarray, centers: np.ndarray) -> dict:
    payloads = select.build_ellipse_candidates(config, split=split, fine_bin=fine_bin, edges=edges, centers=centers)
    per_geom = [len(p[2]) for p in payloads]
    cap = config.cap(split)
    quota = config.ellipse_quota(split)
    total = int(sum(per_geom))
    return {
        "split": split,
        "shape": "ellipse",
        "fine_bin": int(fine_bin),
        "quota": int(quota),
        "total_available": total,
        "candidate_geometries": int(len(per_geom)),
        "geoms_with_band": int(np.count_nonzero(per_geom)),
        "geoms_at_cap": int(np.count_nonzero([g >= cap for g in per_geom])),
        "max_per_geom": int(max(per_geom)) if per_geom else 0,
        "cap": int(cap),
        "deficit": int(max(0, quota - total)),
        "margin": int(total - quota),
    }


def run_preflight(config: DctsConfig, *, out_csv: Path | None = None) -> dict:
    edges = bins.fine_bin_edges(config.eta_min, config.eta_max, config.n_fine_bins)
    centers = bins.fine_bin_centers(edges)

    rows: list[dict] = []
    deficits: list[dict] = []
    for split in SPLITS:
        cq = config.circle_quota(split)
        eq = config.ellipse_quota(split)
        for j in range(config.n_fine_bins):
            # Circles: direct-by-quota, always exact.
            rows.append({
                "split": split, "shape": "circle", "fine_bin": j, "quota": cq,
                "total_available": cq, "candidate_geometries": "", "geoms_with_band": "",
                "geoms_at_cap": "", "max_per_geom": "", "cap": "", "deficit": 0, "margin": 0,
            })
            cap = ellipse_capacity(config, split=split, fine_bin=j, edges=edges, centers=centers)
            rows.append(cap)
            if cap["deficit"] > 0:
                deficits.append(cap)

    if out_csv is not None:
        write_csv(out_csv, rows)

    ell = [r for r in rows if r["shape"] == "ellipse"]
    summary = {
        "tag": config.tag,
        "ellipse_eta_max_scan": config.ellipse_eta_max_scan,
        "ellipse_dense_t": config.ellipse_dense_t,
        "ellipse_cap": config.ellipse_cap,
        "ellipse_quota": {s: config.ellipse_quota(s) for s in SPLITS},
        "deficient_cells": len(deficits),
        "total_deficit": int(sum(d["deficit"] for d in deficits)),
        "min_ellipse_margin": int(min(r["margin"] for r in ell)),
        "min_margin_cell": min(ell, key=lambda r: r["margin"]),
        "full_capacity": len(deficits) == 0,
        "deficits": deficits,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DCTS main-scale capacity preflight.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--main", action="store_true")
    group.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = DctsConfig.main() if args.main else DctsConfig.smoke()
    out = Path(config.output_dir) / "reports" / "capacity_preflight.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = run_preflight(config, out_csv=out)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if not summary["full_capacity"]:
        print(f"\nPREFLIGHT: {summary['deficient_cells']} deficient cells, total deficit {summary['total_deficit']}", file=sys.stderr)
        sys.exit(2)
    print("\nPREFLIGHT: full capacity — every (split, shape, fine_bin) can meet quota without replacement.")


if __name__ == "__main__":
    main()
