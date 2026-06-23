"""DCTS Stage-0 dataset generator CLI.

  python -m train_generate.part2.dcts --smoke   # train 10k packs + all reports + gates
  python -m train_generate.part2.dcts --main    # full 100k/20k/20k

Generates the three splits, writes HDF5 + reports, runs Gate 1-11, and prints the
gate summary. Refuses to be silent about deficiencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train_generate.part2.dcts import bins, gates, reports, select, write
from train_generate.part2.dcts.config import DctsConfig

SPLITS = ("train", "val", "test")


def run_dcts(config: DctsConfig) -> dict:
    out = Path(config.output_dir)
    processed = out / "processed"
    report_dir = out / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    edges = bins.fine_bin_edges(config.eta_min, config.eta_max, config.n_fine_bins)
    centers = bins.fine_bin_centers(edges)

    merged_by_split: dict[str, dict] = {}
    records_by_split: dict[str, list[dict]] = {}
    write_summary: dict[str, dict] = {}
    for split in SPLITS:
        print(f"[dcts] generating split={split} ...", file=sys.stderr, flush=True)
        merged, records = select.generate_split(config, split=split, edges=edges, centers=centers)
        merged_by_split[split] = merged
        records_by_split[split] = records
        write_summary[split] = write.write_split_hdf5(config, merged, split=split, path=processed / f"{split}.h5")
        print(f"[dcts]   {write_summary[split]}", file=sys.stderr, flush=True)

    # Reports
    reports.write_occupancy(report_dir / "occupancy.csv", records_by_split)
    reports.write_ellipse_diversity(report_dir / "ellipse_diversity.csv", merged_by_split, config)
    reports.write_eta_distribution(report_dir / "eta_distribution.csv", merged_by_split, config)
    gate11_worst = reports.write_gate11(report_dir / "gate11_normal_degeneration.csv", merged_by_split, config)

    # Gates
    gate_results = gates.run_gates(config, merged_by_split, records_by_split, edges=edges, gate11_worst=gate11_worst)
    (report_dir / "gate_summary.json").write_text(json.dumps(gate_results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"DCTS Stage 0 — tag={config.tag}  output={out}")
    print("=" * 80)
    for split in SPLITS:
        print(f"  {split}: {write_summary[split]}")
    print("-" * 80)
    print(gates.format_gates(gate_results))
    return {"write": write_summary, "gates": gate_results, "output_dir": str(out.resolve())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the DCTS Stage-0 curvature dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Smoke dataset (train 10k packs).")
    group.add_argument("--main", action="store_true", help="Main dataset (train 100k packs).")
    parser.add_argument("--no-augment", action="store_true", help="Disable D4 x sign augmentation.")
    args = parser.parse_args(argv)

    config = DctsConfig.smoke() if args.smoke else DctsConfig.main()
    if args.no_augment:
        from dataclasses import replace
        config = replace(config, d4_sign_enabled=False)
    result = run_dcts(config)
    if not result["gates"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
