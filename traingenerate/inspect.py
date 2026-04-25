from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from traingenerate.generate import build_dataset_summary_from_hdf5, read_dataset_manifest
else:
    from .generate import build_dataset_summary_from_hdf5, read_dataset_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a generated PINN dataset manifest or HDF5 summary.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset HDF5 file.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path.resolve()}")

    summary = read_dataset_manifest(dataset_path)
    if summary is None:
        summary = build_dataset_summary_from_hdf5(dataset_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
