from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from paper_section4.core import RESULTS_DIR, atomic_json, sha256_file
from paper_section4.validate_package import EXPECTED, validate_package


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _archive_originals(static_dir: Path) -> Path:
    archive = static_dir / "reference_original"
    archive.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    for figure in EXPECTED:
        for suffix in (".png", ".pdf"):
            source = static_dir / f"{figure}{suffix}"
            destination = archive / source.name
            if source.exists() and not destination.exists():
                _atomic_copy(source, destination)
            if destination.exists():
                records.append(
                    {
                        "file": destination.name,
                        "sha256": sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    }
                )
    return atomic_json(
        archive / "manifest.json",
        {
            "schema_version": 1,
            "role": "immutable pre-recompute manuscript reference",
            "files": sorted(records, key=lambda row: str(row["file"])),
        },
    )


def publish(cfd_root: Path) -> dict[str, object]:
    validate_package()
    root = cfd_root.resolve()
    static_dir = root / "figures" / "static"
    if not (root / "figures" / "cfd_style.py").exists():
        raise ValueError(f"Not a CFD checkout with figures/cfd_style.py: {root}")
    archive_manifest = _archive_originals(static_dir)
    published: list[dict[str, str]] = []
    for figure, (source_name, _) in EXPECTED.items():
        for name in (source_name, f"{figure}.provenance.json"):
            source = RESULTS_DIR / name
            destination = static_dir / name
            _atomic_copy(source, destination)
            if sha256_file(source) != sha256_file(destination):
                raise ValueError(f"Published copy hash mismatch: {name}")
            published.append({"file": name, "sha256": sha256_file(destination)})
    manifest = {
        "schema_version": 1,
        "source": "PINN paper_section4/results",
        "archive_manifest": str(archive_manifest.relative_to(root)),
        "files": sorted(published, key=lambda row: row["file"]),
    }
    atomic_json(static_dir / "source_data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish validated Section 4 source data into CFD figures/static")
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = publish(args.cfd_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
