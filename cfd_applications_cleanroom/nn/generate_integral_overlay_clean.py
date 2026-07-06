#!/usr/bin/env python3
"""Generate a clean integral overlay artifact without editing official source."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from cfd_applications_cleanroom.cfd_apps.hashes import sha256_file


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "cfd_applications_cleanroom/vendor/basilisk_clean/src/integral.h"
OUT_DIR = REPO / "cfd_applications_cleanroom/nn/generated"


def generate(source: Path = DEFAULT_SOURCE) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    before = sha256_file(source)
    text = source.read_text()
    overlay = (
        "/* Clean overlay generated from official integral.h.\n"
        " * This file is an overlay artifact; the official source is not edited.\n"
        " */\n"
        + text
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OUT_DIR / "integral_nn_clean.h"
    diff_path = OUT_DIR / "integral_nn_clean.diff"
    meta_path = OUT_DIR / "integral_nn_clean.meta.json"
    overlay_path.write_text(overlay)
    diff_path.write_text(
        "".join(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                overlay.splitlines(keepends=True),
                fromfile="vendor/basilisk_clean/src/integral.h",
                tofile="generated/integral_nn_clean.h",
            )
        )
    )
    after = sha256_file(source)
    meta = {
        "source_path": source.relative_to(REPO).as_posix(),
        "official_integral_h_hash_before": before,
        "official_integral_h_hash_after": after,
        "overlay_contains_no_official_path_write": True,
        "overlay_path": overlay_path.relative_to(REPO).as_posix(),
        "diff_path": diff_path.relative_to(REPO).as_posix(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if before != after:
        raise RuntimeError("official_integral_h_hash_changed")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    meta = generate(args.source)
    print(f"integral_overlay={meta['overlay_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
