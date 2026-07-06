"""Hash helpers for cleanroom source, case, and result provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


GENERATED_NAMES = {
    ".DS_Store",
    "qcc",
    "config",
    "grammar",
    "grammar.h",
    "Makefile.deps",
    "Makefile.tests",
}
GENERATED_SUFFIXES = {
    ".o",
    ".a",
    ".so",
    ".dylib",
    ".pyc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_manifest(root: Path, *, source_only: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    entries: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if source_only and _is_generated(path):
            continue
        if path.is_symlink():
            if source_only and path.name == "config":
                continue
            entries[rel] = {"type": "symlink", "target": path.readlink().as_posix()}
        elif path.is_file():
            entries[rel] = {"type": "file", "sha256": sha256_file(path)}
    return entries


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _is_generated(path: Path) -> bool:
    if any(part.endswith(".dSYM") for part in path.parts):
        return True
    return path.name in GENERATED_NAMES or path.suffix in GENERATED_SUFFIXES
