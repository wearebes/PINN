from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results"
RESOLUTIONS = (32, 64, 128, 256, 512)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return value.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return value.name


def atomic_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, output)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return output


def atomic_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, output)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return output


def atomic_gzip_csv(
    path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        with Path(tmp_name).open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with zipped_text(zipped) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        os.replace(tmp_name, output)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return output


class zipped_text:
    def __init__(self, stream: gzip.GzipFile):
        import io

        self._wrapper = io.TextIOWrapper(stream, encoding="utf-8", newline="")

    def __enter__(self):
        return self._wrapper

    def __exit__(self, exc_type, exc, tb):
        self._wrapper.flush()
        self._wrapper.detach()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    value = Path(path)
    opener = gzip.open if value.suffix == ".gz" else open
    with opener(value, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def model_path(rho: int) -> Path:
    return REPO_ROOT / "out" / str(rho) / f"baseline_{rho}_hgradient.pt"


def normalization_path(rho: int) -> Path:
    return model_path(rho).with_suffix(".csv")


def source_paths() -> list[Path]:
    return [
        PACKAGE_DIR / "core.py",
        PACKAGE_DIR / "recompute.py",
        REPO_ROOT / "train_generate" / "config.py",
        REPO_ROOT / "train_generate" / "generate.py",
        REPO_ROOT / "train_generate" / "geometry_core.py",
        REPO_ROOT / "testdata_generate" / "config.py",
        REPO_ROOT / "testdata_generate" / "generate.py",
        REPO_ROOT / "testdata_generate" / "reinit.py",
        REPO_ROOT / "evaluate" / "shared.py",
        REPO_ROOT / "evaluate" / "flower.py",
    ]


def environment_record() -> dict[str, str]:
    import h5py
    import scipy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "torch": torch.__version__,
    }


def input_hashes(extra: Iterable[Path] = ()) -> dict[str, str]:
    paths = [
        *(model_path(rho) for rho in RESOLUTIONS),
        *(normalization_path(rho) for rho in RESOLUTIONS),
        *source_paths(),
        *extra,
    ]
    return {relative_to_repo(path): sha256_file(path) for path in paths}


def write_provenance(
    figure: str,
    source_data: Path,
    *,
    protocol: dict[str, Any],
    extra_inputs: Iterable[Path] = (),
) -> Path:
    payload = {
        "schema_version": 1,
        "authority": "current local baseline_hgradient checkpoints",
        "figure": figure,
        "source_data": source_data.name,
        "source_data_sha256": sha256_file(source_data),
        "inputs_sha256": input_hashes(extra_inputs),
        "environment": environment_record(),
        "protocol": protocol,
    }
    return atomic_json(RESULTS_DIR / f"{figure}.provenance.json", payload)


@dataclass
class MetricAccumulator:
    count: int = 0
    sum_squared: float = 0.0
    sum_absolute: float = 0.0
    max_absolute: float = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
        truth = np.asarray(target, dtype=np.float64).reshape(-1)
        if pred.shape != truth.shape:
            raise ValueError(f"Prediction/target mismatch: {pred.shape} != {truth.shape}")
        difference = pred - truth
        absolute = np.abs(difference)
        self.count += int(truth.size)
        self.sum_squared += float(np.sum(difference * difference, dtype=np.float64))
        self.sum_absolute += float(np.sum(absolute, dtype=np.float64))
        self.max_absolute = max(self.max_absolute, float(np.max(absolute)))

    def row(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise ValueError("Metric accumulator is empty")
        return {
            "sample_count": self.count,
            "mse": self.sum_squared / self.count,
            "mae": self.sum_absolute / self.count,
            "maxae": self.max_absolute,
        }
