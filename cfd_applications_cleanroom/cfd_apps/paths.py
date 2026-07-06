"""Path helpers for the cleanroom route."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUTE_ROOT = ROOT / "cfd_applications_cleanroom"
RESULTS = ROUTE_ROOT / "results"
RAW = RESULTS / "raw"
REPORTS = RESULTS / "reports"
MANIFESTS = RESULTS / "manifests"
FIGURES = RESULTS / "figures"
SOURCE_DATA = RESULTS / "source_data"


def ensure_result_dirs() -> None:
    for path in (RAW, REPORTS, MANIFESTS, FIGURES, SOURCE_DATA):
        path.mkdir(parents=True, exist_ok=True)


def route_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
