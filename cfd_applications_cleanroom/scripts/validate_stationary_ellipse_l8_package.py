#!/usr/bin/env python3
"""Validate a stationary-ellipse L8 transfer package before extraction."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
CASES = ("E1", "E2")
EXPECTED = {(case, method) for case in CASES for method in METHODS}
RESULT_DIR_RE = re.compile(
    r"^cfd_applications_cleanroom/results/raw/stationary_ellipse/"
    r"stationary_ellipse_curvature_process_(E[12])_[^/]+/"
    r"(NN_DISABLE|NN_PROBE_ONLY|NN27_RAW|NN27_D4)_L8$"
)


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def member_text(package: tarfile.TarFile, name: str) -> str:
    extracted = package.extractfile(name)
    if extracted is None:
        raise ValueError(f"unreadable_member:{name}")
    return extracted.read().decode("utf-8")


def validate_member_names(names: list[str]) -> list[str]:
    errors: list[str] = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"unsafe_tar_member:{name}")
    return errors


def validate_package(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"package_not_found:{path}"]

    try:
        package = tarfile.open(path, "r:gz")
    except tarfile.TarError as exc:
        return [f"unreadable_tar:{path}:{exc}"]

    with package:
        names = package.getnames()
        name_set = set(names)
        errors.extend(validate_member_names(names))

        manifests = [
            name for name in names
            if name.startswith("cfd_applications_cleanroom/results/transfer/")
            and name.endswith(".manifest.txt")
        ]
        path_lists = [
            name for name in names
            if name.startswith("cfd_applications_cleanroom/results/transfer/")
            and name.endswith(".paths.txt")
        ]
        if len(manifests) != 1:
            errors.append(f"manifest_count={len(manifests)}")
        if len(path_lists) != 1:
            errors.append(f"path_list_count={len(path_lists)}")
        if errors:
            return errors

        path_lines = [
            line.strip()
            for line in member_text(package, path_lists[0]).splitlines()
            if line.strip()
        ]
        if len(path_lines) != 8:
            errors.append(f"path_list_row_count={len(path_lines)}")

        found: dict[tuple[str, str], str] = {}
        for line in path_lines:
            match = RESULT_DIR_RE.match(line)
            if not match:
                errors.append(f"unexpected_result_dir:{line}")
                continue
            case, method = match.groups()
            key = (case, method)
            if key in found:
                errors.append(f"duplicate_result_dir:{case}:{method}")
            found[key] = line

        missing = sorted(EXPECTED - set(found))
        extra = sorted(set(found) - EXPECTED)
        for case, method in missing:
            errors.append(f"missing_result_dir:{case}:{method}")
        for case, method in extra:
            errors.append(f"extra_result_dir:{case}:{method}")

        if errors:
            return errors

        for (case, method), result_dir in sorted(found.items()):
            summary_name = f"{result_dir}/summary.json"
            trace_name = f"{result_dir}/surface_tension_trace.csv"
            if summary_name not in name_set:
                errors.append(f"missing_summary_json:{case}:{method}:{summary_name}")
                continue
            if trace_name not in name_set:
                errors.append(f"missing_surface_tension_trace:{case}:{method}:{trace_name}")
            try:
                data: dict[str, Any] = json.loads(member_text(package, summary_name))
            except Exception as exc:
                errors.append(f"unreadable_summary_json:{case}:{method}:{exc}")
                continue
            final_tau = safe_float(data.get("final_tau"))
            if (
                data.get("method") != method
                or int(data.get("level", -1)) != 8
                or data.get("reached_final_time") is not True
                or final_tau < 0.999
            ):
                errors.append(
                    "incomplete_summary_json:"
                    f"{case}:{method}:method={data.get('method')}:"
                    f"level={data.get('level')}:"
                    f"reached_final_time={data.get('reached_final_time')}:"
                    f"final_tau={data.get('final_tau')}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()

    errors = validate_package(args.package)
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(f"PASS package={args.package}")
    print("PASS stationary_ellipse_l8_package_complete=8/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
