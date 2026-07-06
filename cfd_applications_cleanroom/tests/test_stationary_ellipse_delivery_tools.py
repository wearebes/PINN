from __future__ import annotations

import importlib.util
import csv
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "cfd_applications_cleanroom/scripts/validate_stationary_ellipse_l8_package.py"
STATUS_WRITER = ROOT / "cfd_applications_cleanroom/scripts/write_stationary_ellipse_status.py"
IMPORT_SCRIPT = ROOT / "cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh"
CASES = ("E1", "E2")
METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")
LEVELS = (6, 7, 8)


def _load_validator() -> ModuleType:
    return _load_module(VALIDATOR, "validate_stationary_ellipse_l8_package")


def _load_status_writer() -> ModuleType:
    return _load_module(STATUS_WRITER, "write_stationary_ellipse_status")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stationary_ellipse_l8_package_validator_accepts_complete_package(tmp_path: Path) -> None:
    validator = _load_validator()
    package = _write_package(tmp_path)

    assert validator.validate_package(package) == []


def test_stationary_ellipse_l8_package_validator_rejects_missing_method(tmp_path: Path) -> None:
    validator = _load_validator()
    package = _write_package(tmp_path, omit=("E2", "NN27_D4"))

    errors = validator.validate_package(package)

    assert "path_list_row_count=7" in errors
    assert "missing_result_dir:E2:NN27_D4" in errors


def test_stationary_ellipse_l8_package_validator_rejects_incomplete_summary(tmp_path: Path) -> None:
    validator = _load_validator()
    package = _write_package(tmp_path, final_tau_override={("E1", "NN27_D4"): 0.25})

    errors = validator.validate_package(package)

    assert any(error.startswith("incomplete_summary_json:E1:NN27_D4") for error in errors)


def test_stationary_ellipse_l8_package_validator_rejects_unsafe_member(tmp_path: Path) -> None:
    validator = _load_validator()
    package = _write_package(tmp_path, extra_members={"../escape.txt": "bad\n"})

    errors = validator.validate_package(package)

    assert "unsafe_tar_member:../escape.txt" in errors


def test_stationary_ellipse_import_refreshes_status_after_finalizer() -> None:
    script = IMPORT_SCRIPT.read_text()

    finalizer = "finalize_stationary_ellipse_when_complete.py"
    status = "write_stationary_ellipse_status.py"
    verifier = "verify_stationary_ellipse_delivery.py"
    finalizer_index = script.index(finalizer)
    last_status_index = script.rindex(status)
    verifier_index = script.index(verifier)

    assert last_status_index > finalizer_index
    assert verifier_index > last_status_index


def test_stationary_ellipse_status_writer_marks_complete_fixture_ready(tmp_path: Path) -> None:
    writer = _load_status_writer()
    _patch_status_paths(writer, tmp_path)
    _write_summary_csv(writer.SUMMARY_CSV)
    _write_l8_summaries(writer.RAW)
    _write_finalizer_manifest(writer.BACKGROUND)

    status = writer.build_status()

    assert status["diagnostic_ready"] == 12
    assert status["process_ready"] == 24
    assert status["l8_process_ready"] == 8
    assert status["process_png_plates_ready"] == 6
    assert status["final_delivery_status"] == "ready"
    assert status["open_rows"] == []


def test_stationary_ellipse_status_writer_keeps_partial_l8_not_ready(tmp_path: Path) -> None:
    writer = _load_status_writer()
    _patch_status_paths(writer, tmp_path)
    _write_summary_csv(writer.SUMMARY_CSV)
    _write_l8_summaries(writer.RAW, final_tau_override={("E2", "NN27_D4"): 0.24})
    _write_finalizer_manifest(writer.BACKGROUND)

    status = writer.build_status()

    assert status["process_ready"] == 24
    assert status["l8_process_ready"] == 7
    assert status["process_png_plates_ready"] == 6
    assert status["final_delivery_status"] == "not_ready"
    assert {"case": "E2", "method": "NN27_D4"} in status["missing_l8"]


def _write_package(
    tmp_path: Path,
    *,
    omit: tuple[str, str] | None = None,
    final_tau_override: dict[tuple[str, str], float] | None = None,
    extra_members: dict[str, str] | None = None,
) -> Path:
    package = tmp_path / "stationary_ellipse_l8_results_test.tar.gz"
    run_tag = "20260706T000000Z"
    path_list_name = (
        "cfd_applications_cleanroom/results/transfer/"
        f"stationary_ellipse_l8_results_{run_tag}.paths.txt"
    )
    manifest_name = (
        "cfd_applications_cleanroom/results/transfer/"
        f"stationary_ellipse_l8_results_{run_tag}.manifest.txt"
    )
    result_dirs = []
    final_tau_override = final_tau_override or {}
    extra_members = extra_members or {}

    for case in CASES:
        for method in METHODS:
            if omit == (case, method):
                continue
            result_dirs.append(
                "cfd_applications_cleanroom/results/raw/stationary_ellipse/"
                f"stationary_ellipse_curvature_process_{case}_{run_tag}/"
                f"{method}_L8"
            )

    with tarfile.open(package, "w:gz") as tar:
        _add_text(tar, path_list_name, "\n".join(result_dirs) + "\n")
        _add_text(
            tar,
            manifest_name,
            "package_created_utc=20260706T000000Z\nincluded_paths:\n" + "\n".join(result_dirs) + "\n",
        )
        for result_dir in result_dirs:
            case = "E1" if "_E1_" in result_dir else "E2"
            method = result_dir.rsplit("/", 1)[-1].removesuffix("_L8")
            final_tau = final_tau_override.get((case, method), 1.0)
            summary = {
                "method": method,
                "level": 8,
                "reached_final_time": final_tau >= 0.999,
                "final_tau": final_tau,
                "surface_tension_trace_csv": f"{result_dir}/surface_tension_trace.csv",
            }
            _add_text(tar, f"{result_dir}/summary.json", json.dumps(summary) + "\n")
            _add_text(tar, f"{result_dir}/surface_tension_trace.csv", "tau,Ca\n1.0,0.0\n")
        for name, text in extra_members.items():
            _add_text(tar, name, text)
    return package


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _patch_status_paths(writer: ModuleType, tmp_path: Path) -> None:
    writer.ROOT = tmp_path
    writer.REPORT_DIR = tmp_path / "Experiment/report/stationary bubble"
    writer.SUMMARY_CSV = writer.REPORT_DIR / "summary.csv"
    writer.RAW = tmp_path / "cfd_applications_cleanroom/results/raw/stationary_ellipse"
    writer.BACKGROUND = tmp_path / "cfd_applications_cleanroom/results/background"
    writer.STATUS_MD = writer.REPORT_DIR / "stationary_ellipse_completion_status.md"
    writer.OPEN_ROWS_CSV = writer.REPORT_DIR / "stationary_ellipse_open_rows.csv"
    writer.STATUS_JSON = writer.REPORT_DIR / "stationary_ellipse_completion_status.json"
    writer.REPORT_DIR.mkdir(parents=True)


def _write_summary_csv(path: Path) -> None:
    fields = ["benchmark_family", "tier", "case_id", "grid_n", "level", "method", "report_ready"]
    rows = []
    for case in CASES:
        for level in LEVELS:
            for method in ("NN27_RAW", "NN27_D4"):
                rows.append({
                    "benchmark_family": "stationary_ellipse",
                    "tier": "curvature-diagnostic",
                    "case_id": case,
                    "grid_n": str(1 << level),
                    "level": str(level),
                    "method": method,
                    "report_ready": "true",
                })
            for method in METHODS:
                rows.append({
                    "benchmark_family": "stationary_ellipse",
                    "tier": "curvature-process",
                    "case_id": case,
                    "grid_n": str(1 << level),
                    "level": str(level),
                    "method": method,
                    "report_ready": "true",
                })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_l8_summaries(
    raw: Path,
    *,
    final_tau_override: dict[tuple[str, str], float] | None = None,
) -> None:
    final_tau_override = final_tau_override or {}
    for case in CASES:
        for method in METHODS:
            result_dir = raw / f"stationary_ellipse_curvature_process_{case}_20260706T000000Z" / f"{method}_L8"
            result_dir.mkdir(parents=True, exist_ok=True)
            final_tau = final_tau_override.get((case, method), 1.0)
            (result_dir / "summary.json").write_text(json.dumps({
                "method": method,
                "level": 8,
                "reached_final_time": final_tau >= 0.999,
                "final_tau": final_tau,
            }))


def _write_finalizer_manifest(background: Path) -> None:
    log_dir = background / "stationary_ellipse_finalizer_20260706T000000Z"
    figure_dir = background / "figures"
    log_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    plates = []
    for case in CASES:
        for level in LEVELS:
            png = figure_dir / f"{case}_L{level}.png"
            png.write_bytes(b"png")
            plates.append({"case": case, "grid_n": 1 << level, "png": str(png)})
    (log_dir / "figure_manifest.json").write_text(json.dumps({"plates": plates}))
    (background / "stationary_ellipse_finalizer_latest.txt").write_text(str(log_dir))
