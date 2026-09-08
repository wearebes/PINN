"""Source and toolchain immutability audit for the cleanroom route."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfd_applications_cleanroom.cfd_apps.hashes import (
    manifest_digest,
    sha256_file,
    tree_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_TARBALL = ROOT / "tem/rising_hgrad_confirmed_20260629/clean_download/basilisk.tar.gz"
ROUTE_VENDOR = ROOT / "cfd_applications_cleanroom/vendor"
ROUTE_BUILD = ROOT / "cfd_applications_cleanroom/build"
ROUTE_VENDOR_TARBALL = ROUTE_VENDOR / "basilisk.tar.gz"
ROUTE_VENDOR_EXTRACTED = ROUTE_VENDOR / "basilisk"
ROUTE_VENDOR_CLEAN = ROUTE_VENDOR / "basilisk_clean"
ROUTE_VENDOR_SRC = ROUTE_VENDOR_CLEAN / "src"
ROUTE_BUILD_ROOT = ROUTE_BUILD / "basilisk_arm64"
ROUTE_BUILD_SRC = ROUTE_BUILD_ROOT / "src"
ACTIVE_UNTRUSTED_SRC = Path("/private/tmp/basilisk/src")
KNOWN_GOOD_NATIVE_CASE = ROOT / "cfd_applications_cleanroom/cases/stationary/stationary_vof_hf_stock_single.c"
RESULTS_RAW_TOOLCHAIN = ROOT / "cfd_applications_cleanroom/results/raw/toolchain"
RESULTS_MANIFESTS = ROOT / "cfd_applications_cleanroom/results/manifests"
RESULTS_REPORTS = ROOT / "cfd_applications_cleanroom/results/reports"


class SourceAuditError(RuntimeError):
    """Hard audit failure with a stable status code."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def run_source_audit() -> dict[str, Any]:
    RESULTS_RAW_TOOLCHAIN.mkdir(parents=True, exist_ok=True)
    RESULTS_MANIFESTS.mkdir(parents=True, exist_ok=True)
    RESULTS_REPORTS.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_source": "clean",
        "paths": {
            "bootstrap_seed_tarball": _rel(BOOTSTRAP_TARBALL),
            "route_vendor_tarball": _rel(ROUTE_VENDOR_TARBALL),
            "route_vendor_src": _rel(ROUTE_VENDOR_SRC),
            "route_build_src": _rel(ROUTE_BUILD_SRC),
            "active_untrusted_src": str(ACTIVE_UNTRUSTED_SRC),
            "known_good_native_case": _rel(KNOWN_GOOD_NATIVE_CASE),
        },
    }

    try:
        rename_audit = _materialize_vendor_source()
        audit["vendor_source_rename"] = rename_audit
        clean_manifest = tree_manifest(ROUTE_VENDOR_SRC)
        clean_source_manifest = tree_manifest(ROUTE_VENDOR_SRC, source_only=True)
        audit["clean_source"] = {
            "manifest_sha256": manifest_digest(clean_manifest),
            "source_manifest_sha256": manifest_digest(clean_source_manifest),
            "file_count": len(clean_manifest),
            "ast_translate_c_sha256": sha256_file(ROUTE_VENDOR_SRC / "ast/translate.c"),
        }
        audit["active_private_tmp"] = _audit_active_tree()
        audit["bundled_qcc"] = _audit_bundled_qcc()
        build_result = _build_native_qcc(clean_source_manifest)
        audit["native_qcc_build"] = build_result
        canary_result = _compile_native_canary(clean_source_manifest)
        audit["native_no_nn_canary"] = canary_result
        audit["accepted_status"] = {
            "selected_source": "clean",
            "active_private_tmp": audit["active_private_tmp"]["status"],
            "bundled_qcc_usable": audit["bundled_qcc"]["usable_on_host"],
            "native_qcc_required": audit["bundled_qcc"]["native_qcc_required"],
            "official_source_mutated": False,
        }
        audit["overall_status"] = "PASS"
    except SourceAuditError as exc:
        audit["overall_status"] = exc.status
        audit["error"] = str(exc)
    except Exception as exc:  # defensive: keep blocker manifest-backed
        audit["overall_status"] = "source_audit_failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"

    audit_json = RESULTS_MANIFESTS / "source_audit.json"
    audit_report = RESULTS_REPORTS / "source_audit.md"
    audit["audit_json"] = _rel(audit_json)
    audit["audit_report"] = _rel(audit_report)
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    audit_report.write_text(_render_report(audit))
    return audit


def _materialize_vendor_source() -> dict[str, Any]:
    seed_tarball = _select_bootstrap_tarball()

    ROUTE_VENDOR.mkdir(parents=True, exist_ok=True)
    ROUTE_BUILD.mkdir(parents=True, exist_ok=True)
    _remove_path(ROUTE_VENDOR_EXTRACTED)
    _remove_path(ROUTE_VENDOR_CLEAN)
    if seed_tarball.resolve() != ROUTE_VENDOR_TARBALL.resolve():
        shutil.copy2(seed_tarball, ROUTE_VENDOR_TARBALL)

    _run(
        ["tar", "-xzf", str(ROUTE_VENDOR_TARBALL), "-C", str(ROUTE_VENDOR)],
        cwd=ROOT,
        log_path=RESULTS_RAW_TOOLCHAIN / "vendor_extract.log",
    )
    if not (ROUTE_VENDOR_EXTRACTED / "src/Makefile").exists():
        raise SourceAuditError(
            "vendor_source_layout_unrecognized",
            "expected route vendor extraction to create top-level basilisk/src",
        )
    pre_manifest = tree_manifest(ROUTE_VENDOR_EXTRACTED / "src")
    ROUTE_VENDOR_EXTRACTED.rename(ROUTE_VENDOR_CLEAN)
    post_manifest = tree_manifest(ROUTE_VENDOR_SRC)
    if pre_manifest != post_manifest:
        raise SourceAuditError(
            "vendor_rename_hash_mismatch",
            "source file hashes changed across vendor rename",
        )
    required = [
        ROUTE_VENDOR_TARBALL,
        ROUTE_VENDOR_SRC / "Makefile",
        ROUTE_VENDOR_SRC / "test/spurious.c",
        ROUTE_VENDOR_SRC / "test/rising.c",
        ROUTE_VENDOR_SRC / "test/capwave.c",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SourceAuditError("vendor_required_file_missing", ", ".join(missing))
    return {
        "seed_tarball": _rel(seed_tarball),
        "pre_rename_path": _rel(ROUTE_VENDOR / "basilisk"),
        "post_rename_path": _rel(ROUTE_VENDOR_CLEAN),
        "src_hashes_unchanged": True,
        "pre_manifest_sha256": manifest_digest(pre_manifest),
        "post_manifest_sha256": manifest_digest(post_manifest),
        "rename_audit_recorded": True,
    }


def _select_bootstrap_tarball() -> Path:
    if BOOTSTRAP_TARBALL.exists():
        return BOOTSTRAP_TARBALL
    if ROUTE_VENDOR_TARBALL.exists():
        return ROUTE_VENDOR_TARBALL
    raise SourceAuditError(
        "bootstrap_seed_tarball_missing",
        f"{BOOTSTRAP_TARBALL} and {ROUTE_VENDOR_TARBALL}",
    )


def _audit_active_tree() -> dict[str, Any]:
    translate = ACTIVE_UNTRUSTED_SRC / "ast/translate.c"
    qcc_names = ["qcc", "qcc.new", "qcc_rebuilt"]
    qcc_entries = {}
    for name in qcc_names:
        path = ACTIVE_UNTRUSTED_SRC / name
        if path.exists():
            qcc_entries[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "mtime": path.stat().st_mtime,
            }
        else:
            qcc_entries[name] = None
    if translate.exists():
        text = translate.read_text(errors="replace")
        return {
            "status": "untrusted_or_dirty",
            "ast_translate_c_sha256": sha256_file(translate),
            "contains_if_aorder_patch_signature": "if (aorder)" in text,
            "qcc": qcc_entries,
        }
    return {
        "status": "untrusted_or_dirty",
        "ast_translate_c_sha256": None,
        "contains_if_aorder_patch_signature": None,
        "qcc": qcc_entries,
        "missing": str(translate),
    }


def _audit_bundled_qcc() -> dict[str, Any]:
    bundled = ROUTE_VENDOR_SRC / "qcc"
    file_output = None
    if bundled.exists():
        result = subprocess.run(
            ["file", str(bundled)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        file_output = result.stdout.strip()
    host_arch = platform.machine()
    usable = bool(file_output and ("Mach-O" in file_output or host_arch in file_output))
    return {
        "path": _rel(bundled),
        "exists": bundled.exists(),
        "file_output": file_output,
        "host_arch": host_arch,
        "usable_on_host": usable,
        "native_qcc_required": not usable,
        "selected_native_qcc_path": _rel(ROUTE_BUILD_SRC / "qcc"),
    }


def _build_native_qcc(vendor_source_manifest_before: dict[str, Any]) -> dict[str, Any]:
    _remove_path(ROUTE_BUILD_ROOT)
    ROUTE_BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROUTE_VENDOR_CLEAN, ROUTE_BUILD_ROOT, symlinks=True)
    config = ROUTE_BUILD_SRC / "config"
    if config.exists() or config.is_symlink():
        config.unlink()
    config.symlink_to("config.osx")
    config_target = config.readlink().as_posix()

    build_source_before = tree_manifest(ROUTE_BUILD_SRC, source_only=True)
    ast_interpreter_log = RESULTS_RAW_TOOLCHAIN / "make_ast_interpreter.log"
    ast_lib_log = RESULTS_RAW_TOOLCHAIN / "make_ast_libast.log"
    make_log = RESULTS_RAW_TOOLCHAIN / "make_qcc.log"
    ast_interpreter = _run(
        ["make", "-C", "ast/interpreter", "interpreter.o"],
        cwd=ROUTE_BUILD_SRC,
        log_path=ast_interpreter_log,
        check=False,
    )
    ast_lib = _run(
        ["make", "-C", "ast", "libast.a"],
        cwd=ROUTE_BUILD_SRC,
        log_path=ast_lib_log,
        check=False,
    )
    result = _run(["make", "qcc"], cwd=ROUTE_BUILD_SRC, log_path=make_log, check=False)
    file_output = ""
    if (ROUTE_BUILD_SRC / "qcc").exists():
        file_result = subprocess.run(
            ["file", str(ROUTE_BUILD_SRC / "qcc")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        file_output = file_result.stdout.strip()
    vendor_source_after = tree_manifest(ROUTE_VENDOR_SRC, source_only=True)
    build_source_after = tree_manifest(ROUTE_BUILD_SRC, source_only=True)
    vendor_unchanged = vendor_source_manifest_before == vendor_source_after
    build_unchanged = build_source_before == build_source_after
    build_matches_vendor = _build_source_matches_vendor(build_source_after, vendor_source_after)
    native_qcc_exists = (ROUTE_BUILD_SRC / "qcc").exists()
    native_file_ok = "Mach-O" in file_output or "arm64" in file_output

    status = (
        ast_interpreter.returncode == 0
        and ast_lib.returncode == 0
        and result.returncode == 0
        and native_qcc_exists
        and native_file_ok
    )
    if not status:
        raise SourceAuditError("make_qcc_failed", f"see {_rel(make_log)}")
    if not vendor_unchanged:
        raise SourceAuditError("route_vendor_source_mutated", "vendor source changed during make qcc")
    if not build_unchanged:
        raise SourceAuditError("route_build_source_mutated", "build source files changed during make qcc")
    if not build_matches_vendor:
        raise SourceAuditError("route_build_source_mismatch", "build source files differ from vendor")

    return {
        "config_symlink_target": config_target,
        "make_ast_interpreter_rc": ast_interpreter.returncode,
        "make_ast_libast_rc": ast_lib.returncode,
        "make_qcc_rc": result.returncode,
        "ast_interpreter_log": _rel(ast_interpreter_log),
        "ast_libast_log": _rel(ast_lib_log),
        "compile_log": _rel(make_log),
        "native_qcc_exists": native_qcc_exists,
        "file_qcc": file_output,
        "route_vendor_source_hashes_unchanged": vendor_unchanged,
        "route_build_source_hash_before": manifest_digest(build_source_before),
        "route_build_source_hash_after": manifest_digest(build_source_after),
        "route_build_source_hashes_unchanged": build_unchanged,
        "route_build_source_hashes_match_vendor_for_source_files": build_matches_vendor,
    }


def _compile_native_canary(vendor_source_manifest_before: dict[str, Any]) -> dict[str, Any]:
    binary = RESULTS_RAW_TOOLCHAIN / "stationary_vof_hf_stock_single_clean_qcc"
    compile_log = RESULTS_RAW_TOOLCHAIN / "stationary_vof_hf_stock_single_clean_qcc.compile.log"
    if binary.exists():
        binary.unlink()
    build_source_before = tree_manifest(ROUTE_BUILD_SRC, source_only=True)
    cmd = [
        str(ROUTE_BUILD_SRC / "qcc"),
        "-autolink",
        KNOWN_GOOD_NATIVE_CASE.name,
        "-o",
        str(binary),
        "-lm",
    ]
    env = os.environ.copy()
    env["BASILISK"] = str(ROUTE_BUILD_SRC)
    result = _run(
        cmd,
        cwd=KNOWN_GOOD_NATIVE_CASE.parent,
        log_path=compile_log,
        env=env,
        check=False,
    )
    vendor_source_after = tree_manifest(ROUTE_VENDOR_SRC, source_only=True)
    build_source_after = tree_manifest(ROUTE_BUILD_SRC, source_only=True)
    vendor_unchanged = vendor_source_manifest_before == vendor_source_after
    build_unchanged = build_source_before == build_source_after
    status = result.returncode == 0 and binary.exists()
    if not status:
        raise SourceAuditError("native_no_nn_canary_compile_failed", f"see {_rel(compile_log)}")
    if not vendor_unchanged:
        raise SourceAuditError("route_vendor_source_mutated", "vendor source changed during canary compile")
    if not build_unchanged:
        raise SourceAuditError("route_build_source_mutated", "build source files changed during canary compile")
    return {
        "compile_rc": result.returncode,
        "binary": _rel(binary),
        "binary_exists": binary.exists(),
        "compile_command": " ".join(cmd),
        "compile_log": _rel(compile_log),
        "route_vendor_source_hash_before": manifest_digest(vendor_source_manifest_before),
        "route_vendor_source_hash_after": manifest_digest(vendor_source_after),
        "route_vendor_source_hashes_unchanged": vendor_unchanged,
        "route_build_source_hash_before": manifest_digest(build_source_before),
        "route_build_source_hash_after": manifest_digest(build_source_after),
        "route_build_source_hashes_unchanged": build_unchanged,
    }


def _build_source_matches_vendor(
    build_manifest: dict[str, Any], vendor_manifest: dict[str, Any]
) -> bool:
    for rel, entry in vendor_manifest.items():
        if rel == "config":
            continue
        if build_manifest.get(rel) != entry:
            return False
    return True


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n"
        + f"cwd={cwd}\n"
        + f"returncode={result.returncode}\n\n"
        + result.stdout
    )
    if check and result.returncode != 0:
        raise SourceAuditError("command_failed", f"{cmd[0]} failed; see {_rel(log_path)}")
    return result


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _render_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Source And Toolchain Audit",
        "",
        f"- overall_status: `{audit['overall_status']}`",
        f"- selected_source: `{audit.get('selected_source')}`",
    ]
    if "error" in audit:
        lines.append(f"- error: `{audit['error']}`")
    if "vendor_source_rename" in audit:
        rename = audit["vendor_source_rename"]
        lines.extend(
            [
                "",
                "## Vendor Source",
                f"- pre_rename_path: `{rename['pre_rename_path']}`",
                f"- post_rename_path: `{rename['post_rename_path']}`",
                f"- src_hashes_unchanged: `{rename['src_hashes_unchanged']}`",
            ]
        )
    if "native_qcc_build" in audit:
        build = audit["native_qcc_build"]
        lines.extend(
            [
                "",
                "## Native qcc Build",
                f"- config_symlink_target: `{build['config_symlink_target']}`",
                f"- make_qcc_rc: `{build['make_qcc_rc']}`",
                f"- native_qcc_exists: `{build['native_qcc_exists']}`",
                f"- file_qcc: `{build['file_qcc']}`",
            ]
        )
    if "native_no_nn_canary" in audit:
        canary = audit["native_no_nn_canary"]
        lines.extend(
            [
                "",
                "## Native No-NN Canary Compile",
                f"- compile_rc: `{canary['compile_rc']}`",
                f"- binary_exists: `{canary['binary_exists']}`",
                f"- compile_log: `{canary['compile_log']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
