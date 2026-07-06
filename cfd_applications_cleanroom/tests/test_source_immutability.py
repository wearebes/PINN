from __future__ import annotations

from pathlib import Path

from cfd_applications_cleanroom.cfd_apps.hashes import manifest_digest, sha256_file, tree_manifest
from cfd_applications_cleanroom.cfd_apps import source_audit


ROOT = Path(__file__).resolve().parents[2]


def test_toolchain_config_uses_route_local_clean_source() -> None:
    config = (ROOT / "cfd_applications_cleanroom/configs/toolchain.clean.yaml").read_text()

    assert "route_vendor_src: cfd_applications_cleanroom/vendor/basilisk_clean/src" in config
    assert "route_build_src: cfd_applications_cleanroom/build/basilisk_arm64/src" in config
    assert "active_untrusted_src: /private/tmp/basilisk/src" in config


def test_tree_manifest_detects_source_file_mutation(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "case.c"
    target.write_text("int main(void) { return 0; }\n")
    before = tree_manifest(src, source_only=True)

    target.write_text("int main(void) { return 1; }\n")
    after = tree_manifest(src, source_only=True)

    assert before != after
    assert manifest_digest(before) != manifest_digest(after)


def test_tree_manifest_ignores_build_generated_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "solver.c").write_text("int main(void) { return 0; }\n")
    before = tree_manifest(src, source_only=True)

    for name in ["qcc", "grammar", "grammar.h", "Makefile.deps", "Makefile.tests"]:
        (src / name).write_text("generated\n")
    (src / "solver.o").write_text("generated\n")
    after = tree_manifest(src, source_only=True)

    assert before == after


def test_tree_manifest_ignores_macos_metadata_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "solver.c").write_text("int main(void) { return 0; }\n")
    before = tree_manifest(src, source_only=True)

    (src / ".DS_Store").write_bytes(b"metadata\n")
    after = tree_manifest(src, source_only=True)

    assert before == after


def test_build_source_compare_ignores_config_symlink_only(tmp_path: Path) -> None:
    vendor = {
        "config": {"type": "symlink", "target": "config.gcc"},
        "solver.c": {"type": "file", "sha256": "abc"},
    }
    build = {
        "solver.c": {"type": "file", "sha256": "abc"},
    }

    assert source_audit._build_source_matches_vendor(build, vendor)


def test_source_audit_can_reuse_route_vendor_tarball_when_bootstrap_seed_is_gone(
    tmp_path: Path, monkeypatch
) -> None:
    missing_seed = tmp_path / "missing" / "basilisk.tar.gz"
    route_tarball = tmp_path / "vendor" / "basilisk.tar.gz"
    route_tarball.parent.mkdir()
    route_tarball.write_text("route-local tarball\n")
    monkeypatch.setattr(source_audit, "BOOTSTRAP_TARBALL", missing_seed)
    monkeypatch.setattr(source_audit, "ROUTE_VENDOR_TARBALL", route_tarball)

    assert source_audit._select_bootstrap_tarball() == route_tarball


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("same\n")

    assert sha256_file(path) == sha256_file(path)
