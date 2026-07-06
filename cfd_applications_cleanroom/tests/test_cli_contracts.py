from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLI = "cfd_applications_cleanroom.cfd_apps.cli"
SUBCOMMANDS = {
    "audit",
    "reproduce",
    "summarize",
    "figures",
    "compare-old-route",
    "check-final",
    "host-format",
}


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", CLI, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_help_exposes_exact_required_subcommands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0, result.stderr
    for command in SUBCOMMANDS:
        assert command in result.stdout


def test_parser_declares_no_extra_subcommands() -> None:
    from cfd_applications_cleanroom.cfd_apps.cli import SUBCOMMANDS as declared

    assert set(declared) == SUBCOMMANDS


def test_figures_rejects_non_python_backend() -> None:
    result = run_cli("figures", "--figure-backend", "r")

    assert result.returncode != 0
    assert "figure_backend_must_be_python" in result.stderr


def test_readme_contains_required_command_block_and_backend_note() -> None:
    readme = (ROOT / "cfd_applications_cleanroom" / "README.md").read_text()

    expected_lines = [
        "python -m cfd_applications_cleanroom.cfd_apps.cli audit",
        "python -m cfd_applications_cleanroom.cfd_apps.cli reproduce --tier canary --repeat 3",
        "python -m cfd_applications_cleanroom.cfd_apps.cli reproduce --tier accepted-local --repeat 3",
        "python -m cfd_applications_cleanroom.cfd_apps.cli summarize --latest",
        "python -m cfd_applications_cleanroom.cfd_apps.cli figures --latest --figure-backend python",
        "Steps 1-4 below do not require plotting dependencies.",
    ]
    for line in expected_lines:
        assert line in readme
