"""CLI entry point – thin wrapper that preprocesses --main / --nonsdf shortcuts."""

from __future__ import annotations

import sys

from train_generate.part2.generate import main as _generate_main

if __name__ == "__main__":
    argv = list(sys.argv[1:])

    # ---- expand --main shortcut ------------------------------------------
    has_main = "--main" in argv
    if has_main:
        argv.remove("--main")
        if "--rho" not in argv:
            argv += ["--rho", "256"]
        if "--circle-eta-levels" not in argv:
            argv += ["--circle-eta-levels", "320"]
        if "--ellipse-count" not in argv:
            argv += ["--ellipse-count", "12800"]

    # ---- route to main vs main_sdfnonsdf --------------------------------
    has_nonsdf = "--nonsdf" in argv

    if has_main and "--output-dir" not in argv:
        suffix = "_sdfnonsdf" if has_nonsdf else ""
        argv += ["--output-dir", f"dataset/part2_dcts/main{suffix}"]

    _generate_main(argv)