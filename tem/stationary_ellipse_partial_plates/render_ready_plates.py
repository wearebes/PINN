#!/usr/bin/env python3
"""Render the stationary-ellipse process plates that are already complete.

The official finalizer (finalize_stationary_ellipse_when_complete.py) renders all
6 plates (E1/E2 x L6/L7/L8) atomically and aborts if any (case, level, method)
summary is missing. As of 2026-07-06 the L8 (N256) matrix is still partial, so
that gate blocks every plate.

This helper renders only the (case, level) combos whose full 4-method process
matrix is complete (reached_final_time and final_tau >= 0.999), so the ready
plates (E1/E2 x L6/L7 = 4 of 6) can be produced now. It reuses the exact same
render_curvature_process_plate() the finalizer uses -- identical styling and
outputs -- and simply skips the not-yet-ready levels. Nothing is faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfd_applications_cleanroom.cfd_apps.curvature_diagnostic_shared import (  # noqa: E402
    render_curvature_process_plate,
)

RAW = ROOT / "cfd_applications_cleanroom/results/raw/stationary_ellipse"

CASES = ("E1", "E2")
LEVELS = (6, 7, 8)
METHODS = ("NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4")


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_complete_summary(case: str, level: int, method: str) -> Path | None:
    candidates = sorted(
        RAW.glob(f"stationary_ellipse_curvature_process_{case}_*/{method}_L{level}/summary.json")
    )
    for path in reversed(candidates):
        data = _load_json(path)
        if not data:
            continue
        if (
            data.get("method") == method
            and int(data.get("level", -1)) == level
            and data.get("reached_final_time") is True
            and _safe_float(data.get("final_tau")) >= 0.999
        ):
            return path
    return None


def main() -> int:
    # Deterministic run tag: no Date.now() -- keep re-runs reproducible / overwrite-safe.
    run_tag = "partial_ready"
    rendered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case in CASES:
        for level in LEVELS:
            summaries = []
            missing = []
            for method in METHODS:
                summary_path = latest_complete_summary(case, level, method)
                if summary_path is None:
                    missing.append(method)
                else:
                    data = _load_json(summary_path)
                    if data:
                        summaries.append(data)
                    else:
                        missing.append(method)
            if missing:
                skipped.append({"case": case, "level": level, "grid_n": 1 << level, "missing": missing})
                print(f"SKIP {case} L{level} (N{1 << level}) -- missing complete: {', '.join(missing)}")
                continue

            native_summary = next(s for s in summaries if s["method"] == "NN_DISABLE")
            run_id = f"stationary_ellipse_process_plate_{case.lower()}_l{level}_{run_tag}"
            artifact_prefix = f"stationary_ellipse_curvature_process_{case.lower()}_l{level}"
            figures = render_curvature_process_plate(
                run_id=run_id,
                summaries=summaries,
                artifact_prefix=artifact_prefix,
                native_trace_csv_override=Path(native_summary["surface_tension_trace_csv"]),
                trace_yscale="log",
            )
            png = Path(figures["png"])
            ok = png.exists() and png.stat().st_size > 0
            print(f"OK   {case} L{level} (N{1 << level}) -> {png}  ({'ok' if ok else 'EMPTY'})")
            rendered.append({"case": case, "level": level, "grid_n": 1 << level, **figures})

    print()
    print(f"rendered {len(rendered)}/6 plates; skipped {len(skipped)} (not yet complete)")
    manifest = {"rendered": rendered, "skipped": skipped}
    out = Path(__file__).resolve().parent / "partial_figure_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
