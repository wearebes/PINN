"""Host-format migration decision gates for cleanroom CFD applications."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cfd_applications_cleanroom.cfd_apps.paths import REPORTS, ROOT, ensure_result_dirs, route_relative


REGISTRY = ROOT / "cfd_applications_cleanroom/configs/host_format_migration.yaml"


class HostFormatGateError(RuntimeError):
    """Hard host-format gate failure."""


def load_registry() -> dict[str, Any]:
    return yaml.safe_load(REGISTRY.read_text())


def classify_application(application: str) -> dict[str, Any]:
    data = load_registry()
    apps = data["applications"]
    if application not in apps:
        known = ",".join(sorted(apps))
        raise HostFormatGateError(f"unknown_application:{application};known:{known}")
    app = apps[application]
    gates = {
        "HF-G0": app["interface_state"] == "volume_fraction_c" and not app["needs_persistent_signed_distance"],
        "HF-G1": app["host_format"] == "VOF-HF" and Path(ROOT / app["source_case"]).exists(),
        "HF-G2": app["permits_host_format_change"] is True,
        "HF-G3": app["external_curvature_injection_required"] is False,
        "HF-G4": _stock_reference_report_passes(app),
    }
    status = "PASS" if all(gates.values()) else "BLOCKED"
    recommendation = "use_vof_hf_native" if status == "PASS" else "do_not_migrate_in_this_plan"
    return {
        "application": application,
        "status": status,
        "recommendation": recommendation,
        "gates": gates,
        "registry": app,
    }


def _stock_reference_report_passes(app: dict[str, Any]) -> bool:
    report_value = app.get("stock_reference_report", "")
    if not report_value:
        return False
    report_path = ROOT / report_value
    if not report_path.exists():
        return False
    json_path = report_path if report_path.suffix == ".json" else report_path.with_suffix(".json")
    if not json_path.exists():
        return False
    data = json.loads(json_path.read_text())
    if data.get("VOF_HF_REFERENCE_GATE") != "PASS":
        return False
    if data.get("evidence_level") != app.get("accepted_evidence_level"):
        return False
    accepted_gate = float(app["accepted_gate"])
    levels = data.get("levels", {})
    if not levels:
        return False
    return all(
        level_data.get("all_pass") is True
        and level_data.get("all_reached_final_time") is True
        and level_data.get("all_finite_Ca") is True
        and level_data.get("repeat_consistency", {}).get("pass") is True
        and int(level_data.get("repeat_count", 0)) == 3
        and float(level_data.get("max_Ca_tail_max", float("inf"))) <= accepted_gate
        for level_data in levels.values()
    )


def write_decision_report(application: str) -> dict[str, Any]:
    ensure_result_dirs()
    decision = classify_application(application)
    run_id = f"host_format_migration_{application}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = REPORTS / f"{run_id}.json"
    md_path = REPORTS / f"{run_id}.md"
    payload = {"run_id": run_id, **decision}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        f"# Host Format Migration Decision: {application}",
        "",
        f"- run_id: `{run_id}`",
        f"- status: `{decision['status']}`",
        f"- recommendation: `{decision['recommendation']}`",
        f"- source_case: `{decision['registry']['source_case']}`",
        f"- accepted_metric: `{decision['registry']['accepted_metric']}`",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| `{gate}` | `{str(passed).lower()}` |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `PASS` means this application can be presented as a VOF-HF host-format migration positive control.",
            "- `BLOCKED` means the application must not be described as solved by VOF-HF under this plan.",
            "- No neural curvature injection is deployable in the VOF-HF host in this plan.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    payload["decision_json"] = route_relative(json_path)
    payload["decision_report"] = route_relative(md_path)
    return payload
