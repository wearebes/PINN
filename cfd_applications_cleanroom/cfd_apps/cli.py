"""Command-line entrypoint for the cleanroom CFD application route."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


SUBCOMMANDS = (
    "audit",
    "reproduce",
    "summarize",
    "figures",
    "compare-old-route",
    "check-final",
    "host-format",
)


def _add_common_latest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest manifest-backed run bundle.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cfd_applications_cleanroom.cfd_apps.cli",
        description="Cleanroom CFD application evidence runner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Audit source and toolchain only; no solver result is interpreted.",
    )
    audit.set_defaults(handler=_handle_audit)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Run a manifest-backed reproduction ladder.",
    )
    reproduce.add_argument(
        "--tier",
        choices=(
            "canary",
            "accepted-local",
            "stock-reference",
            "convergence",
            "curvature-diagnostic",
            "curvature-process",
        ),
        help="Execution tier to run.",
    )
    reproduce.add_argument("--repeat", type=int, default=1, help="Repeat count.")
    reproduce.add_argument("--benchmark", help="Benchmark selector.")
    reproduce.add_argument("--methods", nargs="+", help="Methods to run.")
    reproduce.add_argument("--levels", nargs="+", type=int, help="Basilisk levels.")
    reproduce.add_argument(
        "--cases",
        nargs="+",
        choices=("E1", "E2"),
        help="Ellipse case selector (benchmark=stationary_ellipse only).",
    )
    reproduce.add_argument(
        "--enable-hk-bias-probe",
        action="store_true",
        help="Enable separated h*kappa bias diagnostics where allowed.",
    )
    reproduce.add_argument(
        "--relax-lambda",
        type=float,
        default=0.25,
        help="One-step local relaxation lambda for explicit *_RELAX curvature methods.",
    )
    reproduce.set_defaults(handler=_handle_reproduce)

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize manifest-backed raw evidence.",
    )
    _add_common_latest(summarize)
    summarize.set_defaults(handler=_handle_not_implemented)

    figures = subparsers.add_parser(
        "figures",
        help="Generate or check Python-only figure artifacts.",
    )
    _add_common_latest(figures)
    figures.add_argument(
        "--figure-backend",
        default="python",
        help="Figure backend. Only 'python' is accepted for this route.",
    )
    figures.add_argument(
        "--check-python-runtime",
        action="store_true",
        help="Check Python plotting runtime without rendering final figures.",
    )
    figures.add_argument(
        "--artifact",
        choices=("curvature-jump",),
        help="Figure/report artifact to build from existing manifest-backed evidence.",
    )
    figures.add_argument("--run-id", help="Explicit source run id, e.g. stationary_curvature_20260703T132941Z.")
    figures.add_argument(
        "--level",
        type=int,
        help="Optional explicit input level. Omit only when the run contains exactly one level.",
    )
    figures.add_argument("--methods", nargs="+", help="Methods to include in figure/report artifact.")
    figures.set_defaults(handler=_handle_figures)

    compare_old = subparsers.add_parser(
        "compare-old-route",
        help="Classify old route evidence after cleanroom evidence exists.",
    )
    compare_old.set_defaults(handler=_handle_not_implemented)

    check_final = subparsers.add_parser(
        "check-final",
        help="Verify final evidence bundle gates.",
    )
    _add_common_latest(check_final)
    check_final.set_defaults(handler=_handle_not_implemented)

    host_format = subparsers.add_parser(
        "host-format",
        help="Classify whether an application may migrate to the VOF-HF host.",
    )
    host_format.add_argument("--application", required=True, help="Application id from host_format_migration.yaml.")
    host_format.set_defaults(handler=_handle_host_format)

    return parser


def _handle_not_implemented(args: argparse.Namespace) -> int:
    command = args.command.replace("-", "_")
    print(f"{command}=IMPLEMENTED_ONLY")
    print("evidence_level=implemented_only")
    return 0


def _handle_audit(args: argparse.Namespace) -> int:
    from cfd_applications_cleanroom.cfd_apps.source_audit import run_source_audit

    result = run_source_audit()
    print(f"source_audit={result['overall_status']}")
    print(f"source_audit_json={result['audit_json']}")
    print(f"source_audit_report={result['audit_report']}")
    if result["overall_status"] != "PASS":
        return 1
    return 0


def _handle_reproduce(args: argparse.Namespace) -> int:
    if args.benchmark == "stationary" and args.tier == "stock-reference":
        from cfd_applications_cleanroom.cfd_apps.stationary import run_stock_reference

        levels = args.levels or [5, 6, 7]
        report = run_stock_reference(levels=levels, repeat=int(args.repeat))
        print(f"run_id={report['run_id']}")
        print(f"VOF_HF_REFERENCE_GATE={report['VOF_HF_REFERENCE_GATE']}")
        print(f"evidence_level={report['evidence_level']}")
        return 0
    if args.benchmark == "stationary" and args.tier == "canary":
        from cfd_applications_cleanroom.cfd_apps.stationary import run_canary

        levels = args.levels or [6]
        inert_methods = ["CLSVOF_LS_NATIVE", "NN_DISABLE", "NN_PROBE_ONLY"]
        methods = args.methods or inert_methods
        report = run_canary(
            methods=methods,
            levels=levels,
            repeat=int(args.repeat),
            relax_lambda=float(args.relax_lambda),
        )
        print(f"run_id={report['run_id']}")
        if "method_labels" in report:
            print(f"SB-G6={report['SB-G6']}")
            for method, label in report["method_labels"].items():
                print(f"{method}={label}")
        else:
            print(f"SB-G2={report['SB-G2']}")
            print(f"SB-G4={report['SB-G4']}")
            print(f"SB-G5={report['SB-G5']}")
        print(f"evidence_level={report['evidence_level']}")
        return 0
    if args.benchmark == "stationary" and args.tier == "curvature-diagnostic":
        from cfd_applications_cleanroom.cfd_apps.stationary import run_curvature_diagnostic

        levels = args.levels or [6]
        methods = args.methods or ["NN27_RAW", "NN27_D4"]
        report = run_curvature_diagnostic(
            methods=methods,
            levels=levels,
            relax_lambda=float(args.relax_lambda),
        )
        print(f"run_id={report['run_id']}")
        print(f"curvature_diagnostic={report['overall_status']}")
        for method, data in report["methods"].items():
            print(
                f"{method}=mean_delta_hk:{data['force_band']['mean_delta_hk']:.6e},"
                f"std_kappa_nn:{data['force_band']['force_band_std_kappa_nn']:.6e},"
                f"sign:{data['sign_scale']['sign_verdict']}"
            )
        print(f"evidence_level={report['evidence_level']}")
        return 0
    if args.benchmark == "stationary" and args.tier == "curvature-process":
        from cfd_applications_cleanroom.cfd_apps.stationary import run_curvature_process

        levels = args.levels or [6]
        methods = args.methods or ["NN27_RAW", "NN27_D4"]
        report = run_curvature_process(
            methods=methods,
            levels=levels,
            relax_lambda=float(args.relax_lambda),
        )
        print(f"run_id={report['run_id']}")
        print(f"curvature_process={report['overall_status']}")
        print(f"primary_metric={report['primary_metric']}")
        print(f"figure_png={report['figures']['png']}")
        print(f"source_data={report['figures']['source_data_csv']}")
        print(f"evidence_level={report['evidence_level']}")
        return 0
    if args.benchmark == "stationary_ellipse" and args.tier == "curvature-diagnostic":
        from cfd_applications_cleanroom.cfd_apps.stationary_ellipse import run_curvature_diagnostic

        levels = args.levels or [6]
        methods = args.methods or ["NN27_RAW", "NN27_D4"]
        cases = args.cases or ["E1"]
        for case in cases:
            report = run_curvature_diagnostic(case=case, methods=methods, levels=levels)
            print(f"case={case}")
            print(f"run_id={report['run_id']}")
            print(f"curvature_diagnostic={report['overall_status']}")
            for method, data in report["methods"].items():
                print(
                    f"{method}=mean_delta_hk:{data['force_band']['mean_delta_hk']:.6e},"
                    f"analytic_std_delta_kappa:{data['analytic_reference']['force_band']['std_delta_kappa']:.6e},"
                    f"sign:{data['sign_scale']['sign_verdict']}"
                )
            if report.get("roughness_comparison"):
                print(f"roughness_comparison={report['roughness_comparison']}")
            print(f"evidence_level={report['evidence_level']}")
        return 0
    if args.benchmark == "stationary_ellipse" and args.tier == "curvature-process":
        from cfd_applications_cleanroom.cfd_apps.stationary_ellipse import run_curvature_process

        levels = args.levels or [6]
        methods = args.methods or ["NN_DISABLE", "NN_PROBE_ONLY", "NN27_RAW", "NN27_D4"]
        cases = args.cases or ["E1"]
        for case in cases:
            report = run_curvature_process(case=case, methods=methods, levels=levels)
            print(f"case={case}")
            print(f"run_id={report['run_id']}")
            print(f"curvature_process={report['overall_status']}")
            print(f"primary_metric={report['primary_metric']}")
            print(f"figure_png={report['figures']['png']}")
            print(f"source_data={report['figures']['source_data_csv']}")
            print(f"evidence_level={report['evidence_level']}")
        return 0
    return _handle_not_implemented(args)


def _handle_figures(args: argparse.Namespace) -> int:
    if args.figure_backend != "python":
        raise SystemExit("figure_backend_must_be_python")
    if args.artifact == "curvature-jump":
        if not args.run_id:
            raise SystemExit("curvature_jump_requires_explicit_run_id")
        from cfd_applications_cleanroom.cfd_apps.curvature_jump import write_curvature_jump_package

        result = write_curvature_jump_package(
            run_id=args.run_id,
            methods=args.methods or [],
            level=args.level,
        )
        print(f"run_id={result['run_id']}")
        print(f"curvature_jump={result['overall_status']}")
        print(f"figure_png={result['figures']['png']}")
        print(f"source_data={result['source_data_csv']}")
        print(f"source_data_manifest={result['source_data_manifest']}")
        print(f"summary_md={result['summary_md']}")
        print(f"summary_json={result['summary_json']}")
        print(f"evidence_level={result['evidence_level']}")
        return 0 if result["overall_status"] == "PASS" else 1
    if args.check_python_runtime:
        try:
            import matplotlib  # noqa: F401

            matplotlib_available = True
        except Exception:
            matplotlib_available = False
        try:
            import seaborn  # noqa: F401

            seaborn_available = True
        except Exception:
            seaborn_available = False
        status = "PASS" if matplotlib_available and seaborn_available else "FAIL"
        print(f"python_plot_runtime={status}")
        print(f"matplotlib_available={str(matplotlib_available).lower()}")
        print(f"seaborn_available={str(seaborn_available).lower()}")
        return 0 if status == "PASS" else 1
    return _handle_not_implemented(args)


def _handle_host_format(args: argparse.Namespace) -> int:
    from cfd_applications_cleanroom.cfd_apps.host_format import write_decision_report

    report = write_decision_report(application=args.application)
    print(f"run_id={report['run_id']}")
    print(f"host_format_status={report['status']}")
    print(f"recommendation={report['recommendation']}")
    print(f"decision_report={report['decision_report']}")
    return 0 if report["status"] == "PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
