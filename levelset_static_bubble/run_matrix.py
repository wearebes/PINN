from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from levelset_static_bubble.contracts import CONFIGS, METHODS, RESOLUTIONS
from levelset_static_bubble.curvature import (
    curvature_exact,
    curvature_lsf_fd,
    curvature_nn27_d4,
    curvature_nn27_raw,
)
from levelset_static_bubble.features import raw27
from levelset_static_bubble.geometry import h_from_N
from levelset_static_bubble.interface_nodes import interface_indices
from levelset_static_bubble.io import write_rows_csv
from levelset_static_bubble.metrics import curvature_stats
from levelset_static_bubble.model_adapter import ModelBundle, load_nn27_bundle


def _csv_to_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_csv_subset(raw: str, allowed: tuple, label: str) -> list:
    values = _csv_to_list(raw)
    bad = sorted(set(values) - set(allowed))
    if bad:
        raise ValueError(f"Unsupported {label}: {bad}. Allowed: {allowed}.")
    return values


def _metadata_for(method: str, bundle: ModelBundle | None) -> dict[str, str | int]:
    if method in ("NN27_RAW", "NN27_D4"):
        if bundle is None:
            raise ValueError(f"{method} requires --checkpoint.")
        return bundle.metadata
    return {
        "raw_feature_dim": "",
        "transform_source": "",
        "feature_order": "",
        "target_scale": "",
        "checkpoint_hash": "",
        "feature_order_assumed": "",
    }


def run_curvature_matrix(
    *,
    configs: list[str],
    methods: list[str],
    resolutions: list[int],
    checkpoint: str | None,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    bundle = None
    if any(method in ("NN27_RAW", "NN27_D4") for method in methods):
        if checkpoint is None:
            raise ValueError("NN27_RAW/NN27_D4 require --checkpoint.")
        bundle = load_nn27_bundle(checkpoint)

    node_rows: list[dict] = []
    summary_rows: list[dict] = []
    for config in configs:
        for N in resolutions:
            h = h_from_N(N)
            idx = interface_indices(config, N)
            for method in methods:
                values: list[float] = []
                for i_raw, j_raw in idx:
                    i = int(i_raw)
                    j = int(j_raw)
                    if method == "EXACT":
                        kappa = curvature_exact(config, N, i, j)
                    elif method == "LS-FD":
                        kappa = curvature_lsf_fd(config, N, i, j)
                    elif method == "NN27_RAW":
                        assert bundle is not None
                        kappa = curvature_nn27_raw(bundle, raw27(config, N, i, j), h)
                    elif method == "NN27_D4":
                        assert bundle is not None
                        kappa = curvature_nn27_d4(bundle, raw27(config, N, i, j), h)
                    else:
                        raise ValueError(f"Unsupported method {method!r}.")
                    values.append(kappa)
                    node_rows.append(
                        {
                            "config": config,
                            "N": N,
                            "h": h,
                            "method": method,
                            "i": i,
                            "j": j,
                            "kappa": kappa,
                        }
                    )
                stats = curvature_stats(np.asarray(values, dtype=np.float64))
                summary_rows.append(
                    {
                        "stage": "curvature",
                        "config": config,
                        "N": N,
                        "h": h,
                        "method": method,
                        "n_interface": len(values),
                        **stats,
                        **_metadata_for(method, bundle),
                    }
                )

    nodes_csv = out_path / "curvature_nodes.csv"
    summary_csv = out_path / "curvature_summary.csv"
    write_rows_csv(nodes_csv, node_rows)
    write_rows_csv(summary_csv, summary_rows)
    return nodes_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run level-set static bubble benchmark stages.")
    parser.add_argument("--stage", choices=("curvature",), required=True)
    parser.add_argument("--configs", default="Q,F")
    parser.add_argument("--methods", default="EXACT,LS-FD")
    parser.add_argument("--resolutions", default="64,128,256,512")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="levelset_static_bubble/results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = _parse_csv_subset(args.configs, CONFIGS, "configs")
    methods = _parse_csv_subset(args.methods, METHODS, "methods")
    resolutions = [int(value) for value in _csv_to_list(args.resolutions)]
    bad_resolutions = sorted(set(resolutions) - set(RESOLUTIONS))
    if bad_resolutions:
        raise ValueError(f"Unsupported resolutions: {bad_resolutions}. Allowed: {RESOLUTIONS}.")
    nodes_csv, summary_csv = run_curvature_matrix(
        configs=configs,
        methods=methods,
        resolutions=resolutions,
        checkpoint=args.checkpoint,
        out_dir=args.out,
    )
    print(f"wrote {nodes_csv}")
    print(f"wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
