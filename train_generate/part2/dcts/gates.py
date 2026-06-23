"""Gate 1-11 evaluation for DCTS Stage 0.

Gates 1-10 are pass/fail (Gate 9 leakage is the hard one); Gate 11 is a mandatory
diagnostic that never fails. measure-don't-reason: every gate returns real numbers.
"""
from __future__ import annotations

import numpy as np

from train_generate.part2.dcts import augment, bins
from train_generate.part2.dcts.config import DctsConfig

SPLITS = ("train", "val", "test")


def run_gates(
    config: DctsConfig,
    merged_by_split: dict[str, dict],
    records_by_split: dict[str, list[dict]],
    *,
    edges: np.ndarray,
    gate11_worst: dict,
) -> dict:
    tol = config.eta_consistency_tol
    results: dict[str, dict] = {}

    def pool(key):
        return {s: np.asarray(merged_by_split[s][key]) for s in SPLITS}

    shape = pool("shape")
    fb = pool("fine_bin")
    eta = pool("eta")
    hk = pool("hk_exact")
    radius = pool("radius")
    q = pool("q"); em = pool("eta_max")

    # ---- Gate 1: budget / quota ----
    circle_ok = True
    ellipse_deficiency = 0
    for s in SPLITS:
        for rec in records_by_split[s]:
            if rec["shape"] == "circle" and int(rec["selected"]) != int(rec["quota"]):
                circle_ok = False
            if rec["shape"] == "ellipse":
                ellipse_deficiency += int(rec["deficiency"])
    results["gate1_budget"] = {
        "circle_quota_exact": bool(circle_ok),
        "ellipse_total_deficiency": int(ellipse_deficiency),
        "passed": bool(circle_ok and (ellipse_deficiency == 0 or config.allow_deficiency)),
    }

    # ---- Gate 2: circle consistency |eta - 1/R| ----
    max_circle_err = 0.0
    for s in SPLITS:
        m = shape[s] == "circle"
        if np.any(m):
            max_circle_err = max(max_circle_err, float(np.max(np.abs(eta[s][m] - 1.0 / radius[s][m]))))
    results["gate2_circle_consistency"] = {"max_abs_error": max_circle_err, "passed": bool(max_circle_err < tol)}

    # ---- Gate 3: ellipse local-eta in [eta_max q^3, eta_max] + bin membership ----
    max_range_viol = 0.0
    membership_fail = 0
    for s in SPLITS:
        m = shape[s] == "ellipse"
        if not np.any(m):
            continue
        e = eta[s][m]; emx = em[s][m]; qq = q[s][m]
        lo = emx * qq ** 3
        viol = np.maximum(0.0, e - emx - tol)
        viol = np.maximum(viol, np.maximum(0.0, lo - e - tol))
        max_range_viol = max(max_range_viol, float(np.max(viol)) if viol.size else 0.0)
        idx = bins.fine_bin_index(e, edges)
        membership_fail += int(np.count_nonzero(idx != fb[s][m]))
    results["gate3_ellipse_eta_range"] = {
        "max_range_violation": max_range_viol, "bin_membership_failures": int(membership_fail),
        "passed": bool(max_range_viol < 1e-9 and membership_fail == 0),
    }

    # ---- Gate 4: fine-bin coverage ----
    coverage = {}
    cov_ok = True
    for s in SPLITS:
        counts = np.bincount(fb[s], minlength=config.n_fine_bins)
        empty = int(np.count_nonzero(counts == 0))
        ell_counts = np.bincount(fb[s][shape[s] == "ellipse"], minlength=config.n_fine_bins)
        coverage[s] = {"empty_bins": empty, "bins_with_ellipse": int(np.count_nonzero(ell_counts > 0))}
        cov_ok = cov_ok and empty == 0
    results["gate4_bin_coverage"] = {**coverage, "passed": bool(cov_ok)}

    # ---- Gate 5: finiteness + unit normals (canonical features) ----
    max_norm_err = 0.0
    all_finite = True
    for s in SPLITS:
        nx = np.asarray(merged_by_split[s]["nx9"]); ny = np.asarray(merged_by_split[s]["ny9"])
        phi9 = np.asarray(merged_by_split[s]["phi9"])
        if not (np.all(np.isfinite(nx)) and np.all(np.isfinite(ny)) and np.all(np.isfinite(phi9))):
            all_finite = False
        if nx.size:
            nm = np.sqrt(nx ** 2 + ny ** 2)
            max_norm_err = max(max_norm_err, float(np.max(np.abs(nm - 1.0))))
    results["gate5_finite_unit_normals"] = {
        "all_finite": bool(all_finite), "max_normal_norm_error": max_norm_err,
        "passed": bool(all_finite and max_norm_err < 1e-9),
    }

    # ---- Gate 6: target/eta consistency (|hk_exact| == eta, positive) ----
    max_target_err = 0.0
    for s in SPLITS:
        if eta[s].size:
            max_target_err = max(max_target_err, float(np.max(np.abs(np.abs(hk[s]) - eta[s]))))
    results["gate6_target_consistency"] = {
        "max_abs_hk_minus_eta": max_target_err,
        "normal_source": config.normal_source, "sdf_mode": config.sdf_mode,
        "passed": bool(max_target_err < 1e-12),
    }

    # ---- Gate 7/8: D4 recompute + sign-flip (evaluate on train) ----
    ve = augment.verify_augmentation(config, merged_by_split["train"])
    results["gate7_d4_consistency"] = {
        "max_d4_hk_central_error": ve["max_d4_hk_central_error"], "passed": bool(ve["max_d4_hk_central_error"] <= 1e-10)}
    results["gate8_sign_flip"] = {
        "max_sign_feature_error": ve["max_sign_feature_error"], "max_sign_target_error": ve["max_sign_target_error"],
        "passed": bool(ve["max_sign_feature_error"] == 0.0 and ve["max_sign_target_error"] == 0.0)}

    # ---- Gate 9: leakage (HARD) ----
    leak = {}
    leak_ok = True
    for key in ("geometry_id", "pack_id"):
        sets = {s: set(merged_by_split[s][key]) for s in SPLITS}
        ov = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        leak[key] = int(len(ov))
        leak_ok = leak_ok and len(ov) == 0
    results["gate9_leakage"] = {**leak, "passed": bool(leak_ok)}

    # ---- Gate 10: log-uniformity (per-bin count == target, modulo deficiency) ----
    uni = {}
    uni_ok = True
    for s in SPLITS:
        counts = np.bincount(fb[s], minlength=config.n_fine_bins)
        target = int(config.per_bin[s])
        off = int(np.count_nonzero(counts != target))
        cv = float(np.std(counts) / np.mean(counts)) if counts.mean() else 0.0
        uni[s] = {"bins_off_target": off, "cv": cv, "min_count": int(counts.min()), "max_count": int(counts.max())}
        # off-target only acceptable if explained by recorded ellipse deficiency
        deficit_bins = {int(r["fine_bin"]) for r in records_by_split[s] if r["shape"] == "ellipse" and int(r["deficiency"]) > 0}
        unexplained = int(np.count_nonzero([(counts[j] != target) and (j not in deficit_bins) for j in range(config.n_fine_bins)]))
        uni[s]["unexplained_off_target"] = unexplained
        uni_ok = uni_ok and (unexplained == 0)
    results["gate10_log_uniformity"] = {**uni, "passed": bool(uni_ok)}

    # ---- Gate 11: normal degeneration diagnostic (mandatory, non-fail) ----
    results["gate11_normal_degeneration"] = {
        "min_fd_grad_overall": float(gate11_worst["min_fd_grad"]),
        "min_medial_dist_overall": float(gate11_worst["min_medial"]),
        "passed": True, "kind": "diagnostic",
    }

    results["all_passed"] = bool(all(
        results[g]["passed"] for g in results if g != "all_passed"
    ))
    return results


def format_gates(results: dict) -> str:
    lines = []
    order = [
        "gate1_budget", "gate2_circle_consistency", "gate3_ellipse_eta_range", "gate4_bin_coverage",
        "gate5_finite_unit_normals", "gate6_target_consistency", "gate7_d4_consistency",
        "gate8_sign_flip", "gate9_leakage", "gate10_log_uniformity", "gate11_normal_degeneration",
    ]
    for g in order:
        r = results[g]
        mark = "PASS" if r["passed"] else "FAIL"
        detail = {k: v for k, v in r.items() if k != "passed"}
        lines.append(f"[{mark}] {g}: {detail}")
    lines.append(f"\nALL GATES: {'GREEN' if results['all_passed'] else 'RED'}")
    return "\n".join(lines)
