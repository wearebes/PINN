from __future__ import annotations

from functools import lru_cache

import mpmath
import numpy as np


STENCIL_OFFSETS = np.asarray(
    [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)],
    dtype=np.int64,
)  # canonical (9, 2); byte-identical to part1's _PHI9_ROW/COL split and part2's STENCIL_OFFSETS

DEFAULT_ELLIPSE_SDF_NEWTON_MAX_ITER = 30
DEFAULT_ELLIPSE_SDF_NEWTON_TOL = 1.0e-12
DEFAULT_ELLIPSE_HP_DPS = 80
DEFAULT_ELLIPSE_HP_NEWTON_MAX_ITER = 100

TWO_PI = float(2.0 * np.pi)


@lru_cache(maxsize=None)
def build_grid(rho: int, grid_convention: str = "endpoint_nodes") -> tuple[np.ndarray, np.ndarray]:
    if grid_convention == "cell_count_cell_centres":
        x = (np.arange(int(rho), dtype=np.float64) + 0.5) / int(rho)
    elif grid_convention == "endpoint_nodes":
        x = np.linspace(0.0, 1.0, int(rho), dtype=np.float64)
    else:
        raise ValueError(f"Unknown grid_convention {grid_convention!r}")
    return np.meshgrid(x, x, indexing="ij")


def build_circle_sdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius


def build_circle_nonsdf(X: np.ndarray, Y: np.ndarray, *, cx: float, cy: float, radius: float) -> np.ndarray:
    return (X - cx) ** 2 + (Y - cy) ** 2 - radius**2


def ellipse_local_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cx: float,
    cy: float,
    psi: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))
    dx = X - cx
    dy = Y - cy
    u = dx * cos_psi + dy * sin_psi
    v = -dx * sin_psi + dy * cos_psi
    return u, v


def build_ellipse_nonsdf(u: np.ndarray, v: np.ndarray, *, a: float, b: float) -> np.ndarray:
    return (u**2) / (a**2) + (v**2) / (b**2) - 1.0


def interface_indices(phi: np.ndarray) -> np.ndarray:
    scx = phi[:-1, :] * phi[1:, :] <= 0.0
    scy = phi[:, :-1] * phi[:, 1:] <= 0.0
    mask = np.zeros_like(phi, dtype=bool)
    ix, jx = np.where(scx)
    iy, jy = np.where(scy)
    mask[ix, jx] = True
    mask[ix + 1, jx] = True
    mask[iy, jy] = True
    mask[iy, jy + 1] = True
    # Exclude 2 boundary layers so ±2 gradient accesses (for extract_grad9) stay in bounds.
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((rows, cols)).astype(np.int64, copy=False)


def project_theta_scalar(u: float, v: float, a: float, b: float, theta0: float, *, max_iter: int, tol: float) -> float:
    theta = float(theta0 % TWO_PI)
    for _ in range(max_iter):
        st = float(np.sin(theta))
        ct = float(np.cos(theta))
        g = (b * b - a * a) * st * ct + a * u * st - b * v * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u * ct + b * v * st
        if abs(g) <= tol:
            return theta
        if abs(gp) < 1.0e-18:
            break
        step = g / gp
        theta = float((theta - step) % TWO_PI)
        if abs(step) <= tol:
            return theta

    coarse_thetas = np.linspace(0.0, TWO_PI, 720, endpoint=False, dtype=np.float64)
    coarse_ct = np.cos(coarse_thetas)
    coarse_st = np.sin(coarse_thetas)
    dist_sq = (a * coarse_ct - u) ** 2 + (b * coarse_st - v) ** 2
    theta = float(coarse_thetas[int(np.argmin(dist_sq))])
    for _ in range(max_iter * 4):
        st = float(np.sin(theta))
        ct = float(np.cos(theta))
        g = (b * b - a * a) * st * ct + a * u * st - b * v * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u * ct + b * v * st
        if abs(g) <= tol:
            break
        if abs(gp) < 1.0e-18:
            break
        step = g / gp
        theta = float((theta - step) % TWO_PI)
        if abs(step) <= tol:
            break
    return theta


def hp_theta_tolerance(dps: int) -> mpmath.mpf:
    if int(dps) <= 20:
        raise ValueError("ellipse_hp_dps must be > 20 so the high-precision tolerance is meaningful.")
    return mpmath.power(10, -(int(dps) - 20))


def project_theta_high_precision_scalar(
    u: float,
    v: float,
    a: float,
    b: float,
    theta0: float,
    *,
    dps: int,
    max_iter: int,
) -> float:
    return float(
        project_theta_high_precision_scalar_diagnostics(
            u,
            v,
            a,
            b,
            theta0,
            dps=dps,
            max_iter=max_iter,
        )["theta"]
    )


def project_theta_high_precision_scalar_diagnostics(
    u: float,
    v: float,
    a: float,
    b: float,
    theta0: float,
    *,
    dps: int,
    max_iter: int,
) -> dict[str, float | int | bool | str]:
    tol = hp_theta_tolerance(dps)

    def _refine(seed_theta: float, *, seed_kind: str) -> dict[str, float | int | bool | str]:
        with mpmath.workdps(int(dps)):
            two_pi = mpmath.mpf(2) * mpmath.pi
            a_mp = mpmath.mpf(a)
            b_mp = mpmath.mpf(b)
            u_mp = mpmath.mpf(u)
            v_mp = mpmath.mpf(v)
            theta = mpmath.fmod(mpmath.mpf(seed_theta), two_pi)
            if theta < 0:
                theta += two_pi
            for _ in range(int(max_iter)):
                st = mpmath.sin(theta)
                ct = mpmath.cos(theta)
                g = (b_mp * b_mp - a_mp * a_mp) * st * ct + a_mp * u_mp * st - b_mp * v_mp * ct
                if mpmath.fabs(g) <= tol:
                    return {
                        "theta": float(theta),
                        "converged": True,
                        "iterations": int(_),
                        "newton_internal_residual": float(mpmath.fabs(g)),
                        "seed_kind": seed_kind,
                    }
                gp = (b_mp * b_mp - a_mp * a_mp) * (ct * ct - st * st) + a_mp * u_mp * ct + b_mp * v_mp * st
                if mpmath.fabs(gp) <= mpmath.mpf("1e-40"):
                    break
                step = g / gp
                theta = mpmath.fmod(theta - step, two_pi)
                if theta < 0:
                    theta += two_pi
                if mpmath.fabs(step) <= tol:
                    st_next = mpmath.sin(theta)
                    ct_next = mpmath.cos(theta)
                    g_next = (
                        (b_mp * b_mp - a_mp * a_mp) * st_next * ct_next
                        + a_mp * u_mp * st_next
                        - b_mp * v_mp * ct_next
                    )
                    return {
                        "theta": float(theta),
                        "converged": True,
                        "iterations": int(_) + 1,
                        "newton_internal_residual": float(mpmath.fabs(g_next)),
                        "seed_kind": seed_kind,
                    }
            st = mpmath.sin(theta)
            ct = mpmath.cos(theta)
            g = (b_mp * b_mp - a_mp * a_mp) * st * ct + a_mp * u_mp * st - b_mp * v_mp * ct
            return {
                "theta": float(theta),
                "converged": False,
                "iterations": int(max_iter),
                "newton_internal_residual": float(mpmath.fabs(g)),
                "seed_kind": seed_kind,
            }

    refined = _refine(theta0, seed_kind="analytic")
    if bool(refined["converged"]):
        return refined

    coarse_thetas = np.linspace(0.0, TWO_PI, 4096, endpoint=False, dtype=np.float64)
    coarse_ct = np.cos(coarse_thetas)
    coarse_st = np.sin(coarse_thetas)
    dist_sq = (float(a) * coarse_ct - float(u)) ** 2 + (float(b) * coarse_st - float(v)) ** 2
    coarse_seed = float(coarse_thetas[int(np.argmin(dist_sq))])
    refined = _refine(coarse_seed, seed_kind="coarse_global")
    if bool(refined["converged"]):
        return refined
    raise RuntimeError(
        f"High-precision ellipse projection failed to converge for point (u={u}, v={v}) "
        f"with a={a}, b={b}, dps={dps}, max_iter={max_iter}."
    )


def project_theta_to_axis_aligned_ellipse_high_precision(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    dps: int,
    max_iter: int,
    initial_theta: np.ndarray | None = None,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f"u and v must have the same shape, got {u_arr.shape} and {v_arr.shape}.")
    if initial_theta is None:
        theta0 = project_theta_to_axis_aligned_ellipse(
            u_arr,
            v_arr,
            a=float(a),
            b=float(b),
            max_iter=20,
            tol=1.0e-12,
        )
    else:
        theta0 = np.asarray(initial_theta, dtype=np.float64)
        if theta0.shape != u_arr.shape:
            raise ValueError(f"initial_theta shape {theta0.shape} does not match input shape {u_arr.shape}.")

    theta_flat = np.empty(u_arr.size, dtype=np.float64)
    for idx, (u_item, v_item, theta_item) in enumerate(
        zip(u_arr.reshape(-1), v_arr.reshape(-1), theta0.reshape(-1), strict=True)
    ):
        theta_flat[idx] = project_theta_high_precision_scalar(
            float(u_item),
            float(v_item),
            float(a),
            float(b),
            float(theta_item),
            dps=int(dps),
            max_iter=int(max_iter),
        )
    return theta_flat.reshape(u_arr.shape)


def project_theta_to_axis_aligned_ellipse(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    theta = np.mod(np.arctan2(a * v_arr, b * u_arr), TWO_PI)
    converged = np.zeros(theta.shape, dtype=bool)

    for _ in range(max_iter):
        st = np.sin(theta)
        ct = np.cos(theta)
        g = (b * b - a * a) * st * ct + a * u_arr * st - b * v_arr * ct
        gp = (b * b - a * a) * (ct * ct - st * st) + a * u_arr * ct + b * v_arr * st
        resolved = np.abs(g) <= tol
        safe = np.abs(gp) >= 1.0e-18
        step = np.zeros_like(theta)
        step[safe] = g[safe] / gp[safe]
        theta = np.mod(theta - step, TWO_PI)
        converged |= resolved | (safe & (np.abs(step) <= tol))
        if bool(np.all(converged)):
            break

    if not bool(np.all(converged)):
        unresolved = np.where(~converged)
        for flat_idx in zip(*unresolved, strict=True):
            theta[flat_idx] = project_theta_scalar(
                float(u_arr[flat_idx]),
                float(v_arr[flat_idx]),
                float(a),
                float(b),
                float(theta[flat_idx]),
                max_iter=max_iter,
                tol=tol,
            )
    return theta.astype(np.float64, copy=False)


def build_ellipse_sdf(
    u: np.ndarray,
    v: np.ndarray,
    *,
    a: float,
    b: float,
    max_iter: int,
    tol: float,
    dps: int,
    hp_max_iter: int,
) -> np.ndarray:
    theta_seed = project_theta_to_axis_aligned_ellipse(u, v, a=a, b=b, max_iter=max_iter, tol=tol)
    theta = project_theta_to_axis_aligned_ellipse_high_precision(
        u,
        v,
        a=a,
        b=b,
        dps=int(dps),
        max_iter=int(hp_max_iter),
        initial_theta=theta_seed,
    )
    qx = a * np.cos(theta)
    qy = b * np.sin(theta)
    dist = np.sqrt((u - qx) ** 2 + (v - qy) ** 2)
    inside = build_ellipse_nonsdf(u, v, a=a, b=b) < 0.0
    return np.where(inside, -dist, dist).astype(np.float64, copy=False)


def ellipse_hkappa_from_theta(theta: np.ndarray, *, h: float, a: float, b: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    denom = (a * a * np.sin(theta) ** 2 + b * b * np.cos(theta) ** 2) ** 1.5
    hkappa = float(h) * (a * b) / denom
    return hkappa.astype(np.float64, copy=False)
