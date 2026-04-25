from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np


class LevelSetReinitializer:
    def __init__(
        self,
        indexing: Literal["ij", "xy"] = "ij",
        cfl: float = 0.01,
        eps_weno: float = 1.0e-6,
        eps_sign_factor: float = 1.0,
        sign_mode: Literal["frozen_phi0", "dynamic_phi"] = "frozen_phi0",
        *,
        time_order: int = 3,
        space_order: int = 5,
    ) -> None:
        if indexing not in ("ij", "xy"):
            raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing!r}")
        if time_order not in (2, 3):
            raise ValueError(f"time_order must be 2 or 3, got {time_order}")
        if space_order not in (3, 4, 5):
            raise ValueError(f"space_order must be 3, 4, or 5, got {space_order}")
        if sign_mode not in ("frozen_phi0", "dynamic_phi"):
            raise ValueError(f"sign_mode must be 'frozen_phi0' or 'dynamic_phi', got {sign_mode!r}")
        self.indexing = indexing
        self.cfl = float(cfl)
        self.eps_weno = float(eps_weno)
        self.eps_sign_factor = float(eps_sign_factor)
        self.sign_mode = sign_mode
        self.time_order = int(time_order)
        self.space_order = int(space_order)

    def _smoothed_sign(self, phi0: np.ndarray, h: float) -> np.ndarray:
        eps = self.eps_sign_factor * h
        return phi0 / np.sqrt(phi0 ** 2 + eps ** 2)

    def _hj_weno5_1d_eps(self, v1, v2, v3, v4, v5) -> np.ndarray:
        b0 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
        b1 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + (1.0 / 4.0) * (v2 - v4) ** 2
        b2 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + (1.0 / 4.0) * (3.0 * v3 - 4.0 * v4 + v5) ** 2
        a0 = 0.1 / (b0 + self.eps_weno) ** 2
        a1 = 0.6 / (b1 + self.eps_weno) ** 2
        a2 = 0.3 / (b2 + self.eps_weno) ** 2
        sa = a0 + a1 + a2
        w0, w1, w2 = a0 / sa, a1 / sa, a2 / sa
        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5
        return w0 * p0 + w1 * p1 + w2 * p2

    def _deriv_space3(self, phi: np.ndarray, h: float):
        n0, n1 = phi.shape
        pp = np.pad(phi, 3, mode="edge")
        if self.indexing == "ij":
            d0_m = (11 * pp[3:n0 + 3, 3:n1 + 3] - 18 * pp[2:n0 + 2, 3:n1 + 3] + 9 * pp[1:n0 + 1, 3:n1 + 3] - 2 * pp[0:n0, 3:n1 + 3]) / (6 * h)
            d0_p = (-11 * pp[3:n0 + 3, 3:n1 + 3] + 18 * pp[4:n0 + 4, 3:n1 + 3] - 9 * pp[5:n0 + 5, 3:n1 + 3] + 2 * pp[6:n0 + 6, 3:n1 + 3]) / (6 * h)
            d1_m = (11 * pp[3:n0 + 3, 3:n1 + 3] - 18 * pp[3:n0 + 3, 2:n1 + 2] + 9 * pp[3:n0 + 3, 1:n1 + 1] - 2 * pp[3:n0 + 3, 0:n1]) / (6 * h)
            d1_p = (-11 * pp[3:n0 + 3, 3:n1 + 3] + 18 * pp[3:n0 + 3, 4:n1 + 4] - 9 * pp[3:n0 + 3, 5:n1 + 5] + 2 * pp[3:n0 + 3, 6:n1 + 6]) / (6 * h)
        else:
            d0_m = (11 * pp[3:n0 + 3, 3:n1 + 3] - 18 * pp[3:n0 + 3, 2:n1 + 2] + 9 * pp[3:n0 + 3, 1:n1 + 1] - 2 * pp[3:n0 + 3, 0:n1]) / (6 * h)
            d0_p = (-11 * pp[3:n0 + 3, 3:n1 + 3] + 18 * pp[3:n0 + 3, 4:n1 + 4] - 9 * pp[3:n0 + 3, 5:n1 + 5] + 2 * pp[3:n0 + 3, 6:n1 + 6]) / (6 * h)
            d1_m = (11 * pp[3:n0 + 3, 3:n1 + 3] - 18 * pp[2:n0 + 2, 3:n1 + 3] + 9 * pp[1:n0 + 1, 3:n1 + 3] - 2 * pp[0:n0, 3:n1 + 3]) / (6 * h)
            d1_p = (-11 * pp[3:n0 + 3, 3:n1 + 3] + 18 * pp[4:n0 + 4, 3:n1 + 3] - 9 * pp[5:n0 + 5, 3:n1 + 3] + 2 * pp[6:n0 + 6, 3:n1 + 3]) / (6 * h)
        return d0_m, d0_p, d1_m, d1_p

    def _deriv_space4(self, phi: np.ndarray, h: float):
        n0, n1 = phi.shape
        pp = np.pad(phi, 4, mode="edge")
        if self.indexing == "ij":
            d0_m = (25 * pp[4:n0 + 4, 4:n1 + 4] - 48 * pp[3:n0 + 3, 4:n1 + 4] + 36 * pp[2:n0 + 2, 4:n1 + 4] - 16 * pp[1:n0 + 1, 4:n1 + 4] + 3 * pp[0:n0, 4:n1 + 4]) / (12 * h)
            d0_p = (-25 * pp[4:n0 + 4, 4:n1 + 4] + 48 * pp[5:n0 + 5, 4:n1 + 4] - 36 * pp[6:n0 + 6, 4:n1 + 4] + 16 * pp[7:n0 + 7, 4:n1 + 4] - 3 * pp[8:n0 + 8, 4:n1 + 4]) / (12 * h)
            d1_m = (25 * pp[4:n0 + 4, 4:n1 + 4] - 48 * pp[4:n0 + 4, 3:n1 + 3] + 36 * pp[4:n0 + 4, 2:n1 + 2] - 16 * pp[4:n0 + 4, 1:n1 + 1] + 3 * pp[4:n0 + 4, 0:n1]) / (12 * h)
            d1_p = (-25 * pp[4:n0 + 4, 4:n1 + 4] + 48 * pp[4:n0 + 4, 5:n1 + 5] - 36 * pp[4:n0 + 4, 6:n1 + 6] + 16 * pp[4:n0 + 4, 7:n1 + 7] - 3 * pp[4:n0 + 4, 8:n1 + 8]) / (12 * h)
        else:
            d0_m = (25 * pp[4:n0 + 4, 4:n1 + 4] - 48 * pp[4:n0 + 4, 3:n1 + 3] + 36 * pp[4:n0 + 4, 2:n1 + 2] - 16 * pp[4:n0 + 4, 1:n1 + 1] + 3 * pp[4:n0 + 4, 0:n1]) / (12 * h)
            d0_p = (-25 * pp[4:n0 + 4, 4:n1 + 4] + 48 * pp[4:n0 + 4, 5:n1 + 5] - 36 * pp[4:n0 + 4, 6:n1 + 6] + 16 * pp[4:n0 + 4, 7:n1 + 7] - 3 * pp[4:n0 + 4, 8:n1 + 8]) / (12 * h)
            d1_m = (25 * pp[4:n0 + 4, 4:n1 + 4] - 48 * pp[3:n0 + 3, 4:n1 + 4] + 36 * pp[2:n0 + 2, 4:n1 + 4] - 16 * pp[1:n0 + 1, 4:n1 + 4] + 3 * pp[0:n0, 4:n1 + 4]) / (12 * h)
            d1_p = (-25 * pp[4:n0 + 4, 4:n1 + 4] + 48 * pp[5:n0 + 5, 4:n1 + 4] - 36 * pp[6:n0 + 6, 4:n1 + 4] + 16 * pp[7:n0 + 7, 4:n1 + 4] - 3 * pp[8:n0 + 8, 4:n1 + 4]) / (12 * h)
        return d0_m, d0_p, d1_m, d1_p

    def _deriv_weno5(self, phi: np.ndarray, h: float):
        n0, n1 = phi.shape
        pp = np.pad(phi, 3, mode="edge")
        if self.indexing == "ij":
            dx = (pp[1:, :] - pp[:-1, :]) / h
            dy = (pp[:, 1:] - pp[:, :-1]) / h
            d0_m = self._hj_weno5_1d_eps(dx[0:n0, 3:-3], dx[1:n0 + 1, 3:-3], dx[2:n0 + 2, 3:-3], dx[3:n0 + 3, 3:-3], dx[4:n0 + 4, 3:-3])
            d0_p = self._hj_weno5_1d_eps(dx[5:n0 + 5, 3:-3], dx[4:n0 + 4, 3:-3], dx[3:n0 + 3, 3:-3], dx[2:n0 + 2, 3:-3], dx[1:n0 + 1, 3:-3])
            d1_m = self._hj_weno5_1d_eps(dy[3:-3, 0:n1], dy[3:-3, 1:n1 + 1], dy[3:-3, 2:n1 + 2], dy[3:-3, 3:n1 + 3], dy[3:-3, 4:n1 + 4])
            d1_p = self._hj_weno5_1d_eps(dy[3:-3, 5:n1 + 5], dy[3:-3, 4:n1 + 4], dy[3:-3, 3:n1 + 3], dy[3:-3, 2:n1 + 2], dy[3:-3, 1:n1 + 1])
        else:
            dx = (pp[:, 1:] - pp[:, :-1]) / h
            dy = (pp[1:, :] - pp[:-1, :]) / h
            d0_m = self._hj_weno5_1d_eps(dx[3:-3, 0:n1], dx[3:-3, 1:n1 + 1], dx[3:-3, 2:n1 + 2], dx[3:-3, 3:n1 + 3], dx[3:-3, 4:n1 + 4])
            d0_p = self._hj_weno5_1d_eps(dx[3:-3, 5:n1 + 5], dx[3:-3, 4:n1 + 4], dx[3:-3, 3:n1 + 3], dx[3:-3, 2:n1 + 2], dx[3:-3, 1:n1 + 1])
            d1_m = self._hj_weno5_1d_eps(dy[0:n0, 3:-3], dy[1:n0 + 1, 3:-3], dy[2:n0 + 2, 3:-3], dy[3:n0 + 3, 3:-3], dy[4:n0 + 4, 3:-3])
            d1_p = self._hj_weno5_1d_eps(dy[5:n0 + 5, 3:-3], dy[4:n0 + 4, 3:-3], dy[3:n0 + 3, 3:-3], dy[2:n0 + 2, 3:-3], dy[1:n0 + 1, 3:-3])
        return d0_m, d0_p, d1_m, d1_p

    def _get_derivatives(self, phi: np.ndarray, h: float):
        if self.space_order == 3:
            return self._deriv_space3(phi, h)
        if self.space_order == 4:
            return self._deriv_space4(phi, h)
        return self._deriv_weno5(phi, h)

    @staticmethod
    def _godunov_grad_norm(dm0, dp0, dm1, dp1, sign_field: np.ndarray) -> np.ndarray:
        gp = np.sqrt(
            np.maximum(np.maximum(dm0, -dp0), 0.0) ** 2
            + np.maximum(np.maximum(dm1, -dp1), 0.0) ** 2
        )
        gm = np.sqrt(
            np.maximum(np.maximum(-dm0, dp0), 0.0) ** 2
            + np.maximum(np.maximum(-dm1, dp1), 0.0) ** 2
        )
        return np.where(sign_field >= 0, gp, gm)

    def _rhs(self, phi: np.ndarray, sign_field: np.ndarray, h: float) -> np.ndarray:
        dm0, dp0, dm1, dp1 = self._get_derivatives(phi, h)
        grad_norm = self._godunov_grad_norm(dm0, dp0, dm1, dp1, sign_field)
        return -sign_field * (grad_norm - 1.0)

    def _sign_field(self, phi_stage: np.ndarray, phi0: np.ndarray, h: float) -> np.ndarray:
        if self.sign_mode == "dynamic_phi":
            return self._smoothed_sign(phi_stage, h)
        return self._smoothed_sign(phi0, h)

    def reinitialize(self, phi0: np.ndarray, h: float, n_steps: int) -> np.ndarray:
        if n_steps <= 0:
            return phi0.copy()

        phi = phi0.astype(np.float64, copy=True)
        dt = self.cfl * h

        for _ in range(n_steps):
            if self.time_order == 2:
                sign_1 = self._sign_field(phi, phi0, h)
                rhs_1 = self._rhs(phi, sign_1, h)
                phi_1 = phi + dt * rhs_1
                sign_2 = self._sign_field(phi_1, phi0, h)
                rhs_2 = self._rhs(phi_1, sign_2, h)
                phi = 0.5 * phi + 0.5 * (phi_1 + dt * rhs_2)
            else:
                sign_1 = self._sign_field(phi, phi0, h)
                rhs_1 = self._rhs(phi, sign_1, h)
                phi_1 = phi + dt * rhs_1
                sign_2 = self._sign_field(phi_1, phi0, h)
                rhs_2 = self._rhs(phi_1, sign_2, h)
                phi_2 = 0.75 * phi + 0.25 * (phi_1 + dt * rhs_2)
                sign_3 = self._sign_field(phi_2, phi0, h)
                rhs_3 = self._rhs(phi_2, sign_3, h)
                phi = (1.0 / 3.0) * phi + (2.0 / 3.0) * (phi_2 + dt * rhs_3)
        return phi


class ReinitFieldPackBuilder:
    def __init__(
        self,
        *,
        cfl: float = 0.01,
        eps_weno: float = 1.0e-6,
        eps_sign_factor: float = 1.0,
        sign_mode: str = "frozen_phi0",
        time_order: int = 3,
        space_order: int = 5,
    ) -> None:
        self.cfl = float(cfl)
        self.eps_weno = float(eps_weno)
        self.eps_sign_factor = float(eps_sign_factor)
        self.sign_mode = str(sign_mode)
        self.time_order = int(time_order)
        self.space_order = int(space_order)
        self.reinitializer = LevelSetReinitializer(
            indexing="ij",
            cfl=self.cfl,
            eps_weno=self.eps_weno,
            eps_sign_factor=self.eps_sign_factor,
            sign_mode=self.sign_mode,
            time_order=self.time_order,
            space_order=self.space_order,
        )

    def build(self, field_pack: dict[str, Any], steps_list: list[int] | None = None) -> dict[str, dict[str, Any]]:
        if steps_list is None:
            steps_list = [5, 10, 15, 20]
        if field_pack["meta"].get("reinit") is not None:
            return {}

        phi0 = field_pack["field"]["phi"]
        h = field_pack["params"]["h"]
        phi_type = field_pack["field"].get("phi_type", "")
        is_sdf = "sdf" in phi_type and "nonsdf" not in phi_type

        if is_sdf:
            pack = copy.deepcopy(field_pack)
            pack["field"]["phi"] = pack["field"]["phi"].astype(np.float32)
            return {"0": pack}

        results: dict[str, dict[str, Any]] = {}
        for steps in steps_list:
            phi_re = self.reinitializer.reinitialize(phi0, h, steps)
            new_pack = copy.deepcopy(field_pack)
            new_pack["field"]["phi"] = phi_re.astype(np.float32)
            new_pack["field"]["phi_type"] += f"_reinit{steps}"
            new_pack["meta"]["reinit"] = {
                "steps": steps,
                "scheme": (
                    f"space_order={self.space_order} + "
                    f"time_order={self.time_order} + "
                    f"sign_mode={self.sign_mode} + Godunov (Rouy-Tourin)"
                ),
            }
            results[str(steps)] = new_pack
        return results
