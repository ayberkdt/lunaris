# -*- coding: utf-8 -*-
"""
Torch spherical-harmonic gravity evaluator.

This module provides the batched GPU/CPU tensor equivalent of the repository's
body-fixed spherical-harmonic acceleration kernel.  It is intentionally small
and runtime-oriented: callers preload a gravity model once, then evaluate many
positions with shape ``(N, 3)`` without per-sample Python loops.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class TorchSHGravityEvaluator:
    """
    Vectorized spherical-harmonic acceleration evaluator backed by PyTorch.

    Parameters
    ----------
    gravity_model:
        Object exposing the normalized gravity contract: ``R_ref_m``,
        ``GM_m3s2``, ``Cnm``, ``Snm``, recurrence tables, and ``scale_m``.
    degree:
        Maximum degree to evaluate.  For ST-LRPS residual runs this is the
        baseline degree ``degree_min``.
    device / dtype:
        Torch device and floating-point dtype used for all internal tensors.
    """

    def __init__(self, gravity_model: Any, *, degree: int, device: Any, dtype: Any) -> None:
        import torch

        self.degree = int(degree)
        self.device = device
        self.dtype = dtype
        self.backend = "torch_sh"
        self.r_ref = torch.tensor(float(getattr(gravity_model, "R_ref_m")), device=device, dtype=dtype)
        self.mu = torch.tensor(float(getattr(gravity_model, "GM_m3s2")), device=device, dtype=dtype)
        self.C = torch.as_tensor(
            np.array(getattr(gravity_model, "Cnm"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        self.S = torch.as_tensor(
            np.array(getattr(gravity_model, "Snm"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        self.diag = torch.as_tensor(
            np.array(getattr(gravity_model, "diag"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        self.subdiag = torch.as_tensor(
            np.array(getattr(gravity_model, "subdiag"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        self.A = torch.as_tensor(
            np.array(getattr(gravity_model, "A"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        self.B = torch.as_tensor(
            np.array(getattr(gravity_model, "B"), dtype=np.float64, copy=True),
            device=device,
            dtype=dtype,
        )
        scale_np = np.asarray(getattr(gravity_model, "scale_m"), dtype=np.float64)
        scale_pad = np.ones(self.degree + 2, dtype=np.float64)
        scale_pad[: min(scale_np.size, scale_pad.size)] = scale_np[: min(scale_np.size, scale_pad.size)]
        self.scale = torch.as_tensor(scale_pad, device=device, dtype=dtype)
        self.m_all = torch.arange(self.degree + 1, device=device, dtype=dtype)

    def acceleration(self, positions_fixed_m: Any) -> Any:
        """Return body-fixed SH acceleration for a position batch ``(N, 3)``."""

        import torch

        x = positions_fixed_m[:, 0]
        y = positions_fixed_m[:, 1]
        z = positions_fixed_m[:, 2]
        rho_sq = x * x + y * y
        r_sq = rho_sq + z * z
        r = torch.sqrt(r_sq).clamp_min(1.0)
        inv_r = 1.0 / r
        inv_r_sq = inv_r * inv_r
        rho = torch.sqrt(rho_sq)

        sin_phi = z * inv_r
        cos_phi = rho * inv_r
        pole = rho > 1e-12
        cos_lon = torch.where(pole, x / rho.clamp_min(1e-30), torch.ones_like(x))
        sin_lon = torch.where(pole, y / rho.clamp_min(1e-30), torch.zeros_like(y))

        u_r = positions_fixed_m * inv_r[:, None]
        u_phi = torch.stack(
            (-sin_phi * cos_lon, -sin_phi * sin_lon, cos_phi),
            dim=1,
        )

        batch_n = positions_fixed_m.shape[0]
        nmax = self.degree
        p_val = torch.zeros((batch_n, nmax + 1, nmax + 2), device=self.device, dtype=self.dtype)
        d_p = torch.zeros_like(p_val)
        p_val[:, 0, 0] = 1.0

        for n in range(1, nmax + 1):
            p_val[:, n, n] = self.diag[n] * cos_phi * p_val[:, n - 1, n - 1]
            p_val[:, n, n - 1] = self.subdiag[n] * sin_phi * p_val[:, n - 1, n - 1]
            if n >= 2:
                m_slice = slice(0, n - 1)
                p_val[:, n, m_slice] = (
                    self.A[n, m_slice][None, :] * sin_phi[:, None] * p_val[:, n - 1, m_slice]
                    - self.B[n, m_slice][None, :] * p_val[:, n - 2, m_slice]
                )

            d_p[:, n, 0] = math.sqrt(n * (n + 1.0)) * p_val[:, n, 1]
            if n >= 1:
                m = torch.arange(1, n + 1, device=self.device, dtype=self.dtype)
                coeff_minus = torch.sqrt((n + m) * (n - m + 1.0))
                term_minus = coeff_minus[None, :] * p_val[:, n, 0:n]
                term_plus = torch.zeros((batch_n, n), device=self.device, dtype=self.dtype)
                if n >= 2:
                    m2 = torch.arange(1, n, device=self.device, dtype=self.dtype)
                    coeff_plus = torch.sqrt((n - m2) * (n + m2 + 1.0))
                    term_plus[:, 0 : n - 1] = coeff_plus[None, :] * p_val[:, n, 2 : n + 1]
                d_p[:, n, 1 : n + 1] = 0.5 * (term_plus - term_minus)

        scale = self.scale[: nmax + 2]
        p_val = p_val * scale[None, None, :]
        d_p = d_p * scale[None, None, :]

        cos_m = torch.empty((batch_n, nmax + 1), device=self.device, dtype=self.dtype)
        sin_m = torch.empty_like(cos_m)
        cos_m[:, 0] = 1.0
        sin_m[:, 0] = 0.0
        if nmax >= 1:
            cos_m[:, 1] = cos_lon
            sin_m[:, 1] = sin_lon
        for m_i in range(2, nmax + 1):
            cos_m[:, m_i] = cos_m[:, m_i - 1] * cos_lon - sin_m[:, m_i - 1] * sin_lon
            sin_m[:, m_i] = sin_m[:, m_i - 1] * cos_lon + cos_m[:, m_i - 1] * sin_lon

        dv_dr = -self.mu * inv_r_sq
        dv_dphi = torch.zeros_like(dv_dr)
        dv_dlambda = torch.zeros_like(dv_dr)

        if nmax >= 2:
            r_ratio_base = self.r_ref * inv_r
            r_ratio_n = r_ratio_base * r_ratio_base
            mu_inv_r = self.mu * inv_r
            mu_inv_r_sq = self.mu * inv_r_sq
            for n in range(2, nmax + 1):
                sl = slice(0, n + 1)
                term_lon = self.C[n, sl][None, :] * cos_m[:, sl] + self.S[n, sl][None, :] * sin_m[:, sl]
                deriv_lon = -self.C[n, sl][None, :] * sin_m[:, sl] + self.S[n, sl][None, :] * cos_m[:, sl]
                m = self.m_all[sl]
                s_r = torch.sum(p_val[:, n, sl] * term_lon, dim=1)
                s_p = torch.sum(d_p[:, n, sl] * term_lon, dim=1)
                s_l = torch.sum(m[None, :] * p_val[:, n, sl] * deriv_lon, dim=1)
                dv_dr = dv_dr - mu_inv_r_sq * (n + 1.0) * r_ratio_n * s_r
                dv_dphi = dv_dphi + mu_inv_r * r_ratio_n * s_p
                dv_dlambda = dv_dlambda + mu_inv_r * r_ratio_n * s_l
                r_ratio_n = r_ratio_n * r_ratio_base

        phi_factor = dv_dphi * inv_r
        inv_rho_sq = torch.where(rho_sq < 1e-24, torch.zeros_like(rho_sq), 1.0 / (rho_sq + 1e-24))
        ax = dv_dr * u_r[:, 0] + phi_factor * u_phi[:, 0] - dv_dlambda * y * inv_rho_sq
        ay = dv_dr * u_r[:, 1] + phi_factor * u_phi[:, 1] + dv_dlambda * x * inv_rho_sq
        az = dv_dr * u_r[:, 2] + phi_factor * u_phi[:, 2]
        return torch.stack((ax, ay, az), dim=1)


__all__ = ["TorchSHGravityEvaluator"]
