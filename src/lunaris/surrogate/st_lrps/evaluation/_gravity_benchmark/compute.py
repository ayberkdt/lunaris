# -*- coding: utf-8 -*-
"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.core.config import SimConfig
from lunaris.core.state import create_state_from_keplerian, calculate_ae_from_altitudes
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagator import propagate
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.surrogate.st_lrps.evaluation import progress

# --- intra-package wiring (auto-generated split) ---
from .types import (
    BatchModelResult,
    GpuBatchTask,
    GravityModelCache,
    SAMPLING_METHODS,
    SCENARIO_UNIT_DIM,
    Scenario,
)

# =============================================================================
# GPU batch comparison helpers
# =============================================================================

def _model_display_name(model_name: str) -> str:
    name = str(model_name).lower()
    base_name, dt_label = _split_gpu_variant_name(name)
    name = base_name
    if name == "st_lrps":
        base = "GPU_ST_LRPS_RK4"
        return f"{base}_DT{dt_label}" if dt_label else base
    if name.startswith("sh"):
        base = f"GPU_{name.upper()}_RK4"
        return f"{base}_DT{dt_label}" if dt_label else base
    return name.upper()


def _parse_model_list_csv(value: str) -> List[str]:
    return [m.strip().lower() for m in str(value).split(",") if m.strip()]


def _parse_float_list_csv(value: Optional[str]) -> List[float]:
    if value is None or str(value).strip() == "":
        return []
    out: List[float] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = float(raw)
        if val <= 0.0:
            raise ValueError("--gpu-rk4-dt-s-list values must be positive.")
        out.append(val)
    return out


def _format_rk4_dt_label(dt_s: float) -> str:
    return f"{float(dt_s):g}"


def _format_rk4_dt_token(dt_s: float) -> str:
    return _format_rk4_dt_label(dt_s).replace("-", "m").replace(".", "p")


def _split_gpu_variant_name(model_name: str) -> Tuple[str, Optional[str]]:
    name = str(model_name).lower()
    marker = "_rk4_dt"
    if marker not in name:
        return name, None
    base, token = name.split(marker, 1)
    token = token.strip("_")
    label = token.replace("m", "-").replace("p", ".")
    return base, label or None


def _gpu_variant_cache_name(model_name: str, rk4_dt_s: float, include_dt: bool) -> str:
    base = str(model_name).strip().lower()
    if not include_dt:
        return base
    return f"{base}_rk4_dt{_format_rk4_dt_token(rk4_dt_s)}"


def _gpu_rk4_dt_values(args: argparse.Namespace) -> List[float]:
    values = _parse_float_list_csv(getattr(args, "gpu_rk4_dt_s_list", None))
    if values:
        return values
    return [float(args.rk4_dt_s if args.rk4_dt_s is not None else args.st_lrps_rk4_dt)]


def _build_gpu_batch_tasks(gpu_models: List[str], args: argparse.Namespace) -> List[GpuBatchTask]:
    dt_values = _gpu_rk4_dt_values(args)
    include_dt = bool(str(getattr(args, "gpu_rk4_dt_s_list", "") or "").strip()) or len(dt_values) > 1
    tasks: List[GpuBatchTask] = []
    for model in gpu_models:
        for dt_s in dt_values:
            cache_name = _gpu_variant_cache_name(model, dt_s, include_dt)
            tasks.append(GpuBatchTask(
                model_name=str(model).lower(),
                cache_name=cache_name,
                display_name=_model_display_name(cache_name),
                rk4_dt_s=float(dt_s),
            ))
    return tasks


def _torch_dtype_from_name(dtype_name: str) -> Any:
    import torch
    return torch.float64 if str(dtype_name).lower() == "float64" else torch.float32


def _quat_rotate_torch(q: Any, v: Any) -> Any:
    """Rotate a batch of vectors by scalar-first quaternion q."""

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (q2 * vz - q3 * vy)
    ty = 2.0 * (q3 * vx - q1 * vz)
    tz = 2.0 * (q1 * vy - q2 * vx)
    cx = q2 * tz - q3 * ty
    cy = q3 * tx - q1 * tz
    cz = q1 * ty - q2 * tx
    return v + torch_stack_like(v, (q0 * tx + cx, q0 * ty + cy, q0 * tz + cz))


def torch_stack_like(reference: Any, cols: Tuple[Any, Any, Any]) -> Any:
    """Stack columns using the torch module that owns *reference*."""

    import torch
    return torch.stack(cols, dim=1).to(device=reference.device, dtype=reference.dtype)


@dataclass
class TorchFrameCache:
    q_i2f: Any
    q_f2i: Any


class TorchFrameProvider:
    """
    Torch-side inertial/body-fixed frame provider for the batch RK4 path.

    ``match_dynamics_engine`` samples the same q_i2f table used by
    ``DynamicsEngine``.  Interpolation is normalized linear interpolation, which
    is close to SLERP for the small ephemeris cadence used here and keeps the
    GPU path free of host round-trips.
    """

    def __init__(self, ephem: Any, *, device: Any, dtype: Any, mode: str) -> None:
        import torch
        self.mode = str(mode)
        self.device = device
        self.dtype = dtype
        if self.mode == "inertial_fixed_legacy":
            self.dt_s = 1.0
            self.q_tab = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0, 0.0]],
                                      device=device, dtype=dtype)
            self.uses_rotation = False
            return

        if self.mode in ("match_dynamics_engine", "precomputed_slerp") and ephem is None:
            raise ValueError(
                f"batch-frame-mode={self.mode} requires an EphemerisManager."
            )
        provider = ephem.get_data_provider()
        q_np = np.asarray(provider["q_i2f_tab"], dtype=np.float64)
        if q_np.ndim != 2 or q_np.shape[1] != 4:
            raise ValueError(f"q_i2f_tab must be shape (N,4), got {q_np.shape}")
        self.dt_s = float(provider["dt_s"])
        self.q_tab = torch.as_tensor(q_np, device=device, dtype=dtype)
        self.uses_rotation = True

    def quat_i2f(self, t_s: float) -> Any:
        import torch
        if self.q_tab.shape[0] <= 1:
            return self.q_tab[0]
        u = max(0.0, float(t_s) / max(self.dt_s, 1e-12))
        i0 = int(math.floor(u))
        if i0 >= self.q_tab.shape[0] - 1:
            return self.q_tab[-1]
        frac = torch.tensor(u - i0, device=self.device, dtype=self.dtype)
        qa = self.q_tab[i0]
        qb = self.q_tab[i0 + 1]
        dot = torch.dot(qa, qb)
        sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
        qb = qb * sign
        dot = torch.clamp(dot * sign, -1.0, 1.0)
        q_linear = (1.0 - frac) * qa + frac * qb
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0).clamp_min(1e-30)
        theta = theta_0 * frac
        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = torch.sin(theta) / sin_theta_0
        q_slerp = s0 * qa + s1 * qb
        q = torch.where(dot > 0.9995, q_linear, q_slerp)
        return q / torch.linalg.norm(q).clamp_min(1e-30)

    def inertial_to_fixed(self, t_s: float, r_i: Any) -> Any:
        if not self.uses_rotation:
            return r_i
        return _quat_rotate_torch(self.quat_i2f(t_s), r_i)

    def fixed_to_inertial(self, t_s: float, a_f: Any) -> Any:
        if not self.uses_rotation:
            return a_f
        q = self.quat_i2f(t_s).clone()
        q[1:] = -q[1:]
        return _quat_rotate_torch(q, a_f)

    def precompute_rk_stage_quaternions(self, total_steps: int, dt_eff: float, gpu_integrator: str) -> "TorchFrameCache":
        import torch
        if gpu_integrator == "light":
            rel_t = [0.0, 0.5 * dt_eff]
        elif gpu_integrator == "robust":
            rel_t = [0.0, 0.5 * dt_eff, 0.5 * dt_eff, dt_eff,
                     0.0, 0.25 * dt_eff, 0.25 * dt_eff, 0.5 * dt_eff,
                     0.5 * dt_eff, 0.75 * dt_eff, 0.75 * dt_eff, dt_eff]
        else:
            rel_t = [0.0, 0.5 * dt_eff, 0.5 * dt_eff, dt_eff]
            
        t_base = torch.arange(total_steps, dtype=torch.float64, device=self.device) * dt_eff
        t_rel = torch.tensor(rel_t, dtype=torch.float64, device=self.device)
        t_all = t_base[:, None] + t_rel[None, :]
        
        if self.q_tab.shape[0] <= 1:
            q_i2f = self.q_tab[0].expand(*t_all.shape, 4).clone()
        else:
            u = (t_all / max(self.dt_s, 1e-12)).clamp_min(0.0)
            i0 = torch.floor(u).long()
            max_idx = self.q_tab.shape[0] - 2
            i0 = i0.clamp(max=max_idx)

            # Match the dynamic quat_i2f() out-of-range behaviour: beyond the
            # final ephemeris interval (u >= N-1) hold the LAST quaternion rather
            # than extrapolating. Clamping i0 alone would leave frac > 1 and
            # extrapolate; clamping frac to [0, 1] reproduces the held endpoint.
            frac = (u - i0).clamp(0.0, 1.0).to(dtype=self.dtype)
            qa = self.q_tab[i0]
            qb = self.q_tab[i0 + 1]
            
            dot = (qa * qb).sum(dim=-1)
            sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
            qb = qb * sign.unsqueeze(-1)
            dot = (dot * sign).clamp(-1.0, 1.0)
            
            q_linear = (1.0 - frac.unsqueeze(-1)) * qa + frac.unsqueeze(-1) * qb
            
            theta_0 = torch.acos(dot)
            sin_theta_0 = torch.sin(theta_0).clamp_min(1e-30)
            theta = theta_0 * frac
            
            s0 = (torch.sin(theta_0 - theta) / sin_theta_0).unsqueeze(-1)
            s1 = (torch.sin(theta) / sin_theta_0).unsqueeze(-1)
            
            q_slerp = s0 * qa + s1 * qb
            q_i2f = torch.where((dot > 0.9995).unsqueeze(-1), q_linear, q_slerp)
            q_i2f = q_i2f / torch.linalg.norm(q_i2f, dim=-1, keepdim=True).clamp_min(1e-30)

        q_f2i = q_i2f.clone()
        q_f2i[..., 1:] = -q_f2i[..., 1:]
        
        return TorchFrameCache(q_i2f=q_i2f, q_f2i=q_f2i)


class TorchSHGravityEvaluator:
    """
    Torch vectorized spherical-harmonic gravity evaluator.

    The implementation mirrors the repository SH kernel recurrence but evaluates
    all scenarios in a position batch at once on the selected torch device.
    """

    def __init__(self, gravity_model: Any, *, degree: int, device: Any, dtype: Any) -> None:
        import torch
        self.degree = int(degree)
        self.device = device
        self.dtype = dtype
        self.backend = "torch_sh"
        self.r_ref = torch.tensor(float(getattr(gravity_model, "R_ref_m")), device=device, dtype=dtype)
        self.mu = torch.tensor(float(getattr(gravity_model, "GM_m3s2")), device=device, dtype=dtype)
        self.C = torch.as_tensor(np.array(getattr(gravity_model, "Cnm"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.S = torch.as_tensor(np.array(getattr(gravity_model, "Snm"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.diag = torch.as_tensor(np.array(getattr(gravity_model, "diag"), dtype=np.float64, copy=True),
                                    device=device, dtype=dtype)
        self.subdiag = torch.as_tensor(np.array(getattr(gravity_model, "subdiag"), dtype=np.float64, copy=True),
                                       device=device, dtype=dtype)
        self.A = torch.as_tensor(np.array(getattr(gravity_model, "A"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        self.B = torch.as_tensor(np.array(getattr(gravity_model, "B"), dtype=np.float64, copy=True),
                                 device=device, dtype=dtype)
        scale_np = np.asarray(getattr(gravity_model, "scale_m"), dtype=np.float64)
        scale_pad = np.ones(self.degree + 2, dtype=np.float64)
        scale_pad[:min(scale_np.size, scale_pad.size)] = scale_np[:min(scale_np.size, scale_pad.size)]
        self.scale = torch.as_tensor(scale_pad, device=device, dtype=dtype)
        self.m_all = torch.arange(self.degree + 1, device=device, dtype=dtype)

    def acceleration(self, positions_fixed_m: Any) -> Any:
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
        P = torch.zeros((batch_n, nmax + 1, nmax + 2), device=self.device, dtype=self.dtype)
        dP = torch.zeros_like(P)
        P[:, 0, 0] = 1.0

        for n in range(1, nmax + 1):
            P[:, n, n] = self.diag[n] * cos_phi * P[:, n - 1, n - 1]
            P[:, n, n - 1] = self.subdiag[n] * sin_phi * P[:, n - 1, n - 1]
            if n >= 2:
                m_slice = slice(0, n - 1)
                P[:, n, m_slice] = (
                    self.A[n, m_slice][None, :] * sin_phi[:, None] * P[:, n - 1, m_slice]
                    - self.B[n, m_slice][None, :] * P[:, n - 2, m_slice]
                )

            dP[:, n, 0] = math.sqrt(n * (n + 1.0)) * P[:, n, 1]
            if n >= 1:
                m = torch.arange(1, n + 1, device=self.device, dtype=self.dtype)
                coeff_minus = torch.sqrt((n + m) * (n - m + 1.0))
                term_minus = coeff_minus[None, :] * P[:, n, 0:n]
                term_plus = torch.zeros((batch_n, n), device=self.device, dtype=self.dtype)
                if n >= 2:
                    m2 = torch.arange(1, n, device=self.device, dtype=self.dtype)
                    coeff_plus = torch.sqrt((n - m2) * (n + m2 + 1.0))
                    term_plus[:, 0:n - 1] = coeff_plus[None, :] * P[:, n, 2:n + 1]
                dP[:, n, 1:n + 1] = 0.5 * (term_plus - term_minus)

        scale = self.scale[:nmax + 2]
        P = P * scale[None, None, :]
        dP = dP * scale[None, None, :]

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
                s_r = torch.sum(P[:, n, sl] * term_lon, dim=1)
                s_p = torch.sum(dP[:, n, sl] * term_lon, dim=1)
                s_l = torch.sum(m[None, :] * P[:, n, sl] * deriv_lon, dim=1)
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


def _make_gpu_accelerator(model_name: str, gravity_model: Any, *, device: Any, dtype: Any) -> Tuple[Any, str]:
    """Create a batched torch acceleration provider for one GPU model."""

    name = str(model_name).lower()
    if name == "st_lrps":
        if str(getattr(gravity_model, "device", "")) != str(device):
            gravity_model.to_device(device)

        def _accel_st(pos_fixed: Any) -> Any:
            return gravity_model.predict_total_accel_torch(pos_fixed).to(device=device, dtype=dtype)

        return _accel_st, "torch_st_lrps"

    if name.startswith("sh"):
        degree = int(name.replace("sh", ""))
        evaluator = TorchSHGravityEvaluator(gravity_model, degree=degree, device=device, dtype=dtype)
        return evaluator.acceleration, evaluator.backend

    raise ValueError(f"Unsupported GPU batch model: {model_name!r}")


# GPU fixed-step integrators. The GPU batch path must use fixed-step schemes
# (no adaptive error control on-device), so three fidelity tiers are offered:
#   light  -> RK2 midpoint           (order 2, 2 RHS evals/step, cheapest)
#   medium -> classic RK4            (order 4, 4 RHS evals/step, default)
#   robust -> RK4 + Richardson       (local order ~6, 12 RHS evals/step, accurate)
# The helpers only use +, *, / and the rhs callable, so they are backend-agnostic
# (work on torch tensors or numpy arrays) and unit-testable without CUDA.
GPU_INTEGRATORS = ("light", "medium", "robust")


def _rk4_step(rhs, t, state, dt):
    """One classic fourth-order Runge-Kutta step."""
    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gpu_fixed_step_advance(rhs, t, state, dt, method: str = "medium"):
    """Advance ``state`` by one ``dt`` using the selected fixed-step method.

    ``method`` is one of :data:`GPU_INTEGRATORS`. Unknown values fall back to the
    medium (classic RK4) scheme. The integrator advances by exactly ``dt`` per
    call; ``robust`` performs internal half-steps and Richardson extrapolation
    but still represents a single output step.
    """
    m = str(method).lower()
    if m == "light":  # midpoint RK2
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        return state + dt * k2
    if m == "robust":  # RK4 with one level of Richardson extrapolation
        full = _rk4_step(rhs, t, state, dt)
        half = _rk4_step(rhs, t, state, 0.5 * dt)
        half2 = _rk4_step(rhs, t + 0.5 * dt, half, 0.5 * dt)
        return (16.0 * half2 - full) / 15.0
    # medium / default: classic RK4
    return _rk4_step(rhs, t, state, dt)


def _gpu_integrator_evals_per_step(method: str) -> int:
    """RHS (acceleration) evaluations per output step for each GPU integrator.

    light = RK2 midpoint (2), medium = classic RK4 (4),
    robust = RK4 + one Richardson level = 3 RK4 steps (12). Mirrors
    :func:`gpu_fixed_step_advance` so throughput metrics are not mis-scaled.
    """
    return {"light": 2, "medium": 4, "robust": 12}.get(str(method).lower(), 4)


def propagate_gpu_batch_model(
    model_name: str,
    gravity_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    rk4_dt_s: float,
    output_dt_s: float,
    ephem: Any,
    *,
    device: Any,
    dtype: Any,
    dtype_name: str,
    frame_mode: str,
    gpu_integrator: str = "medium",
    finite_check_mode: str = "step",
    progress_cb: Optional[Any] = None,
) -> BatchModelResult:
    """Propagate one model for all scenarios using a fixed-step torch integrator.

    ``gpu_integrator`` selects the fidelity tier (light/medium/robust); see
    :func:`gpu_fixed_step_advance`.

    ``finite_check_mode`` controls how often the batch state is scanned for
    NaN/Inf. It is a monitoring policy only and never changes the numbers
    produced — the same trajectory is returned for every mode (modulo whether a
    non-finite result is reported as a failure):

    * ``step`` — scan after every RK step (safest for debugging, highest
      CPU-side overhead; the historical behaviour and the function default).
    * ``snapshot`` — scan once per output snapshot; recommended for benchmark
      throughput and still catches non-finite output before results are
      returned.
    * ``end`` — scan once over the full trajectory after integration; lowest
      checking overhead while still detecting invalid output.
    * ``off`` — skip the scan entirely; fastest but will not flag invalid GPU
      states. Unknown values are treated as ``step`` (fail safe).

    ``progress_cb`` (optional) is invoked as ``cb(current_step, total_steps,
    elapsed_s)`` at step 0, on a throttled cadence during integration, and once
    more on successful completion. It is logging-only and never affects the
    numerical result.
    """

    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if rk4_dt_s <= 0.0 or output_dt_s <= 0.0:
        raise ValueError("rk4_dt_s and output_dt_s must be positive.")
    if rk4_dt_s > output_dt_s:
        print(f"[gpu-batch] WARNING: rk4_dt_s={rk4_dt_s} > output_dt_s={output_dt_s}; "
              "using output_dt_s as the effective RK4 step.", flush=True)
        rk4_dt_s = output_dt_s
    steps_per_snap = max(1, round(output_dt_s / rk4_dt_s))
    dt_eff = output_dt_s / steps_per_snap
    frac = output_dt_s / rk4_dt_s
    if abs(frac - round(frac)) > 1e-6:
        print(f"[gpu-batch] WARNING: output_dt_s={output_dt_s} is not divisible by "
              f"rk4_dt_s={rk4_dt_s}; effective dt={dt_eff:.6f}s.", flush=True)

    n_scenarios = int(y0_batch.shape[0])
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1, dtype=np.float64)

    frame = TorchFrameProvider(ephem, device=device, dtype=dtype, mode=frame_mode)
    accel_fixed, backend = _make_gpu_accelerator(model_name, gravity_model, device=device, dtype=dtype)

    state = torch.as_tensor(y0_batch, device=device, dtype=dtype)
    y_gpu = torch.empty((n_snaps + 1, n_scenarios, 6), device=device, dtype=dtype)
    y_gpu[0].copy_(state)

    total_steps = int(n_snaps * steps_per_snap)

    if frame_mode == "precomputed_slerp":
        frame_cache = frame.precompute_rk_stage_quaternions(total_steps, dt_eff, gpu_integrator)
        step_idx = 0
        stage_idx = 0
        stages_per_step = frame_cache.q_i2f.shape[1]
        
        def _rhs(t_s: float, s: Any) -> Any:
            nonlocal step_idx, stage_idx
            q_i2f = frame_cache.q_i2f[step_idx, stage_idx]
            q_f2i = frame_cache.q_f2i[step_idx, stage_idx]
            
            r_i = s[:, :3]
            v_i = s[:, 3:]
            r_f = _quat_rotate_torch(q_i2f, r_i)
            a_f = accel_fixed(r_f)
            a_i = _quat_rotate_torch(q_f2i, a_f)
            
            stage_idx += 1
            if stage_idx == stages_per_step:
                stage_idx = 0
                step_idx += 1
                
            return torch.cat((v_i, a_i), dim=1)
    else:
        def _rhs(t_s: float, s: Any) -> Any:
            r_i = s[:, :3]
            v_i = s[:, 3:]
            r_f = frame.inertial_to_fixed(t_s, r_i)
            a_f = accel_fixed(r_f)
            a_i = frame.fixed_to_inertial(t_s, a_f)
            return torch.cat((v_i, a_i), dim=1)

    throttle = progress.StepThrottle(total_steps)

    def _emit_progress(step: int, elapsed: float) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(int(step), total_steps, float(elapsed))
        except Exception:
            pass

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_curr = 0.0
    status = "ok"
    failure_reason = ""
    step_count = 0
    bad_state = torch.zeros((), device=device, dtype=torch.bool)

    finite_mode = str(finite_check_mode).lower()
    if finite_mode not in ("step", "snapshot", "end", "off"):
        finite_mode = "step"  # fail safe: unknown policy -> safest behaviour

    def _accumulate_bad_state(tensor: Any) -> None:
        # OR a non-finite flag into ``bad_state`` without materialising a host
        # value in the hot path; the single device->host sync happens once after
        # integration. This monitors the state but never mutates it.
        bad_state.logical_or_(~torch.isfinite(tensor).all())

    _emit_progress(0, 0.0)

    try:
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                state = gpu_fixed_step_advance(_rhs, t_curr, state, dt_eff, gpu_integrator)
                t_curr += dt_eff
                step_count += 1
                if finite_mode == "step":
                    _accumulate_bad_state(state)
                # Consult the clock only when the step gate may be due, so
                # time.perf_counter() stays out of the per-step hot path.
                if throttle.needs_time_check(step_count):
                    now = time.perf_counter()
                    if throttle.update(step_count, now):
                        _emit_progress(step_count, now - t0)
            if finite_mode == "snapshot":
                _accumulate_bad_state(state)
            y_gpu[snap_idx + 1].copy_(state)
    except Exception as exc:
        status = "failed"
        failure_reason = str(exc)
        print(f"[gpu-batch] {model_name.upper()} failed: {exc}", flush=True)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    if status == "ok" and finite_mode == "end":
        _accumulate_bad_state(y_gpu)
    if status == "ok" and bool(bad_state.detach().cpu().item()):
        status = "failed"
        failure_reason = f"non-finite state in {model_name}"
        print(f"[gpu-batch] {model_name.upper()} failed: {failure_reason}", flush=True)
    y_out = y_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    if status == "ok":
        _emit_progress(total_steps, runtime_s)
    return BatchModelResult(
        model_name=str(model_name).lower(),
        display_name=_model_display_name(model_name),
        backend=backend,
        device=str(device),
        dtype=str(dtype_name),
        t=t_out,
        y=y_out,
        runtime_s=float(runtime_s),
        n_steps=int(n_steps),
        n_scenarios=n_scenarios,
        rk4_dt_s=float(dt_eff),
        output_dt_s=float(output_dt_s),
        status=status,
        failure_reason=failure_reason,
    )


# =============================================================================
# Scenario generation
# =============================================================================

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _sobol_note(method: str, n: int) -> str:
    if str(method).startswith("sobol") and n > 0 and not _is_power_of_two(int(n)):
        return ("Sobol sequences have their strongest balance properties when "
                "the scenario count is a power of two; this run generated the "
                "next power-of-two sequence and truncated to the requested count.")
    return ""


def _require_qmc():
    try:
        from scipy.stats import qmc
    except Exception as exc:
        raise ImportError("scipy.stats.qmc is required for LHS/Sobol sampling.") from exc
    return qmc


def generate_unit_samples(
    n: int,
    dim: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Generate unit-hypercube samples for scenario construction."""

    n = int(n)
    dim = int(dim)
    method = str(method)
    if n < 0:
        raise ValueError("n must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if n == 0:
        return np.empty((0, dim), dtype=np.float64)
    if method not in SAMPLING_METHODS:
        raise ValueError(f"Unknown sampling method: {method}")

    if method == "random":
        return np.asarray(np.random.default_rng(int(seed)).random((n, dim)), dtype=np.float64)

    qmc = _require_qmc()
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=int(seed))
        return np.asarray(sampler.random(n), dtype=np.float64)

    scramble = method == "sobol_scrambled"
    sampler = qmc.Sobol(d=dim, scramble=scramble, seed=int(seed) if scramble else None)
    if _is_power_of_two(n):
        m = int(math.log2(n))
        samples = sampler.random_base2(m=m)
    else:
        # Generate a balanced Sobol block and truncate so arbitrary N is allowed
        # without changing the requested scenario count.
        m = int(math.log2(_next_power_of_two(n)))
        samples = sampler.random_base2(m=m)[:n]
    return np.asarray(samples, dtype=np.float64)


def _map_unit_linear(u: float, lo: float, hi: float) -> float:
    return float(lo + float(u) * (hi - lo))


def _map_inclination_deg(u: float, args: argparse.Namespace) -> float:
    inc_min = float(args.inc_min_deg)
    inc_max = float(args.inc_max_deg)
    if str(getattr(args, "inclination_sampling", "uniform_deg")) == "uniform_cos":
        cos_i_min = math.cos(math.radians(inc_max))
        cos_i_max = math.cos(math.radians(inc_min))
        cos_i = _map_unit_linear(float(u), cos_i_min, cos_i_max)
        cos_i = max(-1.0, min(1.0, cos_i))
        return float(math.degrees(math.acos(cos_i)))
    return _map_unit_linear(float(u), inc_min, inc_max)


def _validate_sampling_bounds(args: argparse.Namespace) -> None:
    if float(args.altitude_min_km) > float(args.altitude_max_km):
        raise ValueError("--altitude-min-km must be <= --altitude-max-km")
    if float(args.inc_min_deg) > float(args.inc_max_deg):
        raise ValueError("--inc-min-deg must be <= --inc-max-deg")
    if str(args.scenario_mode) == "near_circular_altitude":
        if float(args.ecc_min) < 0.0 or float(args.ecc_max) >= 1.0:
            raise ValueError("near_circular_altitude requires 0 <= --ecc-min <= --ecc-max < 1")
        if float(args.ecc_min) > float(args.ecc_max):
            raise ValueError("--ecc-min must be <= --ecc-max")


def _state_from_elements(
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    ta_deg: float,
) -> np.ndarray:
    return create_state_from_keplerian(
        semi_major_axis=float(a_m),
        eccentricity=float(e),
        inclination=math.radians(float(inc_deg)),
        raan=math.radians(float(raan_deg)),
        argp=math.radians(float(argp_deg)),
        true_anomaly=math.radians(float(ta_deg)),
        mu=MU_MOON,
    ).y


def generate_scenarios_from_samples(
    samples: np.ndarray,
    args: argparse.Namespace,
) -> List[Scenario]:
    """Map unit-hypercube samples into validation scenarios without propagation."""

    _validate_sampling_bounds(args)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if samples.shape[1] < SCENARIO_UNIT_DIM:
        raise ValueError(f"samples must have at least {SCENARIO_UNIT_DIM} columns")

    scenarios: List[Scenario] = []
    moon_r_km = float(R_MOON) / 1_000.0
    for sid, u in enumerate(samples):
        raw = [float(x) for x in u[:SCENARIO_UNIT_DIM]]
        if str(args.scenario_mode) == "bounded_keplerian":
            raw_alt_1 = _map_unit_linear(raw[0], float(args.altitude_min_km), float(args.altitude_max_km))
            raw_alt_2 = _map_unit_linear(raw[1], float(args.altitude_min_km), float(args.altitude_max_km))
            hp_km = min(raw_alt_1, raw_alt_2)
            ha_km = max(raw_alt_1, raw_alt_2)
            rp_km = moon_r_km + hp_km
            ra_km = moon_r_km + ha_km
            a_km = 0.5 * (rp_km + ra_km)
            e = (ra_km - rp_km) / (ra_km + rp_km)
            inc_u, raan_u, argp_u, ta_u = raw[2], raw[3], raw[4], raw[5]
        else:
            alt_km = _map_unit_linear(raw[0], float(args.altitude_min_km), float(args.altitude_max_km))
            e = _map_unit_linear(raw[1], float(args.ecc_min), float(args.ecc_max))
            a_km = moon_r_km + alt_km
            if abs(e) <= 1e-15:
                hp_km = alt_km
                ha_km = alt_km
            else:
                hp_km = a_km * (1.0 - e) - moon_r_km
                ha_km = a_km * (1.0 + e) - moon_r_km
            inc_u, raan_u, argp_u, ta_u = raw[2], raw[3], raw[4], raw[5]

        if hp_km > ha_km:
            hp_km, ha_km = ha_km, hp_km
        if e < 0.0 or e >= 1.0:
            raise ValueError(f"Generated invalid eccentricity for scenario {sid}: {e}")
        if a_km <= moon_r_km:
            raise ValueError(f"Generated invalid semi-major axis for scenario {sid}: {a_km} km")

        inc_deg = _map_inclination_deg(inc_u, args)
        raan_deg = _map_unit_linear(raan_u, float(args.raan_min_deg), float(args.raan_max_deg))
        argp_deg = _map_unit_linear(argp_u, float(args.argp_min_deg), float(args.argp_max_deg))
        ta_deg = _map_unit_linear(ta_u, float(args.ta_min_deg), float(args.ta_max_deg))
        state = _state_from_elements(a_km * 1_000.0, e, inc_deg, raan_deg, argp_deg, ta_deg)
        if not np.isfinite(state).all():
            raise ValueError(f"Generated non-finite initial state for scenario {sid}")

        scenarios.append(Scenario(
            scenario_id=sid,
            hp_km=float(hp_km),
            ha_km=float(ha_km),
            a_km=float(a_km),
            e=float(e),
            inc_deg=float(inc_deg),
            raan_deg=float(raan_deg),
            argp_deg=float(argp_deg),
            ta_deg=float(ta_deg),
            initial_state=state,
            raw_unit_sample=raw,
            sampling_method=str(getattr(args, "sampling_method", "random")),
        ))
    return scenarios


def generate_random_scenarios(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Scenario]:
    n = args.random_scenarios
    alt_min = args.altitude_min_km
    alt_max = args.altitude_max_km
    inc_min = math.radians(args.inc_min_deg)
    inc_max = math.radians(args.inc_max_deg)

    scenarios: List[Scenario] = []
    attempts = 0
    max_attempts = n * 20

    while len(scenarios) < n and attempts < max_attempts:
        attempts += 1
        sid = len(scenarios)

        try:
            if args.scenario_mode == "bounded_keplerian":
                hp_km = float(rng.uniform(alt_min, max(alt_min + 1.0, alt_max * 0.7)))
                ha_km = float(rng.uniform(hp_km, alt_max))
                a_m, e = calculate_ae_from_altitudes(R_MOON, hp_km, ha_km)
            else:  # near_circular_altitude
                alt_km = float(rng.uniform(alt_min, alt_max))
                e      = float(rng.uniform(args.ecc_min, args.ecc_max))
                a_m    = R_MOON + alt_km * 1_000.0
                if abs(e) <= 1e-15:
                    hp_km = alt_km
                    ha_km = alt_km
                else:
                    hp_km = (a_m * (1.0 - e) - R_MOON) / 1_000.0
                    ha_km = (a_m * (1.0 + e) - R_MOON) / 1_000.0

            if e < 0.0 or e >= 1.0:
                continue
            if a_m <= R_MOON:
                continue

            if str(getattr(args, "inclination_sampling", "uniform_deg")) == "uniform_cos":
                inc_deg = _map_inclination_deg(float(rng.random()), args)
            else:
                inc_deg  = float(math.degrees(rng.uniform(inc_min, inc_max)))
            raan_deg = float(rng.uniform(args.raan_min_deg, args.raan_max_deg))
            argp_deg = float(rng.uniform(args.argp_min_deg, args.argp_max_deg))
            ta_deg   = float(rng.uniform(args.ta_min_deg, args.ta_max_deg))

            state = _state_from_elements(a_m, e, inc_deg, raan_deg, argp_deg, ta_deg)

            if not np.isfinite(state).all():
                continue

            scenarios.append(Scenario(
                scenario_id=sid,
                hp_km=hp_km,
                ha_km=ha_km,
                a_km=a_m / 1_000.0,
                e=e,
                inc_deg=inc_deg,
                raan_deg=raan_deg,
                argp_deg=argp_deg,
                ta_deg=ta_deg,
                initial_state=state,
                sampling_method="random",
            ))
        except Exception:
            continue

    if len(scenarios) < n:
        print(f"WARNING: only generated {len(scenarios)}/{n} valid scenarios "
              f"after {attempts} attempts")

    return scenarios


def generate_validation_scenarios(args: argparse.Namespace) -> List[Scenario]:
    method = str(getattr(args, "sampling_method", "random"))
    if method == "random":
        rng = np.random.default_rng(args.scenario_seed)
        return generate_random_scenarios(args, rng)
    samples = generate_unit_samples(
        int(args.random_scenarios),
        SCENARIO_UNIT_DIM,
        method,
        int(args.scenario_seed),
    )
    return generate_scenarios_from_samples(samples, args)


# =============================================================================
# DOP853 propagation
# =============================================================================

def propagate_for_scenario(
    model_name: str,
    y0: np.ndarray,
    args: argparse.Namespace,
    cfg_base: SimConfig,
    ephem: Any,
    model_cache: GravityModelCache,
) -> Tuple[Optional[Any], float]:
    """Propagate with the named model. Returns (PropagationResult|None, runtime_s)."""
    grav = model_cache.get(model_name)
    cfg  = cfg_base

    if model_name == "st_lrps":
        # ST-LRPS has degree_max=200 (training target), which causes the Nyquist
        # criterion to demand ~5s steps.  We disable Nyquist by:
        # 1. Setting use_nyquist_max_step=False in PropagatorConfig
        # 2. Temporarily overriding grav.degree_max = grav.degree_min so that
        #    _get_sh_degree() returns the base degree (e.g. 10) as a belt-and-suspenders.
        new_prop = replace(cfg_base.propagator, use_nyquist_max_step=False)
        cfg = replace(cfg_base, propagator=new_prop)

        # Belt-and-suspenders: temporarily lower degree_max on the surrogate
        _orig_dmax = getattr(grav, "degree_max", 200)
        _base_deg  = max(1, int(getattr(grav, "degree_min", 20)))
        try:
            grav.degree_max = _base_deg
        except Exception:
            pass

    dyn = DynamicsEngine(
        sc_props=cfg.spacecraft,
        flags=cfg.flags,
        gravity_model=grav,
        ephem_manager=ephem,
        allow_identity_rotation=True,
    )
    t0 = time.perf_counter()
    try:
        res = propagate(dyn, y0, cfg.propagator, time_cfg=cfg.time)
    except Exception as exc:
        print(f"    ERROR propagating {model_name}: {exc}", flush=True)
        res = None
    finally:
        # Restore degree_max on surrogate
        if model_name == "st_lrps":
            try:
                grav.degree_max = _orig_dmax
            except Exception:
                pass
    rt = time.perf_counter() - t0

    if res is None or (res.ode is not None and not res.ode.success):
        return None, rt
    return res, rt


# =============================================================================
# Batched force evaluation (ST-LRPS)
# =============================================================================

def evaluate_st_lrps_forces_batched(
    model: Any,
    positions_m: np.ndarray,   # (N, 3) body-fixed
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Batch evaluate ST-LRPS acceleration at N body-fixed positions.
    Returns (N, 3) in m/s^2.
    """
    N = positions_m.shape[0]
    result = np.empty((N, 3), dtype=np.float64)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        chunk = positions_m[start:end]
        result[start:end] = model.acceleration_fixed_batch(chunk)

    return result


def _synchronize_model_device_if_cuda(model: Any) -> None:
    """Synchronize CUDA timing when *model* is resident on GPU."""

    dev = str(getattr(model, "device", "") or "").lower()
    if "cuda" not in dev:
        return
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


# =============================================================================
# Batch GPU/CPU RK4 for ST-LRPS
# =============================================================================

class _BatchMCCfg:
    """Minimal mc_cfg duck-type for TorchBatchPropagator."""
    def __init__(self, dt_s: float, impact_alt_km: float = 0.0, torch_dtype: str = "float32") -> None:
        self.dt_s = float(dt_s)
        self.impact_alt_km = float(impact_alt_km)
        self.torch_dtype = str(torch_dtype)


def run_st_lrps_batch_rk4(
    surrogate_model: Any,
    y0_batch: np.ndarray,          # (N, 6) SI
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run ST-LRPS fixed-step RK4 for N scenarios.

    GPU path (CUDA available): uses core.torch_batch_propagator.TorchBatchPropagator.
    CPU fallback: sequential numpy batch RK4 using acceleration_fixed_batch.

    NOTE: Both paths evaluate gravity in the inertial frame without applying the
    Moon's rotation matrix (body-fixed != inertial approximation). This matches the
    TorchBatchPropagator's existing contract and is acceptable for short durations.
    """
    requested_chunk = getattr(args, "batch_size", None)
    if requested_chunk is not None and int(requested_chunk) > 0 and y0_batch.shape[0] > int(requested_chunk):
        chunk_size = int(requested_chunk)
        print(f"[batch-rk4] Splitting {y0_batch.shape[0]} scenarios into chunks of {chunk_size}.",
              flush=True)
        child_args = argparse.Namespace(**vars(args))
        child_args.batch_size = None
        chunk_results: List[Dict[str, Any]] = []
        for start in range(0, y0_batch.shape[0], chunk_size):
            end = min(start + chunk_size, y0_batch.shape[0])
            print(f"[batch-rk4] chunk {start}:{end}", flush=True)
            chunk_results.append(run_st_lrps_batch_rk4(
                surrogate_model,
                y0_batch[start:end],
                duration_s=duration_s,
                dt_s=dt_s,
                output_dt_s=output_dt_s,
                args=child_args,
            ))

        t_ref = chunk_results[0]["t"]
        if any(len(r["t"]) != len(t_ref) or np.max(np.abs(r["t"] - t_ref)) > 1e-6 for r in chunk_results):
            raise RuntimeError("Chunked RK4 runs produced inconsistent output time grids.")
        Y = np.concatenate([r["Y"] for r in chunk_results], axis=1)
        impact_flags = np.concatenate([r.get("impact_flags", np.zeros(r["Y"].shape[1])) for r in chunk_results])
        t_impact = np.concatenate([r.get("t_impact", np.full(r["Y"].shape[1], np.nan)) for r in chunk_results])
        runtime_s = float(sum(float(r.get("runtime_s", 0.0)) for r in chunk_results))
        n_steps = int(chunk_results[0].get("n_steps", 0))
        n_scenarios = int(y0_batch.shape[0])
        return {
            "t": t_ref,
            "Y": Y,
            "impact_flags": impact_flags,
            "t_impact": t_impact,
            "runtime_s": runtime_s,
            "device": chunk_results[0].get("device", "?"),
            "dt_s": float(chunk_results[0].get("dt_s", dt_s)),
            "n_scenarios": n_scenarios,
            "n_steps": n_steps,
            "samples_per_second": n_scenarios * n_steps / max(runtime_s, 1e-9),
            "mode": str(chunk_results[0].get("mode", "chunked_rk4")),
            "chunk_size": chunk_size,
            "torch_dtype": getattr(args, "torch_dtype", "float64"),
            "y_layout": "time_scenario_state",
        }

    # Warn if dt > output_dt
    if dt_s > output_dt_s:
        print(f"[batch-rk4] WARNING: rk4-dt ({dt_s}s) > dt-out ({output_dt_s}s); "
              f"clamping to dt_s = output_dt_s.", flush=True)
        dt_s = output_dt_s

    steps_per_snap = max(1, round(output_dt_s / dt_s))
    dt_eff = output_dt_s / steps_per_snap
    frac = output_dt_s / dt_s
    if abs(frac - round(frac)) > 0.01:
        print(f"[batch-rk4] WARNING: output_dt ({output_dt_s}s) not divisible by "
              f"rk4-dt ({dt_s}s). Effective dt = {dt_eff:.3f}s.", flush=True)

    # Try GPU path
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if cuda_ok:
        return _run_batch_rk4_gpu(surrogate_model, y0_batch, duration_s, dt_eff,
                                   output_dt_s, args)
    else:
        fallback = getattr(args, "gpu_fallback", "cpu")
        if fallback == "error":
            raise RuntimeError(
                "GPU batch RK4 requested but CUDA is unavailable. "
                "Use --gpu-fallback cpu."
            )
        print("[batch-rk4] CUDA unavailable; using CPU batch RK4.", flush=True)
        return _run_batch_rk4_cpu(surrogate_model, y0_batch, duration_s, dt_eff, output_dt_s)


def _run_batch_rk4_gpu(
    surrogate_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import torch
    from lunaris.core.torch_batch_propagator import TorchBatchPropagator

    device = torch.device("cuda:0")
    dev_name = torch.cuda.get_device_name(0)

    # Move model to GPU if needed
    if str(surrogate_model.device) != str(device):
        surrogate_model.to_device(device)

    mc_cfg = _BatchMCCfg(dt_s=dt_s, torch_dtype=getattr(args, "torch_dtype", "float32"))
    prop   = TorchBatchPropagator(surrogate_model, mc_cfg, device_id=0)
    N      = y0_batch.shape[0]
    ones_N = np.ones(N, dtype=np.float64)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_out, Y_out, impact_flags, t_impact = prop.propagate(
        y0_batch, ones_N, ones_N, ones_N, ones_N,
        duration_s=duration_s,
        output_dt_s=output_dt_s,
    )
    torch.cuda.synchronize()
    runtime_s = time.perf_counter() - t0

    n_snaps = Y_out.shape[0] - 1
    n_steps = n_snaps * max(1, round(output_dt_s / dt_s))
    print(f"[batch-rk4] GPU done: {runtime_s:.2f}s  device={device} ({dev_name})  "
          f"scenarios={N}  steps={n_steps}  "
          f"throughput={N * n_steps / max(runtime_s, 1e-9):,.0f} traj-steps/s",
          flush=True)

    return {
        "t": t_out, "Y": Y_out,
        "impact_flags": impact_flags, "t_impact": t_impact,
        "runtime_s": runtime_s, "device": f"cuda:0 ({dev_name})",
        "dt_s": dt_s, "n_scenarios": N,
        "n_steps": n_steps,
        "samples_per_second": N * n_steps / max(runtime_s, 1e-9),
        "mode": "gpu_rk4",
        "torch_dtype": getattr(args, "torch_dtype", "float64"),
        "y_layout": "time_scenario_state",
    }


def _run_batch_rk4_cpu(
    surrogate_model: Any,
    y0_batch: np.ndarray,
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
) -> Dict[str, Any]:
    """CPU sequential batch RK4 using acceleration_fixed_batch."""
    N = y0_batch.shape[0]
    steps_per_snap = max(1, round(output_dt_s / dt_s))
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1)
    Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)

    def _batch_accel(Y: np.ndarray) -> np.ndarray:
        return surrogate_model.acceleration_fixed_batch(Y[:, :3])

    def _rhs(Y: np.ndarray) -> np.ndarray:
        a = _batch_accel(Y)
        return np.concatenate([Y[:, 3:], a], axis=1)

    t0 = time.perf_counter()
    Y_curr = y0_batch.copy()
    Y_out[0] = Y_curr

    for snap_idx in range(n_snaps):
        for _ in range(steps_per_snap):
            k1 = _rhs(Y_curr)
            k2 = _rhs(Y_curr + 0.5 * dt_s * k1)
            k3 = _rhs(Y_curr + 0.5 * dt_s * k2)
            k4 = _rhs(Y_curr + dt_s * k3)
            Y_curr = Y_curr + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.isfinite(Y_curr).all():
                print("[batch-rk4] WARNING: non-finite state detected; "
                      "replacing with last good state.", flush=True)
                Y_curr = np.where(np.isfinite(Y_curr), Y_curr, Y_out[snap_idx])
                break
        Y_out[snap_idx + 1] = Y_curr
        if (snap_idx + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  [CPU-RK4] snap {snap_idx+1}/{n_snaps}  {el:.1f}s elapsed", flush=True)

    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    print(f"[batch-rk4] CPU done: {runtime_s:.2f}s  scenarios={N}  steps={n_steps}  "
          f"throughput={N * n_steps / max(runtime_s, 1e-9):,.0f} traj-steps/s", flush=True)

    return {
        "t": t_out, "Y": Y_out,
        "impact_flags": np.zeros(N), "t_impact": np.full(N, np.nan),
        "runtime_s": runtime_s, "device": "cpu",
        "dt_s": dt_s, "n_scenarios": N, "n_steps": n_steps,
        "samples_per_second": N * n_steps / max(runtime_s, 1e-9),
        "mode": "cpu_rk4",
        "torch_dtype": "numpy_float64",
        "y_layout": "time_scenario_state",
    }


# =============================================================================
# SH200 CPU RK4 reference (for error decomposition)
# =============================================================================

def run_sh200_cpu_rk4_reference(
    grav: Any,                 # GravityModel (SH200)
    y0_batch: np.ndarray,     # (N, 6)
    duration_s: float,
    dt_s: float,
    output_dt_s: float,
) -> Dict[str, Any]:
    """
    Run SH200 fixed-step CPU RK4 for N scenarios sequentially.
    Gravity evaluated WITHOUT lunar rotation (same approximation as GPU batch RK4)
    so that the two can be directly subtracted for error decomposition.
    """
    from lunaris.physics.spherical_harmonics import sh_accel_fixed_numba

    N = y0_batch.shape[0]
    steps_per_snap = max(1, round(output_dt_s / dt_s))
    n_snaps = max(1, round(duration_s / output_dt_s))
    t_out = np.linspace(0.0, n_snaps * output_dt_s, n_snaps + 1)
    Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)

    # Pre-allocate workspace once
    ws = grav.make_workspace()

    def sh_accel(r: np.ndarray) -> np.ndarray:
        ax, ay, az = sh_accel_fixed_numba(
            float(r[0]), float(r[1]), float(r[2]),
            grav.max_degree, grav.r_ref, grav.mu,
            grav.c_coeffs, grav.s_coeffs,
            grav.diag_coeffs, grav.subdiag_coeffs,
            grav.a_coeffs, grav.b_coeffs, grav.scale_m_table,
            ws.P, ws.dP, ws.cos_m, ws.sin_m,
        )
        return np.array([ax, ay, az], dtype=np.float64)

    def rhs(y: np.ndarray) -> np.ndarray:
        r, v = y[:3], y[3:]
        return np.concatenate([v, sh_accel(r)])

    t0 = time.perf_counter()
    for i in range(N):
        state = y0_batch[i].copy()
        Y_out[0, i] = state
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                k1 = rhs(state)
                k2 = rhs(state + 0.5 * dt_s * k1)
                k3 = rhs(state + 0.5 * dt_s * k2)
                k4 = rhs(state + dt_s * k3)
                state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y_out[snap_idx + 1, i] = state
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / el
            eta  = (N - i - 1) / max(rate, 1e-9)
            print(f"  [SH200-RK4] {i+1}/{N}  {el:.1f}s  ETA {eta:.0f}s", flush=True)

    runtime_s = time.perf_counter() - t0
    n_steps = n_snaps * steps_per_snap
    print(f"[SH200-RK4] done: {runtime_s:.2f}s  scenarios={N}", flush=True)
    return {"t": t_out, "Y": Y_out, "runtime_s": runtime_s, "device": "cpu",
            "dt_s": dt_s, "n_scenarios": N, "n_steps": n_steps,
            "samples_per_second": N * n_steps / max(runtime_s, 1e-9)}
