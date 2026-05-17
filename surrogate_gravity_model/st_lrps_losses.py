# -*- coding: utf-8 -*-
"""Sobolev losses and curricula for scalar potential-field training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .dataset_parameters import MU_MOON_SI, R_MOON_SI, is_lunar_body_signature
except ImportError:  # pragma: no cover - script execution fallback
    from dataset_parameters import MU_MOON_SI, R_MOON_SI, is_lunar_body_signature


try:
    from .st_lrps_scaling import ScalerPack, compute_base_accel, compute_base_potential
except ImportError:  # pragma: no cover
    from st_lrps_scaling import ScalerPack, compute_base_accel, compute_base_potential

logger = logging.getLogger(__name__)

def _direction_loss_factor(epoch: int, cfg: "TrainConfig") -> float:
    """Effective direction-loss weight lam_dir for the current epoch.

    Ramped linearly from 0 to direction_loss_weight over direction_loss_ramp_epochs,
    starting at direction_loss_start_epoch.  Returns 0 before start_epoch.
    """
    if epoch < cfg.direction_loss_start_epoch:
        return 0.0
    ramp = max(1, int(cfg.direction_loss_ramp_epochs))
    t = min(1.0, (epoch - cfg.direction_loss_start_epoch) / ramp)
    return float(cfg.direction_loss_weight) * t

def _altitude_km_from_positions(x_phys: torch.Tensor, r_ref_m: float) -> torch.Tensor:
    """Return per-sample altitude above the lunar reference radius in kilometres."""

    return (torch.linalg.norm(x_phys, dim=-1) - float(r_ref_m)) / 1000.0

def _altitude_balanced_mean_square(
    sample_sq: torch.Tensor,
    x_phys: torch.Tensor,
    *,
    r_ref_m: float,
    altitude_min_km: float,
    altitude_max_km: float,
    altitude_bin_width_km: float,
) -> torch.Tensor:
    """
    Average a sample-wise squared quantity across altitude bins instead of raw count.

    This keeps easy high-altitude points from dominating the optimisation signal
    when the training shell spans a wide range of orbital heights.
    """

    if sample_sq.ndim != 1:
        raise ValueError("sample_sq must be a 1-D tensor of per-sample squared errors.")

    bin_width = max(float(altitude_bin_width_km), 1e-6)
    alt_lo = float(altitude_min_km)
    alt_hi = float(altitude_max_km)
    alt_km = _altitude_km_from_positions(x_phys, r_ref_m=float(r_ref_m))

    bin_terms: List[torch.Tensor] = []
    cursor = alt_lo
    while cursor < alt_hi - 1e-9:
        upper = min(cursor + bin_width, alt_hi)
        if upper >= alt_hi - 1e-9:
            mask = (alt_km >= cursor) & (alt_km <= alt_hi)
        else:
            mask = (alt_km >= cursor) & (alt_km < upper)
        if torch.any(mask):
            bin_terms.append(sample_sq[mask].mean())
        cursor = upper

    outside_mask = (alt_km < alt_lo) | (alt_km > alt_hi)
    if torch.any(outside_mask):
        bin_terms.append(sample_sq[outside_mask].mean())

    if not bin_terms:
        return sample_sq.mean()
    return torch.stack(bin_terms).mean()

def _radial_cross_components(
    err_vec: torch.Tensor,
    x_phys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose acceleration error into radial and cross-radial magnitudes.

    This mirrors the evaluation-side direction diagnostics without claiming an
    exact RTN frame, because velocity is not part of the surrogate state.
    """

    r_norm = torch.linalg.norm(x_phys, dim=-1, keepdim=True).clamp_min(1e-12)
    r_hat = x_phys / r_norm
    radial = torch.sum(err_vec * r_hat, dim=-1)
    cross = torch.linalg.norm(err_vec - radial.unsqueeze(-1) * r_hat, dim=-1)
    return radial, cross

@dataclass
class GradNormWeights:
    """
    Loss-balance weights for the Sobolev objective (w_u · MSE_U + w_a · MSE_a).

    Three modes controlled by ``mode``:

    ``"ntk_init"`` (default)
        Compute ‖∂L_U/∂W‖ / ‖∂L_a/∂W‖ exactly ONCE on the first training step
        using first-order autograd, then freeze w_a for the rest of training.
        Avoids the instability of repeated Hessian-involving updates that arise
        because a_pred = ∂U/∂x makes ∂L_a/∂W a second-order quantity.

    ``"fixed"``
        Use w_u and w_a exactly as set; no gradient computation.

    ``"dynamic"``
        Legacy EMA-based GradNorm (Chen et al. 2018); amortised every
        ``update_interval`` steps. Kept for ablation studies only.
    """

    w_u: float = 1.0
    w_a: float = 1.0
    mode: str = "ntk_init"          # "ntk_init" | "fixed" | "dynamic"
    dynamic: bool = False           # legacy field; True forces mode="dynamic"
    ema_beta: float = 0.9
    update_interval: int = 10
    w_a_min: float = 0.35
    w_a_max: float = 4.00
    _ema_ratio: float = 1.0
    _step_counter: int = 0
    _ntk_done: bool = False         # True after ntk_init computation is complete

    def _effective_mode(self) -> str:
        """Resolve the active mode, honouring the legacy ``dynamic`` bool."""
        if self.dynamic and self.mode == "ntk_init":
            return "dynamic"   # legacy override: dynamic=True → full EMA mode
        return self.mode

    def _compute_grad_norm_ratio(
        self,
        loss_u: torch.Tensor,
        loss_a: torch.Tensor,
        shared_params: List[torch.nn.Parameter],
    ) -> float:
        """Return ‖∂L_U/∂W‖ / ‖∂L_a/∂W‖, clamped to [w_a_min, w_a_max]."""
        grad_u = torch.autograd.grad(
            loss_u, shared_params, retain_graph=True, create_graph=False, allow_unused=True
        )
        grad_a = torch.autograd.grad(
            loss_a, shared_params, retain_graph=True, create_graph=False, allow_unused=True
        )
        eps = 1e-12
        norm_u = sum(g.detach().norm().item() ** 2 for g in grad_u if g is not None) ** 0.5
        norm_a = sum(g.detach().norm().item() ** 2 for g in grad_a if g is not None) ** 0.5
        raw = norm_u / max(norm_a, eps)
        return float(min(max(raw, float(self.w_a_min)), float(self.w_a_max)))

    def compute_gradnorm_weights(
        self,
        loss_u: torch.Tensor,
        loss_a: torch.Tensor,
        shared_params: List[torch.nn.Parameter],
    ) -> Tuple[float, float]:
        mode = self._effective_mode()

        if mode == "fixed":
            return self.w_u, self.w_a

        if mode == "ntk_init":
            if self._ntk_done:
                return self.w_u, self.w_a
            # Compute once from NTK gradient norms at initialization
            self.w_a = self._compute_grad_norm_ratio(loss_u, loss_a, shared_params)
            self._ntk_done = True
            _gnw_logger = logging.getLogger(__name__)
            _gnw_logger.info(f"NTK-init: w_a={self.w_a:.4f} (frozen for rest of training)")
            return self.w_u, self.w_a

        # mode == "dynamic": legacy EMA GradNorm
        self._step_counter += 1
        if self._step_counter % self.update_interval != 1 and self._step_counter > 1:
            return self.w_u, self.w_a
        raw = self._compute_grad_norm_ratio(loss_u, loss_a, shared_params)
        self._ema_ratio = self.ema_beta * self._ema_ratio + (1.0 - self.ema_beta) * raw
        self._ema_ratio = min(max(self._ema_ratio, float(self.w_a_min)), float(self.w_a_max))
        self.w_u = 1.0
        self.w_a = float(self._ema_ratio)
        return self.w_u, self.w_a

    def needs_grad_compute(self) -> bool:
        """True if any gradient computation is needed on this call."""
        mode = self._effective_mode()
        if mode == "fixed":
            return False
        if mode == "ntk_init":
            return not self._ntk_done
        # dynamic: depends on step counter → caller should always try
        return True

    def get_static_weights(self) -> Tuple[float, float]:
        """Return current weights without computing gradients (for val)."""
        return self.w_u, self.w_a


# --- Loss curriculum ---------------------------------------------------------
# Residual gravity learning has two coupled objectives:
#   1) match residual potential ΔU
#   2) match the acceleration field derived from ∇ΔU
#
# Driving both at full strength from the very first epoch often destabilises
# training. The model is still learning a coarse potential manifold, while the
# acceleration term already differentiates that immature field and amplifies its
# high-frequency errors. The result is exactly the pattern we observed in
# practice: ΔU plateaus early and the acceleration loss starts climbing.
#
# To avoid that failure mode, we stage the optimisation:
#   - a short potential-only warm-up teaches the low-frequency residual shape
#   - the acceleration term is then ramped in smoothly over several epochs
#   - once the ramp completes, the run behaves like the full Sobolev objective
#
# The curriculum affects only the *optimisation objective*. Validation and
# checkpoint selection still monitor the full reference loss so we do not
# accidentally keep an early "potential-only" checkpoint as the best model.

@dataclass(frozen=True)
class LossCurriculum:
    """
    Staged weighting policy for the acceleration branch of the Sobolev loss.

    Parameters
    ----------
    potential_only_epochs:
        Number of initial epochs in the warm-up phase.  During this phase the
        acceleration weight is held at ``accel_min_factor`` (not zero) so the
        derivative field cannot drift freely.
    accel_ramp_epochs:
        Number of epochs used to linearly increase the acceleration weight from
        ``accel_min_factor`` to 1.0 after the warm-up phase.
    accel_min_factor:
        Floor value for the acceleration factor.  The loss always includes at
        least ``accel_min_factor * w_a * MSE_a``, preventing the derivative
        field from becoming completely unconstrained.  Set to 0.0 to restore
        original pure potential-only behaviour (not recommended for SIREN).
    """

    potential_only_epochs: int = 0
    accel_ramp_epochs: int = 0
    accel_min_factor: float = 0.05

    def accel_factor(self, epoch: int) -> float:
        """
        Return the multiplicative factor applied to the acceleration loss.

        The returned factor is always in ``[accel_min_factor, 1]``:

        - ``accel_min_factor`` during the warm-up phase (never exactly 0 unless
          accel_min_factor=0.0, keeping a floor to prevent derivative drift)
        - linearly ramping from ``accel_min_factor`` to 1.0 during ramp phase
        - ``1.0`` once full Sobolev training is enabled
        """

        epoch_i = max(0, int(epoch))
        warmup = max(0, int(self.potential_only_epochs))
        ramp = max(0, int(self.accel_ramp_epochs))
        floor = float(max(0.0, self.accel_min_factor))

        if epoch_i < warmup:
            # Return the floor instead of 0.0: keeps derivative field constrained.
            return floor

        if ramp <= 0:
            return 1.0

        ramp_step = epoch_i - warmup + 1
        linear = float(min(1.0, max(0.0, ramp_step / float(ramp))))
        # Ramp from floor to 1.0 (not from 0.0), so the derivative is never starved.
        return floor + (1.0 - floor) * linear


# --- Sobolev Loss ---

class SobolevLoss(nn.Module):
    """Sobolev loss: w_u·MSE(ΔU_scaled) + w_a·MSE(Δa_scaled). Isometric + GradNorm-ready."""
    def __init__(
        self,
        scaler: "ScalerPack",
        a_sign: float = 1.0,
        mu_si: float = MU_MOON_SI,
        degree_min: int = -1,
        r_ref_m: float = R_MOON_SI,
    ):
        super().__init__()
        self.a_sign = float(a_sign)
        self.mu_si = float(mu_si)
        self.degree_min = int(degree_min)
        self.r_ref_m = float(r_ref_m)

        self.register_buffer("x_mean", torch.tensor(scaler.x.mean))
        self.register_buffer("x_scale", torch.tensor([scaler.x.scale]))

        self.register_buffer("u_mean", torch.tensor(scaler.u.mean))
        self.register_buffer("u_scale", torch.tensor([scaler.u.scale]))

        self.register_buffer("a_mean", torch.tensor(scaler.a.mean))
        self.register_buffer("a_scale", torch.tensor([scaler.a.scale]))

    def scale_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_scale

    def unscale_x(self, x_s: torch.Tensor) -> torch.Tensor:
        return x_s * self.x_scale + self.x_mean

    def scale_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean) / self.u_scale

    def scale_a(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / self.a_scale

    def _maybe_balance(
        self,
        sample_sq: torch.Tensor,
        x_phys: torch.Tensor,
        *,
        enabled: bool,
        altitude_bin_width_km: float,
        altitude_min_km: float,
        altitude_max_km: float,
    ) -> torch.Tensor:
        if not enabled:
            return sample_sq.mean()
        return _altitude_balanced_mean_square(
            sample_sq,
            x_phys,
            r_ref_m=self.r_ref_m,
            altitude_min_km=altitude_min_km,
            altitude_max_km=altitude_max_km,
            altitude_bin_width_km=altitude_bin_width_km,
        )

    def _laplacian_penalty(
        self,
        grad_u_scaled: torch.Tensor,
        x_scaled: torch.Tensor,
        *,
        subset_size: int,
        n_hutchinson_samples: int = 4,
    ) -> torch.Tensor:
        """
        Stochastic Laplacian penalty via the Hutchinson trace estimator.

        Enforces the Laplace equation ∇²U = 0 (satisfied by any gravitational
        potential in free space) as a soft physics constraint.

        Algorithm
        ---------
        Tr(∇²U) ≈ (1/K) Σₖ vₖᵀ ∇²U vₖ,   vₖ ~ Rademacher{±1}³

        Using the identity  vᵀ ∇²U v = ∂(∇U · v)/∂x · v,  each sample
        requires exactly ONE additional autograd call.  Crucially, that call
        uses ``create_graph=False`` (first-order only), so the estimator is:

        * **AMP-compatible**: no second-order graph under mixed precision.
        * **O(K·B)** compute instead of O(3·B) for the exact Laplacian diagonal,
          with K=4 giving ≈50% relative error — sufficient for a regulariser.
        * **Memory-efficient**: no Hessian rows stored.

        The ``grad_u_scaled`` argument must already carry ``create_graph=True``
        (it does: it is produced by ``accel_from_u_scaled`` with
        ``create_graph=is_train``).  This function does not add a new
        computation graph layer.
        """
        k = min(int(subset_size), int(x_scaled.shape[0]))
        if k <= 0:
            return torch.zeros((), device=x_scaled.device, dtype=x_scaled.dtype)

        K = max(1, int(n_hutchinson_samples))
        idx = torch.randperm(int(x_scaled.shape[0]), device=x_scaled.device)[:k]
        g_sub = grad_u_scaled[idx]   # (k, 3), still part of the autograd graph

        trace_acc = torch.zeros((k,), device=x_scaled.device, dtype=x_scaled.dtype)
        for _ in range(K):
            v = 2.0 * (torch.rand_like(g_sub) > 0.5).float() - 1.0  # Rademacher (k, 3)
            Jv = (g_sub * v).sum()                                    # scalar
            # ∂Jv/∂x_scaled — first-order only, no new graph needed.
            # retain_graph=True: the main computational graph (shared with the
            # acceleration loss) must survive for loss.backward() after this call.
            Hv_full = torch.autograd.grad(
                Jv, x_scaled,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]                                     # (B, 3)
            trace_acc = trace_acc + (Hv_full[idx] * v).sum(dim=-1)   # (k,)

        trace_est = trace_acc / float(K)
        # Chain-rule scaling: ∇²U_phys = ∇²U_scaled · (u_scale / x_scale²)
        lap_phys = trace_est * (self.u_scale.squeeze(0) / (self.x_scale.squeeze(0) ** 2))
        return torch.mean(lap_phys ** 2)

    def accel_from_u_scaled(
        self, u_scaled: torch.Tensor, x_scaled: torch.Tensor, *, create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Δa_phys = a_sign · ∂(ΔU_scaled)/∂(x_scaled) · (u_scale/x_scale). Scalar factor only."""
        grad_u_scaled = torch.autograd.grad(
            outputs=u_scaled,
            inputs=x_scaled,
            grad_outputs=torch.ones_like(u_scaled),
            create_graph=bool(create_graph),
            retain_graph=bool(create_graph),
            only_inputs=True,
        )[0]  # Shape: (B, 3)

        # FIX-1: Uniform chain rule factor (scalar / scalar) → isotropy preserved!
        grad_u_phys = grad_u_scaled * (self.u_scale / self.x_scale)
        return self.a_sign * grad_u_phys, grad_u_scaled

    def forward(
        self,
        model: nn.Module,
        x_phys: torch.Tensor,
        u_phys: torch.Tensor,
        a_phys: torch.Tensor,
        weights: "GradNormWeights",
        *,
        is_train: bool,
        accel_factor: float = 1.0,
        allow_dynamic_weight_update: bool = True,
        direction_lambda: float = 0.0,
        direction_floor_abs: float = 3e-6,
        use_altitude_balanced_loss: bool = False,
        altitude_bin_width_km: float = 50.0,
        altitude_min_km: float = 200.0,
        altitude_max_km: float = 600.0,
        use_radial_cross_loss: bool = False,
        radial_lambda: float = 0.0,
        cross_lambda: float = 0.0,
        apply_laplacian: bool = False,
        laplacian_lambda: float = 0.0,
        laplacian_subset_size: int = 512,
        laplacian_n_hutchinson: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the staged Sobolev objective and its reference metrics.

        ``accel_factor`` affects only the optimisation loss returned as the
        first tuple item. The stats dictionary additionally contains
        ``loss_ref``, which always represents the full un-ramped objective and
        is therefore safe to use for validation reporting and checkpoint
        selection.
        """
        # Analytical base (zero when degree_min >= 0; dataset already residual)
        u_base = compute_base_potential(x_phys, self.mu_si, self.a_sign, self.degree_min)   # (B,1)
        a_base = compute_base_accel(x_phys, self.mu_si, self.degree_min)                   # (B,3)

        # Residual targets (what the network must learn)
        delta_u_true = u_phys - u_base   # (B,1)
        delta_a_true = a_phys - a_base   # (B,3)

        x_scaled = self.scale_x(x_phys).requires_grad_(True)
        delta_u_scaled_pred = model(x_scaled)

        delta_u_scaled_true = self.scale_u(delta_u_true)
        delta_u_sample_sq = (delta_u_scaled_pred - delta_u_scaled_true).squeeze(-1) ** 2
        mse_u = self._maybe_balance(
            delta_u_sample_sq,
            x_phys,
            enabled=bool(use_altitude_balanced_loss),
            altitude_bin_width_km=altitude_bin_width_km,
            altitude_min_km=altitude_min_km,
            altitude_max_km=altitude_max_km,
        )

        # Δa via autograd: ∂(ΔU_scaled)/∂(x_scaled) · (u_scale/x_scale)
        delta_a_pred_phys, grad_u_scaled = self.accel_from_u_scaled(
            delta_u_scaled_pred, x_scaled, create_graph=is_train
        )
        delta_a_scaled_err = self.scale_a(delta_a_pred_phys) - self.scale_a(delta_a_true)
        delta_a_sample_sq = torch.mean(delta_a_scaled_err ** 2, dim=-1)
        mse_a = self._maybe_balance(
            delta_a_sample_sq,
            x_phys,
            enabled=bool(use_altitude_balanced_loss),
            altitude_bin_width_km=altitude_bin_width_km,
            altitude_min_km=altitude_min_km,
            altitude_max_km=altitude_max_km,
        )

        if is_train and allow_dynamic_weight_update and weights.needs_grad_compute():
            shared_params = _get_last_hidden_params(model)
            w_u, w_a = weights.compute_gradnorm_weights(mse_u, mse_a, shared_params)
        else:
            w_u, w_a = weights.get_static_weights()

        accel_factor = float(min(1.0, max(0.0, accel_factor)))
        effective_w_a = float(w_a) * accel_factor

        # Direction loss: L_dir = mean(1 - cos_sim(a_pred, a_true)) for ||a_true|| > floor
        lambda_dir = float(direction_lambda)
        loss_dir_val = 0.0
        cossim_mean_val = 1.0
        mask_frac_val = 0.0
        dir_loss_active = False
        angular_mean_deg_val = 0.0
        angular_p90_deg_val = 0.0
        loss_dir_t: Optional[torch.Tensor] = None
        if lambda_dir > 0.0:
            norms_true = delta_a_true.norm(dim=-1, keepdim=True)  # (B,1)
            mask = (norms_true > float(direction_floor_abs)).squeeze(-1)  # (B,)
            mask_frac_val = float(mask.float().mean().item())
            if mask.any():
                a_pred_m = delta_a_pred_phys[mask]
                a_true_m = delta_a_true[mask]
                cos_sim = torch.nn.functional.cosine_similarity(a_pred_m, a_true_m, dim=-1)  # (M,)
                loss_dir_t = (1.0 - cos_sim).mean()
                cossim_mean_val = float(cos_sim.detach().mean().item())
                loss_dir_val = float(loss_dir_t.detach().item())
                _ang_rad = torch.acos(cos_sim.detach().clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                angular_mean_deg_val = float(_ang_rad.mean().item()) * 57.29577951308232
                _ang_p90 = float(torch.quantile(_ang_rad, 0.90).item()) * 57.29577951308232
                angular_p90_deg_val = _ang_p90
                dir_loss_active = True

        radial_lambda = float(max(0.0, radial_lambda))
        cross_lambda = float(max(0.0, cross_lambda))
        loss_radial_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_cross_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_radial_val = 0.0
        loss_cross_val = 0.0
        if use_radial_cross_loss and (radial_lambda > 0.0 or cross_lambda > 0.0):
            radial_err, cross_err = _radial_cross_components(delta_a_pred_phys - delta_a_true, x_phys)
            loss_radial_t = self._maybe_balance(
                radial_err ** 2,
                x_phys,
                enabled=bool(use_altitude_balanced_loss),
                altitude_bin_width_km=altitude_bin_width_km,
                altitude_min_km=altitude_min_km,
                altitude_max_km=altitude_max_km,
            )
            loss_cross_t = self._maybe_balance(
                cross_err ** 2,
                x_phys,
                enabled=bool(use_altitude_balanced_loss),
                altitude_bin_width_km=altitude_bin_width_km,
                altitude_min_km=altitude_min_km,
                altitude_max_km=altitude_max_km,
            )
            loss_radial_val = float(loss_radial_t.detach().item())
            loss_cross_val = float(loss_cross_t.detach().item())

        loss_lap_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_lap_val = 0.0
        if apply_laplacian and float(laplacian_lambda) > 0.0:
            loss_lap_t = self._laplacian_penalty(
                grad_u_scaled,
                x_scaled,
                subset_size=laplacian_subset_size,
                n_hutchinson_samples=int(laplacian_n_hutchinson),
            )
            loss_lap_val = float(loss_lap_t.detach().item())

        loss_ref = (w_u * mse_u) + (w_a * mse_a)
        loss_opt = (w_u * mse_u) + (effective_w_a * mse_a)
        if dir_loss_active and loss_dir_t is not None:
            loss_ref = loss_ref + (lambda_dir * loss_dir_t)
            loss_opt = loss_opt + (lambda_dir * loss_dir_t)
        if use_radial_cross_loss and (radial_lambda > 0.0 or cross_lambda > 0.0):
            loss_ref = loss_ref + (radial_lambda * loss_radial_t) + (cross_lambda * loss_cross_t)
            loss_opt = loss_opt + (radial_lambda * loss_radial_t) + (cross_lambda * loss_cross_t)
        if apply_laplacian and float(laplacian_lambda) > 0.0:
            loss_ref = loss_ref + (float(laplacian_lambda) * loss_lap_t)
            loss_opt = loss_opt + (float(laplacian_lambda) * loss_lap_t)

        stats = {
            "loss": loss_ref.detach().item(),
            "loss_ref": loss_ref.detach().item(),
            "loss_opt": loss_opt.detach().item(),
            "mse_u": mse_u.detach().item(),
            "mse_a": mse_a.detach().item(),
            "w_u": w_u,
            "w_a_raw": float(w_a),
            "w_a_base": float(w_a),     # alias for w_a_raw (pre-accel_factor base weight)
            "w_a": effective_w_a,
            "w_a_eff": effective_w_a,   # alias for w_a (post-accel_factor effective weight)
            "accel_factor": accel_factor,
            "loss_dir": loss_dir_val,
            "cossim_mean": cossim_mean_val,
            "angular_mean_deg": angular_mean_deg_val,
            "angular_p90_deg": angular_p90_deg_val,
            "mask_frac": mask_frac_val,
            "loss_radial": loss_radial_val,
            "loss_cross": loss_cross_val,
            "loss_laplacian": loss_lap_val,
            "altitude_balanced": float(bool(use_altitude_balanced_loss)),
        }
        return loss_opt, stats

def _get_last_hidden_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Return the parameters of the last hidden Linear layer for GradNorm computation.

    Both weight and bias are included so the gradient-norm ratio reflects the
    full affine transformation at the layer boundary.  Excluding bias would
    slightly underestimate norm_u / norm_a, but the effect is negligible for
    typical hidden sizes (512+).  We include it for completeness.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return list(model.parameters())
    last_hidden = linears[-2]
    params = [last_hidden.weight]
    if last_hidden.bias is not None:
        params.append(last_hidden.bias)
    return params


__all__ = [
    'GradNormWeights', 'LossCurriculum', 'SobolevLoss',
    '_direction_loss_factor', '_altitude_km_from_positions',
    '_altitude_balanced_mean_square', '_radial_cross_components',
]
