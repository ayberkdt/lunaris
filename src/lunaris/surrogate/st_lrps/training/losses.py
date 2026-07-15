"""Sobolev losses and curricula for scalar potential-field training."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract
from lunaris.surrogate.st_lrps.shared.scaling import (
    ScalerPack,
    compute_base_potential_accel_from_contract,
)

if TYPE_CHECKING:
    from lunaris.surrogate.st_lrps.training.config import TrainConfig

logger = logging.getLogger(__name__)

def _direction_loss_factor(epoch: int, cfg: TrainConfig) -> float:
    """Effective direction-loss weight lam_dir for the current epoch.

    Ramped linearly from 0 to direction_loss_weight over direction_loss_ramp_epochs,
    starting at direction_loss_start_epoch.  The start epoch is the first active
    ramp step; the factor is 0 only before that epoch.
    """
    if epoch < cfg.direction_loss_start_epoch:
        return 0.0
    ramp = max(1, int(cfg.direction_loss_ramp_epochs))
    ramp_step = int(epoch) - int(cfg.direction_loss_start_epoch) + 1
    t = min(1.0, max(0.0, ramp_step / ramp))
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

    if alt_hi <= alt_lo:
        return sample_sq.mean()

    # A floor-based local bin index preserves the old half-open bins and the
    # inclusive final upper edge.  scatter_add performs one tensorized pass;
    # empty bins are masked exactly as the previous Python loop did.
    n_bins = max(1, int(math.ceil((alt_hi - alt_lo) / bin_width)))
    inside = (alt_km >= alt_lo) & (alt_km <= alt_hi)
    local_idx = torch.floor((alt_km - alt_lo) / bin_width).to(torch.long)
    local_idx = local_idx.clamp(min=0, max=n_bins - 1)
    bin_sums = torch.zeros(n_bins, device=sample_sq.device, dtype=sample_sq.dtype)
    bin_counts = torch.zeros(n_bins, device=sample_sq.device, dtype=sample_sq.dtype)
    bin_sums.scatter_add_(0, local_idx[inside], sample_sq[inside])
    bin_counts.scatter_add_(0, local_idx[inside], torch.ones_like(sample_sq[inside]))

    valid_bins = bin_counts > 0
    bin_means = bin_sums / bin_counts.clamp_min(1.0)

    outside = ~inside
    outside_count = outside.to(sample_sq.dtype).sum()
    outside_mean = sample_sq[outside].sum() / outside_count.clamp_min(1.0)
    outside_term = torch.where(outside_count > 0, outside_mean, torch.zeros_like(outside_mean))
    n_terms = valid_bins.to(sample_sq.dtype).sum() + (outside_count > 0).to(sample_sq.dtype)
    total = bin_means[valid_bins].sum() + outside_term
    return torch.where(n_terms > 0, total / n_terms, sample_sq.mean())

def _radial_cross_components(
    err_vec: torch.Tensor,
    x_phys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        EMA-based GradNorm (Chen et al. 2018); amortised every
        ``update_interval`` steps. For ablation studies only.
    """

    w_u: float = 1.0
    w_a: float = 1.0
    mode: str = "ntk_init"          # "ntk_init" | "fixed" | "dynamic"
    ema_beta: float = 0.9
    update_interval: int = 10
    w_a_min: float = 0.35
    w_a_max: float = 4.00
    _ema_ratio: float = 1.0
    _step_counter: int = 0
    _ntk_done: bool = False         # True after ntk_init computation is complete
    # Diagnostics from the most recent ratio computation (for logging / tests).
    last_gradnorm_status: str = "uninitialized"  # "ok" | "empty_grad_a" | "empty_grad_u" | "nonfinite" | "zero_norm_a"
    last_norm_u: float = float("nan")
    last_norm_a: float = float("nan")
    last_raw_ratio: float = float("nan")
    last_n_grad_u: int = 0
    last_n_grad_a: int = 0

    def _effective_mode(self) -> str:
        """Return the active loss-weighting mode."""
        return self.mode

    def state_dict(self) -> dict[str, Any]:
        """Serialize the mutable loss-weighting state for checkpoint resume.

        Captures the live weights and the NTK/EMA bookkeeping so a resumed run
        does not recompute (or re-freeze) the gradient-norm ratio from scratch.
        Static configuration (mode, clamps, EMA hyperparameters) is intentionally
        NOT restored here — it comes from the resumed TrainConfig.
        """
        return {
            "w_u": float(self.w_u),
            "w_a": float(self.w_a),
            "_ema_ratio": float(self._ema_ratio),
            "_step_counter": int(self._step_counter),
            "_ntk_done": bool(self._ntk_done),
            "last_gradnorm_status": str(self.last_gradnorm_status),
            "last_norm_u": float(self.last_norm_u),
            "last_norm_a": float(self.last_norm_a),
            "last_raw_ratio": float(self.last_raw_ratio),
            "last_n_grad_u": int(self.last_n_grad_u),
            "last_n_grad_a": int(self.last_n_grad_a),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Restore mutable state captured by :meth:`state_dict`.

        Tolerant of missing keys (older checkpoints) and of ``None``; only the
        runtime/bookkeeping fields are overwritten, never the static config.
        """
        if not state:
            return
        for key in ("w_u", "w_a", "_ema_ratio", "last_norm_u", "last_norm_a", "last_raw_ratio"):
            if state.get(key) is not None:
                setattr(self, key, float(state[key]))
        for key in ("_step_counter", "last_n_grad_u", "last_n_grad_a"):
            if state.get(key) is not None:
                setattr(self, key, int(state[key]))
        if "_ntk_done" in state and state["_ntk_done"] is not None:
            self._ntk_done = bool(state["_ntk_done"])
        if state.get("last_gradnorm_status") is not None:
            self.last_gradnorm_status = str(state["last_gradnorm_status"])

    def _compute_grad_norm_ratio(
        self,
        loss_u: torch.Tensor,
        loss_a: torch.Tensor,
        shared_params: list[torch.Tensor],
    ) -> float:
        """Return ‖∂L_U/∂W‖ / ‖∂L_a/∂W‖, clamped to [w_a_min, w_a_max].

        Robustness: if the acceleration loss has no gradient path to the shared
        params (all-None grads), or either norm is non-finite, or ``norm_a`` is
        effectively zero, the ratio is undefined. Rather than let ``norm_u/eps``
        blow up and silently clamp ``w_a`` to ``w_a_max`` (which would freeze a
        meaningless weight for the whole run under ntk_init), we log a detailed
        warning and return the CURRENT ``w_a`` unchanged. ``last_gradnorm_status``
        records the outcome so callers/tests can react.
        """
        _logger = logging.getLogger(__name__)
        eps = 1e-12

        grad_u = torch.autograd.grad(
            loss_u, shared_params, retain_graph=True, create_graph=False, allow_unused=True
        )
        grad_a = torch.autograd.grad(
            loss_a, shared_params, retain_graph=True, create_graph=False, allow_unused=True
        )
        gu = [g for g in grad_u if g is not None]
        ga = [g for g in grad_a if g is not None]
        self.last_n_grad_u = len(gu)
        self.last_n_grad_a = len(ga)

        def _fail(status: str, reason: str) -> float:
            self.last_gradnorm_status = status
            self.last_raw_ratio = float("nan")
            _logger.warning(
                "GradNorm: %s; keeping current w_a=%.4f unchanged "
                "(n_grad_u=%d, n_grad_a=%d, norm_u=%s, norm_a=%s). %s",
                status, float(self.w_a), self.last_n_grad_u, self.last_n_grad_a,
                f"{self.last_norm_u:.3e}", f"{self.last_norm_a:.3e}", reason,
            )
            return float(self.w_a)

        if not gu:
            self.last_norm_u = 0.0
            self.last_norm_a = float("nan")
            return _fail("empty_grad_u",
                         "Potential loss has no gradient path to the shared params.")
        if not ga:
            self.last_norm_u = float(sum(g.detach().norm().item() ** 2 for g in gu) ** 0.5)
            self.last_norm_a = 0.0
            return _fail("empty_grad_a",
                         "Acceleration loss has no gradient path to the shared params "
                         "(da branch disconnected?).")

        norm_u = float(sum(g.detach().norm().item() ** 2 for g in gu) ** 0.5)
        norm_a = float(sum(g.detach().norm().item() ** 2 for g in ga) ** 0.5)
        self.last_norm_u = norm_u
        self.last_norm_a = norm_a

        if not (math.isfinite(norm_u) and math.isfinite(norm_a)):
            return _fail("nonfinite", "Non-finite gradient norm.")
        if norm_a <= eps:
            return _fail("zero_norm_a", "Acceleration-loss gradient norm is ~0.")

        raw = norm_u / norm_a
        self.last_raw_ratio = float(raw)
        self.last_gradnorm_status = "ok"
        return float(min(max(raw, float(self.w_a_min)), float(self.w_a_max)))

    def compute_gradnorm_weights(
        self,
        loss_u: torch.Tensor,
        loss_a: torch.Tensor,
        shared_params: list[torch.Tensor],
    ) -> tuple[float, float]:
        mode = self._effective_mode()

        if mode == "fixed":
            return self.w_u, self.w_a

        if mode == "ntk_init":
            if self._ntk_done:
                return self.w_u, self.w_a
            # Compute once from NTK gradient norms at initialization.
            _new_w_a = self._compute_grad_norm_ratio(loss_u, loss_a, shared_params)
            _gnw_logger = logging.getLogger(__name__)
            if self.last_gradnorm_status == "ok":
                self.w_a = _new_w_a
                self._ntk_done = True   # freeze only on a valid computation
                _gnw_logger.info(
                    "NTK-init: w_a=%.4f (norm_u=%.3e, norm_a=%.3e, raw=%.4f; frozen for rest of "
                    "training)",
                    self.w_a, self.last_norm_u, self.last_norm_a, self.last_raw_ratio,
                )
            else:
                # Do NOT freeze: retry on a later step once gradients connect.
                _gnw_logger.warning(
                    "NTK-init deferred (status=%s); using w_a=%.4f this step and retrying.",
                    self.last_gradnorm_status, self.w_a,
                )
            return self.w_u, self.w_a

        # mode == "dynamic": EMA GradNorm (ablation only)
        self._step_counter += 1
        if self._step_counter % self.update_interval != 1 and self._step_counter > 1:
            return self.w_u, self.w_a
        raw = self._compute_grad_norm_ratio(loss_u, loss_a, shared_params)
        if self.last_gradnorm_status != "ok":
            logging.getLogger(__name__).warning(
                "Dynamic GradNorm update skipped (status=%s); keeping w_a=%.4f unchanged.",
                self.last_gradnorm_status,
                float(self.w_a),
            )
            return self.w_u, self.w_a
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

    def get_static_weights(self) -> tuple[float, float]:
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

    # Registered buffers (torch's Module.__getattr__ stub returns Tensor | Module;
    # these annotations pin the concrete buffer type for the type checker).
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    u_mean: torch.Tensor
    u_scale: torch.Tensor
    a_mean: torch.Tensor
    a_scale: torch.Tensor

    def __init__(
        self,
        scaler: ScalerPack,
        a_sign: float = 1.0,
        mu_si: float = MU_MOON_SI,
        degree_min: int = -1,
        r_ref_m: float = R_MOON_SI,
        target_contract: TargetContract | dict | None = None,
        target_mode: str | None = None,
        degree_max: int | None = None,
        gravity_model: Any | None = None,
    ):
        super().__init__()
        self.a_sign = float(a_sign)
        self.mu_si = float(mu_si)
        self.degree_min = int(degree_min)
        self.r_ref_m = float(r_ref_m)
        # Source SH gravity model, required only for a full-field
        # spherical-harmonics baseline (residual/point-mass paths leave it None).
        # Not a registered buffer/parameter: it is a read-only analytical field.
        self._gravity_model = gravity_model
        if isinstance(target_contract, dict):
            self.target_contract = TargetContract.from_dict(target_contract)
        elif isinstance(target_contract, TargetContract):
            self.target_contract = target_contract
        else:
            self.target_contract = TargetContract.from_resolved_config(
                {
                    "target_mode": target_mode,
                    "degree_min": degree_min,
                    "degree_max": degree_max if degree_max is not None else max(int(degree_min) + 1, 0),
                    "central_body": "moon",
                },
                resolved_mu_si=self.mu_si,
                resolved_r_ref_m=self.r_ref_m,
                a_sign=self.a_sign,
            )

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
        laplacian_mode: str = "diagnostic",
    ) -> torch.Tensor:
        """
        In-batch Laplacian penalty with an exact 3D trace and a Hutchinson
        fallback for non-3D inputs.

        Enforces the Laplace equation ∇²U = 0 (satisfied by any gravitational
        potential in free space) as a soft physics constraint, reusing the
        already-computed in-batch ``grad_u_scaled``.

        Algorithm
        ---------
        In 3D, Tr(∇²U) = Σᵢ eᵢᵀ ∇²U eᵢ. Other input dimensions use
        Tr(∇²U) ≈ (1/K) Σₖ vₖᵀ ∇²U vₖ,   vₖ ~ Rademacher{±1}ᵈ.

        Using the identity  vᵀ ∇²U v = ∂(∇U · v)/∂x · v,  each sample requires
        one additional autograd call.

        Modes
        -----
        ``"diagnostic"`` (default)
            The HVP autograd call uses ``create_graph=False``, so the returned
            scalar is DETACHED from the model parameters: it does NOT
            ``requires_grad`` and contributes **zero** gradient if added to the
            loss. It is therefore a *physics-violation diagnostic only* — cheap,
            AMP-compatible, and safe to log. Use this to monitor ∇²U without
            perturbing optimisation.
        ``"train"``
            The HVP uses ``create_graph=True`` so gradients flow back into the
            model parameters and the penalty can actually be ``.backward()``-ed.
            Requires ``grad_u_scaled`` to carry a graph (it does when produced by
            ``accel_from_u_scaled(..., create_graph=True)`` during training).

        Note: for a dedicated trainable Laplacian regulariser the engine prefers
        :func:`collocation_laplacian_loss` (independent collocation points). This
        in-batch variant stays diagnostic by default.
        """
        mode = str(laplacian_mode).strip().lower()
        if mode not in ("diagnostic", "train"):
            raise ValueError(f"laplacian_mode must be 'diagnostic' or 'train'; got {laplacian_mode!r}")
        create_graph = (mode == "train")

        k = min(int(subset_size), int(x_scaled.shape[0]))
        if k <= 0:
            return torch.zeros((), device=x_scaled.device, dtype=x_scaled.dtype)

        K = max(1, int(n_hutchinson_samples))
        idx = torch.randperm(int(x_scaled.shape[0]), device=x_scaled.device)[:k]
        g_sub = grad_u_scaled[idx]   # (k, 3), still part of the autograd graph

        trace_acc = torch.zeros((k,), device=x_scaled.device, dtype=x_scaled.dtype)
        if int(x_scaled.shape[-1]) == 3:
            # In three dimensions the coordinate-basis HVPs give the exact
            # Hessian trace in three passes.  Hutchinson's squared estimator
            # would add a positive off-diagonal bias even for a truly harmonic
            # potential, so K is intentionally ignored in this path.
            for dim in range(3):
                J_dim = g_sub[:, dim].sum()
                H_full = torch.autograd.grad(
                    J_dim,
                    x_scaled,
                    create_graph=create_graph,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                trace_acc = trace_acc + H_full[idx, dim]
        else:
            for _ in range(K):
                v = 2.0 * (torch.rand_like(g_sub) > 0.5).float() - 1.0
                Jv = (g_sub * v).sum()
                H_full = torch.autograd.grad(
                    Jv, x_scaled,
                    create_graph=create_graph,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                trace_acc = trace_acc + (H_full[idx] * v).sum(dim=-1)
            trace_acc = trace_acc / float(K)

        trace_est = trace_acc
        # Chain-rule scaling to physical units (R25): ∇²U_phys [s⁻²] =
        # ∇²U_scaled · (u_scale / x_scale²). collocation_laplacian_loss() applies
        # the identical factor so both estimators report mean((∇²U_phys)²) [s⁻⁴].
        lap_phys = trace_est * (self.u_scale.squeeze(0) / (self.x_scale.squeeze(0) ** 2))
        loss_lap = torch.mean(lap_phys ** 2)
        if mode == "train" and not loss_lap.requires_grad:
            raise RuntimeError(
                "_laplacian_penalty(laplacian_mode='train'): the computed penalty does not "
                "require grad, so it cannot backpropagate into model parameters. Ensure the "
                "model is in training mode and grad_u_scaled was produced with create_graph=True."
            )
        return loss_lap

    def accel_from_u_scaled(
        self, u_scaled: torch.Tensor, x_scaled: torch.Tensor, *, create_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        weights: GradNormWeights,
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
        laplacian_mode: str = "diagnostic",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute the staged Sobolev objective and its reference metrics.

        ``accel_factor`` affects only the optimisation loss returned as the
        first tuple item. The stats dictionary additionally contains
        ``loss_ref``, which always represents the full un-ramped objective and
        is therefore safe to use for validation reporting and checkpoint
        selection.
        """
        # Analytical base from explicit target semantics. Residual datasets
        # already store residual labels, so base subtraction is zero even when
        # the runtime total field later needs an SH baseline. Potential and
        # acceleration baselines come from one call so a full-field SH baseline
        # is evaluated once per batch, not twice.
        u_base, a_base = compute_base_potential_accel_from_contract(
            x_phys, self.target_contract, self._gravity_model)   # (B,1), (B,3)

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
            shared_params = _gradnorm_shared_params(model, weights._effective_mode())
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
        loss_dir_t: torch.Tensor | None = None
        if lambda_dir > 0.0:
            norms_true = delta_a_true.norm(dim=-1, keepdim=True)  # (B,1)
            mask = (norms_true > float(direction_floor_abs)).squeeze(-1)  # (B,)
            mask_frac_tensor = mask.float().mean()
            mask_frac_val = float(mask_frac_tensor.detach().cpu().item())
            if mask.any():
                a_pred_m = delta_a_pred_phys[mask]
                a_true_m = delta_a_true[mask]
                cos_sim = torch.nn.functional.cosine_similarity(a_pred_m, a_true_m, dim=-1)  # (M,)
                loss_dir_t = (1.0 - cos_sim).mean()
                _ang_rad = torch.acos(cos_sim.detach().clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                # Transfer all scalar diagnostics together so CUDA performs one
                # synchronization instead of one .item() per metric.
                _direction_diag = torch.stack(
                    [
                        cos_sim.detach().mean(),
                        loss_dir_t.detach(),
                        _ang_rad.mean(),
                        torch.quantile(_ang_rad, 0.90),
                    ]
                ).cpu().tolist()
                cossim_mean_val = float(_direction_diag[0])
                loss_dir_val = float(_direction_diag[1])
                angular_mean_deg_val = float(_direction_diag[2]) * 57.29577951308232
                angular_p90_deg_val = float(_direction_diag[3]) * 57.29577951308232
                dir_loss_active = True

        radial_lambda = float(max(0.0, radial_lambda))
        cross_lambda = float(max(0.0, cross_lambda))
        loss_radial_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_cross_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_radial_val = 0.0
        loss_cross_val = 0.0
        radial_cross_equal_weight_fastpath = False
        if use_radial_cross_loss and (radial_lambda > 0.0 or cross_lambda > 0.0):
            accel_err_phys = delta_a_pred_phys - delta_a_true
            if math.isclose(radial_lambda, cross_lambda, rel_tol=0.0, abs_tol=1e-15):
                # With equal weights, radial² + cross² is exactly ||error||².
                # Keep the objective on the one-pass combined path; retain
                # detached component diagnostics for the history fields.
                combined_sq = torch.sum(accel_err_phys ** 2, dim=-1)
                loss_combined_t = self._maybe_balance(
                    combined_sq,
                    x_phys,
                    enabled=bool(use_altitude_balanced_loss),
                    altitude_bin_width_km=altitude_bin_width_km,
                    altitude_min_km=altitude_min_km,
                    altitude_max_km=altitude_max_km,
                )
                with torch.no_grad():
                    radial_err, cross_err = _radial_cross_components(accel_err_phys, x_phys)
                    radial_diag_t = self._maybe_balance(
                        radial_err ** 2,
                        x_phys,
                        enabled=bool(use_altitude_balanced_loss),
                        altitude_bin_width_km=altitude_bin_width_km,
                        altitude_min_km=altitude_min_km,
                        altitude_max_km=altitude_max_km,
                    )
                    cross_diag_t = self._maybe_balance(
                        cross_err ** 2,
                        x_phys,
                        enabled=bool(use_altitude_balanced_loss),
                        altitude_bin_width_km=altitude_bin_width_km,
                        altitude_min_km=altitude_min_km,
                        altitude_max_km=altitude_max_km,
                    )
                    _radial_cross_diag = torch.stack([radial_diag_t, cross_diag_t]).cpu().tolist()
                    loss_radial_val = float(_radial_cross_diag[0])
                    loss_cross_val = float(_radial_cross_diag[1])
                loss_radial_t = loss_combined_t
                loss_cross_t = torch.zeros_like(loss_combined_t)
                radial_cross_equal_weight_fastpath = True
            else:
                radial_err, cross_err = _radial_cross_components(accel_err_phys, x_phys)
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
                _radial_cross_diag = torch.stack([loss_radial_t.detach(), loss_cross_t.detach()]).cpu().tolist()
                loss_radial_val = float(_radial_cross_diag[0])
                loss_cross_val = float(_radial_cross_diag[1])

        # In-batch Laplacian. "diagnostic" is a metric ONLY — it must never enter
        # the objective (loss_ref/loss_opt) or it would pollute the reported loss
        # and the best-checkpoint metric. "train" backpropagates into the weights.
        _lap_mode = str(laplacian_mode).strip().lower()
        loss_lap_t = torch.zeros((), device=x_phys.device, dtype=x_phys.dtype)
        loss_lap_val = 0.0
        loss_lap_diag = 0.0
        loss_lap_train = 0.0
        laplacian_applied = False
        if apply_laplacian and float(laplacian_lambda) > 0.0 and _lap_mode in ("diagnostic", "train"):
            loss_lap_t = self._laplacian_penalty(
                grad_u_scaled,
                x_scaled,
                subset_size=laplacian_subset_size,
                n_hutchinson_samples=int(laplacian_n_hutchinson),
                laplacian_mode=_lap_mode,
            )
            loss_lap_val = float(loss_lap_t.detach().cpu().item())
            laplacian_applied = True
            if _lap_mode == "train":
                loss_lap_train = loss_lap_val
            else:
                loss_lap_diag = loss_lap_val

        loss_ref = (w_u * mse_u) + (w_a * mse_a)
        loss_opt = (w_u * mse_u) + (effective_w_a * mse_a)
        if dir_loss_active and loss_dir_t is not None:
            loss_ref = loss_ref + (lambda_dir * loss_dir_t)
            loss_opt = loss_opt + (lambda_dir * loss_dir_t)
        if use_radial_cross_loss and (radial_lambda > 0.0 or cross_lambda > 0.0):
            loss_ref = loss_ref + (radial_lambda * loss_radial_t) + (cross_lambda * loss_cross_t)
            loss_opt = loss_opt + (radial_lambda * loss_radial_t) + (cross_lambda * loss_cross_t)
        # ONLY the trainable Laplacian enters the objective; diagnostic is logged only.
        if laplacian_applied and _lap_mode == "train":
            loss_opt = loss_opt + (float(laplacian_lambda) * loss_lap_t)
            loss_ref = loss_ref + (float(laplacian_lambda) * loss_lap_t)

        _loss_diag = torch.stack(
            [loss_ref.detach(), loss_opt.detach(), mse_u.detach(), mse_a.detach()]
        ).cpu().tolist()
        stats = {
            "loss": float(_loss_diag[0]),
            "loss_ref": float(_loss_diag[0]),
            "loss_opt": float(_loss_diag[1]),
            "mse_u": float(_loss_diag[2]),
            "mse_a": float(_loss_diag[3]),
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
            "radial_cross_equal_weight_fastpath": bool(radial_cross_equal_weight_fastpath),
            # Laplacian metrics are mean((∇²U_phys)²) in PHYSICAL units [s⁻⁴]
            # (∇²U_phys in s⁻²), consistent with collocation_laplacian_loss (R25).
            "loss_laplacian": loss_lap_val,
            "loss_laplacian_diag": loss_lap_diag,
            "loss_laplacian_train": loss_lap_train,
            "laplacian_mode": (_lap_mode if laplacian_applied else "off"),
            "laplacian_applied": bool(laplacian_applied),
            "altitude_balanced": float(bool(use_altitude_balanced_loss)),
            "target_mode": self.target_contract.target_mode,
            "baseline_kind": self.target_contract.baseline_kind,
            "base_degree": int(self.target_contract.base_degree),
            "target_degree": int(self.target_contract.target_degree),
        }
        return loss_opt, stats

def _get_last_hidden_params(model: nn.Module) -> list[torch.Tensor]:
    """
    Return the parameters of the last hidden Linear layer for GradNorm computation.

    Both weight and bias are included so the gradient-norm ratio reflects the
    full affine transformation at the layer boundary.  Excluding bias would
    slightly underestimate norm_u / norm_a, but the effect is negligible for
    typical hidden sizes (512+).  We include it for completeness.

    This single-layer proxy is cheap, which is why the amortised ``dynamic``
    GradNorm mode (recomputed every interval) uses it. The ``ntk_init`` mode
    computes the ratio once and should use the more representative
    :func:`_get_backbone_shared_params` instead.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return [p for p in model.parameters()]
    last_hidden = linears[-2]
    params: list[torch.Tensor] = [last_hidden.weight]
    if last_hidden.bias is not None:
        params.append(last_hidden.bias)
    return params


def _get_backbone_shared_params(model: nn.Module) -> list[torch.Tensor]:
    """Grad-enabled parameters of the whole backbone except the output head.

    The last-hidden-layer proxy can be unrepresentative for multi-scale /
    additive SIRENs, where ``linears[-2]`` is one band-specific layer rather than
    a shared trunk that every frequency band flows through. For the one-shot
    ``ntk_init`` ratio we can afford to use the full backbone, so the frozen
    ``w_a`` reflects gradients from *all* bands and shared blocks, not a single
    arbitrary layer. The output head (the final ``Linear``) is excluded because
    it is trained on a separate, higher learning rate and starts near zero, so
    its gradients are not representative of the shared backbone balance.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    head_param_ids: set[int] = set()
    if linears:
        head_param_ids = {id(p) for p in linears[-1].parameters()}
    params: list[torch.Tensor] = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    return params or [p for p in model.parameters()]


def _gradnorm_shared_params(model: nn.Module, mode: str) -> list[torch.Tensor]:
    """Pick the GradNorm reference parameters for the active weighting mode.

    ``dynamic`` recomputes the ratio every ``update_interval`` steps, so it keeps
    the cheap single-layer proxy. ``ntk_init`` (and any non-dynamic mode that
    still asks for a gradient compute) runs once, so it uses the representative
    full-backbone set.
    """
    if str(mode).strip().lower() == "dynamic":
        return _get_last_hidden_params(model)
    return _get_backbone_shared_params(model)


def _extract_xu_scale(scaler: Any) -> tuple[float, float]:
    """Return ``(x_scale, u_scale)`` as floats from either a ScalerPack or a SobolevLoss.

    The isometric scalers store a single global characteristic scale per quantity
    (see :class:`IsometricScaleParams`), so the coordinate/potential rescaling is
    isotropic and the chain-rule factor is a scalar. Two carriers expose it:

    * :class:`ScalerPack` — ``scaler.x.scale`` / ``scaler.u.scale`` (Python floats).
    * :class:`SobolevLoss` (nn.Module) — ``scaler.x_scale`` / ``scaler.u_scale``
      (1-element registered buffers).

    Both are duck-typed as "the thing that also provides ``scale_x``", so this
    helper normalises them to plain floats.
    """
    x_obj = getattr(scaler, "x", None)
    if x_obj is not None and hasattr(x_obj, "scale"):
        return float(x_obj.scale), float(scaler.u.scale)
    # Only a SobolevLoss (nn.Module with registered x_scale/u_scale buffers)
    # reaches here; ScalerPack already returned via the x_obj branch above.
    x_scale = scaler.x_scale
    u_scale = scaler.u_scale
    return float(x_scale.reshape(-1)[0]), float(u_scale.reshape(-1)[0])


def collocation_laplacian_loss(
    model: torch.nn.Module,
    scaler: ScalerPack,
    r_min_m: float,
    r_max_m: float,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_hutchinson: int = 4,
    mode: str = "diagnostic",   # "diagnostic" | "train"
) -> torch.Tensor:
    """
    Collocation-style Laplacian regularizer for the residual potential surrogate.

    Generates ``n_points`` random collocation points inside a spherical shell
    ``[r_min_m, r_max_m]`` (in physical metres) and evaluates the squared mean
    Laplacian of the network's prediction using a Hutchinson stochastic-trace
    estimator with ``n_hutchinson`` Rademacher samples. In the supported 3D
    Cartesian path the trace is exact; the knob remains a backward-compatible
    fallback for other input dimensions.

    Units (R25)
    -----------
    The raw trace estimate (exact in 3D, Hutchinson otherwise) represents
    ``∇²U`` in *scaled network coordinates*.
    It is converted to physical units via the isotropic chain-rule factor
    ``u_scale / x_scale²`` so the returned loss is ``mean((∇²U_phys)²)`` with
    ``∇²U_phys`` in ``s⁻²`` (loss in ``s⁻⁴``). This matches
    :meth:`SobolevLoss._laplacian_penalty` exactly, so the same ``laplacian``
    weight carries the same physical meaning in both estimators and the two
    logged Laplacian metrics are directly comparable.

    Modes
    -----
    diagnostic
        ``create_graph=False`` everywhere; the returned loss does NOT require
        grad and is suitable for cheap logging only.
    train
        ``create_graph=True`` on BOTH the first autograd.grad (so the HVP can
        be differentiated through) AND the HVP call (so gradients flow back to
        the model parameters). The returned loss requires_grad and can be
        ``.backward()``-ed to push the Laplace constraint into model weights.
    """
    mode = str(mode).strip().lower()
    if mode not in ("diagnostic", "train"):
        raise ValueError(f"mode must be 'diagnostic' or 'train'; got {mode!r}")

    n_points = max(1, int(n_points))
    K = max(1, int(n_hutchinson))

    # Sample random directions on the unit sphere + radii in [r_min, r_max].
    r_lo = float(min(r_min_m, r_max_m))
    r_hi = float(max(r_min_m, r_max_m))
    dirs = torch.randn(n_points, 3, device=device, dtype=dtype)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radii = torch.rand(n_points, 1, device=device, dtype=dtype) * (r_hi - r_lo) + r_lo
    x_phys = dirs * radii  # (N,3) in metres

    # Scale to network input space; require grad on scaled coords so HVP works.
    x_scaled = scaler.scale_x(x_phys).detach().clone().requires_grad_(True)
    u_pred = model(x_scaled)  # (N,1)

    grad_u = torch.autograd.grad(
        u_pred,
        x_scaled,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    lap_acc = torch.zeros(n_points, device=device, dtype=dtype)
    hvp_cg = mode == "train"
    if int(x_scaled.shape[-1]) == 3:
        # Exact 3D trace: three coordinate-basis HVPs, independent of the
        # legacy Hutchinson knob.  This removes the squared-estimator bias from
        # the diagnostic/train metric while also reducing passes from K to 3.
        for dim in range(3):
            h_col = torch.autograd.grad(
                grad_u[:, dim].sum(),
                x_scaled,
                create_graph=hvp_cg,
                retain_graph=True,
            )[0][:, dim]
            lap_acc = lap_acc + h_col
        lap_scaled = lap_acc
    else:
        for _ in range(K):
            v = torch.randint(0, 2, (n_points, x_scaled.shape[-1]), device=device, dtype=dtype) * 2 - 1
            Jv = (grad_u * v).sum(dim=-1, keepdim=True)
            hvp = torch.autograd.grad(
                Jv, x_scaled, grad_outputs=torch.ones_like(Jv),
                create_graph=hvp_cg, retain_graph=True,
            )[0]
            lap_acc = lap_acc + (hvp * v).sum(dim=-1)
        lap_scaled = lap_acc / float(K)
    # Chain-rule to physical units so this matches SobolevLoss._laplacian_penalty
    # (R25): ∇²U_phys [s⁻²] = ∇²U_scaled · (u_scale / x_scale²). Without this
    # factor the collocation penalty lived in scaled units while the in-batch
    # penalty was physical, so the same `laplacian` weight meant two different
    # physical strengths and the logged metrics were not comparable.
    x_scale, u_scale = _extract_xu_scale(scaler)
    lap_phys = lap_scaled * (u_scale / (x_scale ** 2))
    loss_val = (lap_phys ** 2).mean()
    if mode == "train" and not loss_val.requires_grad:
        raise RuntimeError(
            "collocation_laplacian_loss(mode='train'): computed loss does not require_grad. "
            "This means the Laplacian constraint cannot push gradients into model parameters. "
            "Ensure the model is in training mode and x_scaled.requires_grad_(True) is set."
        )
    return loss_val


__all__ = [
    'GradNormWeights', 'LossCurriculum', 'SobolevLoss',
    '_direction_loss_factor', '_altitude_km_from_positions',
    '_altitude_balanced_mean_square', '_radial_cross_components',
    '_get_last_hidden_params', '_get_backbone_shared_params', '_gradnorm_shared_params',
    'collocation_laplacian_loss',
]
