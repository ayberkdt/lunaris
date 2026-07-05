# lunaris.core.batched_fixed_step
"""
Shared batched fixed-step RK4 + impact loop (roadmap R07).

``torch_sh_propagator`` (classic SH on torch) and ``torch_batch_propagator``
(ST-LRPS surrogate) used to copy the same output-grid / RK4 / alive-mask /
segment-impact / snapshot machinery; impact bugfixes had to be applied twice.
This module is the single implementation. Backends contribute ONLY a
:class:`BatchedAccelerationProvider`; everything else — grid construction,
the RK4 stepper, the t=0 surface check, alive-mask freezing, terrain/sphere
segment-intersection impact detection with sub-step crossing interpolation,
snapshot writing, progress callbacks, and throughput accounting — lives here.

Numerical contract
------------------
The loop reproduces the previous per-backend loops operation-for-operation:
for a given provider, trajectories are bitwise identical to the pre-R07
implementations (chunking never changes the numbers; each chunk is an
independent slice of the sample axis).

No autograd: the whole loop runs under ``torch.no_grad()``. Providers that
need autograd internally (ST-LRPS potential gradients) re-enable it locally.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from lunaris.common.batch_defs import build_batch_output_grid
from lunaris.core.torch_frame import (
    line_sphere_intersection,
    terrain_segment_intersection,
)

if TYPE_CHECKING:
    from torch import Tensor


class BatchedAccelerationProvider(Protocol):
    """The only thing a batch backend must supply to the shared loop."""

    def acceleration(self, t_s: float, state: Tensor) -> Tensor:
        """Inertial acceleration ``[N, 3]`` for state batch ``[N, 6]`` at ``t_s``.

        The provider owns the full gravity composition, including any
        inertial<->body-fixed frame rotation of the position/acceleration.
        """
        ...  # pragma: no cover - protocol


def build_output_grid(duration_s: float, output_dt_s: float) -> tuple[np.ndarray, int, float]:
    """Shared fixed-step snapshot grid contract.

    Thin core-level wrapper for the canonical batch grid helper. Keeping this
    name here makes the R07 loop's public surface explicit while preserving
    ``lunaris.common.batch_defs`` as the single source of truth.
    """
    return build_batch_output_grid(duration_s, output_dt_s)


def rhs_batch(torch_mod: Any, provider: BatchedAccelerationProvider, t_s: float, s: Tensor) -> Tensor:
    """Evaluate ``[v; a]`` for state ``[N, 6]`` at epoch ``t_s``."""
    v_i = s[:, 3:]
    a_i = provider.acceleration(t_s, s)
    return torch_mod.cat((v_i, a_i), dim=1)


def rk4_step(
    torch_mod: Any,
    provider: BatchedAccelerationProvider,
    s: Tensor,
    t_s: float,
    h: float,
) -> Tensor:
    """One classic RK4 step (per-stage provider evaluation)."""
    k1 = rhs_batch(torch_mod, provider, t_s, s)
    k2 = rhs_batch(torch_mod, provider, t_s + 0.5 * h, s + (0.5 * h) * k1)
    k3 = rhs_batch(torch_mod, provider, t_s + 0.5 * h, s + (0.5 * h) * k2)
    k4 = rhs_batch(torch_mod, provider, t_s + h, s + h * k3)
    return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class BatchedFixedStepResult:
    """Everything the shared loop produced for one ensemble propagation."""

    t_out: np.ndarray                       # (T,) snapshot epochs [s]
    Y_out: np.ndarray                       # (T, N, 6) float64 states
    impact_flags: np.ndarray                # (N,) 1.0 impacted / 0.0 alive
    t_impact: np.ndarray                    # (N,) crossing time [s], NaN if none
    impact_positions_inertial: np.ndarray   # (N, 3) crossing position, NaN if none
    metrics: dict[str, Any] = field(default_factory=dict)


def run_batched_fixed_step(
    *,
    torch_mod: Any,
    device: Any,
    dtype: Any,
    provider: BatchedAccelerationProvider,
    frame: Any,
    Y0: np.ndarray,
    duration_s: float,
    output_dt_s: float,
    dt_s: float,
    impact_r_m: float,
    detect_impact: bool = True,
    topo: Any = None,
    impact_alt_m: float = 0.0,
    chunk_size: int | None = None,
    callback: Callable[[float], None] | None = None,
    callback_granularity: str = "chunk",
) -> BatchedFixedStepResult:
    """Propagate ``N`` samples with fixed-step RK4 + shared impact handling.

    Parameters
    ----------
    provider :
        Backend acceleration provider (see :class:`BatchedAccelerationProvider`).
    frame :
        ``TorchMoonFrame`` used by the terrain segment intersection (the
        provider performs its own frame handling for gravity).
    topo :
        Device-resident topography payload for terrain-aware impact freezing,
        or ``None`` for the constant impact sphere at ``impact_r_m``.
    chunk_size :
        Samples per device chunk; ``None`` or ``<= 0`` propagates the whole
        ensemble as a single chunk. Chunking changes only memory use, never
        the numbers.
    callback_granularity :
        ``"chunk"`` invokes ``callback`` once per finished chunk with the
        completed-sample fraction (classic-SH behavior); ``"snapshot"``
        invokes it after every snapshot with the completed-snapshot fraction
        (ST-LRPS behavior).
    """
    torch = torch_mod
    N = int(Y0.shape[0])
    dt = float(dt_s)
    if callback_granularity not in {"chunk", "snapshot"}:
        raise ValueError(
            "callback_granularity must be 'chunk' or 'snapshot', "
            f"got {callback_granularity!r}."
        )

    # Shared output grid contract: t[0]=0, t[-1]=duration_s, uniform.
    t_out, n_snaps, snap_interval = build_output_grid(duration_s, output_dt_s)
    steps_per_snap = max(1, int(round(snap_interval / dt)))
    dt_eff = snap_interval / steps_per_snap
    Y_out = np.empty((n_snaps + 1, N, 6), dtype=np.float64)
    impact_flags = np.zeros(N, dtype=np.float64)
    t_impact = np.full(N, np.nan, dtype=np.float64)
    impact_positions = np.full((N, 3), np.nan, dtype=np.float64)

    chunk = int(chunk_size) if chunk_size and int(chunk_size) > 0 else N
    chunk = max(1, chunk)
    n_chunks = max(1, int(np.ceil(N / chunk)))

    total_raw_state_steps = 0
    total_active_state_steps = 0
    total_steps_per_sample = n_snaps * steps_per_snap

    t_start = time.perf_counter()

    for ci in range(n_chunks):
        a = ci * chunk
        b = min(N, a + chunk)
        chunk_n = b - a
        active_steps = _propagate_chunk(
            torch_mod=torch,
            device=device,
            dtype=dtype,
            provider=provider,
            frame=frame,
            Y0_chunk=Y0[a:b],
            a=a,
            b=b,
            steps_per_snap=steps_per_snap,
            dt_eff=dt_eff,
            n_snaps=n_snaps,
            Y_out=Y_out,
            impact_flags=impact_flags,
            t_impact=t_impact,
            impact_positions=impact_positions,
            impact_r_m=float(impact_r_m),
            detect_impact=bool(detect_impact),
            topo=topo,
            impact_alt_m=float(impact_alt_m),
            callback=callback if callback_granularity == "snapshot" else None,
            snap_fraction_base=float(ci) / float(n_chunks),
            snap_fraction_span=1.0 / float(n_chunks),
        )
        total_raw_state_steps += chunk_n * total_steps_per_sample
        total_active_state_steps += active_steps
        if callback is not None and callback_granularity == "chunk":
            callback(float(b) / float(max(N, 1)))

    if getattr(device, "type", "") == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            # R29b-justified: synchronize only tightens the wall-clock timing
            # below; a failure degrades timing accuracy, never the physics.
            pass
    elapsed = time.perf_counter() - t_start

    n_impacts = int(np.sum(impact_flags > 0.5))
    metrics: dict[str, Any] = {
        "raw_batch_state_steps_per_second": float(total_raw_state_steps) / max(elapsed, 1e-9),
        "active_state_steps_per_second": float(total_active_state_steps) / max(elapsed, 1e-9),
        "active_sample_count": int(N - n_impacts),
        "impacted_sample_count": n_impacts,
        "impact_fraction": float(n_impacts) / max(N, 1),
        "total_raw_state_steps": int(total_raw_state_steps),
        "total_active_state_steps": int(total_active_state_steps),
        "propagation_elapsed_s": float(elapsed),
        "impact_position_method": (
            "terrain_bisection_hybrid" if topo is not None else "line_sphere_quadratic"
        ),
        "impact_time_resolution_s": float(dt_eff),
    }
    return BatchedFixedStepResult(
        t_out=t_out,
        Y_out=Y_out,
        impact_flags=impact_flags,
        t_impact=t_impact,
        impact_positions_inertial=impact_positions,
        metrics=metrics,
    )


def _propagate_chunk(
    *,
    torch_mod: Any,
    device: Any,
    dtype: Any,
    provider: BatchedAccelerationProvider,
    frame: Any,
    Y0_chunk: np.ndarray,
    a: int,
    b: int,
    steps_per_snap: int,
    dt_eff: float,
    n_snaps: int,
    Y_out: np.ndarray,
    impact_flags: np.ndarray,
    t_impact: np.ndarray,
    impact_positions: np.ndarray,
    impact_r_m: float,
    detect_impact: bool,
    topo: Any,
    impact_alt_m: float,
    callback: Callable[[float], None] | None,
    snap_fraction_base: float,
    snap_fraction_span: float,
) -> int:
    """Propagate one chunk of samples through the shared RK4/impact loop.

    Returns the number of active state-step evaluations executed in this chunk
    (for throughput accounting). Impact bookkeeping is accumulated on device
    and resolved with a single host sync at the end of the chunk (no per-step
    ``.item()`` / CPU<->GPU sync in the hot loop).
    """
    torch = torch_mod

    state = torch.as_tensor(
        np.ascontiguousarray(Y0_chunk, dtype=np.float64), device=device, dtype=dtype
    )
    n = int(state.shape[0])
    alive = torch.ones(n, dtype=torch.bool, device=device)

    Y_out[0, a:b, :] = state.detach().cpu().numpy().astype(np.float64)

    active_steps_acc = torch.zeros((), dtype=torch.int64, device=device)
    impact_step = torch.full((n,), -1, dtype=torch.int64, device=device)
    # Interpolated crossing time / inertial position (NaN until a sample hits).
    impact_time = torch.full((n,), float("nan"), dtype=dtype, device=device)
    impact_pos = torch.full((n, 3), float("nan"), dtype=dtype, device=device)

    # t=0 surface check: a sample already at/under the impact radius before any
    # step has impacted at t=0. Flag it (t_impact=0.0, position = initial state)
    # and freeze it from the start instead of propagating through the body.
    r0 = torch.linalg.norm(state[:, :3], dim=1)
    at_surface0 = r0 <= impact_r_m
    if detect_impact:
        impact_step = torch.where(at_surface0, torch.zeros_like(impact_step), impact_step)
        impact_time = torch.where(at_surface0, torch.zeros_like(impact_time), impact_time)
        impact_pos = torch.where(at_surface0.unsqueeze(1), state[:, :3], impact_pos)
        alive = alive & ~at_surface0

    t_curr = 0.0
    global_step = 0
    with torch.no_grad():
        for snap_idx in range(n_snaps):
            for _ in range(steps_per_snap):
                active_steps_acc += alive.sum()
                # Freeze impacted samples: only alive trajectories advance;
                # impacted ones hold their last state (no propagation through
                # the Moon). Fixed batch shape is preserved (no compaction).
                prev_state = state
                candidate = rk4_step(torch, provider, state, t_curr, dt_eff)
                state = torch.where(alive.unsqueeze(1), candidate, state)
                t_curr += dt_eff
                global_step += 1
                if detect_impact:
                    # True segment intersection over the step, then replace the
                    # main state with the crossing state so impacted trajectories
                    # freeze ON the surface (position+velocity), not at the
                    # sub-surface step endpoint. Terrain-aware when a topography
                    # payload is present; otherwise the constant impact sphere.
                    if topo is not None:
                        segment_hit, alpha = terrain_segment_intersection(
                            prev_state[:, :3],
                            state[:, :3],
                            t_prev_s=t_curr - dt_eff,
                            dt_s=dt_eff,
                            frame=frame,
                            topo=topo,
                            impact_alt_m=impact_alt_m,
                        )
                    else:
                        segment_hit, alpha = line_sphere_intersection(
                            prev_state[:, :3],
                            state[:, :3],
                            impact_r_m,
                        )
                    newly = alive & segment_hit
                    cross_state = prev_state + alpha.unsqueeze(1) * (state - prev_state)
                    t_cross = (float(global_step - 1) + alpha) * dt_eff
                    impact_step = torch.where(
                        newly, torch.full_like(impact_step, global_step), impact_step
                    )
                    impact_time = torch.where(newly, t_cross, impact_time)
                    impact_pos = torch.where(
                        newly.unsqueeze(1), cross_state[:, :3], impact_pos
                    )
                    state = torch.where(newly.unsqueeze(1), cross_state, state)
                    alive = alive & ~newly
            Y_out[snap_idx + 1, a:b, :] = state.detach().cpu().numpy().astype(np.float64)
            if callback is not None:
                callback(
                    snap_fraction_base
                    + snap_fraction_span * (float(snap_idx + 1) / float(max(n_snaps, 1)))
                )

    # Single host sync per chunk: resolve impact bookkeeping. The crossing
    # time/position were interpolated on device, so they resolve the exact
    # sub-step crossing instead of the coarse fixed-step endpoint.
    impact_step_host = impact_step.detach().cpu().numpy()
    impact_time_host = impact_time.detach().cpu().numpy().astype(np.float64)
    impact_pos_host = impact_pos.detach().cpu().numpy().astype(np.float64)
    for li in np.nonzero(impact_step_host >= 0)[0].tolist():
        gi = a + int(li)
        if impact_flags[gi] == 0.0:
            impact_flags[gi] = 1.0
            t_impact[gi] = float(impact_time_host[li])
            impact_positions[gi] = impact_pos_host[li]

    return int(active_steps_acc.item())


__all__ = [
    "BatchedAccelerationProvider",
    "BatchedFixedStepResult",
    "build_output_grid",
    "rhs_batch",
    "rk4_step",
    "run_batched_fixed_step",
]
