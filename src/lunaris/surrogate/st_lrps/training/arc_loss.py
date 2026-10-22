"""Adjoint-weighted arc loss: the first-order along-track error of reference orbits.

A directory written by ``tools/st_lrps_arc_proxy.py build`` holds, per orbit k,
body-fixed reference positions ``x_kj``, the true residual acceleration
``a_true_kj`` and quadrature-weighted adjoint sensitivities ``w_kj`` such that
the along-track position error a surrogate adds at the final time is, to first
order,

    s_k(θ) = Σ_j w_kj · (a_θ(x_kj) - a_true_kj)      [m].

The training term targets ``½ mean_k s_k² / s_ref²``. Evaluating ``s_k`` on all
of an orbit's points with a parameter graph would cost tens of batches, so the
term uses ``mean_k s_k · ŝ_k(θ) / s_ref²``. ``s_k`` is evaluated exactly with
no parameter graph; ``ŝ_k`` is an unbiased subsample estimate,
``(M/m) Σ_{j∈S} w_kj · a_θ(x_kj)``, with the parameter graph. Its gradient has
expectation ``∇(½ mean_k s_k² / s_ref²)``.

The orbits used here must be disjoint from the orbits used to evaluate the
model; one scalar per orbit is a weak constraint that the network can meet
orbit by orbit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

AccelFn = Callable[[torch.Tensor, bool], torch.Tensor]


class ArcLossTerm:
    """Adjoint-weighted along-track error of stored reference arcs."""

    def __init__(
        self,
        proxy_dir: str | Path,
        *,
        orbits_per_step: int = 2,
        subsample: int = 4096,
        s_ref_m: float = 1.0,
        seed: int = 0,
        chunk: int = 16384,
        expected_base_degree: int | None = None,
        expected_ref_degree: int | None = None,
    ) -> None:
        proxy = Path(proxy_dir)
        manifest = json.loads((proxy / "proxy_manifest.json").read_text(encoding="utf-8"))
        for key, expected in (("base_degree", expected_base_degree), ("ref_degree", expected_ref_degree)):
            if expected is not None and int(manifest[key]) != int(expected):
                raise ValueError(
                    f"arc-loss data {proxy} was built for {key}={manifest[key]}, but the dataset has "
                    f"{expected}; the stored true residual would not be this model's target."
                )
        files = sorted(proxy.glob("orbit_*.npz"))
        if not files:
            raise ValueError(f"no orbit_*.npz in arc-loss data {proxy}")
        self.orbits: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for f in files:
            z = np.load(f)
            self.orbits.append((
                torch.as_tensor(z["r_fix"], dtype=torch.float64),
                torch.as_tensor(z["a_true"], dtype=torch.float64),
                torch.as_tensor(z["weight"], dtype=torch.float64),
            ))
        self.orbits_per_step = max(1, min(int(orbits_per_step), len(self.orbits)))
        self.subsample = max(1, int(subsample))
        self.s_ref_m = float(s_ref_m)
        self.chunk = max(1, int(chunk))
        self._rng = np.random.default_rng(int(seed))

    def exact_error(self, accel: AccelFn, k: int, device: torch.device, dtype: torch.dtype) -> float:
        """``s_k`` [m] at the current parameters (no parameter graph)."""
        r, a_true, w = self.orbits[k]
        total = 0.0
        for i in range(0, r.shape[0], self.chunk):
            a = accel(r[i : i + self.chunk].to(device=device, dtype=dtype), False).detach().double().cpu()
            total += float((w[i : i + self.chunk] * (a - a_true[i : i + self.chunk])).sum())
        return total

    def evaluate_all(self, accel: AccelFn, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
        """Exact errors of every stored orbit: geometric mean of |s| and RMS [m]."""
        s = np.array([self.exact_error(accel, k, device, dtype) for k in range(len(self.orbits))])
        return {
            "arc_gmean_m": float(np.exp(np.mean(np.log(np.abs(s) + 1e-9)))),
            "arc_rms_m": float(np.sqrt(np.mean(s**2))),
        }

    def term(
        self, accel: AccelFn, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss tensor with the parameter graph, and the exact errors of this step."""
        picks = self._rng.choice(len(self.orbits), size=self.orbits_per_step, replace=False)
        parts = []
        s_exact = []
        for k in picks:
            r, _, w = self.orbits[int(k)]
            s_k = self.exact_error(accel, int(k), device, dtype)
            m = min(self.subsample, r.shape[0])
            idx = torch.as_tensor(self._rng.choice(r.shape[0], size=m, replace=False))
            a = accel(r[idx].to(device=device, dtype=dtype), True)
            s_hat = (float(r.shape[0]) / float(m)) * (w[idx].to(device=device, dtype=a.dtype) * a).sum()
            parts.append(s_k * s_hat)
            s_exact.append(s_k)
        loss = torch.stack(parts).mean() / (self.s_ref_m**2)
        stats = {"arc_s_sq_mean_m2": float(np.mean(np.square(s_exact))), "arc_orbits": float(len(s_exact))}
        return loss, stats


def residual_accel_fn(model: torch.nn.Module, loss_fn: torch.nn.Module) -> AccelFn:
    """Physical residual acceleration of ``model`` at body-fixed positions [m] -> [m/s²].

    ``create_graph=True`` keeps the parameter graph (training); ``False`` gives
    the values only.
    """

    def accel(x_phys: torch.Tensor, create_graph: bool) -> torch.Tensor:
        with torch.enable_grad():
            x_scaled = loss_fn.scale_x(x_phys).detach().requires_grad_(True)
            u_scaled = model(x_scaled)
            a_phys, _ = loss_fn.accel_from_u_scaled(u_scaled, x_scaled, create_graph=bool(create_graph))
        return a_phys

    return accel


__all__ = ["ArcLossTerm", "residual_accel_fn"]
