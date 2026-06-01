# -*- coding: utf-8 -*-
"""Neural model components for the scalar residual lunar potential field."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SIREN primitives
# ---------------------------------------------------------------------------

class Sine(nn.Module):
    """Sinusoidal activation: sin(w0 * x)"""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def siren_init_first_(layer: nn.Linear) -> None:
    """SIREN paper initialization for the FIRST layer."""
    n_in = layer.in_features
    bound = 1.0 / n_in
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


def siren_init_hidden_(layer: nn.Linear, w0: float) -> None:
    """SIREN paper initialization for hidden layers."""
    n_in = layer.in_features
    bound = math.sqrt(6.0 / n_in) / w0
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


# ---------------------------------------------------------------------------
# Harmonic-band utilities
# ---------------------------------------------------------------------------

def _compute_harmonic_w0_bands(
    n_bands: int,
    degree_min: int,
    degree_max: int,
) -> List[float]:
    """
    Geometrically-spaced SIREN w0 values covering the residual harmonic spectrum.

    Degree-n spherical harmonics have a characteristic spatial frequency ∝ n/R_moon.
    Spacing bands in log-degree space gives equal multiplicative SH-spectrum coverage
    per band: low-degree bands resolve large-scale gravity anomalies; high-degree bands
    resolve mascon-level fine structure.

    Parameters
    ----------
    n_bands:
        Number of frequency bands.  1 reproduces the single-w0 SirenMLP behaviour.
    degree_min:
        Maximum degree of the analytical baseline model.  The first band starts at
        ``degree_min + 1`` to cover only the residual range.
    degree_max:
        Target high-fidelity degree (highest degree to be predicted).
    """
    lo = max(1, int(degree_min) + 1)
    hi = max(lo + 1, int(degree_max))
    if n_bands <= 1:
        return [max(10.0, min(100.0, round(math.sqrt(float(hi)) * 3.0, 1)))]
    log_lo = math.log(float(lo))
    log_hi = math.log(float(hi))
    out: List[float] = []
    for i in range(n_bands):
        t = float(i) / float(n_bands - 1)
        deg_c = math.exp(log_lo + t * (log_hi - log_lo))
        out.append(max(10.0, min(100.0, round(math.sqrt(deg_c) * 3.0, 1))))
    return out


# ---------------------------------------------------------------------------
# Residual SIREN block
# ---------------------------------------------------------------------------

class SirenResBlock(nn.Module):
    """
    Pre-norm residual SIREN block.

    Architecture::

        y = x + W₂ · sin(w₀ · LN(W₁ x + b₁))

    Design rationale
    ----------------
    *Pre-norm*: LayerNorm is placed on the branch input (before the sine)
    to keep sine arguments well-conditioned in deep networks, without touching
    the skip path.  The skip carries the raw potential signal unmodified.

    *Zero-init output*: W₂ starts at zero, making each block initially identity.
    Deep residual SIRENs therefore begin as shallow networks and progressively
    activate additional capacity during training — far more stable than standard
    SIREN at depth > 6 where naive initialisation leads to gradient vanishing.

    *Parameter overhead vs. plain SIREN layer*: +dim (LN gamma/beta) ≈ negligible.
    """

    def __init__(self, dim: int, w0: float = 30.0, dropout: float = 0.0):
        super().__init__()
        self.w0 = float(w0)
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=float(dropout)) if dropout > 0 else None

        n = dim
        bound = math.sqrt(6.0 / n) / self.w0
        nn.init.uniform_(self.lin1.weight, -bound, bound)
        nn.init.uniform_(self.lin1.bias,   -bound, bound)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.sin(self.w0 * self.lin1(self.norm(x)))
        if self.drop is not None:
            h = self.drop(h)
        return x + self.lin2(h)


# ---------------------------------------------------------------------------
# SirenMLP (single-scale, optional residual blocks)
# ---------------------------------------------------------------------------

class SirenMLP(nn.Module):
    """
    SIREN MLP: sin(w0 · (Wx + b)) activations with proper first/hidden layer
    initialisation.

    Parameters
    ----------
    use_residual:
        Replace each hidden ``Linear+Sine`` with a ``SirenResBlock``.
        Recommended for ``depth >= 6``.  Adds LayerNorm per block (~negligible
        parameter overhead) and zero-initialises skip outputs so the network
        starts shallow.  Defaults to ``False`` here (the plain SIREN path); the
        trainer enables it by default via ``TrainConfig.use_residual_blocks``.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 256,
        depth: int = 4,
        w0_first: float = 30.0,
        w0_hidden: float = 30.0,
        dropout: float = 0.0,
        use_residual: bool = False,
    ):
        super().__init__()
        self.w0_first = w0_first
        self.w0_hidden = w0_hidden

        layers: List[nn.Module] = []

        # First layer (special init + w0_first)
        first_linear = nn.Linear(in_dim, hidden)
        siren_init_first_(first_linear)
        layers.append(first_linear)
        layers.append(Sine(w0=w0_first))
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))

        # Hidden layers (plain SIREN or residual SIREN blocks)
        for _ in range(depth - 1):
            if use_residual:
                layers.append(SirenResBlock(hidden, w0=w0_hidden, dropout=float(dropout)))
            else:
                lin = nn.Linear(hidden, hidden)
                siren_init_hidden_(lin, w0_hidden)
                layers.append(lin)
                layers.append(Sine(w0=w0_hidden))
                if dropout > 0:
                    layers.append(nn.Dropout(p=float(dropout)))

        # Final output layer: small-amplitude SIREN-style init keeps the
        # initial residual prediction gentle while still providing non-zero
        # gradients to the backbone from the first optimisation step.
        final = nn.Linear(hidden, 1)
        head_bound = 0.1 * (math.sqrt(6.0 / hidden) / max(float(w0_hidden), 1.0))
        nn.init.uniform_(final.weight, -head_bound, head_bound)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return self.net(x_scaled)


# ---------------------------------------------------------------------------
# Plain MLP (non-SIREN activations, for ablation against SIREN)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard MLP backbone for the non-SIREN activations (silu/tanh/softplus)."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 256,
        depth: int = 4,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_map = {"silu": nn.SiLU, "tanh": nn.Tanh, "softplus": nn.Softplus}
        activation = activation.lower()
        if activation not in act_map:
            raise ValueError(f"Activation must be one of {list(act_map.keys())}")
        Act = act_map[activation]

        layers: List[nn.Module] = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            d_in = hidden
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        self._initialize_weights(activation)

    def _initialize_weights(self, activation: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation == "tanh":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return self.net(x_scaled)


# ---------------------------------------------------------------------------
# Multi-scale SIREN (harmonic-band-aware frequency initialisation)
# ---------------------------------------------------------------------------

class MultiScaleSirenMLP(nn.Module):
    """
    Multi-scale SIREN for residual spherical-harmonic gravity fields.

    Motivation
    ----------
    A single-w0 SIREN has one characteristic spatial frequency.  Residual
    gravity fields spanning a wide harmonic range (e.g. degree 11 → 150)
    cover a ~4× frequency ratio; the standard network must simultaneously
    represent slow large-scale anomalies and fast mascon-level detail with
    the same initialisation scale.

    This class projects the input in parallel onto ``n_bands`` frequency
    bands, each initialised with a SIREN w0 tuned to its slice of the SH
    spectrum.  Band activations are concatenated and processed by a shared
    stack of residual SIREN blocks.  Total parameter count is identical to
    a same-depth, same-width ``SirenMLP`` — the hidden dimension is simply
    split across bands for the first layer.

    Architecture
    ------------
    ::

        x(3) ──┬─ sin(w₀_bands[0] · W₀ x) ──┐
               ├─ sin(w₀_bands[1] · W₁ x) ──┤ cat → (hidden,)
               └─ sin(w₀_bands[-1]· Wₖ x) ──┘
                        ↓
               sin(w₀_bands[0] · W_merge · h)   # merge projection
                        ↓
               SirenResBlock × n_shared          # depth - 2 shared blocks
                        ↓
               Linear(hidden → 1)               # output head

    Parameters
    ----------
    w0_bands:
        Per-band SIREN frequencies.  Length determines ``n_bands``.  Use
        ``_compute_harmonic_w0_bands`` to derive these from the SH degree range.
    use_residual:
        Use ``SirenResBlock`` for shared layers.  Always ``True`` for this
        class; the parameter exists for API symmetry.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 512,
        depth: int = 6,
        w0_bands: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        if w0_bands is None:
            w0_bands = [30.0]
        self.w0_bands: List[float] = [float(w) for w in w0_bands]
        n_bands = len(self.w0_bands)
        self.n_bands: int = n_bands

        # Persist the band frequencies in the state_dict so a reload cannot
        # silently reconstruct the network with a DIFFERENT spectrum. The
        # frequencies are not learnable, but if config.json and the checkpoint
        # ever disagree (e.g. degree_min/degree_max drift), this buffer is the
        # authoritative record of the spectrum the weights were trained against.
        self.register_buffer(
            "w0_bands_tensor",
            torch.tensor(self.w0_bands, dtype=torch.float32),
            persistent=True,
        )

        # Split hidden width across bands; last band absorbs the remainder.
        bw_base = hidden // n_bands
        band_widths = [bw_base] * (n_bands - 1) + [hidden - bw_base * (n_bands - 1)]
        self.band_widths = band_widths

        # --- Multi-scale input stage ---
        self.band_layers: nn.ModuleList = nn.ModuleList()
        for i, (w0, bw_i) in enumerate(zip(self.w0_bands, band_widths)):
            lin = nn.Linear(in_dim, bw_i)
            if i == 0:
                siren_init_first_(lin)      # uniform [-1/n, 1/n], frequency-agnostic
            else:
                siren_init_hidden_(lin, w0)
            self.band_layers.append(lin)

        # --- Merge projection: concat(hidden) → hidden ---
        self.merge = nn.Linear(hidden, hidden)
        siren_init_hidden_(self.merge, self.w0_bands[0])

        # --- Shared deep blocks ---
        # input-stage + merge = 2 "layers"; remaining depth goes to shared blocks.
        n_shared = max(0, int(depth) - 2)
        w0_deep = self.w0_bands[-1]
        shared: List[nn.Module] = []
        for _ in range(n_shared):
            if use_residual:
                shared.append(SirenResBlock(hidden, w0=w0_deep, dropout=dropout))
            else:
                lin_h = nn.Linear(hidden, hidden)
                siren_init_hidden_(lin_h, w0_deep)
                shared.append(lin_h)
                shared.append(Sine(w0=w0_deep))
                if dropout > 0:
                    shared.append(nn.Dropout(p=float(dropout)))
        self.shared: nn.Module = nn.Sequential(*shared) if shared else nn.Identity()

        # --- Output head ---
        self.head = nn.Linear(hidden, 1)
        head_bound = 0.1 * (math.sqrt(6.0 / hidden) / max(w0_deep, 1.0))
        nn.init.uniform_(self.head.weight, -head_bound, head_bound)
        nn.init.zeros_(self.head.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Keep the Python-side w0_bands list (used by forward) in sync with the
        # band frequencies stored in the checkpoint. This guards against a
        # reconstruction whose construction-time bands differ from the trained
        # ones: after load, the band-stage spectrum follows the checkpoint.
        key = prefix + "w0_bands_tensor"
        incoming = state_dict.get(key)
        if incoming is not None:
            try:
                self.w0_bands = [float(v) for v in incoming.detach().cpu().tolist()]
            except Exception:
                pass
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        acts = [
            torch.sin(self.w0_bands[i] * self.band_layers[i](x_scaled))
            for i in range(len(self.band_layers))
        ]
        h = torch.cat(acts, dim=-1)                                  # (B, hidden)
        h = torch.sin(self.w0_bands[0] * self.merge(h))             # merge + activate
        h = self.shared(h)
        return self.head(h)


class AdditiveMultiBandSirenMLP(nn.Module):
    """
    Additive multi-band SIREN: ``ΔU(x) = Σ_k ΔU_k(x)``.

    Each band is an independent small SIREN trunk with its own frequency ``w0``,
    and their scalar outputs are summed. This contrasts with
    :class:`MultiScaleSirenMLP`, which concatenates band activations into a single
    shared trunk. The additive form keeps each frequency scale in a separate
    subnetwork, which can be easier to interpret per band.

    Parameter count
    ---------------
    Each band trunk is intentionally narrowed to ``hidden // n_bands`` units to
    keep compute controlled. As a result the total parameter count may be lower
    than the ``concat_shared`` model at the same ``hidden``/``depth``; these two
    modes are not assumed to be parameter-matched. Compute the counts explicitly
    if a fair parameter budget is required.

    Experimental: ``concat_shared`` (:class:`MultiScaleSirenMLP`) remains the
    default multi-scale composition. Evaluate this mode via ablation rather than
    using it as a default; its effect is not benchmarked.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 512,
        depth: int = 6,
        w0_bands: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        if w0_bands is None:
            w0_bands = [30.0]
        self.w0_bands: List[float] = [float(w) for w in w0_bands]
        n_bands = len(self.w0_bands)
        self.n_bands: int = n_bands

        # Persist band frequencies in the state_dict (reload-safety; see
        # MultiScaleSirenMLP for the rationale).
        self.register_buffer(
            "w0_bands_tensor",
            torch.tensor(self.w0_bands, dtype=torch.float32),
            persistent=True,
        )

        band_hidden = max(8, hidden // n_bands)
        self.bands: nn.ModuleList = nn.ModuleList(
            SirenMLP(
                in_dim=in_dim,
                hidden=band_hidden,
                depth=depth,
                w0_first=w0,
                w0_hidden=w0,
                dropout=dropout,
                use_residual=use_residual,
            )
            for w0 in self.w0_bands
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "w0_bands_tensor"
        incoming = state_dict.get(key)
        if incoming is not None:
            try:
                self.w0_bands = [float(v) for v in incoming.detach().cpu().tolist()]
            except Exception:
                pass
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        out = self.bands[0](x_scaled)
        for band in self.bands[1:]:
            out = out + band(x_scaled)
        return out


# ---------------------------------------------------------------------------
# Random Fourier Features (Tancik et al. 2020)
# ---------------------------------------------------------------------------
# φ(v) = [sin(2πBv), cos(2πBv)],  B ~ N(0, σ²)
# Only valid with non-SIREN backbones (activation="silu"/"tanh"/"softplus").

class FourierInputEmbedding(nn.Module):
    """
    Random Fourier Features with an optional raw-coordinate skip path.

    ``append_raw=True`` gives the backbone both:
    - low-frequency geometric context via the scaled (x,y,z)
    - higher-frequency residual cues via sinusoidal Fourier projections
    """

    def __init__(
        self,
        in_dim: int = 3,
        n_features: int = 256,
        sigma: float = 1.0,
        seed: int = 42,
        append_raw: bool = False,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        B = rng.standard_normal((n_features, in_dim)).astype(np.float32) * float(sigma)
        self.register_buffer("B", torch.from_numpy(B))
        self.append_raw = bool(append_raw)
        self.out_dim = (in_dim if self.append_raw else 0) + (2 * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B.T
        encoded = torch.cat(
            [torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)],
            dim=-1,
        )
        if self.append_raw:
            return torch.cat([x, encoded], dim=-1)
        return encoded


# ---------------------------------------------------------------------------
# Radial separation encoding
# ---------------------------------------------------------------------------

class RadialSeparationEncoding(nn.Module):
    """Explicit radial/direction separation: [r_norm, ux, uy, uz] or [r_norm, ux, uy, uz, x, y, z]."""
    def __init__(self, append_raw: bool = False):
        super().__init__()
        self.append_raw = bool(append_raw)

    @property
    def out_dim(self) -> int:
        return 7 if self.append_raw else 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
        encoded = torch.cat([r, x / r], dim=-1)  # (N, 4)
        if self.append_raw:
            return torch.cat([encoded, x], dim=-1)  # (N, 7)
        return encoded


class SHInspiredAngularEncoding(nn.Module):
    """
    This is not an exact spherical harmonic basis.
    It is a smooth Cartesian angular polynomial encoding inspired by low-degree angular harmonic structure.
    """
    def __init__(self, degree_max: int = 4, append_raw: bool = True):
        super().__init__()
        self.degree_max = int(degree_max)
        self.append_raw = bool(append_raw)
        
        if self.degree_max > 8:
            raise ValueError(f"SHInspiredAngularEncoding degree_max={self.degree_max} > 8 is not allowed by policy.")
            
        if not self.append_raw:
            raise ValueError(
                "SHInspiredAngularEncoding with append_raw=False loses radial information. "
                "You must set append_raw=True or include explicit radial features."
            )

        import math
        self.n_features = math.comb(self.degree_max + 3, 3) - 1
        
        from itertools import product
        combos = []
        for i, j, k in product(range(self.degree_max + 1), repeat=3):
            if 1 <= i + j + k <= self.degree_max:
                combos.append((i, j, k))
        combos.sort(key=lambda t: (sum(t), t[0], t[1], t[2]))
        
        self.register_buffer("pow_x", torch.tensor([c[0] for c in combos], dtype=torch.int32))
        self.register_buffer("pow_y", torch.tensor([c[1] for c in combos], dtype=torch.int32))
        self.register_buffer("pow_z", torch.tensor([c[2] for c in combos], dtype=torch.int32))

    @property
    def out_dim(self) -> int:
        return self.n_features + (3 if self.append_raw else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.norm(x, dim=-1, keepdim=True) + 1e-12
        nx = x[:, 0:1] / r
        ny = x[:, 1:2] / r
        nz = x[:, 2:3] / r
        
        features = (nx ** self.pow_x) * (ny ** self.pow_y) * (nz ** self.pow_z)

        if self.append_raw:
            return torch.cat([features, x], dim=-1)
        return features


# ---------------------------------------------------------------------------
# Radial decay-aware encoding (experimental)
# ---------------------------------------------------------------------------

class RadialDecayEncoding(nn.Module):
    """
    Scaled inverse-radius decay features (experimental).

    Inspired by the ``R/r`` radial decay of spherical-harmonic terms (degree-``l``
    contributions fall off roughly as ``(R/r)^(l+1)``), this encoding exposes
    inverse-radius powers to the network rather than forcing it to learn ``1/r``
    behaviour from raw Cartesian coordinates.

    Important: this is NOT exactly ``R_ref / r_phys``. The network input ``x`` is
    already divided by ``x_scale`` (the max training radius), so ``r = ||x||`` is a
    dimensionless *scaled* radius (≈ ``r_phys / r_max``, typically in ``[~0.8, 1.0]``
    for an LLO shell), and ``rho = 1 / r`` is therefore the inverse scaled radius,
    not a physical ``R_ref / r``. The features built are:

        r       = ||x||              (scaled radial magnitude)
        u       = x / r              (unit direction, 3 components)
        rho     = 1 / clamp(r, eps)  (inverse scaled radius)
        rho^k   for k = 1 .. max_power

    Output layout::

        [r, ux, uy, uz, rho, rho^2, ..., rho^max_power, (x, y, z if append_raw)]

    Notes
    -----
    * Working purely in network-scaled coordinates keeps the encoding
      self-contained and reload-stable: it needs no reference-radius / scaler
      metadata, so a reloaded model reproduces identical features.
    * A future physical-radius variant could use scaler metadata to form a true
      ``R_ref / r_phys``; this class intentionally avoids that dependency.
    * Experimental: off by default. Physically motivated for altitude
      generalization, but its effect is not benchmarked — evaluate via ablation.
    """

    def __init__(self, max_power: int = 4, append_raw: bool = True, eps: float = 1e-6):
        super().__init__()
        self.max_power = int(max_power)
        if self.max_power < 1:
            raise ValueError(f"RadialDecayEncoding max_power must be >= 1, got {self.max_power}")
        self.append_raw = bool(append_raw)
        self.eps = float(eps)

    @property
    def out_dim(self) -> int:
        return 1 + 3 + self.max_power + (3 if self.append_raw else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)   # (N, 1)
        u = x / r                                                     # (N, 3)
        rho = 1.0 / r                                                 # (N, 1)
        feats: List[torch.Tensor] = [r, u]
        rho_power = rho
        feats.append(rho_power)
        for _ in range(2, self.max_power + 1):
            rho_power = rho_power * rho
            feats.append(rho_power)
        if self.append_raw:
            feats.append(x)
        return torch.cat(feats, dim=-1)


class PhysicalRadialDecayEncoding(nn.Module):
    """
    Physical radial-decay features using true ``rho = R_ref / r_phys``.

    The network still receives scaled coordinates ``x_scaled``.  This encoding
    reconstructs the physical radius through ``r_phys = ||x_scaled|| *
    x_scale_m`` and then emits differentiable torch-native features:

        [r_scaled? , unit? , rho, rho^2, ..., rho^max_power, raw x_scaled?]
    """

    def __init__(
        self,
        *,
        x_scale_m: float,
        r_ref_m: float,
        max_power: int = 4,
        append_raw: bool = True,
        include_r_scaled: bool = True,
        include_unit: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.x_scale_m = float(x_scale_m)
        self.r_ref_m = float(r_ref_m)
        self.max_power = int(max_power)
        self.append_raw = bool(append_raw)
        self.include_r_scaled = bool(include_r_scaled)
        self.include_unit = bool(include_unit)
        self.eps = float(eps)
        if self.x_scale_m <= 0.0:
            raise ValueError(f"PhysicalRadialDecayEncoding x_scale_m must be positive, got {self.x_scale_m}")
        if self.r_ref_m <= 0.0:
            raise ValueError(f"PhysicalRadialDecayEncoding r_ref_m must be positive, got {self.r_ref_m}")
        if self.max_power < 1:
            raise ValueError(
                f"PhysicalRadialDecayEncoding max_power must be >= 1, got {self.max_power}"
            )

    @property
    def out_dim(self) -> int:
        return (
            (1 if self.include_r_scaled else 0)
            + (3 if self.include_unit else 0)
            + self.max_power
            + (3 if self.append_raw else 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r_scaled = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        r_phys = r_scaled * x.new_tensor(float(self.x_scale_m))
        rho = x.new_tensor(float(self.r_ref_m)) / r_phys.clamp_min(self.eps)

        feats: List[torch.Tensor] = []
        if self.include_r_scaled:
            feats.append(r_scaled)
        if self.include_unit:
            feats.append(x / r_scaled)
        rho_power = rho
        feats.append(rho_power)
        for _ in range(2, self.max_power + 1):
            rho_power = rho_power * rho
            feats.append(rho_power)
        if self.append_raw:
            feats.append(x)
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Real spherical-harmonic angular basis (experimental)
# ---------------------------------------------------------------------------

class RealSHBasisEncoding(nn.Module):
    """
    Genuine real spherical-harmonic angular basis up to degree ``degree_max``.

    This computes the 4π fully-normalized real spherical harmonics (geodesy
    convention, Condon-Shortley phase omitted) for degrees ``l = 0 .. L`` and
    orders ``m = -l .. l`` — ``(L+1)^2`` angular terms — torch-natively, with no
    SciPy dependency.

    Numerical method (pole-safe)
    ----------------------------
    Let ``n = x/||x||`` with ``t = n_z = cosθ``. Define the column-recurrence
    quantities

        C_m = Re((n_x + i n_y)^m) = sinθ^m · cos(mφ)
        S_m = Im((n_x + i n_y)^m) = sinθ^m · sin(mφ)

    computed by a complex-power recurrence that never divides by ``sinθ`` (so it
    is finite at the poles, where ``C_m = S_m = 0`` for ``m ≥ 1``).

    The associated-Legendre polynomial part ``Q_{l,m}(t) = P̄_{l,m}(t)/sinθ^m`` is
    built with the standard fully-normalized forward recurrence (Holmes &
    Featherstone 2002), which is a polynomial in ``t`` and therefore also
    pole-stable. The real SH are then

        Y_{l,0}      = Q_{l,0}(t)
        Y_{l,m}^cos  = √2 · Q_{l,m}(t) · C_m       (m > 0)
        Y_{l,m}^sin  = √2 · Q_{l,m}(t) · S_m       (m > 0)

    Output layout::

        [r (if include_radial), Y_{0,0}, ... Y_{L,L}, (x, y, z if append_raw)]

    Experimental: off by default; intended for angular-generalization ablations.

    TODO: validate low-degree normalization/order against an external SH
    reference; the optional SciPy-gated check in
    tests/test_surrogate_architecture_upgrades.py must remain optional.
    """

    def __init__(
        self,
        degree_max: int = 4,
        append_raw: bool = True,
        include_radial: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.degree_max = int(degree_max)
        if not (0 <= self.degree_max <= 8):
            raise ValueError(f"RealSHBasisEncoding degree_max must be in [0, 8], got {self.degree_max}")
        self.append_raw = bool(append_raw)
        self.include_radial = bool(include_radial)
        self.eps = float(eps)
        self.n_angular = (self.degree_max + 1) ** 2

    @property
    def out_dim(self) -> int:
        return self.n_angular + (1 if self.include_radial else 0) + (3 if self.append_raw else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.degree_max
        r = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)   # (N, 1)
        n = x / r
        nx = n[:, 0:1]
        ny = n[:, 1:2]
        t = n[:, 2:3]                                                 # cosθ
        ones = torch.ones_like(t)

        # C_m = sinθ^m cos(mφ), S_m = sinθ^m sin(mφ) via complex-power recurrence.
        C: List[torch.Tensor] = [ones]
        S: List[torch.Tensor] = [torch.zeros_like(t)]
        for _m in range(1, L + 1):
            C.append(C[-1] * nx - S[-1] * ny)
            S.append(C[-2] * ny + S[-1] * nx)

        # Q[(l, m)] = P̄_{l,m}(t) / sinθ^m  (polynomial in t; sinθ^m carried by C/S).
        Q: Dict[Tuple[int, int], torch.Tensor] = {(0, 0): ones}
        sec = 1.0
        for m in range(1, L + 1):
            sec *= math.sqrt((2 * m + 1) / (2 * m))   # constant sectoral seed (no sinθ)
            Q[(m, m)] = sec * ones
        for m in range(0, L + 1):
            for l in range(m + 1, L + 1):
                a = math.sqrt((2 * l - 1) * (2 * l + 1) / ((l - m) * (l + m)))
                if l - 2 >= m:
                    b = math.sqrt(
                        (2 * l + 1) * (l + m - 1) * (l - m - 1)
                        / ((2 * l - 3) * (l - m) * (l + m))
                    )
                    Q[(l, m)] = a * t * Q[(l - 1, m)] - b * Q[(l - 2, m)]
                else:
                    Q[(l, m)] = a * t * Q[(l - 1, m)]

        sqrt2 = math.sqrt(2.0)
        feats: List[torch.Tensor] = []
        if self.include_radial:
            feats.append(r)
        for l in range(0, L + 1):
            feats.append(Q[(l, 0)])                       # Y_{l,0}
            for m in range(1, l + 1):
                base = sqrt2 * Q[(l, m)]
                feats.append(base * C[m])                 # cos term
                feats.append(base * S[m])                 # sin term
        if self.append_raw:
            feats.append(x)
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# PhysicsNet wrapper
# ---------------------------------------------------------------------------

class PhysicsNet(nn.Module):
    """Optional FourierEmbedding → backbone. ``embedding=None`` is a no-op pass-through."""

    def __init__(self, backbone: nn.Module, embedding: Optional[nn.Module] = None):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x_scaled = self.embedding(x_scaled)
        return self.backbone(x_scaled)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _get_output_head_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Return the parameters of the final scalar output head.

    The head receives a higher learning rate than the backbone (see engine
    param groups) because early training diagnostics showed the backbone
    evolving while the head stayed near zero, locking the surrogate into a
    trivial near-baseline solution.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return list(model.parameters())
    return list(linears[-1].parameters())


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _encoding_flags_from_preset(cfg: Any) -> Dict[str, bool]:
    """Return effective encoding flags after applying a named preset.

    Missing ``model_preset`` means old artifact/config behavior: respect the
    explicit flags exactly.
    """

    flags = {
        "use_fourier": bool(_cfg_value(cfg, "use_fourier", False)),
        "use_sh_encoding": bool(_cfg_value(cfg, "use_sh_encoding", False)),
        "use_radial_separation": bool(_cfg_value(cfg, "use_radial_separation", False)),
        "use_radial_decay_encoding": bool(_cfg_value(cfg, "use_radial_decay_encoding", False)),
        "use_physical_radial_decay_encoding": bool(
            _cfg_value(cfg, "use_physical_radial_decay_encoding", False)
        ),
        "use_real_sh_basis": bool(_cfg_value(cfg, "use_real_sh_basis", False)),
    }
    preset_raw = _cfg_value(cfg, "model_preset", None)
    if preset_raw is None:
        return flags
    preset = str(preset_raw or "custom").strip().lower()
    if preset == "custom":
        return flags
    implied = {name: False for name in flags}
    if preset == "baseline_raw":
        pass
    elif preset == "recommended_physical_radial_decay":
        implied["use_physical_radial_decay_encoding"] = True
    elif preset == "ablation_radial_separation":
        implied["use_radial_separation"] = True
    elif preset == "ablation_radial_decay_scaled":
        implied["use_radial_decay_encoding"] = True
    elif preset == "ablation_real_sh_low_degree":
        implied["use_real_sh_basis"] = True
    else:
        raise ValueError(
            "model_preset must be one of baseline_raw, recommended_physical_radial_decay, "
            "ablation_radial_separation, ablation_radial_decay_scaled, "
            f"ablation_real_sh_low_degree, or custom; got {preset!r}."
        )
    active = {name for name, value in flags.items() if value}
    implied_active = {name for name, value in implied.items() if value}
    if active and active != implied_active:
        raise ValueError(
            f"model_preset={preset!r} conflicts with manual encoding flags {sorted(active)}. "
            "Use model_preset='custom' for manual ablations."
        )
    return implied


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model_from_config(
    cfg: Any,
    *,
    in_dim: int = 3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> PhysicsNet:
    """
    Build ``PhysicsNet`` from a ``TrainConfig``-like object or config dict.

    Supported config keys
    ---------------------
    activation : str
        "sine" (SIREN) | "silu" | "tanh" | "softplus"
    use_residual_blocks : bool
        Wrap hidden SIREN layers in ``SirenResBlock``.  Default False.
    n_bands : int
        Number of harmonic frequency bands.  >1 → ``MultiScaleSirenMLP``.
        Requires degree_min and degree_max in cfg.  Default 1.
    degree_min / degree_max : int
        Harmonic degree range used to derive per-band w0 values when n_bands > 1.
    use_fourier : bool
        Random Fourier Feature embedding (only with non-SIREN activations).
    """
    activation = str(_cfg_value(cfg, "activation", "sine")).lower()
    encoding_flags = _encoding_flags_from_preset(cfg)
    use_fourier = bool(encoding_flags["use_fourier"])
    if activation == "sine" and use_fourier:
        raise ValueError(
            "activation='sine' (SIREN) and use_fourier=True are mutually exclusive. "
            "Disable Fourier/RFF or use a non-sine activation."
        )

    # Optional alternative input encodings. At most one may be active.
    use_sh = bool(encoding_flags["use_sh_encoding"])
    use_radial = bool(encoding_flags["use_radial_separation"])
    use_radial_decay = bool(encoding_flags["use_radial_decay_encoding"])
    use_physical_radial_decay = bool(encoding_flags["use_physical_radial_decay_encoding"])
    use_real_sh = bool(encoding_flags["use_real_sh_basis"])
    sh_degree = int(_cfg_value(cfg, "sh_encoding_degree", 6))
    sh_append_raw = bool(_cfg_value(cfg, "sh_append_raw", True))
    radial_append_raw = bool(_cfg_value(cfg, "radial_append_raw", False))

    _active_encodings = [
        name for name, flag in (
            ("use_fourier", use_fourier),
            ("use_sh_encoding", use_sh),
            ("use_radial_separation", use_radial),
            ("use_radial_decay_encoding", use_radial_decay),
            ("use_physical_radial_decay_encoding", use_physical_radial_decay),
            ("use_real_sh_basis", use_real_sh),
        ) if flag
    ]
    if len(_active_encodings) > 1:
        raise ValueError(
            f"Incompatible encodings: At most one input encoding may be active, but got {_active_encodings}. "
            "These cannot both be True. They are mutually exclusive; enable only one (or none for raw xyz)."
        )
    if use_sh and sh_degree > 8:
        import warnings
        warnings.warn(
            f"sh_encoding_degree={sh_degree} > 8. This significantly increases input "
            "dimensionality and training cost. Consider degree <= 6 for typical residual "
            "SH training shells.",
            UserWarning, stacklevel=2
        )
    if use_sh and not (0 <= sh_degree <= 16):
        raise ValueError(f"sh_encoding_degree must be in [0, 16], got {sh_degree}")

    embedding = None
    backbone_in_dim = int(in_dim)
    if use_fourier:
        embedding = FourierInputEmbedding(
            in_dim=int(in_dim),
            n_features=int(_cfg_value(cfg, "fourier_n_features", _cfg_value(cfg, "fourier_n", 256))),
            sigma=float(_cfg_value(cfg, "fourier_sigma", 1.0)),
            seed=int(_cfg_value(cfg, "fourier_seed", 42)),
            append_raw=bool(_cfg_value(cfg, "fourier_append_raw", True)),
        )
        backbone_in_dim = int(embedding.out_dim)
    elif use_sh:
        embedding = SHInspiredAngularEncoding(
            degree_max=sh_degree,
            append_raw=sh_append_raw
        )
        backbone_in_dim = int(embedding.out_dim)
    elif use_radial:
        embedding = RadialSeparationEncoding(append_raw=radial_append_raw)
        backbone_in_dim = int(embedding.out_dim)
    elif use_radial_decay:
        embedding = RadialDecayEncoding(
            max_power=int(_cfg_value(cfg, "radial_decay_max_power", 4)),
            append_raw=bool(_cfg_value(cfg, "radial_decay_append_raw", True)),
        )
        backbone_in_dim = int(embedding.out_dim)
    elif use_physical_radial_decay:
        x_scale_m = _cfg_value(cfg, "x_scale_m", None)
        r_ref_m = _cfg_value(cfg, "resolved_r_ref_m", _cfg_value(cfg, "r_ref_m", None))
        if x_scale_m is None or r_ref_m is None:
            raise ValueError(
                "PhysicalRadialDecayEncoding requires x_scale_m and resolved_r_ref_m/r_ref_m "
                "in the config. The training engine injects these from the scaler and dataset."
            )
        embedding = PhysicalRadialDecayEncoding(
            x_scale_m=float(x_scale_m),
            r_ref_m=float(r_ref_m),
            max_power=int(_cfg_value(cfg, "physical_radial_decay_max_power", 4)),
            append_raw=bool(_cfg_value(cfg, "physical_radial_decay_append_raw", True)),
            include_unit=bool(_cfg_value(cfg, "physical_radial_decay_include_unit", True)),
            include_r_scaled=bool(_cfg_value(cfg, "physical_radial_decay_include_r_scaled", True)),
        )
        backbone_in_dim = int(embedding.out_dim)
    elif use_real_sh:
        embedding = RealSHBasisEncoding(
            degree_max=int(_cfg_value(cfg, "real_sh_degree", 4)),
            append_raw=bool(_cfg_value(cfg, "real_sh_append_raw", True)),
            include_radial=bool(_cfg_value(cfg, "real_sh_include_radial", True)),
        )
        backbone_in_dim = int(embedding.out_dim)

    n_bands      = max(1, int(_cfg_value(cfg, "n_bands", 1)))
    use_residual = bool(_cfg_value(cfg, "use_residual_blocks", False))

    resolved_w0_bands: Optional[List[float]] = None
    if activation == "sine":
        hidden  = int(_cfg_value(cfg, "hidden",  512))
        depth   = int(_cfg_value(cfg, "depth",   4))
        dropout = float(_cfg_value(cfg, "dropout", 0.0))

        if n_bands > 1:
            # Multi-scale SIREN: the per-band frequencies (w0_bands) define the
            # functional model. They MUST be reconstructed identically at eval
            # time or the loaded weights (which match by shape) will be driven
            # at the wrong frequencies — a silent, catastrophic mismatch.
            # Resolution order:
            #   1. explicit cfg["w0_bands"] (authoritative; written at train time)
            #   2. derive from BOTH degree_min and degree_max (must be present)
            # Silent fallback to (0, 50) is forbidden.
            explicit_bands = _cfg_value(cfg, "w0_bands", None)
            if explicit_bands:
                w0_bands = [float(w) for w in explicit_bands]
                if len(w0_bands) != n_bands:
                    raise ValueError(
                        f"w0_bands has length {len(w0_bands)} but n_bands={n_bands}. "
                        "The number of band frequencies must equal n_bands."
                    )
            else:
                dmin_raw = _cfg_value(cfg, "degree_min", None)
                dmax_raw = _cfg_value(cfg, "degree_max", None)
                if dmin_raw is None or dmax_raw is None:
                    raise ValueError(
                        "MultiScaleSirenMLP (n_bands>1) requires either an explicit "
                        "'w0_bands' list or BOTH 'degree_min' and 'degree_max' in the "
                        "config so the per-band SIREN frequencies can be reconstructed "
                        "deterministically. Refusing to silently default to (degree_min=0, "
                        "degree_max=50): that would build a model with different band "
                        "frequencies than the one the checkpoint was trained with, and the "
                        "state_dict would load by shape while predicting nonsense. "
                        "Re-run training with the current engine (which records w0_bands), "
                        "or add 'w0_bands'/'degree_min'+'degree_max' to config.json."
                    )
                degree_min_cfg = max(-1, int(dmin_raw))
                degree_max_cfg = max(1,  int(dmax_raw))
                w0_bands = _compute_harmonic_w0_bands(n_bands, degree_min_cfg, degree_max_cfg)
            resolved_w0_bands = [float(w) for w in w0_bands]
            multiscale_mode = str(_cfg_value(cfg, "multiscale_mode", "concat_shared")).lower()
            if multiscale_mode not in ("concat_shared", "additive"):
                raise ValueError(
                    f"multiscale_mode must be 'concat_shared' or 'additive', got {multiscale_mode!r}."
                )
            if multiscale_mode == "additive":
                backbone: nn.Module = AdditiveMultiBandSirenMLP(
                    in_dim=backbone_in_dim,
                    hidden=hidden,
                    depth=depth,
                    w0_bands=w0_bands,
                    dropout=dropout,
                    use_residual=use_residual,
                )
            else:
                backbone = MultiScaleSirenMLP(
                    in_dim=backbone_in_dim,
                    hidden=hidden,
                    depth=depth,
                    w0_bands=w0_bands,
                    dropout=dropout,
                    use_residual=True,
                )
        else:
            backbone = SirenMLP(
                in_dim=backbone_in_dim,
                hidden=hidden,
                depth=depth,
                w0_first=float(_cfg_value(cfg, "w0_first",  30.0)),
                w0_hidden=float(_cfg_value(cfg, "w0_hidden", 30.0)),
                dropout=dropout,
                use_residual=use_residual,
            )
    else:
        backbone = MLP(
            in_dim=backbone_in_dim,
            hidden=int(_cfg_value(cfg, "hidden", 512)),
            depth=int(_cfg_value(cfg, "depth",  4)),
            activation=activation,
            dropout=float(_cfg_value(cfg, "dropout", 0.0)),
        )

    model = PhysicsNet(backbone=backbone, embedding=embedding)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)

    # Attach metadata attributes so the engine/evaluator can save them in
    # config.json and checkpoint files without re-deriving them.
    if use_sh:
        _emb_type = "sh_angular"
    elif use_radial:
        _emb_type = "radial_separation"
    elif use_radial_decay:
        _emb_type = "radial_decay"
    elif use_physical_radial_decay:
        _emb_type = "physical_radial_decay"
    elif use_real_sh:
        _emb_type = "real_sh"
    elif use_fourier:
        _emb_type = "fourier_rff"
    else:
        _emb_type = "raw"
    model.embedding_type: str = _emb_type  # type: ignore[assignment]
    model.input_feature_dim: int = int(backbone_in_dim)  # type: ignore[assignment]
    model.model_builder_version: str = MODEL_BUILDER_VERSION  # type: ignore[assignment]
    # Resolved per-band SIREN frequencies (None for single-scale / non-SIREN).
    # The engine persists this so evaluation reconstructs the exact spectrum.
    model.w0_bands = resolved_w0_bands  # type: ignore[assignment]

    return model


# ---------------------------------------------------------------------------
# Architecture signature (reload-safety)
# ---------------------------------------------------------------------------

# Bumped whenever the build logic changes in a way that affects the functional
# architecture for a fixed config. Persisted into config.json and checkpoints.
MODEL_BUILDER_VERSION: str = "v3"

# Fields that fully determine the functional architecture. Two configs that
# agree on all of these must build identical (shape- AND frequency-identical)
# models. Used to detect config/checkpoint drift on reload.
ARCH_SIGNATURE_FIELDS = (
    "activation", "hidden", "depth", "dropout",
    "model_preset", "runtime_model_kind",
    "use_residual_blocks", "n_bands", "multiscale_mode",
    "degree_min", "degree_max", "w0_bands",
    "use_sh_encoding", "sh_encoding_degree", "sh_append_raw",
    "use_radial_separation", "radial_append_raw",
    "use_radial_decay_encoding", "radial_decay_max_power", "radial_decay_append_raw",
    "use_physical_radial_decay_encoding", "physical_radial_decay_max_power",
    "physical_radial_decay_append_raw", "physical_radial_decay_include_unit",
    "physical_radial_decay_include_r_scaled", "x_scale_m", "resolved_r_ref_m",
    "use_real_sh_basis", "real_sh_degree", "real_sh_append_raw", "real_sh_include_radial",
    "use_fourier", "fourier_n_features", "fourier_sigma", "fourier_seed",
    "fourier_append_raw",
    "input_feature_dim", "embedding_type", "model_builder_version",
)


def _normalize_signature_value(key: str, value: Any) -> Any:
    """Normalize a config value so equivalent configs hash identically."""
    if value is None:
        return None
    if key == "w0_bands":
        try:
            return [round(float(v), 4) for v in value]
        except (TypeError, ValueError):
            return value
    if key in ("hidden", "depth", "n_bands", "degree_min", "degree_max",
               "sh_encoding_degree", "fourier_n_features", "fourier_seed",
               "input_feature_dim", "radial_decay_max_power", "real_sh_degree",
               "physical_radial_decay_max_power"):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if key in ("dropout", "fourier_sigma", "x_scale_m", "resolved_r_ref_m"):
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return value
    if key in ("use_residual_blocks", "use_sh_encoding", "sh_append_raw",
               "use_radial_separation", "radial_append_raw", "use_fourier",
               "fourier_append_raw", "use_radial_decay_encoding", "radial_decay_append_raw",
               "use_physical_radial_decay_encoding", "physical_radial_decay_append_raw",
               "physical_radial_decay_include_unit", "physical_radial_decay_include_r_scaled",
               "use_real_sh_basis", "real_sh_append_raw", "real_sh_include_radial"):
        return bool(value)
    return str(value)


def architecture_fingerprint(cfg: Any) -> Dict[str, Any]:
    """Return the normalized architecture-critical fields for ``cfg``."""
    return {
        key: _normalize_signature_value(key, _cfg_value(cfg, key, None))
        for key in ARCH_SIGNATURE_FIELDS
    }


def compute_architecture_signature(cfg: Any) -> str:
    """Stable short hash of the architecture-critical config fields."""
    import hashlib
    import json as _json

    payload = architecture_fingerprint(cfg)
    blob = _json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def architecture_mismatch_fields(cfg_a: Any, cfg_b: Any) -> List[str]:
    """Return field names where two configs disagree on the architecture."""
    fa = architecture_fingerprint(cfg_a)
    fb = architecture_fingerprint(cfg_b)
    out: List[str] = []
    for key in ARCH_SIGNATURE_FIELDS:
        va, vb = fa.get(key), fb.get(key)
        # Treat "absent on one side" as agreement to stay lenient toward
        # partial configs; only flag genuine value disagreements.
        if va is None or vb is None:
            continue
        if va != vb:
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# Reload-safe reconstruction (shared by evaluator, force model, tests)
# ---------------------------------------------------------------------------

# Buffers that may legitimately be absent from older checkpoints. Their absence
# must NOT be treated as a state_dict mismatch; everything else must match.
_OPTIONAL_BUFFER_SUFFIXES = ("w0_bands_tensor",)


def _is_optional_state_key(key: str) -> bool:
    return any(key.endswith(suf) for suf in _OPTIONAL_BUFFER_SUFFIXES)


def _verify_reconstructed_model(model: nn.Module, ref_cfg: Dict[str, Any]) -> None:
    """Fail loudly if the built model disagrees with the checkpoint config.

    Catches the failure mode where a state_dict loads by shape but the functional
    architecture (SIREN band frequencies / input encoding) differs.
    """
    problems: List[str] = []

    ref_n_bands = ref_cfg.get("n_bands")
    if ref_n_bands is not None and int(ref_n_bands) > 1:
        ref_bands = ref_cfg.get("w0_bands")
        model_bands = getattr(model, "w0_bands", None)
        if ref_bands is not None and model_bands is not None:
            rb = [round(float(v), 4) for v in ref_bands]
            mb = [round(float(v), 4) for v in model_bands]
            if rb != mb:
                problems.append(f"w0_bands: checkpoint={rb} but reconstructed={mb}")

    ref_ifd = ref_cfg.get("input_feature_dim")
    model_ifd = getattr(model, "input_feature_dim", None)
    if ref_ifd is not None and model_ifd is not None and int(ref_ifd) != int(model_ifd):
        problems.append(f"input_feature_dim: checkpoint={ref_ifd} but reconstructed={model_ifd}")

    ref_emb = ref_cfg.get("embedding_type")
    model_emb = getattr(model, "embedding_type", None)
    if ref_emb is not None and model_emb is not None and str(ref_emb) != str(model_emb):
        problems.append(f"embedding_type: checkpoint={ref_emb} but reconstructed={model_emb}")

    ref_ver = ref_cfg.get("model_builder_version")
    model_ver = getattr(model, "model_builder_version", None)
    if ref_ver is not None and model_ver is not None and str(ref_ver) != str(model_ver):
        logger.warning(
            "model_builder_version: checkpoint=%s reconstructed=%s (build logic changed)",
            ref_ver, model_ver,
        )

    if problems:
        raise RuntimeError(
            "Reconstructed model does not match the checkpoint architecture: "
            + "; ".join(problems)
            + ". Refusing to evaluate a functionally different model."
        )


def reconstruct_model_from_artifacts(
    cfg_json: Dict[str, Any],
    ckpt: Dict[str, Any],
    device: Optional[torch.device] = None,
    *,
    dtype: torch.dtype = torch.float32,
    allow_config_mismatch: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Reconstruct the EXACT trained model from config.json + checkpoint.

    Reload-safety contract:
      * The checkpoint's own ``config`` block is authoritative for architecture
        fields. If config.json disagrees on any architecture-critical field,
        raise ``RuntimeError`` unless ``allow_config_mismatch``.
      * After ``load_state_dict``, verify n_bands / w0_bands / input_feature_dim /
        embedding_type. Any mismatch fails loudly: a shape-compatible state_dict
        must no longer hide a functional architecture mismatch.

    Returns ``(model, merged_cfg, report)``.
    """
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}

    mismatches = architecture_mismatch_fields(cfg_json, ckpt_cfg) if ckpt_cfg else []
    if mismatches and not allow_config_mismatch:
        details = {f: (cfg_json.get(f), ckpt_cfg.get(f)) for f in mismatches}
        raise RuntimeError(
            "config.json and checkpoint['config'] disagree on architecture-critical "
            f"fields {mismatches}: {details}. The checkpoint was trained with the "
            "checkpoint['config'] architecture; evaluating with a different one would "
            "load weights by shape while predicting nonsense. Pass allow_config_mismatch="
            "True only if you understand the risk."
        )

    merged_cfg: Dict[str, Any] = dict(cfg_json)
    used_ckpt_config = False
    for key in ARCH_SIGNATURE_FIELDS:
        if key in ckpt_cfg and ckpt_cfg[key] is not None:
            merged_cfg[key] = ckpt_cfg[key]
            used_ckpt_config = True
    for key in ("resolved_mu_si", "resolved_a_sign", "resolved_r_ref_m",
                "degree_min", "degree_max", "w0_bands"):
        if key in ckpt and ckpt[key] is not None and merged_cfg.get(key) is None:
            merged_cfg[key] = ckpt[key]

    model = build_model_from_config(merged_cfg, device=device, dtype=dtype)
    model.eval()
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict")
        if state is None:
            state = ckpt.get("model", ckpt)
    else:
        state = ckpt
    load_result = model.load_state_dict(state, strict=False)
    missing = [k for k in load_result.missing_keys if not _is_optional_state_key(k)]
    unexpected = [k for k in load_result.unexpected_keys if not _is_optional_state_key(k)]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state_dict does not match the reconstructed architecture. "
            f"missing_keys={missing}, unexpected_keys={unexpected}. This indicates a "
            "genuine architecture mismatch (not just an optional buffer)."
        )

    _verify_reconstructed_model(model, ckpt_cfg or merged_cfg)

    report: Dict[str, Any] = {
        "checkpoint_config_source": "checkpoint" if used_ckpt_config else "config_json",
        "architecture_mismatch_fields": mismatches,
        "allow_config_mismatch": bool(allow_config_mismatch),
        "architecture_signature": (
            ckpt_cfg.get("architecture_signature")
            or (ckpt.get("architecture", {}) if isinstance(ckpt, dict) else {}).get("signature")
            or compute_architecture_signature(merged_cfg)
        ),
        "model_w0_bands": list(getattr(model, "w0_bands", []) or []),
        "n_bands": int(merged_cfg.get("n_bands", 1) or 1),
    }
    return model, merged_cfg, report


__all__ = [
    "Sine",
    "SirenResBlock",
    "SirenMLP",
    "MultiScaleSirenMLP",
    "AdditiveMultiBandSirenMLP",
    "MLP",
    "FourierInputEmbedding",
    "RadialSeparationEncoding",
    "SHInspiredAngularEncoding",
    "RadialDecayEncoding",
    "PhysicalRadialDecayEncoding",
    "RealSHBasisEncoding",
    "PhysicsNet",
    "siren_init_first_",
    "siren_init_hidden_",
    "_compute_harmonic_w0_bands",
    "_get_output_head_params",
    "build_model_from_config",
    "MODEL_BUILDER_VERSION",
    "ARCH_SIGNATURE_FIELDS",
    "architecture_fingerprint",
    "compute_architecture_signature",
    "architecture_mismatch_fields",
    "reconstruct_model_from_artifacts",
]
