# -*- coding: utf-8 -*-
"""Neural model components for the scalar residual lunar potential field."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


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
        Number of frequency bands.  1 reproduces the legacy single-w0 behaviour.
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
        starts shallow.  Backward-compatible default is ``False``.
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
# Legacy MLP (non-SIREN activations)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard MLP (kept for backward compatibility with non-SIREN modes)."""

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

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        acts = [
            torch.sin(self.w0_bands[i] * self.band_layers[i](x_scaled))
            for i in range(len(self.band_layers))
        ]
        h = torch.cat(acts, dim=-1)                                  # (B, hidden)
        h = torch.sin(self.w0_bands[0] * self.merge(h))             # merge + activate
        h = self.shared(h)
        return self.head(h)


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
    use_fourier = bool(_cfg_value(cfg, "use_fourier", False))
    if activation == "sine" and use_fourier:
        raise ValueError(
            "activation='sine' (SIREN) and use_fourier=True are mutually exclusive. "
            "Disable Fourier/RFF or use a non-sine activation."
        )

    # Optional alternative input encodings (SH / radial separation).
    use_sh = bool(_cfg_value(cfg, "use_sh_encoding", False))
    use_radial = bool(_cfg_value(cfg, "use_radial_separation", False))
    sh_degree = int(_cfg_value(cfg, "sh_encoding_degree", 6))
    sh_append_raw = bool(_cfg_value(cfg, "sh_append_raw", True))
    radial_append_raw = bool(_cfg_value(cfg, "radial_append_raw", False))

    if use_sh and use_radial:
        raise ValueError(
            "use_sh_encoding and use_radial_separation cannot both be True. "
            "SH encoding already subsumes radial information. "
            "Set only one or neither."
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

    n_bands      = max(1, int(_cfg_value(cfg, "n_bands", 1)))
    use_residual = bool(_cfg_value(cfg, "use_residual_blocks", False))

    if activation == "sine":
        hidden  = int(_cfg_value(cfg, "hidden",  512))
        depth   = int(_cfg_value(cfg, "depth",   4))
        dropout = float(_cfg_value(cfg, "dropout", 0.0))

        if n_bands > 1:
            degree_min_cfg = max(-1, int(_cfg_value(cfg, "degree_min", 0)))
            degree_max_cfg = max(1,  int(_cfg_value(cfg, "degree_max", 50)))
            w0_bands = _compute_harmonic_w0_bands(n_bands, degree_min_cfg, degree_max_cfg)
            backbone: nn.Module = MultiScaleSirenMLP(
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
    elif use_fourier:
        _emb_type = "fourier_rff"
    else:
        _emb_type = "raw"
    model.embedding_type: str = _emb_type  # type: ignore[assignment]
    model.input_feature_dim: int = int(backbone_in_dim)  # type: ignore[assignment]
    model.model_builder_version: str = "v2"  # type: ignore[assignment]

    return model


__all__ = [
    "Sine",
    "SirenResBlock",
    "SirenMLP",
    "MultiScaleSirenMLP",
    "MLP",
    "FourierInputEmbedding",
    "RadialSeparationEncoding",
    "SHInspiredAngularEncoding",
    "PhysicsNet",
    "siren_init_first_",
    "siren_init_hidden_",
    "_compute_harmonic_w0_bands",
    "_get_output_head_params",
    "build_model_from_config",
]
