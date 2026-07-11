"""Inference-time neural network definitions for ST-LRPS surrogate artifacts."""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.surrogate.runtime.device import _require_torch, nn, torch

if torch is not None and nn is not None:

    class Sine(nn.Module):
        """SIREN activation used by some surrogate runs."""

        def __init__(self, w0: float = 30.0) -> None:
            super().__init__()
            self.w0 = float(w0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.w0 * x)


    class SirenMLP(nn.Module):
        """Small SIREN MLP that matches the training artifact contract."""

        def __init__(
            self,
            *,
            in_dim: int = 3,
            hidden: int = 256,
            depth: int = 4,
            w0_first: float = 30.0,
            w0_hidden: float = 30.0,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden), Sine(w0=w0_first)]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            for _ in range(max(0, depth - 1)):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(Sine(w0=w0_hidden))
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
            layers.append(nn.Linear(hidden, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
            return self.net(x_scaled)


    class MLP(nn.Module):
        """Legacy activation-based MLP used by older checkpoints."""

        def __init__(
            self,
            *,
            in_dim: int,
            hidden: int,
            depth: int,
            activation: str,
            dropout: float,
        ) -> None:
            super().__init__()
            act_name = str(activation).strip().lower()
            if act_name == "silu":
                act_factory = nn.SiLU
            elif act_name == "tanh":
                act_factory = nn.Tanh
            elif act_name == "softplus":
                act_factory = nn.Softplus
            else:
                raise ValueError(
                    "Unsupported surrogate activation. "
                    "Expected one of: sine, silu, tanh, softplus. "
                    f"Got {activation!r}."
                )

            layers: list[nn.Module] = []
            width_in = int(in_dim)
            for _ in range(max(0, int(depth))):
                layers.append(nn.Linear(width_in, int(hidden)))
                layers.append(act_factory())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
                width_in = int(hidden)
            layers.append(nn.Linear(width_in, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
            return self.net(x_scaled)


    class FourierInputEmbedding(nn.Module):
        """
        Random Fourier Feature embedding used by newer lunar surrogate runs.

        The runtime keeps this tiny implementation locally so inference does not
        depend on the experimental training script environment.
        """

        def __init__(
            self,
            *,
            in_dim: int = 3,
            n_features: int = 256,
            sigma: float = 1.0,
            seed: int = 42,
            append_raw: bool = False,
        ) -> None:
            super().__init__()
            rng = np.random.default_rng(int(seed))
            B = rng.standard_normal((int(n_features), int(in_dim))).astype(np.float32) * float(sigma)
            self.register_buffer("B", torch.from_numpy(B))
            self.append_raw = bool(append_raw)
            self.out_dim = (int(in_dim) if self.append_raw else 0) + (2 * int(n_features))

        def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
            proj = x_scaled @ self.B.T
            encoded = torch.cat(
                [torch.sin(2.0 * math.pi * proj), torch.cos(2.0 * math.pi * proj)],
                dim=-1,
            )
            if self.append_raw:
                return torch.cat([x_scaled, encoded], dim=-1)
            return encoded


    class PhysicsNet(nn.Module):
        """Inference-time wrapper for optional Fourier preprocessing + backbone."""

        def __init__(self, *, backbone: nn.Module, embedding: FourierInputEmbedding | None) -> None:
            super().__init__()
            self.backbone = backbone
            self.embedding = embedding

        def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
            if self.embedding is not None:
                x_scaled = self.embedding(x_scaled)
            return self.backbone(x_scaled)


def _build_model_from_config(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate the network architecture encoded in ``config.json``."""

    if cfg.get("architecture") in ("MultiScale", "Residual") or int(cfg.get("n_bands", 1)) > 1:
        raise ValueError(
            "This legacy surrogate provider does not support MultiScale or advanced Residual models. "
            "Please use the st_lrps module."
        )
    _require_torch()

    activation = str(cfg.get("activation", "sine")).strip().lower()
    hidden = int(cfg.get("hidden", 256))
    depth = int(cfg.get("depth", 4))
    dropout = float(cfg.get("dropout", 0.0) or 0.0)
    use_fourier = bool(cfg.get("use_fourier", False))

    # Audit F7: this legacy builder reconstructs the architecture from config
    # values, and some hyperparameters are NOT recoverable from the checkpoint
    # state_dict — a wrong default silently changes the computed field. The
    # defaults stay usable (pre-contract artifacts never recorded them) but the
    # assumption must be visible, never silent.
    if activation == "sine" and ("w0_first" not in cfg or "w0_hidden" not in cfg):
        warnings.warn(
            "[ST-LRPS legacy loader] sine (SIREN) artifact config does not record "
            "w0_first/w0_hidden; assuming w0=30.0. w0 is not stored in the "
            "checkpoint state_dict, so a wrong default silently changes the "
            "computed field. Regenerate the artifact with explicit w0 metadata "
            "for paper-safe use.",
            RuntimeWarning,
            stacklevel=2,
        )
    if use_fourier:
        _missing_fourier = [
            key
            for key in ("fourier_n_features", "fourier_sigma", "fourier_seed", "fourier_append_raw")
            if key not in cfg
        ]
        if _missing_fourier:
            warnings.warn(
                "[ST-LRPS legacy loader] Fourier-embedding artifact config does not "
                f"record {', '.join(_missing_fourier)}; reconstructing the embedding "
                "from defaults. The projection matrix itself is restored from the "
                "checkpoint state_dict (strict load), but shape-affecting defaults "
                "can fail the load or misdescribe the run's provenance. Regenerate "
                "the artifact with explicit Fourier metadata for paper-safe use.",
                RuntimeWarning,
                stacklevel=2,
            )

    embedding: FourierInputEmbedding | None = None
    backbone_in_dim = 3
    if use_fourier:
        embedding = FourierInputEmbedding(
            in_dim=3,
            n_features=int(cfg.get("fourier_n_features", 256)),
            sigma=float(cfg.get("fourier_sigma", 1.0)),
            seed=int(cfg.get("fourier_seed", cfg.get("seed", 42))),
            append_raw=bool(cfg.get("fourier_append_raw", False)),
        )
        backbone_in_dim = int(embedding.out_dim)

    if activation == "sine":
        backbone = SirenMLP(
            in_dim=backbone_in_dim,
            hidden=hidden,
            depth=depth,
            w0_first=float(cfg.get("w0_first", 30.0) or 30.0),
            w0_hidden=float(cfg.get("w0_hidden", 30.0) or 30.0),
            dropout=dropout,
        )
    else:
        backbone = MLP(
            in_dim=backbone_in_dim,
            hidden=hidden,
            depth=depth,
            activation=activation,
            dropout=dropout,
        )
    return PhysicsNet(backbone=backbone, embedding=embedding)


def _extract_state_dict(checkpoint_obj: dict[str, Any]) -> dict[str, Any]:
    """Extract the model state dictionary from a checkpoint payload."""

    for key in ("model", "model_state", "state_dict"):
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("Checkpoint does not contain a model state dictionary.")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint with compatibility across PyTorch versions.

    The safe ``weights_only=True`` loader is tried first; only legacy payloads
    holding arbitrary pickled objects fall back to full unpickling, which
    executes code embedded in the file — never load untrusted checkpoints.
    """

    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch too old to know the weights_only kwarg: plain load (unpickle).
        obj = torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError):
        warnings.warn(
            f"Safe (weights_only=True) load of {path} failed; falling back to "
            "full unpickling. This executes code embedded in the checkpoint — "
            "only load checkpoints from trusted sources.",
            RuntimeWarning,
            stacklevel=2,
        )
        obj = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(obj)!r}")
    return obj


# =============================================================================
# 4.                        PUBLIC RUNTIME PROVIDER
# =============================================================================

__all__ = [
    "_build_model_from_config",
    "_extract_state_dict",
    "_load_checkpoint",
]

if torch is not None and nn is not None:
    __all__.extend(["Sine", "SirenMLP", "MLP", "FourierInputEmbedding", "PhysicsNet"])
