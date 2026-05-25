"""
Surrogate Gravity Runtime
=========================

Runtime helpers for neural-network gravity surrogates stored under
``st_lrps/``.

Why this module exists
----------------------
The training / evaluation scripts in ``st_lrps/`` are valuable
for experimentation, but they are not a safe runtime dependency for the main
simulation loop:

- they pull in plotting / dataset-evaluation dependencies
- they assume a standalone script environment
- they do not expose a small, stable inference API for the propagator

This module turns those experiment artifacts into a production-facing provider
that the rest of the application can treat like a gravity model.

Supported artifact families
---------------------------
Two artifact layouts are supported because the repository contains both newer
and older ST-LRPS runs:

1. Residual potential models
   - network predicts ``ΔU``
   - total acceleration = SH(degree_min) baseline + neural correction
   - typically paired with isometric scaling (``scale`` fields)

2. Absolute potential models
   - network predicts the full potential ``U``
   - acceleration is obtained directly from ``a_sign * ∇U``
   - older runs often store classic Z-score statistics (``std`` fields)

The loader detects the contract from the artifact metadata and scaler shape so
the main application can fail fast on malformed runs instead of silently using
the wrong physics path.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from common.constants import MU_MOON, R_MOON
from models.spherical_harmonics import GravityModel
from models.torch_spherical_harmonics import TorchSHGravityEvaluator
from st_lrps.dataset_parameters import (
    DEFAULT_DATASET_CONFIG,
    looks_like_lunar_run_config,
    resolve_lunar_gravity_path,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _TORCH_IMPORT_ERROR = None


# =============================================================================
# 1.                           DISCOVERY HELPERS
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ST_LRPS_RUNS_DIR = _REPO_ROOT / "st_lrps" / "runs"


def _is_valid_surrogate_run(path: Path) -> bool:
    """
    Return ``True`` when ``path`` looks like a complete surrogate gravity run.

    Accepts either ``ckpt_best.pt`` (fully trained) or ``ckpt_last.pt``
    (in-progress / interrupted run) so the UI can offer recently-started
    runs for inspection without requiring a completed best-checkpoint.
    """

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    ckpt_dir = path / "checkpoints"
    return (ckpt_dir / "ckpt_best.pt").is_file() or (ckpt_dir / "ckpt_last.pt").is_file()


def _find_checkpoint_for_run(run_dir: Path) -> Path:
    """
    Return the best available checkpoint inside *run_dir/checkpoints/*.

    Preference order: ``ckpt_best.pt`` (finished training) then
    ``ckpt_last.pt`` (interrupted run, suitable for inference with a warning).
    Raises ``FileNotFoundError`` when neither exists.
    """

    ckpt_dir = run_dir / "checkpoints"
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. Expected ckpt_best.pt or ckpt_last.pt."
    )


def find_checkpoint_for_st_lrps_run(run_dir: Path | str) -> Path:
    """Public wrapper used by validation tools to report the selected weights."""

    return _find_checkpoint_for_run(Path(run_dir).expanduser().resolve())


def _extract_degree_metadata(config: Dict[str, Any]) -> tuple:
    """
    Resolve ``degree_min`` and ``degree_max`` from a run ``config.json``.

    Resolution order (mirrors ``st_lrps/st_lrps_evaluate.py``):
    1. Top-level ``degree_min`` / ``degree_max`` keys.
    2. ``dataset_meta.degree_min`` / ``dataset_meta.degree_max`` fallback.
    3. ``dataset_meta.requested_degree`` as a last resort for ``degree_max``.
    Raises ``ValueError`` when ``degree_max`` cannot be resolved.
    """

    dm = config.get("dataset_meta") or {}

    deg_min = config.get("degree_min")
    if deg_min is None:
        deg_min = dm.get("degree_min")

    deg_max = config.get("degree_max")
    if deg_max is None:
        deg_max = dm.get("degree_max")
    if deg_max is None:
        deg_max = dm.get("requested_degree")

    if deg_max is None:
        raise ValueError(
            "ST-LRPS model is missing degree metadata. "
            "Expected 'degree_max' in config.json at the top level or under 'dataset_meta'. "
            "Re-generate the dataset with spatial_cloud_generator.py >= v2.0 which writes "
            "degree_max to config.json automatically."
        )

    return int(deg_min if deg_min is not None else 0), int(deg_max)


def _config_path_value(config: Dict[str, Any], *keys: str) -> Optional[str]:
    """Return the first non-empty path-like value from config or dataset_meta."""

    mappings: list[Dict[str, Any]] = [config]
    dataset_meta = config.get("dataset_meta")
    if isinstance(dataset_meta, dict):
        mappings.append(dataset_meta)

    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        cloud_json = mapping.get("cloud_config_json")
        if isinstance(cloud_json, str) and cloud_json.strip():
            try:
                nested = json.loads(cloud_json)
            except Exception:
                nested = {}
            if isinstance(nested, dict):
                for key in keys:
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    return None


def _resolve_baseline_gravity_path(config: Dict[str, Any]) -> Path:
    """
    Resolve the SH coefficient file used for the ST-LRPS baseline.

    Older run configs may not carry the path explicitly.  In that case we use
    the surrogate pipeline SSOT default, which is the repository-local lunar
    JGGRX file used by the generator defaults.
    """

    path_value = _config_path_value(
        config,
        "gravity_model_path",
        "gfc_path",
        "gravity_gfc_path",
        "gravity_file_path",
    )
    if path_value:
        return resolve_lunar_gravity_path(path_value)
    return resolve_lunar_gravity_path(getattr(DEFAULT_DATASET_CONFIG, "gravity_gfc_path"))


def _looks_like_lunar_run(path: Path) -> bool:
    """
    Return ``True`` when run metadata clearly targets the Moon.

    Discovery helpers should avoid auto-selecting old Earth-era experiments,
    even if those folders still contain syntactically valid checkpoints.
    """

    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(looks_like_lunar_run_config(cfg))


def discover_st_lrps_model_dirs(root: Path | str = DEFAULT_ST_LRPS_RUNS_DIR) -> list[Path]:
    """
    Discover available surrogate gravity run directories.

    Results are sorted by modification time (newest first) so the UI can offer
    the most recent trained model as the first suggestion.
    """

    runs_root = Path(root).expanduser().resolve()
    if not runs_root.is_dir():
        return []

    candidates = [
        p.resolve()
        for p in runs_root.iterdir()
        if _is_valid_surrogate_run(p) and _looks_like_lunar_run(p)
    ]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates


def find_latest_st_lrps_model_dir(root: Path | str = DEFAULT_ST_LRPS_RUNS_DIR) -> Optional[Path]:
    """Return the newest valid surrogate run directory, if any."""

    candidates = discover_st_lrps_model_dirs(root)
    return candidates[0] if candidates else None


# =============================================================================
# 2.                          SCALER NORMALIZATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class _ScaleVector:
    """
    Scaling parameters for one quantity.

    ``scale`` may be either:
    - shape ``(1,)``  -> isotropic / scalar scaling
    - shape ``(N,)``  -> legacy per-axis scaling
    """

    mean: np.ndarray
    scale: np.ndarray

    @property
    def is_isometric(self) -> bool:
        return int(self.scale.size) == 1


@dataclass(frozen=True, slots=True)
class _ScalerBundle:
    """Normalized view of the artifact scaler pack."""

    x: _ScaleVector
    u: _ScaleVector
    a: Optional[_ScaleVector]


def _normalize_scale_mapping(mapping: Dict[str, Any], expected_dim: int, name: str) -> _ScaleVector:
    """
    Normalize legacy/new scaler JSON into a common in-memory representation.

    Legacy runs store ``std`` arrays while newer residual runs store a single
    ``scale`` value. The runtime keeps both formats alive so older experiments
    remain usable from the desktop app.
    """

    if "mean" not in mapping:
        raise ValueError(f"Scaler entry '{name}' is missing 'mean'.")

    mean = np.asarray(mapping["mean"], dtype=np.float64).reshape(-1)
    if mean.size != expected_dim:
        raise ValueError(
            f"Scaler entry '{name}.mean' must have {expected_dim} values, got {mean.size}."
        )

    raw_scale = mapping.get("scale", mapping.get("std"))
    if raw_scale is None:
        raise ValueError(f"Scaler entry '{name}' is missing 'scale'/'std'.")

    scale = np.asarray(raw_scale, dtype=np.float64).reshape(-1)
    if scale.size not in (1, expected_dim):
        raise ValueError(
            f"Scaler entry '{name}.scale' must be scalar or length {expected_dim}, got {scale.size}."
        )
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"Scaler entry '{name}.scale' must contain positive finite values.")

    return _ScaleVector(mean=mean, scale=scale)


def _load_scaler_bundle(model_dir: Path, checkpoint_obj: Dict[str, Any]) -> _ScalerBundle:
    """
    Load scaler metadata from checkpoint first, then ``scaler.json`` as fallback.

    Checkpoints tend to be the most self-consistent source because they are
    written at training time together with the model weights.
    """

    scaler_obj = checkpoint_obj.get("scaler")
    if not isinstance(scaler_obj, dict):
        scaler_path = model_dir / "scaler.json"
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
        scaler_obj = json.loads(scaler_path.read_text(encoding="utf-8"))

    x = _normalize_scale_mapping(dict(scaler_obj.get("x", {})), expected_dim=3, name="x")
    u = _normalize_scale_mapping(dict(scaler_obj.get("u", {})), expected_dim=1, name="u")
    a_raw = scaler_obj.get("a")
    a = (
        _normalize_scale_mapping(dict(a_raw), expected_dim=3, name="a")
        if isinstance(a_raw, dict)
        else None
    )
    return _ScalerBundle(x=x, u=u, a=a)


# =============================================================================
# 3.                         NETWORK ARCHITECTURE
# =============================================================================


class Sine(nn.Module):
    """SIREN activation used by some surrogate runs."""

    def __init__(self, w0: float = 30.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
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

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        return self.net(x_scaled)


class MLP(nn.Module):
    """Legacy activation-based MLP used by older checkpoints."""

    def __init__(self, *, in_dim: int, hidden: int, depth: int, activation: str, dropout: float) -> None:
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
                "Unsupported surrogate activation. Expected one of: sine, silu, tanh, softplus. "
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

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
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

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
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

    def __init__(self, *, backbone: nn.Module, embedding: Optional[FourierInputEmbedding]) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding = embedding

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        if self.embedding is not None:
            x_scaled = self.embedding(x_scaled)
        return self.backbone(x_scaled)


def _build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the network architecture encoded in ``config.json``."""

    if cfg.get("architecture") in ("MultiScale", "Residual") or int(cfg.get("n_bands", 1)) > 1:
        raise ValueError(
            "This legacy surrogate provider does not support MultiScale or advanced Residual models. "
            "Please use the st_lrps module."
        )

    activation = str(cfg.get("activation", "sine")).strip().lower()
    hidden = int(cfg.get("hidden", 256))
    depth = int(cfg.get("depth", 4))
    dropout = float(cfg.get("dropout", 0.0) or 0.0)
    use_fourier = bool(cfg.get("use_fourier", False))

    embedding: Optional[FourierInputEmbedding] = None
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


def _extract_state_dict(checkpoint_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the model state dictionary from a checkpoint payload."""

    for key in ("model", "model_state", "state_dict"):
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("Checkpoint does not contain a model state dictionary.")


def _load_checkpoint(path: Path, device: "torch.device") -> Dict[str, Any]:
    """Load a checkpoint with compatibility across PyTorch versions."""

    try:
        obj = torch.load(path, map_location=device, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(obj)!r}")
    return obj


# =============================================================================
# 4.                        PUBLIC RUNTIME PROVIDER
# =============================================================================


@dataclass(frozen=True, slots=True)
class SurrogateGravityMetadata:
    """User-facing summary of a loaded surrogate gravity run."""

    model_dir: str
    training_mode: str
    scaler_kind: str
    activation: str
    hidden: int
    depth: int
    a_sign: float
    mu_m3s2: float
    r_ref_m: float
    device: str


class SurrogateGravityModel:
    """
    Runtime gravity provider backed by a neural potential surrogate.

    The provider exposes ``acceleration_fixed(...)`` so the dynamics engine can
    evaluate it in body-fixed coordinates and rotate the result back into the
    inertial propagation frame.
    """

    model_kind = "st_lrps"

    def __init__(
        self,
        *,
        model_dir: Path,
        model: nn.Module,
        device: "torch.device",
        scaler: _ScalerBundle,
        training_mode: str,
        a_sign: float,
        mu_m3s2: float,
        r_ref_m: float,
        config: Dict[str, Any],
        baseline_gravity_model: Optional[Any] = None,
        baseline_gravity_path: Optional[Path] = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.model = model
        self.device = device
        self.scaler = scaler
        self.training_mode = str(training_mode)
        self.a_sign = float(a_sign)
        self.GM_m3s2 = float(mu_m3s2)
        self.gm_m3s2 = float(mu_m3s2)
        self.R_ref_m = float(r_ref_m)
        self.r_ref_m = float(r_ref_m)
        self.config = dict(config)
        self.baseline_gravity_model = baseline_gravity_model
        self.baseline_gravity_path = str(baseline_gravity_path) if baseline_gravity_path is not None else None
        self._baseline_torch_evaluator: Optional[Any] = None
        self._baseline_torch_signature: Optional[tuple[str, str, int]] = None

        # Degree metadata — required by core.propagator._get_sh_degree() and
        # MC result provenance.  Raised at construction time so the error fires
        # once at load, not N times inside the Monte Carlo sample loop.
        _deg_min, _deg_max = _extract_degree_metadata(config)
        self.degree_min: int = _deg_min
        self.degree_max: int = _deg_max
        self.base_degree: int = _deg_min        # SH baseline the surrogate sits on
        self.baseline_degree: int = _deg_min if baseline_gravity_model is not None else 0
        self.target_degree: int = _deg_max      # high-fidelity SH equivalent
        self.effective_degree_max: int = _deg_max

        self._x_mean = torch.as_tensor(self.scaler.x.mean, device=self.device, dtype=torch.float32)
        self._x_scale = torch.as_tensor(self.scaler.x.scale, device=self.device, dtype=torch.float32)
        self._u_mean = torch.as_tensor(self.scaler.u.mean, device=self.device, dtype=torch.float32)
        self._u_scale = torch.as_tensor(self.scaler.u.scale, device=self.device, dtype=torch.float32)
        self._mu_tensor = torch.tensor(float(mu_m3s2), device=self.device, dtype=torch.float32)

        self.metadata = SurrogateGravityMetadata(
            model_dir=str(self.model_dir),
            training_mode=self.training_mode,
            scaler_kind=("isometric" if self.scaler.x.is_isometric else "legacy_zscore"),
            activation=str(self.config.get("activation", "sine")),
            hidden=int(self.config.get("hidden", 256)),
            depth=int(self.config.get("depth", 4)),
            a_sign=float(self.a_sign),
            mu_m3s2=float(self.GM_m3s2),
            r_ref_m=float(self.R_ref_m),
            device=str(self.device),
        )
        self._validate_body_compatibility()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        *,
        mu_override: Optional[float] = None,
        r_ref_override: Optional[float] = None,
        device_preference: str = "cpu",
    ) -> "SurrogateGravityModel":
        """
        Load a surrogate gravity provider from a trained run directory.

        Parameters
        ----------
        model_dir:
            Directory containing ``config.json`` and ``checkpoints/ckpt_best.pt``
            (or ``ckpt_last.pt`` for in-progress runs, loaded with a warning).
        mu_override / r_ref_override:
            Optional mission-side overrides used when the artifact does not carry
            explicit central-body metadata.
        device_preference:
            ``"cpu"``, ``"cuda"``, or ``"auto"``. The desktop app defaults to
            CPU because solver RHS calls are fine-grained and frequent.
        """

        if torch is None:  # pragma: no cover - depends on machine setup
            raise ImportError(
                "PyTorch is required for surrogate gravity inference but could not be imported."
            ) from _TORCH_IMPORT_ERROR

        run_dir = Path(model_dir).expanduser().resolve()
        if not _is_valid_surrogate_run(run_dir):
            raise FileNotFoundError(
                "Surrogate gravity run directory is incomplete. "
                f"Expected config.json and checkpoints/ckpt_best.pt (or ckpt_last.pt) "
                f"under: {run_dir}"
            )

        device = cls._select_device(device_preference)
        cfg_path = run_dir / "config.json"
        ckpt_path = _find_checkpoint_for_run(run_dir)

        if ckpt_path.name == "ckpt_last.pt":
            import warnings as _warnings
            _warnings.warn(
                f"[ST-LRPS] Loading from ckpt_last.pt in {run_dir} — "
                "training may be incomplete. ckpt_best.pt not found.",
                RuntimeWarning,
                stacklevel=2,
            )

        cfg_file = json.loads(cfg_path.read_text(encoding="utf-8"))
        checkpoint_obj = _load_checkpoint(ckpt_path, device=device)
        cfg_ckpt = checkpoint_obj.get("config")
        config = dict(cfg_file)
        if isinstance(cfg_ckpt, dict):
            config.update(cfg_ckpt)

        scaler = _load_scaler_bundle(run_dir, checkpoint_obj)
        training_mode = cls._infer_training_mode(config, scaler)
        a_sign = float(config.get("resolved_a_sign", config.get("a_sign", 1.0)) or 1.0)

        mu_guess = config.get("resolved_mu_si")
        if mu_guess is None:
            mu_guess = mu_override if mu_override is not None else MU_MOON

        r_ref_guess = config.get("resolved_r_ref_m")
        if r_ref_guess is None:
            r_ref_guess = config.get("r_ref_m", config.get("r_ref_m_fallback"))
        if r_ref_guess is None:
            r_ref_guess = r_ref_override if r_ref_override is not None else R_MOON

        model = _build_model_from_config(config).to(device=device, dtype=torch.float32)
        model.load_state_dict(_extract_state_dict(checkpoint_obj), strict=True)
        model.eval()

        baseline_model = None
        baseline_path = None
        deg_min, _deg_max = _extract_degree_metadata(config)
        if training_mode == "residual_potential" and int(deg_min) >= 2:
            baseline_path = _resolve_baseline_gravity_path(config)
            baseline_model = GravityModel.from_file(
                str(baseline_path),
                requested_degree=int(deg_min),
            )
            logger.info(
                "ST-LRPS residual run uses SH%d baseline from %s; "
                "total acceleration is SH(degree_min) + neural residual.",
                int(deg_min),
                baseline_path,
            )

        return cls(
            model_dir=run_dir,
            model=model,
            device=device,
            scaler=scaler,
            training_mode=training_mode,
            a_sign=a_sign,
            mu_m3s2=float(mu_guess),
            r_ref_m=float(r_ref_guess),
            config=config,
            baseline_gravity_model=baseline_model,
            baseline_gravity_path=baseline_path,
        )

    @staticmethod
    def _select_device(preference: str) -> "torch.device":
        """Resolve the requested inference device."""

        pref = str(preference or "cpu").strip().lower()
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for surrogate gravity but is not available.")
            return torch.device("cuda")
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    @staticmethod
    def _infer_training_mode(config: Dict[str, Any], scaler: _ScalerBundle) -> str:
        """
        Infer whether the model predicts the full potential or a residual.

        Newer residual models store ``scale`` fields and/or resolved central-body
        metadata. Older checkpoints in this repository store ``std`` fields and
        were trained against the full potential directly.
        """

        explicit = str(config.get("potential_target_mode", "") or "").strip().lower()
        if explicit in {"absolute_potential", "residual_potential"}:
            return explicit

        if "resolved_mu_si" in config or "scaler_kind" in config or scaler.x.is_isometric:
            return "residual_potential"
        return "absolute_potential"

    def _validate_body_compatibility(self) -> None:
        """
        Fail fast when artifact statistics are wildly inconsistent with the body.

        The repository currently contains experimental runs with thin metadata.
        A simple scale sanity check is better than silently using an Earth-like
        model inside a lunar propagator and producing believable-but-wrong
        accelerations.
        """

        resolved_mu = self.config.get("resolved_mu_si")
        if resolved_mu is not None:
            resolved_mu = float(resolved_mu)
            rel_err = abs(resolved_mu - float(self.GM_m3s2)) / max(abs(float(self.GM_m3s2)), 1.0)
            if rel_err > 0.20:
                raise ValueError(
                    "Surrogate gravity artifact central-body GM does not match the active simulation body. "
                    f"artifact={resolved_mu:.6e} m^3/s^2, active={float(self.GM_m3s2):.6e} m^3/s^2"
                )

        if self.training_mode == "absolute_potential":
            mean_u = abs(float(self.scaler.u.mean[0]))
            reference_u = abs(float(self.GM_m3s2) / max(float(self.R_ref_m), 1.0))
            if reference_u > 0.0:
                ratio = mean_u / reference_u
                if ratio > 10.0 or ratio < 0.1:
                    raise ValueError(
                        "Surrogate gravity artifact looks incompatible with the active central body. "
                        f"|mean(U)|/mu_over_rref ratio={ratio:.3f} is outside the accepted range."
                    )

    # ------------------------------------------------------------------
    # Physics evaluation
    # ------------------------------------------------------------------
    def _scale_x(self, x_phys: "torch.Tensor") -> "torch.Tensor":
        return (x_phys - self._x_mean) / self._x_scale

    def _unscale_u(self, u_scaled: "torch.Tensor") -> "torch.Tensor":
        return u_scaled * self._u_scale + self._u_mean

    def _base_potential(self, x_phys: "torch.Tensor") -> "torch.Tensor":
        # Potential is rarely consumed by the propagator.  We keep the monopole
        # potential as a conservative scalar baseline while acceleration below
        # uses the physically required SH(degree_min) baseline.
        r = torch.linalg.norm(x_phys, dim=1, keepdim=True).clamp_min(1.0)
        return self.a_sign * self._mu_tensor / r

    def _point_mass_acceleration(self, x_phys: "torch.Tensor") -> "torch.Tensor":
        r = torch.linalg.norm(x_phys, dim=1, keepdim=True).clamp_min(1.0)
        return (-self._mu_tensor.to(device=x_phys.device, dtype=x_phys.dtype) * x_phys) / (r * r * r)

    def _base_acceleration(self, x_phys: "torch.Tensor") -> "torch.Tensor":
        """Return the CPU-safe baseline acceleration for residual models."""

        if self.baseline_gravity_model is None:
            return self._point_mass_acceleration(x_phys)

        pos_np = x_phys.detach().cpu().numpy().astype(np.float64, copy=False)
        out = np.empty((pos_np.shape[0], 3), dtype=np.float64)
        degree = int(self.baseline_degree)
        for idx, row in enumerate(pos_np):
            out[idx, :] = self.baseline_gravity_model.accel_fixed(row, degree=degree)
        return torch.as_tensor(out, device=x_phys.device, dtype=x_phys.dtype)

    def _base_acceleration_torch(self, x_phys: "torch.Tensor") -> "torch.Tensor":
        """Return batched SH(degree_min) baseline acceleration on the tensor device."""

        if self.baseline_gravity_model is None:
            return self._point_mass_acceleration(x_phys)

        degree = int(self.baseline_degree)
        signature = (str(x_phys.device), str(x_phys.dtype), degree)
        if self._baseline_torch_signature != signature or self._baseline_torch_evaluator is None:
            self._baseline_torch_evaluator = TorchSHGravityEvaluator(
                self.baseline_gravity_model,
                degree=degree,
                device=x_phys.device,
                dtype=x_phys.dtype,
            )
            self._baseline_torch_signature = signature
        return self._baseline_torch_evaluator.acceleration(x_phys)

    def predict_potential_and_acceleration_fixed(
        self,
        positions_m: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict gravitational potential and acceleration in the body-fixed frame.

        Parameters
        ----------
        positions_m:
            Either one position ``(3,)`` or a batch ``(N, 3)`` in meters.
        """

        if torch is None:  # pragma: no cover - depends on machine setup
            raise RuntimeError("PyTorch is not available.")

        pos = np.asarray(positions_m, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(1, 3)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions_m must be shape (3,) or (N,3), got {pos.shape}")

        x_phys = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        with torch.enable_grad():
            x_scaled = self._scale_x(x_phys).requires_grad_(True)
            u_scaled = self.model(x_scaled)
            grad_u_scaled = torch.autograd.grad(
                outputs=u_scaled,
                inputs=x_scaled,
                grad_outputs=torch.ones_like(u_scaled),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

            grad_u_phys = grad_u_scaled * (self._u_scale / self._x_scale)
            accel = self.a_sign * grad_u_phys
            potential = self._unscale_u(u_scaled)

            if self.training_mode == "residual_potential":
                accel = accel + self._base_acceleration(x_phys)
                potential = potential + self._base_potential(x_phys)

        u_np = potential.detach().cpu().numpy().astype(np.float64, copy=False)
        a_np = accel.detach().cpu().numpy().astype(np.float64, copy=False)
        return u_np, a_np

    def acceleration_fixed(
        self,
        x_m: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return one body-fixed acceleration vector ``(3,)`` in ``m/s²``."""

        _u, a = self.predict_potential_and_acceleration_fixed(x_m)
        return np.asarray(a[0], dtype=np.float64)

    def acceleration_fixed_batch(
        self,
        positions_m: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return body-fixed acceleration vectors ``(N, 3)`` in ``m/s²``."""

        _u, a = self.predict_potential_and_acceleration_fixed(positions_m)
        return np.asarray(a, dtype=np.float64)

    # ------------------------------------------------------------------
    # Batched GPU inference — torch tensor I/O
    # ------------------------------------------------------------------

    def to_device(self, device: "torch.device") -> None:
        """
        Move the model and all cached scaling tensors to *device* in-place.

        Call this once before starting a GPU MC run to transfer everything to
        CUDA.  After the call, ``predict_residual_accel_torch`` and
        ``predict_total_accel_torch`` will run natively on that device with no
        host-device transfers per call.
        """

        self.model = self.model.to(device=device)
        # Reassign in-place: tensors are not registered as nn.Module parameters
        # so model.to() does not move them automatically.
        self._x_mean = self._x_mean.to(device=device)
        self._x_scale = self._x_scale.to(device=device)
        self._u_mean = self._u_mean.to(device=device)
        self._u_scale = self._u_scale.to(device=device)
        self._mu_tensor = self._mu_tensor.to(device=device)
        self._baseline_torch_evaluator = None
        self._baseline_torch_signature = None
        # Update the stored device so _scale_x and _base_acceleration stay
        # consistent with future inputs.
        object.__setattr__(self, "device", device)  # bypass any frozen guard

    def predict_residual_accel_torch(
        self,
        x_m: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Return the neural residual acceleration ΔA in m/s² as a torch tensor.

        Parameters
        ----------
        x_m : torch.Tensor
            Body-fixed position batch, shape ``(N, 3)``, float32.
            The tensor may live on any device; it is moved to the model device
            automatically.

        Returns
        -------
        torch.Tensor
            Residual acceleration ``Δa = a_sign * ∇ΔU``, shape ``(N, 3)``,
            float32, on the same device as the model.

        Notes
        -----
        - Uses ``torch.autograd.grad`` with ``create_graph=False`` for
          inference-only operation (no double-diff overhead).
        - Requires ``torch.enable_grad()`` because autograd is called.
        - Does **not** include the SH(degree_min) baseline; use
          ``predict_total_accel_torch`` for the full acceleration.
        """

        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is not available.")

        out_dtype = x_m.dtype if x_m.is_floating_point() else torch.float32
        x = x_m.to(device=self.device, dtype=torch.float32)

        with torch.enable_grad():
            x_scaled = self._scale_x(x).requires_grad_(True)
            u_scaled = self.model(x_scaled)
            (grad_u_scaled,) = torch.autograd.grad(
                outputs=(u_scaled.sum(),),
                inputs=(x_scaled,),
                create_graph=False,
                retain_graph=False,
            )

        grad_u_phys = grad_u_scaled * (self._u_scale / self._x_scale)
        return (self.a_sign * grad_u_phys).detach().to(dtype=out_dtype)

    def predict_total_accel_torch(
        self,
        x_m: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Return the total acceleration (SH baseline + neural residual) in m/s².

        This is the GPU-tensor equivalent of ``acceleration_fixed_batch``.
        The result matches the CPU path: for ``residual_potential`` mode the
        SH(degree_min) baseline is added to the neural correction.

        Parameters
        ----------
        x_m : torch.Tensor
            Body-fixed position batch, shape ``(N, 3)``, float32.

        Returns
        -------
        torch.Tensor
            Total acceleration, shape ``(N, 3)``, float32, on the model device.
        """

        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is not available.")

        out_dtype = x_m.dtype if x_m.is_floating_point() else torch.float32
        x = x_m.to(device=self.device, dtype=out_dtype)
        delta_a = self.predict_residual_accel_torch(x_m)

        if self.training_mode == "residual_potential":
            a_base = self._base_acceleration_torch(x)
            return (delta_a + a_base).detach().to(dtype=out_dtype)
        return delta_a.detach().to(dtype=out_dtype)


__all__ = [
    "DEFAULT_ST_LRPS_RUNS_DIR",
    "SurrogateGravityMetadata",
    "SurrogateGravityModel",
    "discover_st_lrps_model_dirs",
    "find_checkpoint_for_st_lrps_run",
    "find_latest_st_lrps_model_dir",
]
