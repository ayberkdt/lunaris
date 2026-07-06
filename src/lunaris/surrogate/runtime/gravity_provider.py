"""Production gravity-provider wrapper for ST-LRPS surrogate artifacts."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator
from lunaris.surrogate.runtime.artifact import _find_checkpoint_for_run, _is_valid_surrogate_run
from lunaris.surrogate.runtime.device import _TORCH_IMPORT_ERROR, nn, torch
from lunaris.surrogate.runtime.metadata import (
    SurrogateGravityMetadata,
    _extract_degree_metadata,
    _resolve_baseline_gravity_path,
)
from lunaris.surrogate.runtime.networks import (
    _build_model_from_config,
    _extract_state_dict,
    _load_checkpoint,
)
from lunaris.surrogate.runtime.scalers import _load_scaler_bundle, _ScalerBundle

logger = logging.getLogger(__name__)


def _coerce_torch_dtype(dtype: Any, default: Any | None = None) -> Any:
    if torch is None:  # pragma: no cover - import guard
        return None
    if dtype is None:
        return default if default is not None else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).replace("torch.", "").strip().lower()
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported surrogate runtime dtype {dtype!r}; expected float32 or float64.")


def _module_dtype(model: Any, default: Any | None = None) -> Any:
    if torch is None:  # pragma: no cover - import guard
        return None
    resolved_default = default if default is not None else torch.float32
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return resolved_default


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _cfg_values_differ(file_val: Any, ckpt_val: Any) -> bool:
    """Compare config values across the JSON/pickle boundary.

    ``config.json`` round-trips through JSON (tuples become lists, numeric types
    may shift), so a naive ``!=`` would report false divergence for equal
    content. Values are divergent only when plain equality AND a canonical JSON
    dump both disagree.
    """
    if file_val == ckpt_val:
        return False
    try:
        return json.dumps(file_val, sort_keys=True, default=str) != json.dumps(
            ckpt_val, sort_keys=True, default=str
        )
    except (TypeError, ValueError):  # pragma: no cover - non-serializable payloads
        return True


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
        device: torch.device,
        scaler: _ScalerBundle,
        training_mode: str,
        a_sign: float,
        mu_m3s2: float,
        r_ref_m: float,
        config: dict[str, Any],
        baseline_gravity_model: Any | None = None,
        baseline_gravity_path: Path | None = None,
        force_runtime: Any | None = None,
        strict_domain: bool = False,
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
        self._force_runtime = force_runtime
        self._baseline_torch_evaluator: Any | None = None
        self._baseline_torch_signature: tuple[str, str, int] | None = None
        self._torch_dtype = _module_dtype(model, torch.float32)

        # Domain guard for the GPU tensor path. The canonical ``_fixed`` runtime
        # enforces the training-altitude envelope, but the tensor entry points
        # (``predict_*_accel_torch``) used by the GPU batch propagator did not,
        # so a trajectory leaving the training shell could silently extrapolate.
        # Resolve the envelope here (prefer the canonical runtime's resolved
        # bounds) and guard every torch prediction below.
        self.strict_domain = bool(strict_domain)
        self._warned_out_of_domain = False
        self._alt_min_km, self._alt_max_km = self._resolve_altitude_envelope(config, force_runtime)

        # Degree metadata — required by core.propagation.propagator._get_sh_degree() and
        # batch result provenance.  Raised at construction time so the error fires
        # once at load, not N times inside the batch sample loop.
        _deg_min, _deg_max = _extract_degree_metadata(config)
        self.degree_min: int = _deg_min
        self.degree_max: int = _deg_max
        self.base_degree: int = _deg_min        # SH baseline the surrogate sits on
        self.baseline_degree: int = _deg_min if baseline_gravity_model is not None else 0
        self.target_degree: int = _deg_max      # high-fidelity SH equivalent
        self.effective_degree_max: int = _deg_max

        self._x_mean = torch.as_tensor(self.scaler.x.mean, device=self.device, dtype=self._torch_dtype)
        self._x_scale = torch.as_tensor(self.scaler.x.scale, device=self.device, dtype=self._torch_dtype)
        self._u_mean = torch.as_tensor(self.scaler.u.mean, device=self.device, dtype=self._torch_dtype)
        self._u_scale = torch.as_tensor(self.scaler.u.scale, device=self.device, dtype=self._torch_dtype)
        self._mu_tensor = torch.tensor(float(mu_m3s2), device=self.device, dtype=self._torch_dtype)

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
        mu_override: float | None = None,
        r_ref_override: float | None = None,
        device_preference: str = "cpu",
        strict_domain: bool = False,
    ) -> SurrogateGravityModel:
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
            # The checkpoint's embedded config wins the merge, but a divergence
            # from config.json is a provenance problem (edited config.json,
            # mixed-up run dirs, ...) and must not pass silently.
            diverged = sorted(
                key
                for key, ckpt_val in cfg_ckpt.items()
                if key in cfg_file and _cfg_values_differ(cfg_file[key], ckpt_val)
            )
            if diverged:
                shown = ", ".join(diverged[:8]) + (", ..." if len(diverged) > 8 else "")
                warnings.warn(
                    "[ST-LRPS] config.json and the checkpoint's embedded config "
                    f"disagree on {len(diverged)} key(s) ({shown}); the checkpoint "
                    f"values take precedence. Run dir: {run_dir}",
                    RuntimeWarning,
                    stacklevel=2,
                )
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

        # Declared runtime kind decides whether a legacy fallback is even legal.
        # The legacy local path (_build_model_from_config) only ever builds a
        # scalar-potential MLP and evaluates it via autograd. That is the WRONG
        # physics for a force_direct artifact (which predicts residual
        # acceleration directly from a 3-output head), so a force_direct artifact
        # must NEVER silently fall back to it — it has to load through the
        # canonical runtime or fail loudly.
        declared_runtime_kind = str(
            config.get("runtime_model_kind")
            or (config.get("artifact_contract") or {}).get("runtime_model_kind")
            or ""
        ).strip().lower()

        force_runtime = None
        try:
            from lunaris.surrogate.runtime.force_runtime import load_force_runtime

            force_runtime = load_force_runtime(
                run_dir,
                device=str(device),
                chunk_size=8192,
            )
            model = force_runtime.model
            logger.info(
                "lunaris.surrogate.runtime delegated neural inference "
                "to st_lrps.runtime.force_model."
            )
        except Exception as exc:
            if declared_runtime_kind == "force_direct":
                raise RuntimeError(
                    "force_direct ST-LRPS artifacts are archived in the "
                    "experimental/force-direct-archive branch and are no longer loadable "
                    "on main: only the conservative 'potential_autograd' surrogate is "
                    "supported. There is deliberately NO legacy fallback (the legacy local "
                    "path builds a scalar-potential model, which is the wrong physics for a "
                    f"direct residual-acceleration artifact). Original error: {exc}"
                ) from exc
            # The legacy local path skips strict checkpoint-contract validation
            # (validate_checkpoint_contract), the frame guard, and the scaler
            # contract. That is only tolerable for pre-contract artifacts: an
            # artifact that *declares* a versioned contract must load through the
            # canonical runtime or fail loudly, never degrade to an unvalidated
            # model.
            contract_era = bool(
                config.get("artifact_contract")
                or config.get("target_contract")
                or declared_runtime_kind
            )
            if contract_era:
                raise RuntimeError(
                    "ST-LRPS artifact declares a versioned contract "
                    "(artifact_contract / target_contract / runtime_model_kind) but "
                    "could not be loaded through the canonical runtime "
                    "(st_lrps.runtime.force_model). Refusing the legacy fallback, "
                    "which would skip strict contract validation. Fix the artifact "
                    f"or environment instead of degrading. Original error: {exc}"
                ) from exc
            warnings.warn(
                "[ST-LRPS] Canonical runtime unavailable for pre-contract artifact "
                f"{run_dir}; falling back to the legacy local runtime path "
                "(potential_autograd only). Strict checkpoint-contract validation "
                f"is SKIPPED on this path. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(
                "Could not initialize canonical st_lrps.runtime.force_model adapter; "
                "falling back to the legacy local runtime path (potential_autograd only): %s",
                exc,
            )
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
            force_runtime=force_runtime,
            strict_domain=strict_domain,
        )

    @staticmethod
    def _select_device(preference: str) -> torch.device:
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
    def _infer_training_mode(config: dict[str, Any], scaler: _ScalerBundle) -> str:
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

    @property
    def is_conservative(self) -> bool:
        """Whether the surrogate acceleration is the gradient of a scalar potential.

        Main supports only the conservative ``potential_autograd`` surrogate, so
        this is True in practice; it still mirrors the runtime taxonomy flag on
        the wrapped runtime rather than relying on ``isinstance()`` checks.
        """
        fr = self._force_runtime
        if fr is not None and hasattr(fr, "is_conservative"):
            return bool(fr.is_conservative)
        return True

    # ------------------------------------------------------------------
    # Physics evaluation
    # ------------------------------------------------------------------
    def _scale_x(self, x_phys: torch.Tensor) -> torch.Tensor:
        return (x_phys - self._x_mean) / self._x_scale

    def _unscale_u(self, u_scaled: torch.Tensor) -> torch.Tensor:
        return u_scaled * self._u_scale + self._u_mean

    @staticmethod
    def _resolve_altitude_envelope(
        config: dict[str, Any], force_runtime: Any | None
    ) -> tuple[float | None, float | None]:
        """Resolve the surrogate's training altitude envelope [km].

        Prefers the canonical runtime's already-resolved bounds (single source of
        truth); otherwise reads the artifact config / contract / provenance.
        Returns ``(None, None)`` when no envelope is recorded, in which case the
        tensor-path guard cannot fire (it never blocks on unknown bounds).
        """
        if force_runtime is not None:
            lo = getattr(force_runtime, "_train_alt_min_km", None)
            hi = getattr(force_runtime, "_train_alt_max_km", None)
            if lo is not None and hi is not None:
                return float(lo), float(hi)

        sources = [config]
        for key in ("artifact_contract", "provenance", "meta", "metadata"):
            sub = config.get(key)
            if isinstance(sub, dict):
                sources.append(sub)

        def _get(*keys: str) -> float | None:
            for src in sources:
                for k in keys:
                    v = src.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            continue
            return None

        return _get("altitude_min_km", "alt_min_km"), _get("altitude_max_km", "alt_max_km")

    def _enforce_domain_torch(self, x_m: torch.Tensor, *, caller: str) -> None:
        """Guard the tensor path against out-of-training-envelope extrapolation.

        R11: delegates to the canonical runtime's single-source envelope guard
        (same ~1 km slack, strict raise / warn-once semantics). This closes the
        gap where the GPU batch propagator's per-step ``predict_*_accel_torch``
        calls bypassed the domain check that the CPU ``_fixed`` path performs.
        """
        from lunaris.surrogate.st_lrps.runtime.canonical_runtime import (
            enforce_altitude_envelope_torch,
        )

        self._warned_out_of_domain = enforce_altitude_envelope_torch(
            x_m,
            r_ref_m=float(self.R_ref_m),
            alt_min_km=self._alt_min_km,
            alt_max_km=self._alt_max_km,
            strict=self.strict_domain,
            already_warned=self._warned_out_of_domain,
            caller=caller,
        )

    def _base_potential(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Geodesy potential baseline, consistent with :meth:`_base_acceleration`.

        The acceleration baseline for residual models is the SH field truncated at
        ``baseline_degree`` (``accel_fixed``), so the potential baseline MUST be the
        matching SH potential (``potential_fixed`` at the same degree), not the bare
        monopole. With ``a = +grad(U)`` holding for both, this keeps total potential
        and total acceleration self-consistent; the old monopole-only baseline made
        any energy / Hamiltonian / conservative-field check on the total field
        invalid. No ``a_sign`` factor: the baseline acceleration carries none either
        (``a_sign`` belongs to the neural residual).
        """
        if self.baseline_gravity_model is None:
            # Point-mass geodesy potential U = mu/r, whose +gradient is the monopole
            # acceleration -mu x / r^3 (matches ``_point_mass_acceleration``).
            r = torch.linalg.norm(x_phys, dim=1, keepdim=True).clamp_min(1.0)
            return self._mu_tensor / r

        pos_np = x_phys.detach().cpu().numpy().astype(np.float64, copy=False)
        out = np.empty((pos_np.shape[0], 1), dtype=np.float64)
        degree = int(self.baseline_degree)
        for idx, row in enumerate(pos_np):
            out[idx, 0] = float(self.baseline_gravity_model.potential_fixed(row, degree=degree))
        return torch.as_tensor(out, device=x_phys.device, dtype=x_phys.dtype)

    def _point_mass_acceleration(self, x_phys: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(x_phys, dim=1, keepdim=True).clamp_min(1.0)
        return (-self._mu_tensor.to(device=x_phys.device, dtype=x_phys.dtype) * x_phys) / (r * r * r)

    def _base_acceleration(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Return the CPU-safe baseline acceleration for residual models."""

        if self.baseline_gravity_model is None:
            return self._point_mass_acceleration(x_phys)

        pos_np = x_phys.detach().cpu().numpy().astype(np.float64, copy=False)
        out = np.empty((pos_np.shape[0], 3), dtype=np.float64)
        degree = int(self.baseline_degree)
        for idx, row in enumerate(pos_np):
            out[idx, :] = self.baseline_gravity_model.accel_fixed(row, degree=degree)
        return torch.as_tensor(out, device=x_phys.device, dtype=x_phys.dtype)

    def _base_acceleration_torch(self, x_phys: torch.Tensor) -> torch.Tensor:
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

        x_phys = torch.as_tensor(pos, device=self.device, dtype=self._torch_dtype)
        if self._force_runtime is not None:
            delta_u_np = np.asarray(self._force_runtime.predict_residual_potential(pos), dtype=np.float64).reshape(-1, 1)
            delta_a_np = np.asarray(self._force_runtime.predict_residual_accel(pos), dtype=np.float64).reshape(-1, 3)
            potential = torch.as_tensor(delta_u_np, device=self.device, dtype=self._torch_dtype)
            accel = torch.as_tensor(delta_a_np, device=self.device, dtype=self._torch_dtype)
            if self.training_mode == "residual_potential":
                accel = accel + self._base_acceleration(x_phys)
                potential = potential + self._base_potential(x_phys)
            return (
                potential.detach().cpu().numpy().astype(np.float64, copy=False),
                accel.detach().cpu().numpy().astype(np.float64, copy=False),
            )
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

    def to_device(self, device: torch.device, dtype: torch.dtype | str | None = None) -> None:
        """
        Move the model and all cached scaling tensors to *device* in-place.

        Call this once before starting a GPU batch run to transfer everything to
        CUDA.  After the call, ``predict_residual_accel_torch`` and
        ``predict_total_accel_torch`` will run natively on that device with no
        host-device transfers per call.
        """

        resolved_dtype = _coerce_torch_dtype(dtype, default=getattr(self, "_torch_dtype", torch.float32))
        self.model = self.model.to(device=device, dtype=resolved_dtype)
        # Reassign in-place: tensors are not registered as nn.Module parameters
        # so model.to() does not move them automatically.
        self._x_mean = self._x_mean.to(device=device, dtype=resolved_dtype)
        self._x_scale = self._x_scale.to(device=device, dtype=resolved_dtype)
        self._u_mean = self._u_mean.to(device=device, dtype=resolved_dtype)
        self._u_scale = self._u_scale.to(device=device, dtype=resolved_dtype)
        self._mu_tensor = self._mu_tensor.to(device=device, dtype=resolved_dtype)
        self._baseline_torch_evaluator = None
        self._baseline_torch_signature = None
        if self._force_runtime is not None:
            if hasattr(self._force_runtime, "to_device"):
                self._force_runtime.to_device(device, dtype=resolved_dtype)
                self.model = self._force_runtime.model
            else:
                self._force_runtime.device = device
                self._force_runtime.model = self.model
                if hasattr(self._force_runtime.scaler, "to_tensors"):
                    self._force_runtime.scaler.to_tensors(device=device, dtype=resolved_dtype)
        # Update the stored device so _scale_x and _base_acceleration stay
        # consistent with future inputs.
        object.__setattr__(self, "device", device)  # bypass any frozen guard
        self._torch_dtype = resolved_dtype

    def dtype_diagnostics(self, requested_dtype: torch.dtype | str | None = None) -> dict[str, Any]:
        """Return requested/effective dtype provenance for paper-safe diagnostics."""
        requested = _coerce_torch_dtype(requested_dtype, default=None) if requested_dtype is not None else None
        model_dtype = _module_dtype(self.model, getattr(self, "_torch_dtype", torch.float32))
        scaler_dtype = getattr(self._x_mean, "dtype", None)
        force_scaler = None
        if self._force_runtime is not None:
            force_scaler = getattr(getattr(self._force_runtime, "scaler", None), "_x_mean", None)
        force_scaler_dtype = getattr(force_scaler, "dtype", None)
        effective = model_dtype
        downgraded = bool(requested is not None and requested != effective)
        return {
            "requested_dtype": _dtype_name(requested) if requested is not None else None,
            "effective_dtype": _dtype_name(effective),
            "dtype_downgraded": downgraded,
            "model_dtype": _dtype_name(model_dtype),
            "scaler_dtype": _dtype_name(scaler_dtype),
            "force_runtime_scaler_dtype": (
                _dtype_name(force_scaler_dtype) if force_scaler_dtype is not None else None
            ),
        }

    def predict_residual_accel_torch(
        self,
        x_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the neural residual acceleration ΔA in m/s² as a torch tensor.

        Parameters
        ----------
        x_m : torch.Tensor
            Body-fixed position batch, shape ``(N, 3)``. The tensor may live on
            any device; it is moved to the model device and runtime dtype
            automatically.

        Returns
        -------
        torch.Tensor
            Residual acceleration ``Δa = a_sign * ∇ΔU``, shape ``(N, 3)``,
            in the caller's floating dtype, on the same device as the model.

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

        # Domain guard (covers predict_total_accel_torch too, which routes here).
        self._enforce_domain_torch(x_m, caller="predict_residual_accel_torch")

        out_dtype = x_m.dtype if x_m.is_floating_point() else self._torch_dtype
        x = x_m.to(device=self.device, dtype=self._torch_dtype)
        # Single source of truth for scaling: when the canonical runtime is
        # loaded, scale with ITS scaler (the model is already shared via
        # from_model_dir). This collapses the dual-scaler "second truth" — the
        # same chain rule the canonical runtime applies in its own autograd
        # (SurrogateForceModel._predict_chunk). gravity_provider's own scaler
        # tensors are used only on the legacy fallback path (_force_runtime is None).
        fr = self._force_runtime
        if fr is not None:
            scale_x = fr.scaler.scale_x
        else:
            scale_x = self._scale_x

        with torch.enable_grad():
            x_scaled = scale_x(x).requires_grad_(True)
            if fr is not None:
                u_over_x = fr.scaler._u_scale / fr.scaler._x_scale
            else:
                u_over_x = self._u_scale / self._x_scale
            u_scaled = self.model(x_scaled)
            (grad_u_scaled,) = torch.autograd.grad(
                outputs=(u_scaled.sum(),),
                inputs=(x_scaled,),
                create_graph=False,
                retain_graph=False,
            )

        grad_u_phys = grad_u_scaled * u_over_x
        return (self.a_sign * grad_u_phys).detach().to(dtype=out_dtype)

    def predict_total_accel_torch(
        self,
        x_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the total acceleration (SH baseline + neural residual) in m/s².

        This is the GPU-tensor equivalent of ``acceleration_fixed_batch``.
        The result matches the CPU path: for ``residual_potential`` mode the
        SH(degree_min) baseline is added to the neural correction.

        Parameters
        ----------
        x_m : torch.Tensor
            Body-fixed position batch, shape ``(N, 3)``; converted to the
            runtime dtype internally.

        Returns
        -------
        torch.Tensor
            Total acceleration, shape ``(N, 3)``, in the caller's floating
            dtype, on the model device.
        """

        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is not available.")

        out_dtype = x_m.dtype if x_m.is_floating_point() else self._torch_dtype
        x = x_m.to(device=self.device, dtype=self._torch_dtype)
        delta_a = self.predict_residual_accel_torch(x)

        if self.training_mode == "residual_potential":
            a_base = self._base_acceleration_torch(x)
            return (delta_a + a_base).detach().to(dtype=out_dtype)
        return delta_a.detach().to(dtype=out_dtype)

__all__ = ["SurrogateGravityMetadata", "SurrogateGravityModel"]
