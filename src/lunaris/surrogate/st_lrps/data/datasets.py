"""Dataset, HDF5 streaming, and strict lunar metadata validation."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from lunaris.surrogate.st_lrps.data.dataset_contract import (
    REQUIRED_DERIVATIVE_CONVENTION,
    DatasetContract,
    DatasetContractError,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    MU_MOON_SI,
    R_MOON_SI,
    is_lunar_body_signature,
)

logger = logging.getLogger(__name__)
DTYPE = torch.float32

def collate_xyz_u_a(
    batch: list[tuple[Any, Any, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pickle-safe collate for (x, u, a) samples from either dataset backend.

    Handles both:
      * ``np.ndarray`` items (``H5BlockDataset.__getitem__``) → stacked via
        ``torch.as_tensor(np.stack(...))`` (single copy, no per-sample tensor
        construction).
      * ``torch.Tensor`` items (``TensorMemoryDataset.__getitem__``) → stacked
        via ``torch.stack`` with zero NumPy round-trip, which keeps the fast
        all-in-RAM path tensor-native and ``pin_memory``-friendly.

    Returns float tensors of shape ``(B, 3)``, ``(B, 1)``, ``(B, 3)``.
    """
    first_x = batch[0][0]
    if isinstance(first_x, torch.Tensor):
        x = torch.stack([b[0] for b in batch], dim=0).to(DTYPE)
        u = torch.stack([b[1] for b in batch], dim=0).to(DTYPE)
        a = torch.stack([b[2] for b in batch], dim=0).to(DTYPE)
    else:
        x = torch.as_tensor(np.stack([b[0] for b in batch], axis=0), dtype=DTYPE)
        u = torch.as_tensor(np.stack([b[1] for b in batch], axis=0), dtype=DTYPE)
        a = torch.as_tensor(np.stack([b[2] for b in batch], axis=0), dtype=DTYPE)
    return x, u, a


# Backward-compatible alias: older code (and the engine) imports ``collate_h5``.
collate_h5 = collate_xyz_u_a


# --- Utilities ---

def _resolve_loader_worker_count(
    data_path: Path,
    requested_workers: int,
    *,
    os_name: str | None = None,
) -> int:
    """
    Return a safe DataLoader worker count for the dataset backend in use.

    Why this exists
    ---------------
    The surrogate training stack stores its clouds in HDF5. On Windows, that
    combination is fragile when ``h5py`` datasets are read from multiple worker
    processes, especially when compression filters are enabled. The observed
    failure mode is an intermittent

    ``OSError: Can't synchronously read data (filter returned failure during read)``

    during otherwise valid training runs. Since correctness matters more than
    marginal input-pipeline speed here, we automatically force single-process
    HDF5 loading on Windows. PT datasets remain free to use multiple workers.
    """

    workers = max(0, int(requested_workers))
    platform_name = os.name if os_name is None else str(os_name)
    if workers <= 0:
        return 0

    if Path(data_path).suffix.lower() in {".h5", ".hdf5"} and platform_name == "nt":
        return 0
    return workers


# --- Dataset Metadata ---

def _safe_float(d: dict[str, Any], key: str) -> float | None:
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _safe_str(d: dict[str, Any], key: str) -> str | None:
    val = d.get(key)
    if val is None:
        return None
    try:
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        s = str(val).strip()
        return s or None
    except (ValueError, TypeError, UnicodeDecodeError):
        return None

def _parse_bool(x: Any) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8")
        except UnicodeDecodeError:
            return None
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f"):
        return False
    return None


# --- Main Meta Class ---

@dataclass(frozen=True)
class DatasetMeta:
    """Metadata read from HDF5 attrs: units, physical constants, degree range, altitude bounds."""
    unit_system: str  # Expected: "si", "canonical", or "unknown"
    mu_si: float | None
    r_ref_m: float | None
    DU_m: float | None
    TU_s: float | None
    VU_m_s: float | None
    requested_degree: int | None
    degree_min: int | None
    gravity_model_path: str | None
    include_potential: bool | None
    raw_attrs: dict[str, Any]
    # Cloud generation parameters (from cloud_config_json HDF5 attr)
    alt_min_km: float | None = None
    alt_max_km: float | None = None
    cloud_config: dict[str, Any] | None = None
    degree_max: int | None = None
    target_mode: str | None = None
    a_sign_convention: str | None = None
    columns: str | None = None
    derivative_convention_version: str | None = None
    central_body: str | None = None

    def can_convert_to_si(self) -> bool:
        return bool(
            self.unit_system == "canonical"
            and self.DU_m is not None
            and self.TU_s is not None
            and self.VU_m_s is not None
            and np.isfinite(self.DU_m)
            and np.isfinite(self.TU_s)
            and np.isfinite(self.VU_m_s)
        )

    def convert_xyz_U_a_to_si(
        self, x: np.ndarray, u: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert canonical (nondimensional) arrays to SI. Returns copies; never mutates."""
        if self.unit_system != "canonical":
            return x, u, a
        if not self.can_convert_to_si():
            raise ValueError(
                "Dataset is marked as 'canonical' but is missing required "
                "scaling constants (DU_m, TU_s, VU_m_s). Cannot convert to SI."
            )

        # Guaranteed non-None by can_convert_to_si() above; assert narrows for mypy.
        assert self.DU_m is not None and self.TU_s is not None and self.VU_m_s is not None
        DU = float(self.DU_m)
        TU = float(self.TU_s)
        VU = float(self.VU_m_s)

        x_si = x * DU
        a_si = a * (DU / (TU * TU))
        u_si = u * (VU * VU)

        return x_si, u_si, a_si

    @classmethod
    def from_h5(cls, h5_path: Path) -> DatasetMeta:
        with h5py.File(h5_path, "r") as f:
            attrs = {str(k): f.attrs[k] for k in f.attrs.keys()}

        unit_system = str(attrs.get("unit_system", "unknown")).lower()
        if unit_system not in ("si", "canonical"):
            unit_system = "unknown"

        def _parse_int(key: str) -> int | None:
            val = attrs.get(key)
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        req_deg = _parse_int("requested_degree")
        deg_min = _parse_int("degree_min")
        deg_max = _parse_int("degree_max")

        # Parse cloud_config_json if present (written by spatial_cloud_generator)
        cloud_cfg: dict[str, Any] | None = None
        cloud_cfg_raw = attrs.get("cloud_config_json")
        if cloud_cfg_raw is not None:
            try:
                if isinstance(cloud_cfg_raw, bytes):
                    cloud_cfg_raw = cloud_cfg_raw.decode("utf-8")
                cloud_cfg = json.loads(str(cloud_cfg_raw))
            except Exception:
                cloud_cfg = None

        # Fallback: recover degree info from cloud_config_json for old datasets
        # that stored degree_min/max only inside the JSON blob
        if cloud_cfg is not None:
            if req_deg is None:
                try:
                    _v = cloud_cfg.get("degree_max")
                    if _v is not None:
                        # NOTE: do NOT use ``int(_v) or None`` — that turns a
                        # legitimate degree of 0 into None. Only treat negative
                        # as "absent"; keep 0 as a real (if unusual) value.
                        _iv = int(_v)
                        req_deg = _iv if _iv >= 0 else None
                except (TypeError, ValueError):
                    pass
            if deg_min is None:
                try:
                    _v = cloud_cfg.get("degree_min")
                    if _v is not None:
                        deg_min = int(_v)
                except (TypeError, ValueError):
                    pass
            if deg_max is None:
                try:
                    _v = cloud_cfg.get("degree_max")
                    if _v is not None:
                        deg_max = int(_v)
                except (TypeError, ValueError):
                    pass

        alt_min: float | None = None
        alt_max: float | None = None
        if cloud_cfg is not None:
            try:
                alt_min = float(cloud_cfg.get("alt_min_km", attrs.get("alt_min_km") or 0))
                alt_max = float(cloud_cfg.get("alt_max_km", attrs.get("alt_max_km") or 0))
            except (TypeError, ValueError):
                pass
        else:
            try:
                alt_min = float(attrs["alt_min_km"]) if "alt_min_km" in attrs else None
                alt_max = float(attrs["alt_max_km"]) if "alt_max_km" in attrs else None
            except (TypeError, ValueError):
                pass

        deriv_conv = _safe_str(attrs, "derivative_convention_version")
        if deriv_conv is None:
            logger.warning(
                "Dataset is missing 'derivative_convention_version'. "
                "If generated before the dP_dphi sign fix (derivative_convention_version="
                f"{REQUIRED_DERIVATIVE_CONVENTION!r}), the latitude acceleration components are "
                "sign-flipped and the dataset must be regenerated before training."
            )

        # Resolve central_body from attrs or cloud_config
        _cb_candidates = [
            attrs.get("central_body"),
            (cloud_cfg or {}).get("central_body") if cloud_cfg is not None else None,
        ]
        resolved_central_body: str | None = None
        for _cb in _cb_candidates:
            _cb_s = str(_cb or "").strip().lower()
            if _cb_s:
                resolved_central_body = _cb_s
                break

        return cls(
            unit_system=unit_system,
            mu_si=_safe_float(attrs, "mu_si"),
            r_ref_m=_safe_float(attrs, "r_ref_m"),
            DU_m=_safe_float(attrs, "DU_m"),
            TU_s=_safe_float(attrs, "TU_s"),
            VU_m_s=_safe_float(attrs, "VU_m_s"),
            requested_degree=req_deg,
            degree_min=deg_min,
            gravity_model_path=_safe_str(attrs, "gravity_model_path"),
            include_potential=_parse_bool(attrs.get("include_potential")),
            raw_attrs=attrs,
            alt_min_km=alt_min,
            alt_max_km=alt_max,
            cloud_config=cloud_cfg,
            degree_max=deg_max,
            target_mode=_safe_str(attrs, "target_mode"),
            a_sign_convention=_safe_str(attrs, "a_sign_convention"),
            columns=_safe_str(attrs, "columns"),
            derivative_convention_version=deriv_conv,
            central_body=resolved_central_body,
        )


# -----------------------------------------------------------------------------
# Dataset body-contract helpers
# -----------------------------------------------------------------------------
_LUNAR_ALIASES = {"moon", "lunar", "selene"}

def _normalized_dataset_body_name(meta: DatasetMeta) -> str | None:
    """
    Return the dataset's declared central-body name, if any.

    The value can come either from top-level HDF5 attributes or the serialized
    ``cloud_config_json`` block. Normalizing it in one place keeps training,
    evaluation, and runtime checks consistent.
    """

    candidates = [
        meta.raw_attrs.get("central_body"),
        (meta.cloud_config or {}).get("central_body") if meta.cloud_config is not None else None,
    ]
    for candidate in candidates:
        name = str(candidate or "").strip().lower()
        if name:
            return name
    return None

def _resolve_lunar_dataset_contract(meta: DatasetMeta, *, data_path: Path) -> tuple[str, float, float]:
    """
    Validate that an HDF5 dataset really belongs to the lunar surrogate stack.

    Why this exists
    ---------------
    Older experimental folders sometimes lacked complete metadata, and a prior
    version of the training script papered over that ambiguity by unconditionally
    stamping ``central_body="moon"`` into the output config. That was dangerous:
    an Earth-era dataset could be retrained and then silently promoted as a
    lunar surrogate artifact.

    This helper fail-fast validates the dataset before training begins:

    - explicit non-lunar body names are rejected immediately
    - explicit lunar names must agree with lunar GM / reference radius when
      those numeric fields are present
    - metadata-free datasets are accepted only when their numeric body
      signature still looks lunar
    """

    body_name = _normalized_dataset_body_name(meta)
    has_lunar_signature = is_lunar_body_signature(mu_si=meta.mu_si, r_ref_m=meta.r_ref_m)

    if body_name is not None and body_name not in _LUNAR_ALIASES:
        raise ValueError(
            f"Dataset {data_path.name!r} declares central_body={body_name!r}, which is not lunar. "
            "Refusing to train a Moon surrogate on a non-lunar dataset."
        )

    if body_name in _LUNAR_ALIASES and not has_lunar_signature and (
        meta.mu_si is not None or meta.r_ref_m is not None
    ):
        raise ValueError(
            f"Dataset {data_path.name!r} is labeled lunar but its body constants do not look lunar "
            f"(mu_si={meta.mu_si!r}, r_ref_m={meta.r_ref_m!r})."
        )

    if body_name is None and not has_lunar_signature:
        raise ValueError(
            f"Dataset {data_path.name!r} is missing reliable lunar metadata. "
            "Provide a dataset generated by st_lrps/spatial_cloud_generator.py."
        )

    resolved_body = "moon"
    resolved_mu = float(meta.mu_si) if meta.mu_si is not None else float(MU_MOON_SI)
    resolved_r_ref = float(meta.r_ref_m) if meta.r_ref_m is not None else float(R_MOON_SI)
    return resolved_body, resolved_mu, resolved_r_ref


def build_dataset_contract(
    meta: DatasetMeta,
    *,
    data_path: Path,
    n_samples: int | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the versioned dataset-contract block used by artifacts."""

    raw_contract = meta.raw_attrs.get("dataset_contract_json") or meta.raw_attrs.get("contract_json")
    if raw_contract:
        if isinstance(raw_contract, bytes):
            raw_contract = raw_contract.decode("utf-8")
        payload_obj = json.loads(str(raw_contract))
        contract = DatasetContract.from_dict(payload_obj)
        payload = contract.to_dict()
        if dataset_sha256 and not payload.get("content_sha256"):
            payload["content_sha256"] = dataset_sha256
        payload["path"] = str(Path(data_path))
        payload["dataset_sha256"] = payload.get("content_sha256")
        payload["derivative_convention_version"] = payload.get("derivative_convention")
        payload["alt_min_km"] = payload.get("altitude_min_km")
        payload["alt_max_km"] = payload.get("altitude_max_km")
        return payload

    raise DatasetContractError(
        f"Dataset {Path(data_path).name!r} is missing dataset_contract_json. "
        "Regenerate it with the current ST-LRPS generator."
    )


def read_dataset_contract_from_h5(
    h5_path: Path,
    *,
    dataset_name: str = "data",
) -> dict[str, Any]:
    """Read dataset metadata and return the normalized dataset contract."""

    path = Path(h5_path)
    return DatasetContract.from_hdf5(path, dataset_name=dataset_name).to_dict()


def validate_dataset_contract(
    meta: DatasetMeta,
    *,
    data_path: Path,
) -> dict[str, Any]:
    """Validate dataset metadata and return a normalized dataset contract."""

    validate_training_dataset_convention(
        meta,
        data_path=data_path,
    )
    return DatasetContract.from_hdf5(data_path).to_dict()


def validate_training_dataset_convention(
    meta: DatasetMeta,
    *,
    data_path: Path,
) -> None:
    """Fail-fast guard against training on a silently-wrong dataset.

    Checks (all raise ``ValueError`` on violation):
      * derivative_convention_version == "dP_dphi_corrected_v1". Datasets made
        before the dP_dphi sign fix have sign-flipped latitude acceleration; the
        model would learn an inverted field with no error signal.
      * central_body is lunar.
      * target_mode, when present, is "residual" or "full".
      * degree_max > degree_min when both are known.
      * a_sign_convention, when present, is parseable as +1/-1.

    Column-vs-mode consistency is a soft warning (logged), since column labels
    vary across generator versions.
    """
    name = Path(data_path).name

    # --- derivative convention (the dangerous, silent one) ---
    deriv = meta.derivative_convention_version
    if deriv != REQUIRED_DERIVATIVE_CONVENTION:
        msg = (
            f"Dataset {name!r} has derivative_convention_version={deriv!r}, expected "
            f"{REQUIRED_DERIVATIVE_CONVENTION!r}. Datasets generated before the dP_dphi "
            "sign fix have sign-flipped latitude acceleration components; training on "
            "them learns an inverted field with no error signal. Regenerate with the "
            "current spatial_cloud_generator.py."
        )
        raise ValueError(msg)

    # --- central body ---
    body = _normalized_dataset_body_name(meta)
    if body is not None and body not in _LUNAR_ALIASES:
        raise ValueError(
            f"Dataset {name!r} declares central_body={body!r}, which is not lunar."
        )

    # --- target_mode ---
    if meta.target_mode is None:
        msg = (
            f"Dataset {name!r} is missing target_mode. New datasets must declare "
            "whether labels are residual or full-field."
        )
        raise ValueError(msg)
    tmode = str(meta.target_mode).strip().lower()
    if tmode not in ("residual", "full"):
        raise ValueError(
            f"Dataset {name!r} has target_mode={meta.target_mode!r}; expected "
            "'residual' or 'full'."
        )

    # --- degree ordering ---
    dmax = meta.degree_max if meta.degree_max is not None else meta.requested_degree
    if meta.degree_min is None or dmax is None:
        msg = f"Dataset {name!r} is missing degree_min/degree_max metadata."
        raise ValueError(msg)
    if meta.degree_min is not None and dmax is not None:
        if int(dmax) <= int(meta.degree_min):
            raise ValueError(
                f"Dataset {name!r} has degree_max={dmax} <= degree_min={meta.degree_min}; "
                "a residual band requires degree_max > degree_min."
            )

    # --- units and altitude envelope ---
    if meta.unit_system not in ("si", "canonical"):
        raise ValueError(f"Dataset {name!r} has missing or unsupported unit_system={meta.unit_system!r}.")
    # Canonical datasets are non-dimensional; the SI field is ambiguous without
    # the DU/TU/VU scaling constants. Reject at preflight rather than failing
    # deep inside data loading (convert_xyz_U_a_to_si) after training has begun.
    if meta.unit_system == "canonical" and not meta.can_convert_to_si():
        raise ValueError(
            f"Dataset {name!r} declares unit_system='canonical' but is missing the "
            "DU_m/TU_s/VU_m_s scaling constants required to convert to SI; the physical "
            "field is ambiguous. Regenerate with explicit canonical scaling constants "
            "or provide an SI dataset."
        )
    if meta.alt_min_km is None or meta.alt_max_km is None:
        msg = f"Dataset {name!r} is missing altitude bounds."
        raise ValueError(msg)
    if float(meta.alt_max_km) <= float(meta.alt_min_km):
        raise ValueError(
            f"Dataset {name!r} has invalid altitude bounds: "
            f"{meta.alt_min_km} >= {meta.alt_max_km}."
        )

    # --- a_sign convention parseable ---
    if meta.a_sign_convention is not None:
        if str(meta.a_sign_convention).strip() not in ("+1", "1", "-1"):
            logger.warning(
                "Dataset %s has unrecognised a_sign_convention=%r; a_sign will be "
                "auto-inferred from data.", name, meta.a_sign_convention,
            )

    # --- columns vs mode (soft) ---
    if meta.columns is not None and meta.degree_min is not None:
        cols = str(meta.columns).lower()
        residual_cols = ("du" in cols or "dax" in cols)
        if int(meta.degree_min) >= 0 and not residual_cols and "[x,y,z,u,ax,ay,az]" in cols:
            logger.warning(
                "Dataset %s columns look full-field (%r) but degree_min=%s suggests "
                "residual. Verify generation parameters.",
                name, meta.columns, meta.degree_min,
            )


# --- Isometric scaling ---
# Gravitational gradient ∇U depends on Euclidean norm ‖r‖, not per-axis
# variances. A single global scale (max ‖x‖) preserves aspect ratio so
# the chain rule Δa = ∇(ΔU) stays isotropy-correct.

def _discover_dataset_name(h5_path: Path, preferred: str = "data") -> str:
    dataset_name = None

    def visitor(name: str, obj: object) -> bool | None:
        nonlocal dataset_name
        if isinstance(obj, h5py.Dataset):
            dataset_name = name
            return True
        return None

    with h5py.File(h5_path, "r") as f:
        if preferred in f and isinstance(f[preferred], h5py.Dataset):
            return preferred
        f.visititems(visitor)

    if dataset_name is None:
        raise RuntimeError(f"No valid HDF5 Dataset found inside: {h5_path}")
    return dataset_name

class H5BlockDataset(Dataset):
    """HDF5 Dataset that reads contiguous blocks to minimise seek time and RAM pressure."""
    def __init__(
        self,
        h5_path: Path,
        dset_name: str,
        start: int,
        end: int,
        meta: DatasetMeta,
        use_si: bool = True,
        cache_rows: int = 65536,
        indices: np.ndarray | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.dset_name = str(dset_name)
        self.start = int(start)
        self.end = int(end)
        self.cache_rows = int(cache_rows)
        self.meta = meta
        self.use_si = bool(use_si)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

        assert 0 <= self.start < self.end, "Invalid dataset slice boundaries."

        self._h5: h5py.File | None = None
        self._dset: h5py.Dataset | None = None

        self._cache_start = -1
        self._cache_end = -1

        self._cache_x: np.ndarray | None = None
        self._cache_u: np.ndarray | None = None
        self._cache_a: np.ndarray | None = None

        with h5py.File(self.h5_path, "r") as f:
            ds = f[self.dset_name]
            if ds.ndim != 2 or ds.shape[1] != 7:
                raise ValueError(f"Expected array shape [N, 7], got {ds.shape} at '{self.dset_name}'")
            if self.end > ds.shape[0]:
                raise ValueError(f"Requested slice end={self.end} exceeds total dataset size N={ds.shape[0]}")
            if self.indices is not None:
                if self.indices.ndim != 1 or self.indices.size == 0:
                    raise ValueError("indices must be a non-empty 1-D array when provided.")
                if int(np.min(self.indices)) < 0 or int(np.max(self.indices)) >= int(ds.shape[0]):
                    raise ValueError("indices contain out-of-range dataset rows.")

    def __len__(self) -> int:
        if self.indices is not None:
            return int(self.indices.size)
        return self.end - self.start

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._dset = self._h5[self.dset_name]

    def _load_cache_for(self, global_index: int) -> None:
        assert self._dset is not None

        # When ``indices`` is present, the sampler operates in the local split
        # index space while the old cache was keyed by global HDF5 rows.  A
        # shuffled local block can therefore span multiple global cache
        # windows, causing avoidable cache thrashing.  Cache the requested
        # local block as one logical batch instead.  Split indices are normally
        # sorted, but sorting here keeps the dataset correct for callers that
        # provide an arbitrary permutation.
        if self.indices is not None:
            local_start = (int(global_index) // self.cache_rows) * self.cache_rows
            local_end = min(local_start + self.cache_rows, int(self.indices.size))
            local_global = np.asarray(self.indices[local_start:local_end], dtype=np.int64)
            order = np.argsort(local_global, kind="stable")
            sorted_global = local_global[order]
            arr_sorted = np.asarray(self._dset[sorted_global, :])
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size, dtype=order.dtype)
            arr = arr_sorted[inverse]
            block_start = local_start
            block_end = local_end
        else:
            block_start = (int(global_index) // self.cache_rows) * self.cache_rows
            block_end = min(block_start + self.cache_rows, self.end)
            arr = np.asarray(self._dset[block_start:block_end, :])

        x = arr[:, 0:3]
        u = arr[:, 3:4]
        a = arr[:, 4:7]

        if self.use_si and self.meta.unit_system == "canonical":
            x, u, a = self.meta.convert_xyz_U_a_to_si(
                x.astype(np.float64, copy=False),
                u.astype(np.float64, copy=False),
                a.astype(np.float64, copy=False)
            )

        self._cache_x = np.ascontiguousarray(x, dtype=np.float32)
        self._cache_u = np.ascontiguousarray(u, dtype=np.float32)
        self._cache_a = np.ascontiguousarray(a, dtype=np.float32)

        self._cache_start = int(block_start)
        self._cache_end = int(block_end)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_open()
        local_query_idx = int(idx)
        if self.indices is not None:
            if not (0 <= local_query_idx < int(self.indices.size)):
                raise IndexError(f"Index {idx} out of bounds.")
            global_idx = int(self.indices[local_query_idx])
        else:
            global_idx = self.start + local_query_idx

        if not (self.start <= global_idx < self.end):
            raise IndexError(f"Index {idx} out of bounds.")

        cache_query_idx = local_query_idx if self.indices is not None else global_idx
        if self._cache_x is None or not (self._cache_start <= cache_query_idx < self._cache_end):
            self._load_cache_for(cache_query_idx)

        local_idx = cache_query_idx - self._cache_start

        # _load_cache_for above guarantees the caches are populated here.
        assert self._cache_x is not None and self._cache_u is not None and self._cache_a is not None
        return (
            self._cache_x[local_idx],
            self._cache_u[local_idx],
            self._cache_a[local_idx]
        )

    def __del__(self) -> None:
        if self._h5 is not None:
            with contextlib.suppress(Exception):
                self._h5.close()

class BlockShuffleSampler(Sampler[int]):
    """
    Shuffles block order per epoch AND shuffles indices within each block.

    Block-order shuffle breaks inter-block spatial correlation; intra-block
    shuffle removes within-block correlations that arise because the cloud
    generator writes spatially contiguous chunks.  Without intra-block shuffle,
    a single mini-batch could be drawn entirely from a tight spatial cluster,
    biasing the gradient and inflating any batch-statistics estimates.
    """

    def __init__(self, data_len: int, block_size: int, seed: int):
        self.data_len = int(data_len)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        n = self.data_len
        bs = self.block_size
        block_starts = list(range(0, n, bs))
        rng = random.Random(self.seed + self.epoch * 1337)
        rng.shuffle(block_starts)
        for start in block_starts:
            end = min(start + bs, n)
            block_indices = list(range(start, end))
            rng.shuffle(block_indices)   # intra-block: break spatial correlation within block
            yield from block_indices

    def __len__(self) -> int:
        return self.data_len

class TensorMemoryDataset(Dataset):
    """
    All-in-RAM dataset: zero HDF5 overhead once loaded.

    Enables pin_memory=True and multi-worker DataLoaders on Windows because
    h5py is not involved at all.  Used automatically when the dataset is small
    enough to fit in memory (see auto_preload_mb in TrainConfig).
    """

    def __init__(self, x: np.ndarray, u: np.ndarray, a: np.ndarray) -> None:
        self._x = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32))
        self._u = torch.as_tensor(np.ascontiguousarray(u, dtype=np.float32))
        self._a = torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return torch tensors directly: the previous ``.numpy()`` round-trip was
        # pure overhead because the collate function only re-wrapped them as
        # tensors again. Returning tensors keeps the all-in-RAM path tensor-native
        # and pin_memory-friendly (collate_xyz_u_a stacks tensors without a copy).
        return self._x[idx], self._u[idx], self._a[idx]


class TensorBatchSampler(Sampler[np.ndarray]):
    """Yield whole index arrays for tensor-backed datasets.

    ``DataLoader`` normally calls ``__getitem__`` once per sample and then
    stacks the results in Python.  This sampler is paired with
    ``batch_size=None`` so :class:`TensorMemoryDataset` receives one index
    array per batch and performs a single tensor fancy-index operation.
    """

    def __init__(
        self,
        data_len: int,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.data_len = int(data_len)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch * 1337)
            order = rng.permutation(self.data_len)
        else:
            order = np.arange(self.data_len, dtype=np.int64)
        stop = self.data_len - (self.data_len % self.batch_size) if self.drop_last else self.data_len
        for start in range(0, stop, self.batch_size):
            batch = order[start : min(start + self.batch_size, stop)]
            if batch.size:
                yield np.asarray(batch, dtype=np.int64)

    def __len__(self) -> int:
        if self.drop_last:
            return self.data_len // self.batch_size
        return (self.data_len + self.batch_size - 1) // self.batch_size


def identity_collate(batch: Any) -> Any:
    """Pickle-safe identity collate for ``TensorBatchSampler``."""

    return batch


# --- SIREN: Sinusoidal Representation Network (Sitzmann et al. 2020) ---
# sin(w0 · (Wx + b)) avoids spectral bias for high-frequency mascon fields.
# Init: first layer W~U(-1/n, 1/n); hidden W~U(-sqrt(6/n)/w0, sqrt(6/n)/w0).

def _build_train_val_indices(n_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a deterministic shuffled split while preserving ascending row order.

    We randomize membership first so validation is not just the final contiguous
    tail of the HDF5 file, then sort each subset so the block-cached dataset can
    still read reasonably contiguous regions from disk.
    """

    n_total = int(n_rows)
    frac = float(val_fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1). Got {val_fraction!r}.")

    n_val = max(1, int(round(n_total * frac)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(f"Validation fraction ({val_fraction}) is too large for dataset size N={n_total}.")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_total)
    val_idx = np.sort(perm[:n_val].astype(np.int64, copy=False))
    train_idx = np.sort(perm[n_val:].astype(np.int64, copy=False))
    return train_idx, val_idx

def _find_latest_dataset(start_dir: Path) -> Path | None:
    """
    Return the newest lunar-compatible HDF5 dataset near ``start_dir``.

    This deliberately ignores arbitrary HDF5 files that cannot prove they are
    surrogate gravity clouds for the Moon. The training CLI should auto-pick a
    safe dataset or nothing at all.
    """

    def _candidate_score(path: Path) -> tuple[int, float] | None:
        try:
            dset_name = _discover_dataset_name(path, preferred="data")
            with h5py.File(path, "r") as handle:
                ds = handle[dset_name]
                if ds.ndim != 2 or int(ds.shape[1]) != 7 or int(ds.shape[0]) <= 0:
                    return None
            meta = DatasetMeta.from_h5(path)
            body_name = _normalized_dataset_body_name(meta)
            has_signature = is_lunar_body_signature(mu_si=meta.mu_si, r_ref_m=meta.r_ref_m)
            if body_name in _LUNAR_ALIASES and has_signature:
                return (3, path.stat().st_mtime)
            if body_name in _LUNAR_ALIASES:
                return (2, path.stat().st_mtime)
            if has_signature:
                return (1, path.stat().st_mtime)
        except Exception:
            return None
        return None

    candidates: list[tuple[int, float, Path]] = []
    search_dirs = ["runs", "run", "outputs", "output", "data", "datasets", "out", "."]
    for rel in search_dirs:
        p = start_dir / rel
        if not (p.exists() and p.is_dir()):
            continue
        found = list(p.rglob("*.h5")) + list(p.rglob("*.hdf5"))
        for candidate in found:
            if candidate.stat().st_size <= 1024:
                continue
            score = _candidate_score(candidate)
            if score is not None:
                candidates.append((score[0], score[1], candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]

def infer_a_sign_from_data(
    h5_path: Path,
    dset_name: str,
    meta: DatasetMeta,
    use_si: bool,
    n_probe: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Automatically infers the mathematical sign convention of the dataset.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        ds = f[dset_name]
        total_rows = int(ds.shape[0])
        n = min(int(n_probe), total_rows)
        selected = np.sort(rng.choice(total_rows, size=n, replace=False).astype(np.int64, copy=False))
        arr = np.asarray(ds[selected, :], dtype=np.float64)

    pos = arr[:, 0:3]
    pot = arr[:, 3]
    accel = arr[:, 4:7]

    if use_si and meta.unit_system == "canonical":
        pos, pot_2d, accel = meta.convert_xyz_U_a_to_si(pos, pot[:, None], accel)
        pot = pot_2d[:, 0]

    r_norm = np.linalg.norm(pos, axis=1)
    a_dot_r = np.sum(accel * pos, axis=1)

    valid_mask = np.isfinite(pot) & np.isfinite(a_dot_r) & (r_norm > 0)
    pot = pot[valid_mask]
    a_dot_r = a_dot_r[valid_mask]

    if pot.size < 1000:
        logger.warning("Not enough valid points to infer a_sign. Defaulting to +1.0.")
        return 1.0

    c1 = float(np.corrcoef(pot, a_dot_r)[0, 1])
    if not np.isfinite(c1):
        logger.warning("Acceleration-sign correlation is non-finite; defaulting to +1.0.")
        return 1.0

    # corr(potential, -a·r) is exactly -corr(potential, a·r); comparing both
    # added work without changing the decision rule.  Random row sampling also
    # avoids bias from a contiguous, spatially correlated HDF5 block.
    inferred_sign = -1.0 if c1 >= 0.0 else +1.0
    logger.info(f"Inferred acceleration sign convention: {inferred_sign:+.1f}")
    return inferred_sign


# -----------------------------
# Training Configuration & Loop
# -----------------------------


__all__ = [
    'DTYPE', 'DatasetMeta', 'H5BlockDataset', 'TensorMemoryDataset',
    'DatasetContract', 'DatasetContractError',
    'BlockShuffleSampler', 'TensorBatchSampler', 'collate_xyz_u_a', 'collate_h5',
    'identity_collate', '_resolve_loader_worker_count',
    '_build_train_val_indices', '_find_latest_dataset', '_discover_dataset_name',
    '_resolve_lunar_dataset_contract', 'infer_a_sign_from_data',
    'build_dataset_contract', 'read_dataset_contract_from_h5', 'validate_dataset_contract',
]
