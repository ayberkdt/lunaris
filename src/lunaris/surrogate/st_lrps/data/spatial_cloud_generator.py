#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial_cloud_generator_refactored.py

Generate a large spatial point-cloud for SH gravity potential + acceleration:
    [x, y, z, U, ax, ay, az]

SSOT
----
- Physics SSOT: dataset_parameters.py (μ, R, coeff loader expectations).
- Cloud-parameter SSOT: cloud_parameters.py (alias of spatial_cloud_parameters.py)
  where you set altitudes, sample count, chunking, output format, etc.

This script is a refactor of the original spatial_cloud_generator.py to remove
configuration duplication and make runs reproducible/config-driven.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import heapq
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Numba is required for this script. Install: pip install numba") from e

# ---- Physics SSOT (local lunar dataset parameters) ----
from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    DEFAULT_DATASET_CONFIG,
    MU_MOON_SI,
    R_MOON_SI,
    canonical_scales,
    is_lunar_body_signature,
    load_icgem_gfc,
)
from lunaris.surrogate.st_lrps.data.dataset_contract import (
    contract_from_generation_attrs,
    ensure_output_path_allowed,
    stamp_hdf5_content_hash,
)

# ---- Cloud-parameter SSOT ----
from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
    SpatialCloudConfig,
    DEFAULT_SPATIAL_CLOUD_CONFIG,
    SamplingStrategy,
    CloudSuiteConfig,
    DEFAULT_CLOUD_SUITE_CONFIG,
)

# =============================================================================
# Utilities
# =============================================================================
def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _resolve_path(p: str | Path, base: Optional[Path] = None) -> Path:
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return pp
    candidates = []
    if base is not None:
        candidates.append(base / pp)
    candidates.append(_script_dir() / pp)
    candidates.append(Path.cwd() / pp)
    candidates.append(pp)  # last
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            pass
    return (base or _script_dir()) / pp


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def _file_sha256(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Historical Earth/EGM96 embedded coefficients were retired from the active
# lunar workflow. New datasets should use the repository lunar gravity file via
# coeff_source="gfc". Legacy users can still provide their own built-in arrays
# through dataset_parameters.py if they explicitly need that path.


# =============================================================================
# SH constants precompute (pure NumPy, called once)
# =============================================================================
def precompute_legendre_constants(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = int(N)
    a_nm = np.zeros((N + 1, N + 1), dtype=np.float64)
    b_nm = np.zeros_like(a_nm)
    diag_f = np.zeros(N + 1, dtype=np.float64)
    subdiag_f = np.zeros(N + 1, dtype=np.float64)
    k_ratio = np.zeros_like(a_nm)

    for n in range(1, N + 1):
        diag_f[n] = math.sqrt((2.0 * n + 1.0) / (2.0 * n))
        subdiag_f[n] = math.sqrt(2.0 * n + 1.0)

    for n in range(2, N + 1):
        for m in range(0, n - 1):
            if m <= n - 2:
                num_a = (2.0 * n + 1.0) * (2.0 * n - 1.0)
                den_a = (n - m) * (n + m)
                a_nm[n, m] = math.sqrt(num_a / den_a)

                num_b = (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
                den_b = (2.0 * n - 3.0) * (n - m) * (n + m)
                b_nm[n, m] = math.sqrt(num_b / den_b)

    for n in range(1, N + 1):
        for m in range(0, n + 1):
            if n == 0 or (n + m) == 0:
                k_ratio[n, m] = 0.0
            else:
                if (2 * n - 1) > 0:
                    k_ratio[n, m] = math.sqrt(((2.0 * n + 1.0) / (2.0 * n - 1.0)) * ((n - m) / (n + m)))
                else:
                    k_ratio[n, m] = 0.0

    return a_nm, b_nm, diag_f, subdiag_f, k_ratio


# =============================================================================
# Numba kernels
# =============================================================================
@njit(cache=True, fastmath=True)
def _sh_potential_accel_batch_serial(
    xyz_m: np.ndarray,   # (M,3) [m]
    C: np.ndarray,       # (N+1,N+1)
    S: np.ndarray,       # (N+1,N+1)
    a_nm: np.ndarray,    # (N+1,N+1)
    b_nm: np.ndarray,    # (N+1,N+1)
    diag_f: np.ndarray,  # (N+1,)
    subdiag_f: np.ndarray,  # (N+1,)
    k_ratio: np.ndarray, # (N+1,N+1)
    mu_si: float,
    r_ref_m: float,
    degree_max: int,
    degree_min: int,
) -> Tuple[np.ndarray, np.ndarray]:
    M = xyz_m.shape[0]
    N = degree_max
    V_out = np.empty(M, dtype=np.float64)
    a_out = np.empty((M, 3), dtype=np.float64)

    eps_r = 1e-12
    eps_rho = 1e-14
    eps_c = 1e-14

    for k in range(M):
        x = float(xyz_m[k, 0])
        y = float(xyz_m[k, 1])
        z = float(xyz_m[k, 2])

        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        if r < eps_r:
            V_out[k] = 0.0
            a_out[k, 0] = 0.0
            a_out[k, 1] = 0.0
            a_out[k, 2] = 0.0
            continue

        rho2 = x * x + y * y
        rho = math.sqrt(rho2)

        s = z / r
        c = rho / r

        if rho > eps_rho:
            cosl = x / rho
            sinl = y / rho
        else:
            cosl = 1.0
            sinl = 0.0

        cos_m = np.empty(N + 1, dtype=np.float64)
        sin_m = np.empty(N + 1, dtype=np.float64)
        cos_m[0] = 1.0
        sin_m[0] = 0.0
        for m in range(1, N + 1):
            cos_m[m] = cos_m[m - 1] * cosl - sin_m[m - 1] * sinl
            sin_m[m] = sin_m[m - 1] * cosl + cos_m[m - 1] * sinl

        P_nm2 = np.zeros(N + 1, dtype=np.float64)
        P_nm1 = np.zeros(N + 1, dtype=np.float64)
        P_n = np.zeros(N + 1, dtype=np.float64)

        P_nm1[0] = 1.0

        q = r_ref_m / r
        qpow = 1.0

        V_sum = 0.0
        dr_sum = 0.0
        dphi_sum = 0.0
        dlam_sum = 0.0

        term_cs0 = float(C[0, 0])
        if degree_min < 0:
            V_sum += P_nm1[0] * term_cs0
            dr_sum += P_nm1[0] * term_cs0

        for n in range(1, N + 1):
            qpow *= q

            for i in range(n + 1):
                P_n[i] = 0.0

            if n == 1:
                P_n[0] = math.sqrt(3.0) * s
                P_n[1] = -math.sqrt(3.0) * c
            else:
                P_n[n] = -diag_f[n] * c * P_nm1[n - 1]
                P_n[n - 1] = subdiag_f[n] * s * P_nm1[n - 1]
                for m in range(0, n - 1):
                    if m <= n - 2:
                        P_n[m] = a_nm[n, m] * s * P_nm1[m] - b_nm[n, m] * P_nm2[m]

            nn = float(n)
            for m in range(0, n + 1):
                cnm = float(C[n, m])
                snm = float(S[n, m])

                term_cs = cnm * cos_m[m] + snm * sin_m[m]
                P = P_n[m]

                if n > degree_min:
                    V_sum += qpow * P * term_cs
                    dr_sum += (nn + 1.0) * qpow * P * term_cs

                    if m > 0:
                        term_lon = (-cnm * sin_m[m] + snm * cos_m[m]) * float(m)
                        dlam_sum += qpow * P * term_lon

                    if c > eps_c:
                        P_nm1_m = 0.0
                        if m <= n - 1:
                            P_nm1_m = P_nm1[m]
                        kfac = 0.0
                        if m <= n:
                            kfac = k_ratio[n, m]
                        # Correct derivative: dP̄_n^m/dφ = [-n sinφ P̄_n^m + (n+m) k_{n,m} P̄_{n-1}^m] / cosφ
                        # WARNING: datasets generated before this fix have sign-flipped latitude
                        # acceleration components and must be regenerated.
                        dP_dphi = (-nn * s * P + (nn + float(m)) * kfac * P_nm1_m) / c
                        dphi_sum += qpow * dP_dphi * term_cs

            P_nm2, P_nm1, P_n = P_nm1, P_n, P_nm2

        V = (mu_si / r) * V_sum

        inv_r2 = 1.0 / r2
        a_r = -mu_si * inv_r2 * dr_sum
        a_phi = mu_si * inv_r2 * dphi_sum
        a_lam = 0.0
        if c > eps_c:
            a_lam = mu_si * inv_r2 * (dlam_sum / c)

        rx = c * cosl
        ry = c * sinl
        rz = s
        phix = -s * cosl
        phiy = -s * sinl
        phiz = c
        lamx = -sinl
        lamy = cosl

        ax = a_r * rx + a_phi * phix + a_lam * lamx
        ay = a_r * ry + a_phi * phiy + a_lam * lamy
        az = a_r * rz + a_phi * phiz

        V_out[k] = V
        a_out[k, 0] = ax
        a_out[k, 1] = ay
        a_out[k, 2] = az

    return V_out, a_out


# =============================================================================
# Sampling
# =============================================================================
def sample_uniform_shell_xyz(n: int, r_min_m: float, r_max_m: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a shell volumetrically uniformly.

    The ``r^3`` inverse-CDF produces a constant density in shell volume.
    This is statistically clean, but it tends to under-emphasize the lower
    altitudes that matter most for high-degree residual gravity learning.
    """

    n = int(n)
    u = rng.random(n, dtype=np.float64)
    r = (r_min_m**3 + u * (r_max_m**3 - r_min_m**3)) ** (1.0 / 3.0)

    u1 = rng.random(n, dtype=np.float64)
    u2 = rng.random(n, dtype=np.float64)
    z = 2.0 * u1 - 1.0
    t = 2.0 * math.pi * u2
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = xy * np.cos(t)
    y = xy * np.sin(t)

    xyz = np.empty((n, 3), dtype=np.float64)
    xyz[:, 0] = r * x
    xyz[:, 1] = r * y
    xyz[:, 2] = r * z
    return xyz


def sample_inverse_r2_shell_xyz(
    n: int,
    r_min_m: float,
    r_max_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a shell with more points near the lunar surface.

    For radial PDF ``p(r) ∝ 1/r^2`` the cumulative distribution becomes linear
    in ``1/r``. This focuses training density toward smaller radii, where the
    residual field is less smooth and acceleration supervision is richer.
    """

    n = int(n)
    inv_r_min = 1.0 / float(r_min_m)
    inv_r_max = 1.0 / float(r_max_m)
    u = rng.random(n, dtype=np.float64)
    inv_r = inv_r_min - u * (inv_r_min - inv_r_max)
    r = 1.0 / inv_r

    u1 = rng.random(n, dtype=np.float64)
    u2 = rng.random(n, dtype=np.float64)
    z = 2.0 * u1 - 1.0
    t = 2.0 * math.pi * u2
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = xy * np.cos(t)
    y = xy * np.sin(t)

    xyz = np.empty((n, 3), dtype=np.float64)
    xyz[:, 0] = r * x
    xyz[:, 1] = r * y
    xyz[:, 2] = r * z
    return xyz


def sample_mixed_shell_xyz(
    n: int,
    r_min_m: float,
    r_max_m: float,
    rng: np.random.Generator,
    *,
    surface_bias_ratio: float,
) -> np.ndarray:
    """
    Blend uniform and surface-biased samples in one batch.

    This is the most robust default for the lunar surrogate workflow. It keeps
    enough far-field coverage to avoid over-specializing the network while
    still feeding the harder near-surface harmonics that drive ``dU``/``da``.
    """

    ratio = min(1.0, max(0.0, float(surface_bias_ratio)))
    n = int(n)
    n_surface = int(round(n * ratio))
    n_uniform = n - n_surface

    chunks = []
    if n_surface > 0:
        chunks.append(sample_inverse_r2_shell_xyz(n_surface, r_min_m, r_max_m, rng))
    if n_uniform > 0:
        chunks.append(sample_uniform_shell_xyz(n_uniform, r_min_m, r_max_m, rng))
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)

    xyz = np.concatenate(chunks, axis=0)
    if xyz.shape[0] > 1:
        rng.shuffle(xyz, axis=0)
    return xyz


def sample_shell_xyz(
    n: int,
    r_min_m: float,
    r_max_m: float,
    rng: np.random.Generator,
    *,
    strategy: str,
    surface_bias_ratio: float,
) -> np.ndarray:
    """
    Dispatch shell sampling using the configured strategy.

    The dispatcher keeps the worker code simple and makes the sampling law part
    of the explicit experiment contract written into dataset metadata.
    """

    mode = str(strategy).strip().lower()
    if mode == SamplingStrategy.UNIFORM.value:
        return sample_uniform_shell_xyz(n, r_min_m, r_max_m, rng)
    if mode == SamplingStrategy.INVERSE_R2.value:
        return sample_inverse_r2_shell_xyz(n, r_min_m, r_max_m, rng)
    if mode == SamplingStrategy.MIXED.value:
        return sample_mixed_shell_xyz(
            n,
            r_min_m,
            r_max_m,
            rng,
            surface_bias_ratio=surface_bias_ratio,
        )
    raise ValueError(f"Unsupported sampling strategy: {strategy!r}")


# =============================================================================
# Writers
# =============================================================================
def write_h5_streaming(out_path: Path, n_samples: int, dtype: np.dtype, chunks_rows: int, attrs: Dict[str, str]) -> "h5py.File":
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("h5py required for --format h5. Install: pip install h5py") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(out_path), "w")
    chunk_rows_eff = max(1, min(int(chunks_rows), int(n_samples)))
    _ = f.create_dataset(
        "data",
        shape=(int(n_samples), 7),
        dtype=dtype,
        chunks=(chunk_rows_eff, 7),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    for k, v in attrs.items():
        f.attrs[str(k)] = str(v)
    contract = contract_from_generation_attrs(attrs, n_samples=int(n_samples), dataset_name="data")
    generation_config = {}
    try:
        generation_config = json.loads(str(attrs.get("cloud_config_json", "{}")))
    except Exception:
        generation_config = {}
    contract.write_hdf5_attrs(f, generation_config=generation_config)
    return f


def finalize_pt_from_memmap(memmap_path: Path, out_path: Path, n_samples: int, dtype: np.dtype, attrs: Dict[str, str], *, delete_memmap: bool = True) -> None:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch required for --format pt. Install: pip install torch") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(str(memmap_path), mode="r", dtype=dtype, shape=(int(n_samples), 7))
    data_t = torch.from_numpy(mm)  # type: ignore[arg-type]
    _cols_str = str(attrs.get("columns", "[x,y,z,U,ax,ay,az]"))
    _cols_list = [c.strip() for c in _cols_str.strip("[]").split(",") if c.strip()]
    torch.save({"data": data_t, "columns": _cols_list, "meta": attrs}, str(out_path))

    if delete_memmap:
        try:
            del mm
        except Exception:
            pass
        try:
            memmap_path.unlink(missing_ok=True)
        except Exception:
            pass


# =============================================================================
# Coeff loader (physics SSOT)
# =============================================================================

def load_coeffs_from_ssot(*, degree_max: int, gfc_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cfg = DEFAULT_DATASET_CONFIG
    meta: Dict[str, object] = {
        "degree_max": int(degree_max),
        "coeff_source": "gfc",
    }

    p = (gfc_path or "").strip()
    if not p:
        p = str(getattr(cfg, "gravity_gfc_path"))
    gfc = _resolve_path(p, base=_script_dir())
    C, S, gmeta = load_icgem_gfc(
        file_path=str(gfc),
        max_degree=int(degree_max),
        expected_norm=str(getattr(cfg, "gravity_expected_norm", "fully_normalized")),
        strict=bool(getattr(cfg, "gravity_strict_norm", True)),
    )
    mu_si = float(gmeta["mu_si"])
    r_ref_m = float(gmeta["r_ref_m"])
    central_body = str(gmeta.get("central_body", "") or "").strip().lower() or "unknown"
    loaded_degree = int(gmeta.get("degree", degree_max))

    if central_body != "moon" or not is_lunar_body_signature(mu_si=mu_si, r_ref_m=r_ref_m):
        raise ValueError(
            "Loaded gravity model is not lunar-compatible. "
            f"central_body={central_body!r}, mu_si={mu_si!r}, r_ref_m={r_ref_m!r}"
        )
    if loaded_degree != int(degree_max):
        raise ValueError(
            f"Gravity model could only provide degree={loaded_degree}, but degree_max={int(degree_max)} was requested. "
            "Refusing to generate a cloud with silently truncated physics."
        )

    meta["mu_si"] = mu_si
    meta["r_ref_m"] = r_ref_m
    meta["central_body"] = central_body
    meta["gfc_path"] = str(gfc)
    meta["loaded_degree"] = loaded_degree
    if "modelname" in gmeta:
        meta["gfc_modelname"] = str(gmeta["modelname"])
    if "norm" in gmeta:
        meta["gfc_norm"] = str(gmeta["norm"])
    return C, S, meta


# =============================================================================
# Multiprocessing globals
# =============================================================================
_G: Dict[str, object] = {}


def _init_worker(globals_blob: Dict[str, object]) -> None:
    global _G
    _G = globals_blob


def _worker_compute_chunk(start: int, n: int, seed: int) -> Tuple[int, np.ndarray]:
    global _G
    rng = np.random.default_rng(int(seed))
    xyz = sample_shell_xyz(
        int(n),
        float(_G["r_min_m"]),
        float(_G["r_max_m"]),
        rng,
        strategy=str(_G["sampling_strategy"]),
        surface_bias_ratio=float(_G["surface_bias_ratio"]),
    )

    V, a = _sh_potential_accel_batch_serial(
        xyz,
        _G["C"],  # type: ignore[arg-type]
        _G["S"],  # type: ignore[arg-type]
        _G["a_nm"],  # type: ignore[arg-type]
        _G["b_nm"],  # type: ignore[arg-type]
        _G["diag_f"],  # type: ignore[arg-type]
        _G["subdiag_f"],  # type: ignore[arg-type]
        _G["k_ratio"],  # type: ignore[arg-type]
        float(_G["mu_si"]),
        float(_G["r_ref_m"]),
        int(_G["degree_max"]),
        int(_G["degree_min"]),
    )

    out = np.empty((int(n), 7), dtype=np.float64)
    out[:, 0:3] = xyz
    out[:, 3] = V
    out[:, 4:7] = a


    if bool(_G["canonical"]):
        DU = float(_G["DU"]); TU = float(_G["TU"]); VU = float(_G["VU"])
        out[:, 0:3] /= DU
        out[:, 3] /= (VU * VU)
        out[:, 4:7] /= (DU / (TU * TU))

    dtype_out = _G["dtype_out"]  # type: ignore[assignment]
    return int(start), out.astype(dtype_out, copy=False)  # type: ignore[arg-type]


def _worker_write_memmap(memmap_path: str, n_total: int, start: int, n: int, seed: int) -> int:
    start_i, chunk = _worker_compute_chunk(start=start, n=n, seed=seed)
    dtype_out = _G["dtype_out"]  # type: ignore[assignment]
    mm = np.memmap(memmap_path, mode="r+", dtype=dtype_out, shape=(int(n_total), 7))  # type: ignore[arg-type]
    mm[start_i : start_i + int(n), :] = chunk
    mm.flush()
    return int(n)


# =============================================================================
# Config plumbing
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial SH potential/acceleration point-cloud generator (config-driven).")

    p.add_argument("--preset", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--config-json", type=str, default="", help="Load SpatialCloudConfig from JSON.")

    p.add_argument("--degree-max", type=int, default=None)
    p.add_argument("--degree-min", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--alt-range", nargs=2, type=float, default=None, metavar=("H_MIN_KM", "H_MAX_KM"))
    p.add_argument(
        "--sampling-strategy",
        choices=[item.value for item in SamplingStrategy],
        default=None,
        help="Radial shell sampling law for the generated dataset.",
    )
    p.add_argument(
        "--surface-bias-ratio",
        type=float,
        default=None,
        help="Only used by --sampling-strategy mixed. 0=uniform, 1=fully surface-biased.",
    )
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)

    p.add_argument("--format", choices=["pt", "h5"], default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default=None)
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Allow replacing an existing generated dataset file.")

    canon = p.add_mutually_exclusive_group()
    canon.add_argument("--canonical", dest="canonical", action="store_true")
    canon.add_argument("--si", dest="canonical", action="store_false")
    p.set_defaults(canonical=None)

    p.add_argument("--gfc-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    mp = p.add_mutually_exclusive_group()
    mp.add_argument("--no-multiprocessing", dest="no_multiprocessing", action="store_true")
    mp.add_argument("--multiprocessing", dest="no_multiprocessing", action="store_false")
    p.set_defaults(no_multiprocessing=None)

    p.add_argument("--dump-config", type=str, default="", help="Write resolved config JSON (and continue).")
    p.add_argument("--print-config", action="store_true", help="Print resolved config JSON.")

    # ------------------------------------------------------------------
    # Suite mode
    # ------------------------------------------------------------------
    p.add_argument("--generate-suite", action="store_true", default=False,
                   help="Generate a full dataset suite instead of a single cloud.")
    p.add_argument("--suite-name", type=str, default="",
                   help="Optional human-readable name for the suite folder.")
    p.add_argument("--suite-out-dir", type=str, default="",
                   help="Parent directory for suite output. Default: <repo>/outputs/datasets/cloud_suites/")

    # Suite physics
    p.add_argument("--train-alt-min-km", type=float, default=None)
    p.add_argument("--train-alt-max-km", type=float, default=None)
    p.add_argument("--ood-margin-km", type=float, default=None)

    # Suite train allocation
    p.add_argument("--train-stratified-uniform-n", type=int, default=None)
    p.add_argument("--train-inverse-r2-n", type=int, default=None)
    p.add_argument("--train-residual-mag-n", type=int, default=None)
    p.add_argument("--train-boundary-n", type=int, default=None)

    # Suite val/test/ood sizes
    p.add_argument("--val-n", type=int, default=None)
    p.add_argument("--test-n", type=int, default=None)
    p.add_argument("--ood-low-n", type=int, default=None)
    p.add_argument("--ood-high-n", type=int, default=None)

    # Suite seeds
    p.add_argument("--base-seed", type=int, default=None)
    p.add_argument("--train-uniform-seed", type=int, default=None)
    p.add_argument("--train-inverse-r2-seed", type=int, default=None)
    p.add_argument("--train-residual-mag-seed", type=int, default=None)
    p.add_argument("--train-boundary-seed", type=int, default=None)
    p.add_argument("--val-seed", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=None)
    p.add_argument("--ood-low-seed", type=int, default=None)
    p.add_argument("--ood-high-seed", type=int, default=None)

    # Suite residual-mag params
    p.add_argument("--residual-mag-candidate-multiplier", type=int, default=None)
    p.add_argument("--residual-mag-weight-power", type=float, default=None)
    rm_stream_grp = p.add_mutually_exclusive_group()
    rm_stream_grp.add_argument("--residual-mag-streaming", dest="residual_mag_streaming",
                               action="store_true", default=None,
                               help="Use memory-bounded streaming (weighted reservoir) residual-mag "
                                    "sampling (default).")
    rm_stream_grp.add_argument("--no-residual-mag-streaming", dest="residual_mag_streaming",
                               action="store_false", default=None,
                               help="Use the exact in-memory residual-mag sampling (higher peak RAM; "
                                    "reproduces legacy datasets bit-for-bit).")

    # Suite boundary params
    p.add_argument("--boundary-mode", choices=["strict", "soft"], default=None)
    p.add_argument("--boundary-width-km", type=float, default=None)

    # Suite post-processing
    p.add_argument("--combine-ood", action="store_true", default=True,
                   help="Combine ood_low and ood_high into ood_combined (default: True).")
    p.add_argument("--no-combine-ood", dest="combine_ood", action="store_false")

    grp_active = p.add_argument_group("Active Error Refinement")
    grp_active.add_argument("--active-from-error-points", type=str, default=None,
        help="Path to CSV of top-K error points (from evaluator --save-error-points). "
             "When set, generates jittered points around error sources instead of global cloud.")
    grp_active.add_argument("--active-jitter-radial-km", type=float, default=10.0,
        help="Std dev of radial perturbation in km (default: 10).")
    grp_active.add_argument("--active-jitter-tangent-km", type=float, default=20.0,
        help="Std dev of tangential perturbation in km (default: 20).")
    grp_active.add_argument("--active-samples-per-point", type=int, default=50,
        help="New samples generated per source error point (default: 50).")
    grp_active.add_argument("--active-max-source-points", type=int, default=1000,
        help="Maximum number of source error points to use (default: 1000).")
    grp_active.add_argument("--active-gfc-file", type=str, default=None,
        help="Path to ICGEM GFC file used for SH labeling of active-refinement points.")
    grp_active.add_argument("--active-degree-max", type=int, default=None,
        help="Maximum SH degree for active-refinement labeling (default: uses --degree-max).")
    grp_active.add_argument("--active-degree-min", type=int, default=None,
        help="Baseline SH degree for residual active-refinement labeling (default: uses --degree-min).")
    grp_active.add_argument("--active-out", type=str, default=None,
        help="Output HDF5 path for active-refinement labeled cloud (default: <out_dir>/active_refinement_labeled.h5).")
    grp_active.add_argument("--active-seed", type=int, default=42,
        help="Random seed for active-refinement jitter generation (default: 42).")
    grp_active.add_argument("--active-clip-to-alt-range", action="store_true", default=False,
        help="Clip jittered active-refinement points to the configured altitude range before labeling.")
    grp_active.add_argument("--active-reject-outside-alt-range", action="store_true", default=False,
        help="Reject (discard) jittered points outside the configured altitude range instead of clipping.")
    grp_active.add_argument("--active-save-positions-only", action="store_true", default=False,
        help="Debug flag: save only the jittered positions as NPZ without SH labeling, then exit.")

    return p.parse_args()


def resolve_cloud_config(args: argparse.Namespace) -> SpatialCloudConfig:
    cfg: SpatialCloudConfig = DEFAULT_SPATIAL_CLOUD_CONFIG

    if str(args.preset).strip():
        # Visible, on-stderr warning (a DeprecationWarning alone is hidden by
        # default and let scripts believe --preset still worked).
        import warnings
        msg = (
            f"--preset={args.preset!r} is DEPRECATED and IGNORED. Presets no longer "
            "control generation; use --config-json or explicit CLI flags. The run will "
            "proceed with the default/explicit configuration, NOT the named preset."
        )
        print(f"[WARN] {msg}", file=sys.stderr)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    if str(args.config_json).strip():
        cfg = SpatialCloudConfig.from_json(str(args.config_json).strip())

    if args.degree_max is not None:
        cfg = replace(cfg, degree_max=int(args.degree_max))
    if args.degree_min is not None:
        cfg = replace(cfg, degree_min=int(args.degree_min))
    if args.n_samples is not None:
        cfg = replace(cfg, n_samples=int(args.n_samples))
    if args.alt_range is not None:
        cfg = replace(cfg, alt_min_km=float(args.alt_range[0]), alt_max_km=float(args.alt_range[1]))
    if args.sampling_strategy is not None:
        cfg = replace(cfg, sampling_strategy=str(args.sampling_strategy))
    if args.surface_bias_ratio is not None:
        cfg = replace(cfg, surface_bias_ratio=float(args.surface_bias_ratio))
    if args.chunk_size is not None:
        cfg = replace(cfg, chunk_size=int(args.chunk_size))
    if args.workers is not None:
        cfg = replace(cfg, workers=int(args.workers))

    if args.format is not None:
        cfg = replace(cfg, out_format=str(args.format))
    if args.out is not None:
        cfg = replace(cfg, out_path=str(args.out))
    if args.dtype is not None:
        cfg = replace(cfg, dtype=str(args.dtype))

    if args.canonical is not None:
        cfg = replace(cfg, canonical=bool(args.canonical))

    if args.gfc_path is not None:
        cfg = replace(cfg, gfc_path=str(args.gfc_path))
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))

    if args.no_multiprocessing is not None:
        cfg = replace(cfg, no_multiprocessing=bool(args.no_multiprocessing))

    return cfg


def run_generation(cfg: SpatialCloudConfig, *, overwrite: bool = False) -> None:
    C, S, meta = load_coeffs_from_ssot(degree_max=int(cfg.degree_max), gfc_path=cfg.gfc_path)
    mu_si = float(meta["mu_si"])
    r_ref_m = float(meta["r_ref_m"])
    DU, TU, VU = canonical_scales(mu_si=mu_si, du_m=r_ref_m)

    t0 = time.time()
    a_nm, b_nm, diag_f, subdiag_f, k_ratio = precompute_legendre_constants(int(cfg.degree_max))
    t1 = time.time()

    r_min_m = r_ref_m + float(cfg.alt_min_km) * 1_000.0
    r_max_m = r_ref_m + float(cfg.alt_max_km) * 1_000.0

    base_dir = _script_dir().parents[4]
    data_dir = (base_dir / "outputs" / "datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    resolved = cfg.resolved_out_path()  # e.g. "potential_cloud_moon_deg50.h5" if out_path empty
    p = Path(resolved)
    if p.is_absolute():
        out_path = p
    else:
        out_path = (data_dir / p).resolve()
    out_path = ensure_output_path_allowed(out_path, overwrite=bool(overwrite))

    fmt = str(cfg.out_format).lower()
    dtype_out = np.float32 if str(cfg.dtype) == "float32" else np.float64

    unit_system = "canonical" if bool(cfg.canonical) else "si"
    target_mode = "residual" if int(cfg.degree_min) >= 0 else "full"
    columns_str = "[x,y,z,dU,dax,day,daz]" if target_mode == "residual" else "[x,y,z,U,ax,ay,az]"
    cloud_cfg_payload = dict(cfg.to_dict())
    cloud_cfg_payload.update(
        {
            "mu_si": float(mu_si),
            "r_ref_m": float(r_ref_m),
            "central_body": str(meta.get("central_body", "moon")),
            "loaded_degree": int(meta.get("loaded_degree", cfg.degree_max)),
            "gravity_model_path": str(meta.get("gfc_path", cfg.resolved_gfc_path())),
        }
    )
    attrs: Dict[str, str] = {
        "schema_version": "1",
        "dataset_kind": "st_lrps_spatial_cloud",
        **{str(key): str(value) for key, value in meta.items()},
        "unit_system": unit_system,
        "central_body": str(meta.get("central_body", "moon")),
        "degree_min": str(int(cfg.degree_min)),
        "degree_max": str(int(cfg.degree_max)),
        "requested_degree": str(int(cfg.degree_max)),
        "target_mode": target_mode,
        "baseline_kind": "spherical_harmonics" if target_mode == "residual" else "none",
        "columns": columns_str,
        "a_sign_convention": "+1",
        "derivative_convention_version": "dP_dphi_corrected_v1",
        "gravity_model_path": str(meta.get("gfc_path", cfg.resolved_gfc_path())),
        "source_gravity_model": str(meta.get("gfc_path", cfg.resolved_gfc_path())),
        "source_gravity_file_path": str(meta.get("gfc_path", cfg.resolved_gfc_path())),
        "source_gravity_file_sha256": str(_file_sha256(meta.get("gfc_path", cfg.resolved_gfc_path())) or ""),
        "alt_min_km": str(float(cfg.alt_min_km)),
        "alt_max_km": str(float(cfg.alt_max_km)),
        "coordinate_frame": "moon_fixed_cartesian",
        "units": json.dumps({"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"}, sort_keys=True),
        "generator_version": "spatial_cloud_generator_contract_v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "sampling_strategy": str(cfg.sampling_strategy),
        "surface_bias_ratio": str(float(cfg.surface_bias_ratio)),
        "n_samples": str(int(cfg.n_samples)),
        "random_seed": str(int(cfg.seed)),
        "seed": str(int(cfg.seed)),
        "dtype": str(cfg.dtype),
        "DU_m": str(DU),
        "TU_s": str(TU),
        "VU_m_s": str(VU),
        "cloud_config_json": json.dumps(cloud_cfg_payload, sort_keys=True),
        "created_by": "spatial_cloud_generator_refactored.py",
    }

    chunk_size = int(cfg.chunk_size)
    n_samples = int(cfg.n_samples)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    est_bytes = n_samples * 7 * (4 if dtype_out == np.float32 else 8)

    print(f"[info] degree_max={int(cfg.degree_max)} | degree_min={int(cfg.degree_min)} | target_mode={target_mode} | format={fmt} | dtype={cfg.dtype}")
    print(f"[info] samples={n_samples:,} | chunks={n_chunks} | chunk_size={chunk_size:,} | est_size={_human_bytes(est_bytes)}")
    print(f"[info] alt_range=[{float(cfg.alt_min_km):.1f}, {float(cfg.alt_max_km):.1f}] km | unit_system={unit_system}")
    print(f"[info] lunar constants: mu_si={mu_si:.6e} m^3/s^2 | r_ref_m={r_ref_m:.3f} m")
    print(
        f"[info] sampling={cfg.sampling_strategy}"
        f" | surface_bias_ratio={float(cfg.surface_bias_ratio):.2f}"
    )
    print(f"[info] precompute constants took {t1 - t0:.2f} s")
    print(f"[info] output: {out_path}")

    globals_blob: Dict[str, object] = {
        "C": C,
        "S": S,
        "a_nm": a_nm,
        "b_nm": b_nm,
        "diag_f": diag_f,
        "subdiag_f": subdiag_f,
        "k_ratio": k_ratio,
        "mu_si": float(mu_si),
        "r_ref_m": float(r_ref_m),
        "degree_max": int(cfg.degree_max),
        "degree_min": int(cfg.degree_min),
        "r_min_m": float(r_min_m),
        "r_max_m": float(r_max_m),
        "sampling_strategy": str(cfg.sampling_strategy),
        "surface_bias_ratio": float(cfg.surface_bias_ratio),
        "canonical": bool(cfg.canonical),
        "DU": float(DU),
        "TU": float(TU),
        "VU": float(VU),
        "dtype_out": dtype_out,
    }

    base_seed = int(cfg.seed)

    if fmt == "h5":
        with write_h5_streaming(out_path, n_samples, dtype_out, min(chunk_size, 1_000_000), attrs) as f:
            dset = f["data"]
            if bool(cfg.no_multiprocessing) or int(cfg.workers) <= 1:
                _init_worker(globals_blob)
                offset = 0
                for i in range(n_chunks):
                    n_i = min(chunk_size, n_samples - offset)
                    seed_i = base_seed + i
                    start_i, chunk = _worker_compute_chunk(offset, n_i, seed_i)
                    dset[start_i : start_i + n_i, :] = chunk
                    offset += n_i
                    if (i + 1) % max(1, n_chunks // 20) == 0:
                        print(f"[progress] {offset:,}/{n_samples:,}")
            else:
                workers = int(cfg.workers)
                print(f"[info] multiprocessing enabled (workers={workers}). HDF5 writes are serialized in main.")
                futures = []
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(globals_blob,)) as ex:
                    offset = 0
                    for i in range(n_chunks):
                        n_i = min(chunk_size, n_samples - offset)
                        seed_i = base_seed + i
                        futures.append(ex.submit(_worker_compute_chunk, offset, n_i, seed_i))
                        offset += n_i

                    done = 0
                    for fut in as_completed(futures):
                        start_i, chunk = fut.result()
                        dset[start_i : start_i + chunk.shape[0], :] = chunk
                        done += chunk.shape[0]
                        if done % max(1, n_samples // 20) < chunk.shape[0]:
                            print(f"[progress] {done:,}/{n_samples:,}")

            f.flush()
        stamp_hdf5_content_hash(out_path, dataset_name="data")
        print("[done] HDF5 saved.")

    elif fmt == "pt":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        memmap_path = out_path.with_suffix(".mmap")

        mm = np.memmap(str(memmap_path), mode="w+", dtype=dtype_out, shape=(n_samples, 7))
        mm[:] = 0
        mm.flush()
        del mm

        print(f"[info] writing memmap to: {memmap_path}")

        if bool(cfg.no_multiprocessing) or int(cfg.workers) <= 1:
            _init_worker(globals_blob)
            offset = 0
            for i in range(n_chunks):
                n_i = min(chunk_size, n_samples - offset)
                seed_i = base_seed + i
                _worker_write_memmap(str(memmap_path), n_samples, offset, n_i, seed_i)
                offset += n_i
                if (i + 1) % max(1, n_chunks // 20) == 0:
                    print(f"[progress] {offset:,}/{n_samples:,}")
        else:
            workers = int(cfg.workers)
            print(f"[info] multiprocessing enabled (workers={workers}). Workers write non-overlapping memmap slices.")
            futures = []
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(globals_blob,)) as ex:
                offset = 0
                for i in range(n_chunks):
                    n_i = min(chunk_size, n_samples - offset)
                    seed_i = base_seed + i
                    futures.append(ex.submit(_worker_write_memmap, str(memmap_path), n_samples, offset, n_i, seed_i))
                    offset += n_i

                done = 0
                for fut in as_completed(futures):
                    done += int(fut.result())
                    if done % max(1, n_samples // 20) < chunk_size:
                        print(f"[progress] {done:,}/{n_samples:,}")

        print(f"[info] finalizing .pt to: {out_path}")
        finalize_pt_from_memmap(memmap_path, out_path, n_samples, dtype_out, attrs, delete_memmap=True)
        print("[done] PT saved.")

    else:
        raise ValueError(f"Unknown out_format: {fmt!r}")


# =============================================================================
# Suite helpers
# =============================================================================

def _compute_labels_for_xyz(
    xyz: np.ndarray,
    globals_blob: Dict[str, object],
) -> np.ndarray:
    """Compute SH potential + acceleration for xyz (M,3) using current globals_blob.

    Returns (M,7) float64 array [x,y,z,dU,dax,day,daz].
    """
    V, a = _sh_potential_accel_batch_serial(
        xyz,
        globals_blob["C"],         # type: ignore[arg-type]
        globals_blob["S"],         # type: ignore[arg-type]
        globals_blob["a_nm"],   # type: ignore[arg-type]
        globals_blob["b_nm"],   # type: ignore[arg-type]
        globals_blob["diag_f"],       # type: ignore[arg-type]
        globals_blob["subdiag_f"], # type: ignore[arg-type]
        globals_blob["k_ratio"],     # type: ignore[arg-type]
        float(globals_blob["mu_si"]),
        float(globals_blob["r_ref_m"]),
        int(globals_blob["degree_max"]),
        int(globals_blob["degree_min"]),
    )
    out = np.empty((xyz.shape[0], 7), dtype=np.float64)
    out[:, 0:3] = xyz
    out[:, 3] = V
    out[:, 4:7] = a
    return out


def _write_suite_h5(
    out_path: Path,
    data: np.ndarray,
    attrs: Dict[str, str],
    chunk_size: int,
    dtype: np.dtype,
) -> None:
    """Write (N,7) array to HDF5 with suite metadata attributes."""
    import h5py  # type: ignore
    n = int(data.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_rows = max(1, min(chunk_size, n))
    with h5py.File(str(out_path), "w") as f:
        ds = f.create_dataset(
            "data",
            shape=(n, 7),
            dtype=dtype,
            chunks=(chunk_rows, 7),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        ds[:] = data.astype(dtype, copy=False)
        for k, v in attrs.items():
            f.attrs[str(k)] = str(v)
        contract = contract_from_generation_attrs(attrs, n_samples=n, dataset_name="data")
        generation_config = {}
        try:
            generation_config = json.loads(str(attrs.get("cloud_config_json", "{}")))
        except Exception:
            generation_config = {}
        contract.write_hdf5_attrs(f, generation_config=generation_config)
    stamp_hdf5_content_hash(out_path, dataset_name="data")
    print(f"[suite] wrote {n:,} rows -> {out_path.name}")


def _build_suite_attrs(
    *,
    globals_blob: Dict[str, object],
    cfg: "CloudSuiteConfig",
    dataset_role: str,
    sampling_strategy: str,
    alt_min_km: float,
    alt_max_km: float,
    seed: int,
    suite_id: str,
    suite_dir: Path,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build the HDF5 attribute dict for one suite component."""
    DU, TU, VU = canonical_scales(
        mu_si=float(globals_blob["mu_si"]),
        du_m=float(globals_blob["r_ref_m"]),
    )
    cfg_dict = cfg.to_dict()
    attrs: Dict[str, str] = {
        "schema_version": "1",
        "dataset_kind": "st_lrps_spatial_cloud",
        "central_body": "moon",
        "mu_si": str(float(globals_blob["mu_si"])),
        "r_ref_m": str(float(globals_blob["r_ref_m"])),
        "unit_system": "si",
        "degree_min": str(int(cfg.degree_min)),
        "degree_max": str(int(cfg.degree_max)),
        "target_mode": "residual",
        "columns": "[x,y,z,dU,dax,day,daz]",
        "alt_min_km": str(float(alt_min_km)),
        "alt_max_km": str(float(alt_max_km)),
        "coordinate_frame": "moon_fixed_cartesian",
        "units": json.dumps({"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"}, sort_keys=True),
        "generator_version": "spatial_cloud_generator_suite_contract_v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_role": str(dataset_role),
        "sampling_strategy": str(sampling_strategy),
        "suite_id": str(suite_id),
        "seed": str(int(seed)),
        "random_seed": str(int(seed)),
        "cloud_config_json": json.dumps(cfg_dict, sort_keys=True),
        "suite_manifest_path": str(suite_dir / "manifest.json"),
        "a_sign_convention": "+1",
        "baseline_kind": "spherical_harmonics" if int(cfg.degree_min) >= 0 else "none",
        "source_gravity_model": str(globals_blob.get("gfc_path", "")),
        "source_gravity_file_path": str(globals_blob.get("gfc_path", "")),
        "source_gravity_file_sha256": str(globals_blob.get("source_gravity_file_sha256", "")),
        "derivative_convention_version": "dP_dphi_corrected_v1",
        "DU_m": str(DU),
        "TU_s": str(TU),
        "VU_m_s": str(VU),
        "created_by": "spatial_cloud_generator.py:suite",
    }
    if extra:
        attrs.update(extra)
    return attrs


def _sample_stratified_uniform(
    n: int,
    r_min_m: float,
    r_max_m: float,
    r_ref_m: float,
    rng: np.random.Generator,
    *,
    bin_width_km: float = 50.0,
) -> np.ndarray:
    """Stratified uniform: equal points per altitude bin."""
    alt_min_km = (r_min_m - r_ref_m) / 1_000.0
    alt_max_km = (r_max_m - r_ref_m) / 1_000.0
    total_km = alt_max_km - alt_min_km
    n_bins = max(1, int(math.ceil(total_km / float(bin_width_km))))
    bin_edges_km = np.linspace(alt_min_km, alt_max_km, n_bins + 1)

    base = n // n_bins
    remainder = n - base * n_bins
    counts = [base + (1 if i < remainder else 0) for i in range(n_bins)]

    chunks: List[np.ndarray] = []
    for i, cnt in enumerate(counts):
        if cnt <= 0:
            continue
        lo_km = float(bin_edges_km[i])
        hi_km = float(bin_edges_km[i + 1])
        lo_r = r_ref_m + lo_km * 1_000.0
        hi_r = r_ref_m + hi_km * 1_000.0
        chunks.append(sample_uniform_shell_xyz(cnt, lo_r, hi_r, rng))

    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _generate_component(
    n: int,
    r_min_m: float,
    r_max_m: float,
    r_ref_m: float,
    seed: int,
    strategy: str,
    globals_blob: Dict[str, object],
    chunk_size: int,
    *,
    bin_width_km: float = 50.0,
) -> np.ndarray:
    """Generate n labeled points using the given strategy in chunks."""
    if n <= 0:
        return np.empty((0, 7), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    results: List[np.ndarray] = []
    generated = 0
    while generated < n:
        cnt = min(chunk_size, n - generated)
        if strategy == "stratified_uniform":
            xyz = _sample_stratified_uniform(cnt, r_min_m, r_max_m, r_ref_m, rng, bin_width_km=bin_width_km)
        elif strategy == SamplingStrategy.UNIFORM.value:
            xyz = sample_uniform_shell_xyz(cnt, r_min_m, r_max_m, rng)
        elif strategy == SamplingStrategy.INVERSE_R2.value:
            xyz = sample_inverse_r2_shell_xyz(cnt, r_min_m, r_max_m, rng)
        else:
            xyz = sample_uniform_shell_xyz(cnt, r_min_m, r_max_m, rng)
        labeled = _compute_labels_for_xyz(xyz, globals_blob)
        results.append(labeled)
        generated += cnt
    return np.concatenate(results, axis=0) if results else np.empty((0, 7), dtype=np.float64)


def _residual_mag_bin_counts(
    n: int, r_min_m: float, r_max_m: float, n_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-bin sample counts (sum == n) and the r-space bin edges."""
    r_edges = np.linspace(float(r_min_m), float(r_max_m), n_bins + 1)
    bin_vols = np.array(
        [r_edges[i + 1] ** 3 - r_edges[i] ** 3 for i in range(n_bins)], dtype=np.float64
    )
    bin_vols /= bin_vols.sum()
    bin_counts_f = bin_vols * float(n)
    bin_counts = np.floor(bin_counts_f).astype(int)
    deficit = int(n) - int(bin_counts.sum())
    if deficit > 0:
        extra_idx = np.argsort(bin_counts_f - bin_counts)[::-1][:deficit]
        bin_counts[extra_idx] += 1
    return bin_counts, r_edges


def _residual_mag_stream_bin(
    n_bin: int,
    lo_r: float,
    hi_r: float,
    rng: np.random.Generator,
    globals_blob: Dict[str, object],
    chunk_size: int,
    *,
    candidate_multiplier: int,
    weight_power: float,
    probability_floor: float,
) -> np.ndarray:
    """Memory-bounded weighted reservoir sampling of ``n_bin`` rows for one bin.

    Uses the Efraimidis-Spirakis A-Res weighted reservoir algorithm: each
    candidate gets key ``log(u)/w`` (u~Uniform, w = floor + ||da||^power) and we
    retain the ``n_bin`` largest keys in a fixed-size min-heap. Candidates are
    generated and scored in ``chunk_size`` sub-batches and discarded immediately,
    so peak memory is O(n_bin + chunk_size) instead of O(n_bin * multiplier).

    Reproducibility: the result depends only on the candidate stream and the key
    draws, both taken from ``rng`` in a fixed order — same seed → same rows.
    """
    n_cand = n_bin * max(1, int(candidate_multiplier))
    # min-heap of (key, tie_breaker, row); the smallest key sits at heap[0].
    heap: List[Tuple[float, int, np.ndarray]] = []
    tie = 0
    gen = 0
    while gen < n_cand:
        cnt = min(int(chunk_size), n_cand - gen)
        xyz = sample_uniform_shell_xyz(cnt, lo_r, hi_r, rng)
        labeled = _compute_labels_for_xyz(xyz, globals_blob)
        scores = np.linalg.norm(labeled[:, 4:7], axis=1).astype(np.float64)
        weights = float(probability_floor) + np.power(np.maximum(scores, 0.0), float(weight_power))
        u = rng.random(cnt)
        # key = u^(1/w); rank in log-space for numerical stability: log(key)=log(u)/w.
        keys = np.log(np.clip(u, 1e-300, 1.0)) / np.maximum(weights, 1e-12)
        for j in range(cnt):
            item = (float(keys[j]), tie, labeled[j])
            tie += 1
            if len(heap) < n_bin:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        gen += cnt

    if not heap:
        return np.empty((0, 7), dtype=np.float64)
    return np.stack([h[2] for h in heap], axis=0)


def _generate_residual_mag_component(
    n: int,
    r_min_m: float,
    r_max_m: float,
    r_ref_m: float,
    seed: int,
    globals_blob: Dict[str, object],
    chunk_size: int,
    *,
    candidate_multiplier: int = 5,
    weight_power: float = 0.5,
    probability_floor: float = 1e-3,
    n_altitude_bins: int = 20,
    streaming: bool = True,
) -> np.ndarray:
    """Residual-acceleration magnitude weighted sampling.

    Points are allocated across altitude bins (proportional to r-volume) and,
    within each bin, drawn with probability rising with ``||da||``.

    ``streaming`` (default True)
        Memory-bounded weighted reservoir sampling (A-Res): candidates are scored
        in ``chunk_size`` sub-batches and only an ``n_bin``-sized pool is ever
        retained, so the full ``n * candidate_multiplier`` candidate set is never
        materialised. Weights use raw ``||da||^power`` (no per-bin median
        normalisation), a small approximation of the exact method that keeps
        memory bounded.

    ``streaming=False``
        The exact legacy method: materialise every candidate in a bin, normalise
        scores by the bin median, and draw with ``rng.choice``. Higher peak
        memory but reproduces prior datasets bit-for-bit.

    Both paths return exactly ``n`` rows and are reproducible for a fixed seed.
    """
    if n <= 0:
        return np.empty((0, 7), dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    n_bins = max(1, int(n_altitude_bins))
    bin_counts, r_edges = _residual_mag_bin_counts(int(n), float(r_min_m), float(r_max_m), n_bins)

    total_candidates = int(n) * max(1, int(candidate_multiplier))
    print(
        f"[suite/residual_mag] {n:,} points in {n_bins} altitude bins "
        f"(~{total_candidates:,} total candidates, {chunk_size:,}/chunk, "
        f"mode={'streaming-reservoir' if streaming else 'exact-in-memory'}) ..."
    )

    results: List[np.ndarray] = []
    for i in range(n_bins):
        n_bin = int(bin_counts[i])
        if n_bin <= 0:
            continue
        lo_r = float(r_edges[i])
        hi_r = float(r_edges[i + 1])

        if streaming:
            chosen = _residual_mag_stream_bin(
                n_bin, lo_r, hi_r, rng, globals_blob, chunk_size,
                candidate_multiplier=int(candidate_multiplier),
                weight_power=float(weight_power),
                probability_floor=float(probability_floor),
            )
            results.append(chosen)
            continue

        # Exact (legacy) path: materialise all bin candidates, then weighted choice.
        n_cand_bin = n_bin * max(1, int(candidate_multiplier))
        bin_chunks: List[np.ndarray] = []
        gen = 0
        while gen < n_cand_bin:
            cnt = min(chunk_size, n_cand_bin - gen)
            xyz = sample_uniform_shell_xyz(cnt, lo_r, hi_r, rng)
            labeled = _compute_labels_for_xyz(xyz, globals_blob)
            bin_chunks.append(labeled)
            gen += cnt

        bin_cands = np.concatenate(bin_chunks, axis=0)

        # Score by ||da||
        da = bin_cands[:, 4:7]
        scores = np.linalg.norm(da, axis=1).astype(np.float64)
        median_score = float(np.median(scores))
        if median_score > 0.0:
            s_norm = scores / median_score
        else:
            s_norm = np.ones_like(scores)
        probs = float(probability_floor) + np.power(np.maximum(s_norm, 0.0), float(weight_power))
        probs /= probs.sum()

        replace = n_bin > len(bin_cands)
        chosen_idx = rng.choice(len(bin_cands), size=n_bin, replace=replace, p=probs)
        results.append(bin_cands[chosen_idx])

    if not results:
        return np.empty((0, 7), dtype=np.float64)
    return np.concatenate(results, axis=0)


def _generate_boundary_component(
    n: int,
    r_min_m: float,
    r_max_m: float,
    r_ref_m: float,
    seed: int,
    globals_blob: Dict[str, object],
    chunk_size: int,
    *,
    boundary_mode: str = "strict",
    boundary_width_km: float = 20.0,
    train_alt_min_km: float = 200.0,
    train_alt_max_km: float = 600.0,
) -> np.ndarray:
    """Boundary buffer: points near the edges of the training altitude range."""
    if n <= 0:
        return np.empty((0, 7), dtype=np.float64)

    bw = float(boundary_width_km)
    if boundary_mode == "soft":
        lo_lo_km = train_alt_min_km - bw / 2.0
        lo_hi_km = train_alt_min_km + bw / 2.0
        hi_lo_km = train_alt_max_km - bw / 2.0
        hi_hi_km = train_alt_max_km + bw / 2.0
    else:  # strict
        lo_lo_km = train_alt_min_km
        lo_hi_km = train_alt_min_km + bw
        hi_lo_km = train_alt_max_km - bw
        hi_hi_km = train_alt_max_km

    n_lower = n // 2
    n_upper = n - n_lower

    rng = np.random.default_rng(int(seed))

    parts: List[np.ndarray] = []
    for (cnt, lo_km, hi_km) in [(n_lower, lo_lo_km, lo_hi_km), (n_upper, hi_lo_km, hi_hi_km)]:
        if cnt <= 0:
            continue
        lo_r = r_ref_m + lo_km * 1_000.0
        hi_r = r_ref_m + hi_km * 1_000.0
        lo_r = max(lo_r, r_min_m)
        hi_r = min(hi_r, r_max_m)
        if hi_r <= lo_r:
            hi_r = lo_r + 1_000.0
        labeled = _generate_component(cnt, lo_r, hi_r, r_ref_m, int(rng.integers(0, 2**31)), "uniform", globals_blob, chunk_size)
        parts.append(labeled)

    if not parts:
        return np.empty((0, 7), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def resolve_suite_config(args: argparse.Namespace) -> "CloudSuiteConfig":
    """Build a CloudSuiteConfig from CLI args, starting from defaults."""
    cfg = DEFAULT_CLOUD_SUITE_CONFIG
    kw: Dict[str, object] = {}

    # Override from args if provided
    if args.degree_max is not None:
        kw["degree_max"] = int(args.degree_max)
    if args.degree_min is not None:
        kw["degree_min"] = int(args.degree_min)
    if args.gfc_path is not None:
        kw["gfc_path"] = str(args.gfc_path)
    if getattr(args, "train_alt_min_km", None) is not None:
        kw["train_alt_min_km"] = float(args.train_alt_min_km)
    if getattr(args, "train_alt_max_km", None) is not None:
        kw["train_alt_max_km"] = float(args.train_alt_max_km)
    if getattr(args, "ood_margin_km", None) is not None:
        kw["ood_margin_km"] = float(args.ood_margin_km)
    if getattr(args, "train_stratified_uniform_n", None) is not None:
        kw["train_stratified_uniform_n"] = int(args.train_stratified_uniform_n)
    if getattr(args, "train_inverse_r2_n", None) is not None:
        kw["train_inverse_r2_n"] = int(args.train_inverse_r2_n)
    if getattr(args, "train_residual_mag_n", None) is not None:
        kw["train_residual_mag_n"] = int(args.train_residual_mag_n)
    if getattr(args, "train_boundary_n", None) is not None:
        kw["train_boundary_n"] = int(args.train_boundary_n)
    if getattr(args, "val_n", None) is not None:
        kw["val_n"] = int(args.val_n)
    if getattr(args, "test_n", None) is not None:
        kw["test_n"] = int(args.test_n)
    if getattr(args, "ood_low_n", None) is not None:
        kw["ood_low_n"] = int(args.ood_low_n)
    if getattr(args, "ood_high_n", None) is not None:
        kw["ood_high_n"] = int(args.ood_high_n)
    if getattr(args, "base_seed", None) is not None:
        kw["base_seed"] = int(args.base_seed)
    if getattr(args, "train_uniform_seed", None) is not None:
        kw["train_uniform_seed"] = int(args.train_uniform_seed)
    if getattr(args, "train_inverse_r2_seed", None) is not None:
        kw["train_inverse_r2_seed"] = int(args.train_inverse_r2_seed)
    if getattr(args, "train_residual_mag_seed", None) is not None:
        kw["train_residual_mag_seed"] = int(args.train_residual_mag_seed)
    if getattr(args, "train_boundary_seed", None) is not None:
        kw["train_boundary_seed"] = int(args.train_boundary_seed)
    if getattr(args, "val_seed", None) is not None:
        kw["val_seed"] = int(args.val_seed)
    if getattr(args, "test_seed", None) is not None:
        kw["test_seed"] = int(args.test_seed)
    if getattr(args, "ood_low_seed", None) is not None:
        kw["ood_low_seed"] = int(args.ood_low_seed)
    if getattr(args, "ood_high_seed", None) is not None:
        kw["ood_high_seed"] = int(args.ood_high_seed)
    if getattr(args, "residual_mag_candidate_multiplier", None) is not None:
        kw["residual_mag_candidate_multiplier"] = int(args.residual_mag_candidate_multiplier)
    if getattr(args, "residual_mag_weight_power", None) is not None:
        kw["residual_mag_weight_power"] = float(args.residual_mag_weight_power)
    if getattr(args, "residual_mag_streaming", None) is not None:
        kw["residual_mag_streaming"] = bool(args.residual_mag_streaming)
    if getattr(args, "boundary_mode", None) is not None:
        kw["boundary_mode"] = str(args.boundary_mode)
    if getattr(args, "boundary_width_km", None) is not None:
        kw["boundary_width_km"] = float(args.boundary_width_km)
    if args.chunk_size is not None:
        kw["chunk_size"] = int(args.chunk_size)
    if args.workers is not None:
        kw["workers"] = int(args.workers)
    if getattr(args, "suite_name", ""):
        kw["suite_name"] = str(args.suite_name)
    if args.dtype is not None:
        kw["dtype"] = str(args.dtype)

    if kw:
        cfg = replace(cfg, **kw)
    return cfg


def run_suite_generation(
    cfg: "CloudSuiteConfig",
    *,
    suite_out_dir: Optional[Path] = None,
    combine_ood: bool = True,
) -> Path:
    """
    Generate the full dataset suite: train_hybrid, val, test, ood_low, ood_high,
    ood_combined, and manifest.json.  Returns the suite directory.
    """
    import h5py  # type: ignore

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_label = str(cfg.suite_name).strip() or f"moon_deg{cfg.degree_min}to{cfg.degree_max}_alt{int(cfg.train_alt_min_km)}to{int(cfg.train_alt_max_km)}km"
    suite_id = f"{suite_label}_{ts}"

    if suite_out_dir is None:
        suite_out_dir = _script_dir().parents[4] / "outputs" / "datasets" / "cloud_suites"
    suite_dir = ensure_output_path_allowed(Path(suite_out_dir) / suite_id, overwrite=False)
    suite_dir.mkdir(parents=True, exist_ok=True)
    print(f"[suite] output directory: {suite_dir}")

    # ------------------------------------------------------------------
    # Load coefficients once
    # ------------------------------------------------------------------
    print(f"[suite] loading GFC coefficients (degree_max={cfg.degree_max})...")
    C, S, meta = load_coeffs_from_ssot(degree_max=int(cfg.degree_max), gfc_path=cfg.gfc_path)
    mu_si = float(meta["mu_si"])
    r_ref_m = float(meta["r_ref_m"])
    DU, TU, VU = canonical_scales(mu_si=mu_si, du_m=r_ref_m)

    t0 = time.time()
    a_nm, b_nm, diag_f, subdiag_f, k_ratio = precompute_legendre_constants(int(cfg.degree_max))
    print(f"[suite] Legendre constants precomputed in {time.time()-t0:.2f}s")

    r_min_m = r_ref_m + float(cfg.train_alt_min_km) * 1_000.0
    r_max_m = r_ref_m + float(cfg.train_alt_max_km) * 1_000.0
    dtype_np = np.float32 if str(cfg.dtype) == "float32" else np.float64

    globals_blob: Dict[str, object] = {
        "C": C, "S": S, "a_nm": a_nm, "b_nm": b_nm,
        "diag_f": diag_f, "subdiag_f": subdiag_f, "k_ratio": k_ratio,
        "mu_si": mu_si, "r_ref_m": r_ref_m,
        "gfc_path": str(meta.get("gfc_path", cfg.resolved_gfc_path())),
        "source_gravity_file_sha256": str(_file_sha256(meta.get("gfc_path", cfg.resolved_gfc_path())) or ""),
        "degree_max": int(cfg.degree_max),
        "degree_min": int(cfg.degree_min),
        "r_min_m": r_min_m, "r_max_m": r_max_m,
        "sampling_strategy": "uniform",
        "surface_bias_ratio": 0.0,
        "canonical": False,
        "DU": DU, "TU": TU, "VU": VU,
        "dtype_out": dtype_np,
    }
    _init_worker(globals_blob)

    chunk_size = int(cfg.chunk_size)

    def _attrs(role: str, strategy: str, lo_km: float, hi_km: float, seed: int,
               extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        return _build_suite_attrs(
            globals_blob=globals_blob, cfg=cfg,
            dataset_role=role, sampling_strategy=strategy,
            alt_min_km=lo_km, alt_max_km=hi_km,
            seed=seed, suite_id=suite_id, suite_dir=suite_dir,
            extra=extra,
        )

    train_alt_min = float(cfg.train_alt_min_km)
    train_alt_max = float(cfg.train_alt_max_km)
    output_files: Dict[str, str] = {}
    component_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # A) Stratified uniform train component
    # ------------------------------------------------------------------
    n_su = int(cfg.train_stratified_uniform_n)
    su_data = np.empty((0, 7), dtype=np.float64)
    if n_su > 0:
        print(f"[suite] generating stratified_uniform train component ({n_su:,} points)...")
        su_data = _generate_component(n_su, r_min_m, r_max_m, r_ref_m, int(cfg.train_uniform_seed),
                                      "stratified_uniform", globals_blob, chunk_size)
    component_counts["stratified_uniform"] = len(su_data)

    # ------------------------------------------------------------------
    # B) Inverse-r2 train component
    # ------------------------------------------------------------------
    n_ir2 = int(cfg.train_inverse_r2_n)
    ir2_data = np.empty((0, 7), dtype=np.float64)
    if n_ir2 > 0:
        print(f"[suite] generating inverse_r2 train component ({n_ir2:,} points)...")
        ir2_data = _generate_component(n_ir2, r_min_m, r_max_m, r_ref_m, int(cfg.train_inverse_r2_seed),
                                       SamplingStrategy.INVERSE_R2.value, globals_blob, chunk_size)
    component_counts["inverse_r2"] = len(ir2_data)

    # ------------------------------------------------------------------
    # C) Residual-magnitude weighted train component
    # ------------------------------------------------------------------
    n_rm = int(cfg.train_residual_mag_n)
    rm_data = np.empty((0, 7), dtype=np.float64)
    if n_rm > 0:
        print(f"[suite] generating residual_mag train component ({n_rm:,} points)...")
        rm_data = _generate_residual_mag_component(
            n_rm, r_min_m, r_max_m, r_ref_m, int(cfg.train_residual_mag_seed),
            globals_blob, chunk_size,
            candidate_multiplier=int(cfg.residual_mag_candidate_multiplier),
            weight_power=float(cfg.residual_mag_weight_power),
            probability_floor=float(cfg.residual_mag_probability_floor),
            streaming=bool(getattr(cfg, "residual_mag_streaming", True)),
        )
    component_counts["residual_mag"] = len(rm_data)

    # ------------------------------------------------------------------
    # D) Boundary buffer train component
    # ------------------------------------------------------------------
    n_bb = int(cfg.train_boundary_n)
    bb_data = np.empty((0, 7), dtype=np.float64)
    if n_bb > 0:
        print(f"[suite] generating boundary train component ({n_bb:,} points)...")
        bb_data = _generate_boundary_component(
            n_bb, r_min_m, r_max_m, r_ref_m, int(cfg.train_boundary_seed),
            globals_blob, chunk_size,
            boundary_mode=str(cfg.boundary_mode),
            boundary_width_km=float(cfg.boundary_width_km),
            train_alt_min_km=train_alt_min,
            train_alt_max_km=train_alt_max,
        )
    component_counts["boundary"] = len(bb_data)

    # ------------------------------------------------------------------
    # Combine + shuffle -> train_hybrid.h5
    # ------------------------------------------------------------------
    train_parts = [p for p in [su_data, ir2_data, rm_data, bb_data] if len(p) > 0]
    train_data = np.concatenate(train_parts, axis=0) if train_parts else np.empty((0, 7), dtype=np.float64)
    rng_shuffle = np.random.default_rng(int(cfg.base_seed))
    if len(train_data) > 1:
        idx = rng_shuffle.permutation(len(train_data))
        train_data = train_data[idx]

    # Verify actual vs expected row count
    expected_train_total = (
        int(cfg.train_stratified_uniform_n)
        + int(cfg.train_inverse_r2_n)
        + int(cfg.train_residual_mag_n)
        + int(cfg.train_boundary_n)
    )
    actual_train_total = len(train_data)
    if actual_train_total != expected_train_total:
        print(
            f"[suite/WARNING] train row count mismatch: "
            f"expected {expected_train_total:,}, got {actual_train_total:,}. "
            f"Components: su={component_counts['stratified_uniform']}, "
            f"ir2={component_counts['inverse_r2']}, "
            f"rm={component_counts['residual_mag']}, "
            f"bb={component_counts['boundary']}"
        )

    hybrid_components_json = json.dumps({
        "stratified_uniform": {"n": component_counts["stratified_uniform"]},
        "inverse_r2": {"n": component_counts["inverse_r2"]},
        "residual_mag": {
            "n": component_counts["residual_mag"],
            "candidate_multiplier": int(cfg.residual_mag_candidate_multiplier),
            "weight_power": float(cfg.residual_mag_weight_power),
            "streaming": bool(getattr(cfg, "residual_mag_streaming", True)),
        },
        "boundary": {
            "n": component_counts["boundary"],
            "mode": str(cfg.boundary_mode),
            "width_km": float(cfg.boundary_width_km),
        },
    }, sort_keys=True)

    total_suffix = f"_{len(train_data)//1_000_000}M" if len(train_data) >= 1_000_000 else f"_{len(train_data)//1_000}k"
    train_fname = f"train_hybrid{total_suffix}.h5"
    train_path = suite_dir / train_fname
    _write_suite_h5(
        train_path, train_data,
        _attrs("train", "hybrid", train_alt_min, train_alt_max, int(cfg.train_uniform_seed),
               {"hybrid_components_json": hybrid_components_json}),
        chunk_size, dtype_np,
    )
    output_files["train"] = str(train_path)

    # ------------------------------------------------------------------
    # E) Validation
    # ------------------------------------------------------------------
    n_val = int(cfg.val_n)
    if n_val > 0:
        print(f"[suite] generating val_uniform ({n_val:,} points)...")
        val_data = _generate_component(n_val, r_min_m, r_max_m, r_ref_m, int(cfg.val_seed),
                                       "stratified_uniform", globals_blob, chunk_size)
        val_suffix = f"_{n_val//1_000_000}M" if n_val >= 1_000_000 else f"_{n_val//1_000}k"
        val_path = suite_dir / f"val_uniform{val_suffix}.h5"
        _write_suite_h5(val_path, val_data, _attrs("val", "stratified_uniform", train_alt_min, train_alt_max, int(cfg.val_seed)), chunk_size, dtype_np)
        output_files["val"] = str(val_path)

    # ------------------------------------------------------------------
    # F) Test
    # ------------------------------------------------------------------
    n_test = int(cfg.test_n)
    if n_test > 0:
        print(f"[suite] generating test_uniform ({n_test:,} points)...")
        test_data = _generate_component(n_test, r_min_m, r_max_m, r_ref_m, int(cfg.test_seed),
                                        "stratified_uniform", globals_blob, chunk_size)
        test_suffix = f"_{n_test//1_000_000}M" if n_test >= 1_000_000 else f"_{n_test//1_000}k"
        test_path = suite_dir / f"test_uniform{test_suffix}.h5"
        _write_suite_h5(test_path, test_data, _attrs("test", "stratified_uniform", train_alt_min, train_alt_max, int(cfg.test_seed)), chunk_size, dtype_np)
        output_files["test"] = str(test_path)

    # ------------------------------------------------------------------
    # G) OOD low
    # ------------------------------------------------------------------
    n_ood_low = int(cfg.ood_low_n)
    ood_low_data = np.empty((0, 7), dtype=np.float64)
    ood_low_path: Optional[Path] = None
    if n_ood_low > 0:
        ood_lo_min = float(cfg.ood_low_alt_min_km)
        ood_lo_max = float(cfg.ood_low_alt_max_km)
        ood_lo_r_min = r_ref_m + ood_lo_min * 1_000.0
        ood_lo_r_max = r_ref_m + ood_lo_max * 1_000.0
        print(f"[suite] generating ood_low ({n_ood_low:,} points, alt={ood_lo_min:.0f}-{ood_lo_max:.0f} km)...")
        ood_low_data = _generate_component(n_ood_low, ood_lo_r_min, ood_lo_r_max, r_ref_m,
                                           int(cfg.ood_low_seed), "stratified_uniform", globals_blob, chunk_size)
        ood_low_suffix = f"_{n_ood_low//1_000}k"
        ood_low_path = suite_dir / f"ood_low_{int(ood_lo_min)}to{int(ood_lo_max)}km{ood_low_suffix}.h5"
        _write_suite_h5(ood_low_path, ood_low_data, _attrs("ood_low", "stratified_uniform", ood_lo_min, ood_lo_max, int(cfg.ood_low_seed)), chunk_size, dtype_np)
        output_files["ood_low"] = str(ood_low_path)

    # ------------------------------------------------------------------
    # H) OOD high
    # ------------------------------------------------------------------
    n_ood_high = int(cfg.ood_high_n)
    ood_high_data = np.empty((0, 7), dtype=np.float64)
    ood_high_path: Optional[Path] = None
    if n_ood_high > 0:
        ood_hi_min = float(cfg.ood_high_alt_min_km)
        ood_hi_max = float(cfg.ood_high_alt_max_km)
        ood_hi_r_min = r_ref_m + ood_hi_min * 1_000.0
        ood_hi_r_max = r_ref_m + ood_hi_max * 1_000.0
        print(f"[suite] generating ood_high ({n_ood_high:,} points, alt={ood_hi_min:.0f}-{ood_hi_max:.0f} km)...")
        ood_high_data = _generate_component(n_ood_high, ood_hi_r_min, ood_hi_r_max, r_ref_m,
                                            int(cfg.ood_high_seed), "stratified_uniform", globals_blob, chunk_size)
        ood_high_suffix = f"_{n_ood_high//1_000}k"
        ood_high_path = suite_dir / f"ood_high_{int(ood_hi_min)}to{int(ood_hi_max)}km{ood_high_suffix}.h5"
        _write_suite_h5(ood_high_path, ood_high_data, _attrs("ood_high", "stratified_uniform", ood_hi_min, ood_hi_max, int(cfg.ood_high_seed)), chunk_size, dtype_np)
        output_files["ood_high"] = str(ood_high_path)

    # ------------------------------------------------------------------
    # I) OOD combined
    # ------------------------------------------------------------------
    ood_combined_path: Optional[Path] = None
    if combine_ood and (len(ood_low_data) > 0 or len(ood_high_data) > 0):
        parts_ood = [p for p in [ood_low_data, ood_high_data] if len(p) > 0]
        ood_combined_data = np.concatenate(parts_ood, axis=0)
        combined_attrs = _attrs(
            "ood_combined", "stratified_uniform",
            float(cfg.ood_low_alt_min_km), float(cfg.ood_high_alt_max_km),
            int(cfg.base_seed),
            extra={
                "ood_low_n": str(len(ood_low_data)),
                "ood_high_n": str(len(ood_high_data)),
                "ood_low_alt_min_km": str(float(cfg.ood_low_alt_min_km)),
                "ood_low_alt_max_km": str(float(cfg.ood_low_alt_max_km)),
                "ood_high_alt_min_km": str(float(cfg.ood_high_alt_min_km)),
                "ood_high_alt_max_km": str(float(cfg.ood_high_alt_max_km)),
                "train_alt_min_km": str(train_alt_min),
                "train_alt_max_km": str(train_alt_max),
            },
        )
        n_combined = len(ood_combined_data)
        combined_suffix = f"_{n_combined//1_000}k"
        ood_combined_path = suite_dir / f"ood_combined{combined_suffix}.h5"
        _write_suite_h5(ood_combined_path, ood_combined_data, combined_attrs, chunk_size, dtype_np)
        output_files["ood_combined"] = str(ood_combined_path)

    # ------------------------------------------------------------------
    # J) manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "suite_id": suite_id,
        "suite_name": str(cfg.suite_name),
        "timestamp": ts,
        "central_body": "moon",
        "mu_si": float(mu_si),
        "r_ref_m": float(r_ref_m),
        "degree_min": int(cfg.degree_min),
        "degree_max": int(cfg.degree_max),
        "train_alt_min_km": float(cfg.train_alt_min_km),
        "train_alt_max_km": float(cfg.train_alt_max_km),
        "ood_low_alt_min_km": float(cfg.ood_low_alt_min_km),
        "ood_low_alt_max_km": float(cfg.ood_low_alt_max_km),
        "ood_high_alt_min_km": float(cfg.ood_high_alt_min_km),
        "ood_high_alt_max_km": float(cfg.ood_high_alt_max_km),
        "ood_margin_km": float(cfg.ood_margin_km),
        "train_components": {
            "stratified_uniform": {"n": component_counts["stratified_uniform"], "seed": int(cfg.train_uniform_seed)},
            "inverse_r2": {"n": component_counts["inverse_r2"], "seed": int(cfg.train_inverse_r2_seed)},
            "residual_mag": {
                "n": component_counts["residual_mag"],
                "seed": int(cfg.train_residual_mag_seed),
                "candidate_multiplier": int(cfg.residual_mag_candidate_multiplier),
                "weight_power": float(cfg.residual_mag_weight_power),
                "streaming": bool(getattr(cfg, "residual_mag_streaming", True)),
            },
            "boundary": {
                "n": component_counts["boundary"],
                "seed": int(cfg.train_boundary_seed),
                "mode": str(cfg.boundary_mode),
                "width_km": float(cfg.boundary_width_km),
            },
        },
        "train_total_n": len(train_data),
        "val_n": n_val if n_val > 0 else 0,
        "test_n": n_test if n_test > 0 else 0,
        "ood_low_n": len(ood_low_data),
        "ood_high_n": len(ood_high_data),
        "ood_combined_n": len(ood_combined_data) if ood_combined_path is not None else 0,
        "val_seed": int(cfg.val_seed),
        "test_seed": int(cfg.test_seed),
        "ood_low_seed": int(cfg.ood_low_seed),
        "ood_high_seed": int(cfg.ood_high_seed),
        "output_files": {k: str(v) for k, v in output_files.items()},
        "suite_dir": str(suite_dir),
        "generator_config": cfg.to_dict(),
    }
    manifest_path = suite_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[suite] manifest -> {manifest_path}")

    print("\n[suite] COMPLETE")
    print(f"  train_hybrid   : {len(train_data):,} rows")
    print(f"  val_uniform    : {int(cfg.val_n):,} rows")
    print(f"  test_uniform   : {int(cfg.test_n):,} rows")
    print(f"  ood_low        : {len(ood_low_data):,} rows")
    print(f"  ood_high       : {len(ood_high_data):,} rows")
    if ood_combined_path is not None:
        print(f"  ood_combined   : {len(ood_combined_data):,} rows")
    print(f"  suite dir      : {suite_dir}")
    return suite_dir


def _load_error_points(path: Path, max_source: int = 5000) -> np.ndarray:
    """
    Load top-K error points from a CSV produced by the evaluator.

    Expected columns: x,y,z,u_true,u_pred,ax_true,ay_true,az_true,
                      ax_pred,ay_pred,az_pred,abs_a_error,rel_a_error,altitude_km
    Returns float64 array shape (M, 14).
    """
    import csv as _csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= int(max_source):
                break
            rows.append([float(row[c]) for c in
                         ["x","y","z","u_true","u_pred",
                          "ax_true","ay_true","az_true",
                          "ax_pred","ay_pred","az_pred",
                          "abs_a_error","rel_a_error","altitude_km"]])
    if not rows:
        raise ValueError(f"No error points loaded from {path}")
    return np.array(rows, dtype=np.float64)


def _resolve_active_alt_bounds(a) -> "Tuple[Optional[float], Optional[float], str]":
    """Resolve the active-refinement altitude shell [min, max] km and its source.

    Priority: explicit CLI args -> source dataset HDF5 metadata -> suite manifest.
    Returns ``(alt_min_km, alt_max_km, source_label)``; min/max are None when
    unresolved so the caller can fail loudly instead of using a bogus default.
    """
    def _pair(lo_name: str, hi_name: str):
        lo = getattr(a, lo_name, None)
        hi = getattr(a, hi_name, None)
        if lo is not None and hi is not None:
            try:
                return float(lo), float(hi)
            except (TypeError, ValueError):
                return None
        return None

    # 1. Explicit args (active-specific first, then generic training-shell args).
    for lo_name, hi_name in (
        ("active_alt_min_km", "active_alt_max_km"),
        ("altitude_min_km", "altitude_max_km"),
        ("alt_min_km", "alt_max_km"),
    ):
        got = _pair(lo_name, hi_name)
        if got is not None:
            return got[0], got[1], f"args:{lo_name}/{hi_name}"

    # 2. Source dataset metadata (HDF5 attrs / cloud_config_json).
    for ds_attr in ("active_source_dataset", "source_dataset", "data"):
        ds_path = getattr(a, ds_attr, None)
        if not ds_path:
            continue
        p = Path(str(ds_path))
        if p.suffix.lower() not in (".h5", ".hdf5") or not p.exists():
            continue
        try:
            import h5py as _h5
            with _h5.File(p, "r") as _f:
                _at = {str(k): _f.attrs[k] for k in _f.attrs.keys()}
            lo = _at.get("alt_min_km")
            hi = _at.get("alt_max_km")
            if lo is None or hi is None:
                _cc = _at.get("cloud_config_json")
                if _cc is not None:
                    if isinstance(_cc, bytes):
                        _cc = _cc.decode("utf-8")
                    _cc = json.loads(str(_cc))
                    lo = lo if lo is not None else _cc.get("alt_min_km")
                    hi = hi if hi is not None else _cc.get("alt_max_km")
            if lo is not None and hi is not None:
                return float(lo), float(hi), f"dataset:{p.name}"
        except Exception:
            continue

    # 3. Suite manifest.
    sm = getattr(a, "suite_manifest", None)
    if sm:
        try:
            manifest = json.loads(Path(str(sm)).read_text(encoding="utf-8"))
            lo = manifest.get("alt_min_km")
            hi = manifest.get("alt_max_km")
            if lo is not None and hi is not None:
                return float(lo), float(hi), "suite_manifest"
        except Exception:
            pass

    return None, None, "unresolved"


def _jitter_around_point(
    x_src: np.ndarray,
    n_samples: int,
    jitter_radial_km: float,
    jitter_tangent_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n_samples points jittered around a single source position.

    Perturbation is decomposed into:
    - radial: along r_hat
    - two perpendicular tangential directions using Gram-Schmidt
    """
    r_norm = float(np.linalg.norm(x_src))
    if r_norm < 1e-10:
        return np.tile(x_src, (n_samples, 1))
    r_hat = x_src / r_norm

    # Build two orthogonal tangential directions via Gram-Schmidt
    ref = np.array([1.0, 0.0, 0.0]) if abs(r_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - np.dot(ref, r_hat) * r_hat
    t1 /= max(np.linalg.norm(t1), 1e-12)
    t2 = np.cross(r_hat, t1)
    t2 /= max(np.linalg.norm(t2), 1e-12)

    dr = rng.normal(0.0, jitter_radial_km * 1000.0, n_samples)
    dt1 = rng.normal(0.0, jitter_tangent_km * 1000.0, n_samples)
    dt2 = rng.normal(0.0, jitter_tangent_km * 1000.0, n_samples)

    pts = (x_src[None, :]
           + dr[:, None] * r_hat[None, :]
           + dt1[:, None] * t1[None, :]
           + dt2[:, None] * t2[None, :])
    return pts


def _run_active_refinement(a, ap) -> None:
    """Generate active refinement cloud from evaluator error points, with SH labeling.

    Produces a labeled HDF5 file in the same column format as normal clouds:
    [x, y, z, U, ax, ay, az] (full field) or [x, y, z, dU, dax, day, daz] (residual),
    depending on target_mode derived from degree_min/degree_max settings.

    If --active-save-positions-only is set, saves only positions NPZ (debug mode).
    """
    from pathlib import Path
    import json
    try:
        import h5py as _h5py
    except ImportError:
        ap.error("h5py is required for active refinement HDF5 output. pip install h5py")

    error_path = Path(a.active_from_error_points)
    if not error_path.exists():
        ap.error(f"Error points file not found: {error_path}")

    src = _load_error_points(error_path, max_source=int(a.active_max_source_points))
    n_src = src.shape[0]
    n_per = int(a.active_samples_per_point)
    _seed = int(getattr(a, "active_seed", getattr(a, "seed", 42)))
    rng = np.random.default_rng(_seed)

    print(f"[active-refinement] {n_src} source points x {n_per} samples = {n_src * n_per} total")

    all_pts = []
    for i in range(n_src):
        x_src = src[i, :3]
        pts = _jitter_around_point(
            x_src, n_per,
            float(a.active_jitter_radial_km),
            float(a.active_jitter_tangent_km),
            rng,
        )
        all_pts.append(pts)

    x_all = np.vstack(all_pts)  # (N_total, 3)
    total_generated = int(x_all.shape[0])
    print(f"[active-refinement] Generated {total_generated} positions.")

    _active_out_path = getattr(a, "active_out", None)
    if _active_out_path:
        out_dir = Path(_active_out_path).parent
    else:
        out_dir = Path(getattr(a, "out", ".") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Altitude clipping/rejection
    _r_ref_m = float(R_MOON_SI)
    _clip = bool(getattr(a, "active_clip_to_alt_range", False))
    _reject = bool(getattr(a, "active_reject_outside_alt_range", False))

    if _clip or _reject:
        # Resolve the training shell from a trustworthy source. Never silently
        # fall back to (0, 10000) km — that admits points far outside the Moon's
        # neighbourhood (R_moon = 1737 km) into the refinement cloud with no error.
        # Priority: explicit args -> source dataset metadata -> suite manifest.
        _alt_min_km, _alt_max_km, _alt_src = _resolve_active_alt_bounds(a)
        if _alt_min_km is None or _alt_max_km is None:
            raise ValueError(
                "Active refinement was asked to clip/reject by altitude "
                "(--active-clip-to-alt-range / --active-reject-outside-alt-range) but the "
                "training altitude shell could not be resolved. Provide --active-alt-min-km "
                "and --active-alt-max-km (or --altitude-min-km/--altitude-max-km), or a "
                "source dataset / suite manifest that records alt_min_km/alt_max_km. "
                "Refusing to default to a 0..10000 km shell."
            )
        if not (np.isfinite(_alt_min_km) and np.isfinite(_alt_max_km)) or _alt_max_km <= _alt_min_km:
            raise ValueError(
                f"Active refinement altitude shell is invalid: "
                f"[{_alt_min_km}, {_alt_max_km}] km (source={_alt_src})."
            )
        print(f"[active-refinement] Altitude shell [{_alt_min_km:.1f}, {_alt_max_km:.1f}] km "
              f"(source={_alt_src})")
        r_norms = np.linalg.norm(x_all, axis=1)
        r_min_lim = _r_ref_m + _alt_min_km * 1000.0
        r_max_lim = _r_ref_m + _alt_max_km * 1000.0
        if _clip:
            # Project outside-range points to nearest boundary by scaling radius
            _n_out = int(np.sum((r_norms < r_min_lim) | (r_norms > r_max_lim)))
            dirs = x_all / r_norms[:, None].clip(1e-10)
            r_clipped = np.clip(r_norms, r_min_lim, r_max_lim)
            x_all = dirs * r_clipped[:, None]
            print(f"[active-refinement] Clipped {_n_out}/{total_generated} out-of-range radii "
                  f"to [{r_min_lim:.3e}, {r_max_lim:.3e}] m")
        elif _reject:
            # Resample points outside range (up to 3 attempts per bad point)
            for _attempt in range(3):
                r_norms = np.linalg.norm(x_all, axis=1)
                bad_mask = (r_norms < r_min_lim) | (r_norms > r_max_lim)
                n_bad = int(np.sum(bad_mask))
                if n_bad == 0:
                    break
                print(f"[active-refinement] Attempt {_attempt+1}: resampling {n_bad} out-of-range points")
                bad_idx = np.where(bad_mask)[0]
                # Resample from random source points
                src_replace_idx = rng.integers(0, n_src, size=n_bad)
                for _j, _bi in enumerate(bad_idx):
                    x_src_rep = src[src_replace_idx[_j], :3]
                    new_pts = _jitter_around_point(x_src_rep, 1, float(a.active_jitter_radial_km),
                                                   float(a.active_jitter_tangent_km), rng)
                    x_all[_bi] = new_pts[0]
            # Drop any points still out of range so the cloud never contains
            # far-shell junk (jitter may exceed the shell for tight bounds).
            r_norms = np.linalg.norm(x_all, axis=1)
            still_bad = (r_norms < r_min_lim) | (r_norms > r_max_lim)
            n_still_bad = int(np.sum(still_bad))
            if n_still_bad > 0:
                frac = n_still_bad / max(1, x_all.shape[0])
                print(f"[active-refinement] WARNING: {n_still_bad} points ({frac:.1%}) still "
                      "out-of-range after resampling; dropping them.")
                x_all = x_all[~still_bad]
                if frac > 0.10:
                    print("[active-refinement] WARNING: >10% of points were unrecoverable; "
                          "consider widening the shell or reducing jitter radius.")

    # Debug path: save positions only (NPZ) and return
    if bool(getattr(a, "active_save_positions_only", False)):
        positions_path = ensure_output_path_allowed(
            out_dir / "active_refinement_positions.npz",
            overwrite=bool(getattr(a, "overwrite", False)),
        )
        np.savez(str(positions_path), x=x_all.astype(np.float64))
        meta_debug = {
            "component_name": "active_error_refinement",
            "source_error_file": str(error_path.resolve()),
            "n_source_points": int(n_src),
            "active_jitter_radial_km": float(a.active_jitter_radial_km),
            "active_jitter_tangent_km": float(a.active_jitter_tangent_km),
            "active_samples_per_point": int(n_per),
            "total_generated_positions": total_generated,
            "mode": "positions_only_debug",
        }
        (out_dir / "active_refinement_meta.json").write_text(json.dumps(meta_debug, indent=2))
        print(f"[active-refinement] DEBUG: saved positions to {positions_path}")
        return

    # Require GFC file for SH labeling
    _gfc_file = getattr(a, "active_gfc_file", None)
    if not _gfc_file:
        raise ValueError(
            "Active refinement requires --active-gfc-file to be set for SH labeling. "
            "Set --active-save-positions-only to skip labeling (debug mode)."
        )
    gfc_path = Path(str(_gfc_file))
    if not gfc_path.exists():
        ap.error(f"--active-gfc-file not found: {gfc_path}")

    # Explicit None check: using `or` would override degree_min=0 and degree_max=0.
    _adm = getattr(a, "active_degree_max", None)
    _degree_max_active = int(_adm if _adm is not None else getattr(a, "degree_max", 50))
    _adm_min = getattr(a, "active_degree_min", None)
    _degree_min_active = int(_adm_min if _adm_min is not None else getattr(a, "degree_min", -1))

    print(f"[active-refinement] Loading GFC: {gfc_path.name} (degree_max={_degree_max_active})")
    C, S, gmeta = load_icgem_gfc(file_path=str(gfc_path), max_degree=_degree_max_active)
    # load_icgem_gfc normalizes the header keys into mu_si and r_ref_m.
    # Do NOT use raw ICGEM keys ("earth_gravity_constant", "radius") here.
    mu_gfc = float(gmeta.get("mu_si", MU_MOON_SI))
    r_ref_gfc = float(gmeta.get("r_ref_m", R_MOON_SI))

    print(f"[active-refinement] Precomputing Legendre constants (degree_max={_degree_max_active})...")
    a_nm, b_nm, diag_f, subdiag_f, k_ratio = precompute_legendre_constants(_degree_max_active)

    # Label with SH
    print(f"[active-refinement] Labeling {x_all.shape[0]} points with SH (degree_min={_degree_min_active}, degree_max={_degree_max_active})...")
    V_sh, a_sh = _sh_potential_accel_batch_serial(
        x_all.astype(np.float64),
        C, S,
        a_nm, b_nm,
        diag_f, subdiag_f, k_ratio,
        mu_gfc, r_ref_gfc,
        _degree_max_active,
        _degree_min_active,
    )
    V_sh = V_sh.reshape(-1, 1)
    a_sh = a_sh.reshape(-1, 3)

    _is_residual = (_degree_min_active >= 0)
    target_mode_active = "residual" if _is_residual else "full"
    columns_label = "[x,y,z,dU,dax,day,daz]" if _is_residual else "[x,y,z,U,ax,ay,az]"

    # Build output dataset (N, 7)
    data_out = np.concatenate([x_all.astype(np.float64), V_sh, a_sh], axis=1)  # (N, 7)

    # Resolve output path
    _active_out_path = getattr(a, "active_out", None)
    if _active_out_path:
        h5_path = Path(str(_active_out_path))
    else:
        h5_path = out_dir / "active_refinement_labeled.h5"

    h5_path = ensure_output_path_allowed(h5_path, overwrite=bool(getattr(a, "overwrite", False)))
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute alt bounds from generated points
    r_norms_out = np.linalg.norm(x_all, axis=1)
    alt_km_out = (r_norms_out - r_ref_gfc) / 1000.0
    alt_min_out = float(alt_km_out.min())
    alt_max_out = float(alt_km_out.max())
    if alt_max_out <= alt_min_out:
        alt_max_out = alt_min_out + 1e-6

    print(f"[active-refinement] Saving labeled HDF5 to {h5_path} (shape={data_out.shape}) ...")
    with _h5py.File(str(h5_path), "w") as hf:
        ds = hf.create_dataset("data", data=data_out.astype(np.float64),
                                chunks=(min(65536, data_out.shape[0]), 7),
                                compression="gzip", compression_opts=4)
        # Required HDF5 attrs
        hf.attrs["schema_version"] = 1
        hf.attrs["dataset_kind"] = "st_lrps_spatial_cloud"
        hf.attrs["component_name"] = "active_error_refinement"
        hf.attrs["source_error_file"] = str(error_path.resolve())
        hf.attrs["active_jitter_radial_km"] = float(a.active_jitter_radial_km)
        hf.attrs["active_jitter_tangent_km"] = float(a.active_jitter_tangent_km)
        hf.attrs["active_samples_per_point"] = int(n_per)
        hf.attrs["active_max_source_points"] = int(a.active_max_source_points)
        hf.attrs["n_source_points_used"] = int(n_src)
        hf.attrs["total_generated_points"] = int(total_generated)
        hf.attrs["degree_min"] = int(_degree_min_active)
        hf.attrs["degree_max"] = int(_degree_max_active)
        hf.attrs["target_mode"] = str(target_mode_active)
        hf.attrs["central_body"] = "moon"
        hf.attrs["mu_si"] = float(mu_gfc)
        hf.attrs["r_ref_m"] = float(r_ref_gfc)
        hf.attrs["alt_min_km"] = alt_min_out
        hf.attrs["alt_max_km"] = alt_max_out
        hf.attrs["coordinate_frame"] = "moon_fixed_cartesian"
        hf.attrs["units"] = json.dumps({"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"}, sort_keys=True)
        hf.attrs["generator_version"] = "spatial_cloud_generator_active_contract_v1"
        hf.attrs["created_at_utc"] = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        hf.attrs["columns"] = columns_label
        hf.attrs["a_sign_convention"] = "+1"
        hf.attrs["unit_system"] = "si"
        hf.attrs["baseline_kind"] = "spherical_harmonics" if _is_residual else "none"
        hf.attrs["source_gravity_model"] = str(gfc_path)
        hf.attrs["source_gravity_file_path"] = str(gfc_path)
        hf.attrs["source_gravity_file_sha256"] = str(_file_sha256(gfc_path) or "")
        hf.attrs["derivative_convention_version"] = "dP_dphi_corrected_v1"
        hf.attrs["seed"] = int(getattr(a, "active_seed", 42))
        hf.attrs["created_by"] = "spatial_cloud_generator._run_active_refinement"
        contract = contract_from_generation_attrs(
            {str(k): hf.attrs[k] for k in hf.attrs.keys()},
            n_samples=int(data_out.shape[0]),
            dataset_name="data",
        )
        contract.write_hdf5_attrs(
            hf,
            generation_config={
                "active_jitter_radial_km": float(a.active_jitter_radial_km),
                "active_jitter_tangent_km": float(a.active_jitter_tangent_km),
                "active_samples_per_point": int(n_per),
                "active_seed": int(getattr(a, "active_seed", 42)),
            },
        )
    stamp_hdf5_content_hash(h5_path, dataset_name="data")

    # Save metadata JSON
    meta = {
        "component_name": "active_error_refinement",
        "source_error_file": str(error_path.resolve()),
        "active_jitter_radial_km": float(a.active_jitter_radial_km),
        "active_jitter_tangent_km": float(a.active_jitter_tangent_km),
        "active_samples_per_point": int(n_per),
        "active_max_source_points": int(a.active_max_source_points),
        "n_source_points_used": int(n_src),
        "total_generated_points": total_generated,
        "degree_min": int(_degree_min_active),
        "degree_max": int(_degree_max_active),
        "target_mode": str(target_mode_active),
        "central_body": "moon",
        "mu_si": float(mu_gfc),
        "r_ref_m": float(r_ref_gfc),
        "alt_min_km": alt_min_out,
        "alt_max_km": alt_max_out,
        "columns": columns_label,
        "a_sign_convention": "+1",
        "output_h5": str(h5_path),
    }
    meta_path = out_dir / "active_refinement_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[active-refinement] Done. Labeled {data_out.shape[0]} points -> {h5_path}")
    print(f"[active-refinement] Metadata saved to {meta_path}")


def main() -> None:
    args = parse_args()

    if getattr(args, "active_from_error_points", None):
        # Build a tiny stub parser-like object so _run_active_refinement can call ap.error.
        class _ArgErr:
            @staticmethod
            def error(msg: str) -> None:
                raise SystemExit(f"error: {msg}")
        _run_active_refinement(args, _ArgErr())
        return

    if bool(args.generate_suite):
        suite_cfg = resolve_suite_config(args)
        suite_out = Path(str(args.suite_out_dir).strip()) if str(getattr(args, "suite_out_dir", "") or "").strip() else None
        run_suite_generation(suite_cfg, suite_out_dir=suite_out, combine_ood=bool(args.combine_ood))
        return

    cfg = resolve_cloud_config(args)

    if bool(args.print_config):
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))

    if str(args.dump_config).strip():
        cfg.to_json(str(args.dump_config).strip())

    run_generation(cfg, overwrite=bool(getattr(args, "overwrite", False)))


if __name__ == "__main__":
    main()
