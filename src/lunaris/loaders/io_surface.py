# lunaris/loaders/io_surface.py
"""
Surface Data Layer (Topography & Albedo Grids)
=============================================

This module owns *data acquisition and sampling* for Moon surface products
used by the simulation (e.g., LOLA/LDEM topography and LOLA-derived albedo).

It is intentionally **physics-free**:
- parses PDS3 labels / resolves product files
- loads rasters into contiguous NumPy arrays
- provides lightweight provider interfaces (TopographyProvider/AlbedoProvider)
- offers a small facade (SurfaceProvider) that bundles tables for the dynamics layer

The actual force/acceleration models (SRP, albedo radiation pressure, thermal
reradiation, etc.) live in their respective effect modules.
"""

# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import warnings
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, Union, Callable, TypeVar

import numpy as np
import numpy.typing as npt

from lunaris.common.constants import R_MOON_MEAN
from lunaris.common.math_utils import (
    clamp,
    latlon_from_xyz_m,
    sample_2d_bilinear,
    sample_2d_nearest,
    sample_2d_scaled_bilinear,
    sample_2d_scaled_nearest,
    wrap_lon_deg,
)

_LABEL_NAME_SUFFIXES: tuple[str, ...] = (
    ".lbl",
    ".lbl.txt",
    ".lab",
    ".lab.txt",
    ".label",
    ".label.txt",
)

# -----------------------------------------------------------------------------
# Local type aliases
# -----------------------------------------------------------------------------
Number = Union[int, float]
PathLike = Union[str, Path]


# =============================================================================
# 1.                           Provider abstractions
# =============================================================================

class TopographyProvider(ABC):
    """
    Topography provider interface.

    Contract
    --------
    Inputs
      lat_rad : latitude [rad]
      lon_rad : longitude [rad] (any real; implementations may wrap)

    Output
      Surface radius from body center [m].
    """

    @abstractmethod
    def radius_m(self, lat_rad: float, lon_rad: float) -> float:
        """Return surface radius from the body center [m]."""
        raise NotImplementedError

    def __call__(self, lat_rad: float, lon_rad: float) -> float:
        return self.radius_m(lat_rad, lon_rad)


@dataclass(frozen=True, slots=True)
class ConstantTopography(TopographyProvider):
    """
    Constant-radius topography (perfect sphere).

    Parameters
    ----------
    radius0_m
        Spherical radius [m]. Must be > 0.
    """
    radius0_m: float  # e.g., default at call site: ConstantTopography(R_MOON)

    def __post_init__(self) -> None:
        r = float(self.radius0_m)
        if not (r > 0.0):
            raise ValueError("ConstantTopography.radius0_m must be > 0.")
        object.__setattr__(self, "radius0_m", r)

    def radius_m(self, lat_rad: float, lon_rad: float) -> float:
        # lat/lon intentionally unused: constant sphere model
        return self.radius0_m


class AlbedoProvider(ABC):
    """
    Albedo provider interface.

    Contract
    --------
    Inputs
      lat_rad : latitude in radians
      lon_rad : longitude in radians (any real value; implementations may wrap)

    Output
      Local albedo in [0, 1].

    Notes
    -----
    - Minimal interface on purpose: can be backed by a constant, a raster/grid,
      an analytic model, or spherical harmonics.
    """

    @abstractmethod
    def albedo(self, lat_rad: float, lon_rad: float) -> float:
        """Return local albedo in [0, 1]."""
        raise NotImplementedError

    def __call__(self, lat_rad: float, lon_rad: float) -> float:
        """Allow provider(lat, lon) syntax."""
        return self.albedo(lat_rad, lon_rad)


@dataclass(frozen=True, slots=True)
class ConstantAlbedo(AlbedoProvider):
    """
    Constant albedo everywhere.

    Parameters
    ----------
    value
        Constant albedo value. Will be clamped to [0, 1] at construction time.
    """
    value: float = 0.12

    def __post_init__(self) -> None:
        a = float(self.value)
        # Clamp once at construction time (keeps hot-path minimal).
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
        object.__setattr__(self, "value", a)

    def albedo(self, lat_rad: float, lon_rad: float) -> float:
        return self.value





# =============================================================================
# 6.                       PDS3 LABEL & FILE UTILITIES
# =============================================================================

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class PDS3ParseError(ValueError):
    """Raised when a PDS3 label cannot be parsed in a consistent way."""
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


# --- precompiled regex (faster + keeps patterns centralized) ---
_RE_PDS_COMMENT = re.compile(r"/\*.*?\*/", flags=re.DOTALL)

_RE_IMAGE_STR = re.compile(r'\^IMAGE\s*=\s*"([^"]+)"', flags=re.IGNORECASE)
_RE_IMAGE_TUP = re.compile(r'\^IMAGE\s*=\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)', flags=re.IGNORECASE)
_RE_IMAGE_REC = re.compile(r"\^IMAGE\s*=\s*(\d+)", flags=re.IGNORECASE)

_RE_FILE_NAME = re.compile(r'\bFILE_NAME\s*=\s*"([^"]+)"', flags=re.IGNORECASE | re.MULTILINE)
_RE_RECORD_BYTES = re.compile(r"\bRECORD_BYTES\s*=\s*(\d+)", flags=re.IGNORECASE | re.MULTILINE)


def _read_text(path: Union[str, Path]) -> str:
    """Read text robustly (UTF-8), ignoring decoding errors."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _strip_pds_comments(text: str) -> str:
    """Remove PDS3-style block comments: /* ... */."""
    return _RE_PDS_COMMENT.sub("", text)


def _re_find_one(pattern: Union[str, re.Pattern[str]], text: str, cast: Callable[[str], T] = str) -> T:
    """
    Find a required PDS field using a regex capturing group (group 1).

    Raises
    ------
    PDS3ParseError
        If the pattern is not found or the value cannot be cast.
    """
    rx = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE) if isinstance(pattern, str) else pattern
    m = rx.search(text)
    if not m:
        raise PDS3ParseError(f"PDS3 parse error: missing required field for pattern: {rx.pattern!r}")
    raw = m.group(1)
    try:
        return cast(raw)
    except Exception as e:
        raise PDS3ParseError(
            f"PDS3 parse error: failed to cast value {raw!r} for pattern {rx.pattern!r}"
        ) from e


def _re_find_optional(
    pattern: Union[str, re.Pattern[str]],
    text: str,
    cast: Callable[[str], T] = str,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Find an optional PDS field using a regex capturing group (group 1).

    Returns
    -------
    value or default
        If pattern is not found, returns default.
        If found but casting fails, raises PDS3ParseError (fail-fast).
    """
    rx = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE) if isinstance(pattern, str) else pattern
    m = rx.search(text)
    if not m:
        return default
    raw = m.group(1)
    try:
        return cast(raw)
    except Exception as e:
        raise PDS3ParseError(
            f"PDS3 parse error: failed to cast value {raw!r} for pattern {rx.pattern!r}"
        ) from e


def _strip_known_label_suffixes(p: Union[str, Path]) -> Path:
    """
    Remove common label suffixes to recover the base stem.

    Example:
      product.lbl.txt -> product
    """
    base = Path(p)
    while base.suffix.lower() in (".lbl", ".txt", ".label", ".xml"):
        base = base.with_suffix("")
    return base


def _find_case_insensitive(path: Union[str, Path]) -> Optional[Path]:
    """
    Return an existing path even if the filename casing differs.

    Useful for PDS archives where detached IMG files are often uppercase on disk.
    """
    path = Path(path)
    if path.exists():
        return path

    parent = path.parent
    if not parent.exists():
        return None

    target = path.name.lower()
    try:
        for cand in parent.iterdir():
            if cand.name.lower() == target:
                return cand
    except OSError:
        return None

    return None


def _parse_pds3_image_pointer(label_text: str, *, strict: bool = True) -> Tuple[Optional[str], int]:
    """
    Parse the PDS3 '^IMAGE' pointer.

    Supported formats:
      ^IMAGE = "FILE.IMG"             -> (FILE.IMG, 1)
      ^IMAGE = ("FILE.IMG", 123)      -> (FILE.IMG, 123)
      ^IMAGE = 123                    -> (None, 123)   (data in the label file itself)

    Parameters
    ----------
    strict
        If True, missing ^IMAGE raises PDS3ParseError.
        If False, returns (None, 1).

    Returns
    -------
    (filename, start_record)
        filename is None if the label points to itself.
    """
    m = _RE_IMAGE_STR.search(label_text)
    if m:
        return m.group(1), 1

    m = _RE_IMAGE_TUP.search(label_text)
    if m:
        return m.group(1), int(m.group(2))

    m = _RE_IMAGE_REC.search(label_text)
    if m:
        return None, int(m.group(1))

    if strict:
        raise PDS3ParseError("PDS3 parse error: missing required ^IMAGE pointer.")
    return None, 1


def _resolve_img_from_label(label_path: Union[str, Path], *, strict: bool = True) -> Tuple[Path, int]:
    """
    Resolve the binary IMG path and byte offset from a PDS3 label file.

    Logic:
      1) Read label
      2) Parse ^IMAGE pointer (and optional FILE_NAME / RECORD_BYTES)
      3) Compute offset in bytes = RECORD_BYTES * (start_record - 1)
      4) Resolve the target file using:
         - ptr filename (from ^IMAGE)
         - FILE_NAME (if present)
         - label stem + (.IMG/.img) fallback
      5) Case-insensitive lookup on disk

    Parameters
    ----------
    label_path
        Path to the detached PDS3 label.
    strict
        If True, missing ^IMAGE is an error. If False, allows fallback behavior.

    Returns
    -------
    (img_path, offset_bytes)
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    raw_text = _read_text(label_path)
    raw_text = _strip_pds_comments(raw_text)

    # 1) Parse metadata
    ptr_file, start_record = _parse_pds3_image_pointer(raw_text, strict=strict)
    explicit_file_name = _re_find_optional(_RE_FILE_NAME, raw_text, str, default=None)
    record_bytes = _re_find_optional(_RE_RECORD_BYTES, raw_text, int, default=None)

    # 2) Compute offset
    offset_bytes = 0
    if start_record > 1:
        if record_bytes is None:
            # In strict mode, if ^IMAGE implies an offset but RECORD_BYTES is missing, fail fast.
            if strict:
                raise PDS3ParseError(
                    "PDS3 parse error: ^IMAGE specifies a start record > 1 but RECORD_BYTES is missing."
                )
        else:
            offset_bytes = int(record_bytes) * (int(start_record) - 1)

    # 3) Candidate list
    candidates: list[Path] = []

    # Priority A: explicit names from label
    for name in (ptr_file, explicit_file_name):
        if name:
            candidates.append(label_path.parent / name)

    # Priority B: implicit convention (same stem)
    stem = _strip_known_label_suffixes(label_path)
    candidates.append(stem.with_suffix(".IMG"))
    candidates.append(stem.with_suffix(".img"))

    # 4) Resolve
    img_path: Optional[Path] = None

    # Special case: data inside label (ptr_file None) with a non-trivial start record
    if ptr_file is None and start_record > 1:
        img_path = label_path
    else:
        for cand in candidates:
            found = _find_case_insensitive(cand)
            if found is not None:
                img_path = found
                break

    if img_path is None:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Could not resolve binary IMG file for label: {label_path.name}\n"
            f"Tried locations:\n  {tried}"
        )

    return img_path, int(offset_bytes)





# =============================================================================
# 7.                   TOPOGRAPHY (LOLA / LDEM) MODELS
# =============================================================================

@dataclass(frozen=True, slots=True)
class PDS3MapInfo:
    """
    Minimal PDS3 metadata required to locate and interpret LOLA/LDEM raster products.

    Notes
    -----
    - `lines` and `samples` define the raster shape (lines, samples).
    - `sample_type` + `sample_bits` define the binary dtype.
    - `scaling_factor` and `offset_km` convert stored samples to physical values:
        value_km = stored * scaling_factor + offset_km
      (the exact convention is product-dependent; keep this as parsed metadata).
    - Longitudes are interpreted using `positive_lon_direction` together with the
      provided west/east extents.
    """
    # --- image structure ---
    lines: int
    samples: int
    sample_type: str
    sample_bits: int
    unit: str
    scaling_factor: float
    offset_km: float

    # --- map projection ---
    map_projection_type: str
    map_resolution_ppd: float  # pixels per degree
    max_lat_deg: float
    min_lat_deg: float
    west_lon_deg: float
    east_lon_deg: float
    positive_lon_direction: str
    center_lon_deg: float
    center_lat_deg: float

    # --- optional reference ellipsoid radii (km) ---
    a_axis_radius_km: Optional[float] = None
    b_axis_radius_km: Optional[float] = None
    c_axis_radius_km: Optional[float] = None


@lru_cache(maxsize=128)
def parse_ldem_label(label_path: Union[str, Path]) -> PDS3MapInfo:
    """
    Parse the minimal LOLA LDEM-style metadata from a PDS3 label.

    This parser is intentionally conservative and fail-fast: required keys must
    exist, optional keys are captured when present.

    Parameters
    ----------
    label_path
        Path to the PDS3 label file (.LBL/.TXT/.XML-style detached label).

    Returns
    -------
    PDS3MapInfo
        Parsed metadata container.
    """
    txt = _strip_pds_comments(_read_text(label_path))

    # --- raster geometry ---
    lines = _re_find_one(r"\bLINES\s*=\s*(\d+)", txt, int)
    samples = _re_find_one(r"\bLINE_SAMPLES\s*=\s*(\d+)", txt, int)
    sample_type = _re_find_one(r"\bSAMPLE_TYPE\s*=\s*([A-Z0-9_]+)", txt, str)
    sample_bits = _re_find_one(r"\bSAMPLE_BITS\s*=\s*(\d+)", txt, int)

    # --- value scaling ---
    unit = _re_find_one(r"\bUNIT\s*=\s*([A-Z]+)", txt, str)
    scaling_factor = _re_find_one(r"\bSCALING_FACTOR\s*=\s*([0-9.+-Ee]+)", txt, float)
    offset_km = _re_find_one(r"\bOFFSET\s*=\s*([0-9.+-Ee]+)", txt, float)

    # --- projection ---
    map_projection_type = _re_find_one(r'\bMAP_PROJECTION_TYPE\s*=\s*"([^"]+)"', txt, str)
    map_resolution_ppd = _re_find_one(r"\bMAP_RESOLUTION\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)

    # --- extents / conventions ---
    max_lat_deg = _re_find_one(r"\bMAXIMUM_LATITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    min_lat_deg = _re_find_one(r"\bMINIMUM_LATITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    west_lon_deg = _re_find_one(r"\bWESTERNMOST_LONGITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    east_lon_deg = _re_find_one(r"\bEASTERNMOST_LONGITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    positive_lon_direction = _re_find_one(r'\bPOSITIVE_LONGITUDE_DIRECTION\s*=\s*"([^"]+)"', txt, str)

    center_lat_deg = _re_find_one(r"\bCENTER_LATITUDE\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)
    center_lon_deg = _re_find_one(r"\bCENTER_LONGITUDE\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)

    # --- optional ellipsoid radii (km) ---
    a_axis = _re_find_optional(r"\bA_AXIS_RADIUS\s*=\s*([0-9.+-Ee]+)", txt, float, default=None)
    b_axis = _re_find_optional(r"\bB_AXIS_RADIUS\s*=\s*([0-9.+-Ee]+)", txt, float, default=None)
    c_axis = _re_find_optional(r"\bC_AXIS_RADIUS\s*=\s*([0-9.+-Ee]+)", txt, float, default=None)

    return PDS3MapInfo(
        lines=lines,
        samples=samples,
        sample_type=sample_type,
        sample_bits=sample_bits,
        unit=unit,
        scaling_factor=scaling_factor,
        offset_km=offset_km,
        map_projection_type=map_projection_type,
        map_resolution_ppd=map_resolution_ppd,
        max_lat_deg=max_lat_deg,
        min_lat_deg=min_lat_deg,
        west_lon_deg=west_lon_deg,
        east_lon_deg=east_lon_deg,
        positive_lon_direction=positive_lon_direction,
        center_lon_deg=center_lon_deg,
        center_lat_deg=center_lat_deg,
        a_axis_radius_km=a_axis,
        b_axis_radius_km=b_axis,
        c_axis_radius_km=c_axis,
    )


def dtype_from_pds3_sample(sample_type: str, sample_bits: int) -> np.dtype:
    """
    Convert PDS3 SAMPLE_TYPE / SAMPLE_BITS to a NumPy dtype.

    The PDS3 standard allows several endian- and signedness-qualified encodings.
    The LOLA LDEM/LDAM products commonly used in this project are typically:

    - PC_REAL (little-endian IEEE float32/float64)
    - MSB_INTEGER (big-endian signed int16)

    This mapper is intentionally conservative but supports the most common integer
    variants as well.

    Parameters
    ----------
    sample_type
        PDS3 SAMPLE_TYPE value (case-insensitive).
    sample_bits
        PDS3 SAMPLE_BITS value.

    Returns
    -------
    np.dtype
        NumPy dtype with correct endian/sign.

    Raises
    ------
    ValueError
        If the type/bit-depth combination is unsupported.
    """
    st = str(sample_type).strip().upper()
    bits = int(sample_bits)

    # --- Floating point ----------------------------------------------------
    # "PC_REAL" is the canonical little-endian IEEE encoding in many PDS3 labels.
    if st in {"PC_REAL", "IEEE_REAL", "REAL"}:
        if bits == 32:
            return np.dtype("<f4")
        if bits == 64:
            return np.dtype("<f8")
        raise ValueError(f"Unsupported REAL bit depth: {bits}")

    # --- Signed integers ---------------------------------------------------
    if st in {"MSB_INTEGER", "INTEGER"}:
        if bits == 8:
            return np.dtype(">i1")  # endian irrelevant
        if bits == 16:
            return np.dtype(">i2")
        if bits == 32:
            return np.dtype(">i4")
        if bits == 64:
            return np.dtype(">i8")
        raise ValueError(f"Unsupported INTEGER bit depth: {bits}")

    if st == "LSB_INTEGER":
        if bits == 8:
            return np.dtype("<i1")
        if bits == 16:
            return np.dtype("<i2")
        if bits == 32:
            return np.dtype("<i4")
        if bits == 64:
            return np.dtype("<i8")
        raise ValueError(f"Unsupported LSB_INTEGER bit depth: {bits}")

    # --- Unsigned integers -------------------------------------------------
    if st in {"MSB_UNSIGNED_INTEGER", "MSB_UNSIGNED"}:
        if bits == 8:
            return np.dtype(">u1")
        if bits == 16:
            return np.dtype(">u2")
        if bits == 32:
            return np.dtype(">u4")
        if bits == 64:
            return np.dtype(">u8")
        raise ValueError(f"Unsupported MSB_UNSIGNED_INTEGER bit depth: {bits}")

    if st in {"LSB_UNSIGNED_INTEGER", "LSB_UNSIGNED"}:
        if bits == 8:
            return np.dtype("<u1")
        if bits == 16:
            return np.dtype("<u2")
        if bits == 32:
            return np.dtype("<u4")
        if bits == 64:
            return np.dtype("<u8")
        raise ValueError(f"Unsupported LSB_UNSIGNED_INTEGER bit depth: {bits}")

    raise ValueError(f"Unsupported PDS3 sample encoding: SAMPLE_TYPE={st!r}, SAMPLE_BITS={bits}")



# =============================================================================
# 8.                              GRID MANAGER
# =============================================================================

class TopographyGrid(TopographyProvider):
    """
    High-performance reader & sampler for LOLA LDEM topography rasters (PDS3).

    Features
    --------
    - Lazy load via np.memmap (default) or full RAM load.
    - Nearest / bilinear sampling in (lat, lon).
    - Longitude wrapping to product bounds.
    - Pixel-center registration consistent with PDS3 cylindrical products.

    Parameters
    ----------
    label_path
        PDS3 label path.
    img_path
        Optional explicit IMG path. If None, it is resolved from the label.
    mmap
        If True uses memory mapping; otherwise loads the full array into RAM.
    flip_lat
        If True reverses latitude axis interpretation (useful if product is stored flipped).
    """

    def __init__(
        self,
        label_path: PathLike,
        img_path: Optional[PathLike] = None,
        *,
        mmap: bool = True,
        flip_lat: bool = False,
    ) -> None:
        self.label_path = Path(label_path)
        self.info = parse_ldem_label(self.label_path)

        resolved_img, offset_bytes = _resolve_img_from_label(self.label_path)

        self.img_path = Path(resolved_img if img_path is None else img_path)
        self._offset_bytes = int(offset_bytes)
        self._dtype = dtype_from_pds3_sample(self.info.sample_type, self.info.sample_bits)
        self._flip_lat = bool(flip_lat)

        # Grid angular resolution (deg/pixel)
        self.ddeg = 1.0 / float(self.info.map_resolution_ppd)

        # Pixel-center coordinates (deg). For cylindrical maps, lines run N->S by convention.
        lat_n = float(self.info.max_lat_deg) - 0.5 * self.ddeg
        lat_s = float(self.info.min_lat_deg) + 0.5 * self.ddeg
        if not self._flip_lat:
            self._lat_centers_deg = np.linspace(lat_n, lat_s, int(self.info.lines), dtype=np.float64)
        else:
            self._lat_centers_deg = np.linspace(lat_s, lat_n, int(self.info.lines), dtype=np.float64)

        lon_w = float(self.info.west_lon_deg) + 0.5 * self.ddeg
        lon_e = float(self.info.east_lon_deg) - 0.5 * self.ddeg
        self._lon_centers_deg = np.linspace(lon_w, lon_e, int(self.info.samples), dtype=np.float64)

        self._arr: Optional[npt.NDArray[np.generic]] = None
        if mmap:
            self.open(mmap=True)

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    def open(self, *, mmap: bool = True) -> "TopographyGrid":
        """Open/attach the raster array (idempotent)."""
        if self._arr is not None:
            return self

        shape = (int(self.info.lines), int(self.info.samples))

        if mmap:
            self._arr = np.memmap(
                self.img_path,
                dtype=self._dtype,
                mode="r",
                shape=shape,
                order="C",
                offset=self._offset_bytes,
            )
        else:
            with self.img_path.open("rb") as f:
                f.seek(self._offset_bytes)
                count = shape[0] * shape[1]
                self._arr = np.fromfile(f, dtype=self._dtype, count=count).reshape(shape)

        return self

    @property
    def dn_km(self) -> npt.NDArray[np.generic]:
        """
        Raw Data Numbers array (as stored in the file).

        For LOLA LDEM products this is typically in km (or integer DN later scaled),
        but interpretation must follow (SCALING_FACTOR, OFFSET) in the label.
        """
        if self._arr is None:
            self.open(mmap=True)
        # mypy: _arr non-None after open()
        return self._arr  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    @property
    def reference_radius_m(self) -> float:
        """
        Reference radius [m] implied by the product.

        Policy:
        - Prefer OFFSET if present and positive (common in LDEM: radius reference in km).
        - Else if ellipsoid radii exist, use their mean.
        - Else fallback to global constant reference (R_MOON_MEAN).
        """
        off_km = float(self.info.offset_km)
        if off_km > 0.0:
            return off_km * 1_000.0

        axes = (self.info.a_axis_radius_km, self.info.b_axis_radius_km, self.info.c_axis_radius_km)
        valid = [float(v) for v in axes if v is not None]
        if valid:
            return (sum(valid) / len(valid)) * 1_000.0

        return float(R_MOON_MEAN)

    def altitude_above_reference_m(self, x_m: Number, y_m: Number, z_m: Number) -> float:
        """Altitude [m] above the product reference radius for a Cartesian point."""
        _, _, r_m = latlon_from_xyz_m(float(x_m), float(y_m), float(z_m), lon_0_360=True)
        return float(r_m) - self.reference_radius_m

    def _ij_from_latlon(self, lat_deg: Number, lon_deg: Number) -> Tuple[float, float]:
        """
        Convert (lat, lon) in degrees -> fractional (i, j) indices into the raster.

        i grows downward in array coordinates.
        """
        lat = float(lat_deg)
        lon = wrap_lon_deg(float(lon_deg), float(self.info.west_lon_deg), float(self.info.east_lon_deg))

        # Clamp latitude to avoid edge overflow for interpolation
        lat_min = float(self._lat_centers_deg.min())
        lat_max = float(self._lat_centers_deg.max())
        lat = clamp(lat, min(lat_min, lat_max), max(lat_min, lat_max))

        lat0 = float(self._lat_centers_deg[0])
        if not self._flip_lat:
            # N -> S storage: latitude decreases as i increases
            i = (lat0 - lat) / self.ddeg
        else:
            # flipped: latitude increases as i increases
            i = (lat - lat0) / self.ddeg

        lon0 = float(self._lon_centers_deg[0])
        j = (lon - lon0) / self.ddeg
        return i, j

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def sample_nearest(self, lat_deg: Number, lon_deg: Number, *, kind: str = "height_km") -> float:
        """Nearest-neighbor sampling with unit conversion via `kind`."""
        i_f, j_f = self._ij_from_latlon(lat_deg, lon_deg)
        dn = float(
            sample_2d_nearest(
                self.dn_km, float(i_f), float(j_f),
                int(self.info.lines), int(self.info.samples),
            )
        )
        return self._convert_scalar(dn, kind)

    def sample_bilinear(self, lat_deg: Number, lon_deg: Number, *, kind: str = "height_km") -> float:
        """Bilinear sampling with unit conversion via `kind`."""
        i_f, j_f = self._ij_from_latlon(lat_deg, lon_deg)
        dn = float(
            sample_2d_bilinear(
                self.dn_km, float(i_f), float(j_f),
                int(self.info.lines), int(self.info.samples),
            )
        )
        return self._convert_scalar(dn, kind)

    def _convert_scalar(self, dn_val: float, kind: str) -> float:
        """
        Convert stored DN to the requested quantity.

        Conventions (typical for LDEM):
        - height_km = DN * scaling_factor
        - radius_km = height_km + offset_km
        """
        k = kind.strip().lower()
        sf = float(self.info.scaling_factor)
        off_km = float(self.info.offset_km)

        if k in ("dn",):
            return float(dn_val)

        # height relative to reference (km / m)
        h_km = float(dn_val) * sf
        if k == "height_km":
            return h_km
        if k == "height_m":
            return h_km * 1_000.0

        # absolute radius from center (km / m)
        r_km = h_km + off_km
        if k == "radius_km":
            return r_km
        if k == "radius_m":
            return r_km * 1_000.0

        raise ValueError(f"Unknown sample kind: {kind!r}")

    # -------------------------------------------------------------------------
    # Provider API
    # -------------------------------------------------------------------------
    def radius_m(self, lat_rad: float, lon_rad: float) -> float:
        """Return surface radius [m] at (lat, lon) in radians."""
        lat_deg = math.degrees(float(lat_rad))
        lon_deg = math.degrees(float(lon_rad))
        return float(self.sample_bilinear(lat_deg, lon_deg, kind="radius_m"))

    def height_m(self, lat_rad: float, lon_rad: float) -> float:
        """Return terrain height [m] above reference at (lat, lon) in radians."""
        lat_deg = math.degrees(float(lat_rad))
        lon_deg = math.degrees(float(lon_rad))
        return float(self.sample_bilinear(lat_deg, lon_deg, kind="height_m"))


class PDS3CylindricalGrid:
    """
    Generic reader/sampler for PDS3 cylindrical rasters with linear scaling.

        value = DN * scaling_factor + offset

    Typical use:
    - LOLA LDAM albedo products
    - other cylindrical maps (slope/roughness/etc.)
    """

    def __init__(
        self,
        label_path: PathLike,
        img_path: Optional[PathLike] = None,
        *,
        mmap: bool = True,
        flip_lat: bool = False,
    ) -> None:
        self.label_path = Path(label_path)
        self.info = parse_pds3_cyl_label(self.label_path)

        resolved_img, offset_bytes = _resolve_img_from_label(self.label_path)
        self.img_path = Path(resolved_img if img_path is None else img_path)

        self._offset_bytes = int(offset_bytes)
        self._dtype = dtype_from_pds3_sample(self.info.sample_type, self.info.sample_bits)
        self._flip_lat = bool(flip_lat)

        self.ddeg = 1.0 / float(self.info.map_resolution_ppd)

        lat_n = float(self.info.max_lat_deg) - 0.5 * self.ddeg
        lat_s = float(self.info.min_lat_deg) + 0.5 * self.ddeg
        if not self._flip_lat:
            self._lat_centers_deg = np.linspace(lat_n, lat_s, int(self.info.lines), dtype=np.float64)
        else:
            self._lat_centers_deg = np.linspace(lat_s, lat_n, int(self.info.lines), dtype=np.float64)

        lon_w = float(self.info.west_lon_deg) + 0.5 * self.ddeg
        lon_e = float(self.info.east_lon_deg) - 0.5 * self.ddeg
        self._lon_centers_deg = np.linspace(lon_w, lon_e, int(self.info.samples), dtype=np.float64)

        self._arr: Optional[npt.NDArray[np.generic]] = None
        if mmap:
            self.open(mmap=True)

    def open(self, *, mmap: bool = True) -> "PDS3CylindricalGrid":
        """Open/attach the raster array (idempotent)."""
        if self._arr is not None:
            return self

        shape = (int(self.info.lines), int(self.info.samples))

        if mmap:
            self._arr = np.memmap(
                self.img_path,
                dtype=self._dtype,
                mode="r",
                shape=shape,
                order="C",
                offset=self._offset_bytes,
            )
        else:
            with self.img_path.open("rb") as f:
                f.seek(self._offset_bytes)
                count = shape[0] * shape[1]
                self._arr = np.fromfile(f, dtype=self._dtype, count=count).reshape(shape)

        return self

    @property
    def dn(self) -> npt.NDArray[np.generic]:
        """Raw DN array as stored in the file."""
        if self._arr is None:
            self.open(mmap=True)
        return self._arr  # type: ignore[return-value]

    @property
    def lat_centers_deg(self) -> npt.NDArray[np.float64]:
        return self._lat_centers_deg

    @property
    def lon_centers_deg(self) -> npt.NDArray[np.float64]:
        return self._lon_centers_deg

    def _ij_from_latlon(self, lat_deg: Number, lon_deg: Number) -> Tuple[float, float]:
        lat = float(lat_deg)
        lon = wrap_lon_deg(float(lon_deg), float(self.info.west_lon_deg), float(self.info.east_lon_deg))

        lat_min = float(self._lat_centers_deg.min())
        lat_max = float(self._lat_centers_deg.max())
        lat = clamp(lat, min(lat_min, lat_max), max(lat_min, lat_max))

        lat0 = float(self._lat_centers_deg[0])
        if not self._flip_lat:
            i = (lat0 - lat) / self.ddeg
        else:
            i = (lat - lat0) / self.ddeg

        lon0 = float(self._lon_centers_deg[0])
        j = (lon - lon0) / self.ddeg
        return i, j

    def sample_nearest(self, lat_deg: Number, lon_deg: Number) -> float:
        """Nearest sample with scaling + missing handling."""
        i_f, j_f = self._ij_from_latlon(lat_deg, lon_deg)
        return float(
            sample_2d_scaled_nearest(
                self.dn, float(i_f), float(j_f),
                int(self.info.lines), int(self.info.samples),
                float(self.info.scaling_factor),
                float(self.info.offset),
                float(self.info.missing_constant),
            )
        )

    def sample_bilinear(self, lat_deg: Number, lon_deg: Number) -> float:
        """Bilinear sample with scaling + missing handling."""
        i_f, j_f = self._ij_from_latlon(lat_deg, lon_deg)
        return float(
            sample_2d_scaled_bilinear(
                self.dn, float(i_f), float(j_f),
                int(self.info.lines), int(self.info.samples),
                float(self.info.scaling_factor),
                float(self.info.offset),
                float(self.info.missing_constant),
            )
        )


class LOLAAlbedoGrid(PDS3CylindricalGrid, AlbedoProvider):
    """
    AlbedoProvider backed by LOLA LDAM cylindrical products.

    Returns albedo clamped to [0, 1]. If the sample is NaN/missing, falls back
    to a conservative default (typical global-average lunar albedo).
    """

    _FALLBACK_ALBEDO: float = 0.12

    def albedo(self, lat_rad: float, lon_rad: float) -> float:
        lat_deg = math.degrees(float(lat_rad))
        lon_deg = math.degrees(float(lon_rad))

        val = float(self.sample_bilinear(lat_deg, lon_deg))

        if not math.isfinite(val):
            return self._FALLBACK_ALBEDO
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val


@dataclass(frozen=True, slots=True)
class SurfaceGrids:
    """
    Optional surface raster grids (topography + albedo).

    This container makes it easy to pass around datasets without tuple unpacking.
    """
    topo: Optional[TopographyGrid] = None
    albedo: Optional[LOLAAlbedoGrid] = None


def load_surface_grids(
    *,
    ldem_root: Optional[PathLike] = None,
    albedo_root: Optional[PathLike] = None,
    ldem_ppd: Optional[int] = None,
    mmap: bool = True,
    flip_lat_ldem: bool = False,
    flip_lat_albedo: bool = False,
) -> SurfaceGrids:
    """
    Load available surface grids.

    - If a dataset root is None, that dataset is skipped.
    - Missing datasets return as None in the resulting container.

    Returns
    -------
    SurfaceGrids
        topo and/or albedo may be None.
    """
    topo: Optional[TopographyGrid] = None
    alb: Optional[LOLAAlbedoGrid] = None

    if ldem_root is not None:
        ldem_lbl, ldem_img = find_ldem_product(ldem_root, ppd=ldem_ppd)
        topo = TopographyGrid(ldem_lbl, ldem_img, mmap=mmap, flip_lat=flip_lat_ldem)

    if albedo_root is not None:
        alb_lbl, alb_img = find_lola_albedo_product(albedo_root)
        alb = LOLAAlbedoGrid(alb_lbl, alb_img, mmap=mmap, flip_lat=flip_lat_albedo)

    return SurfaceGrids(topo=topo, albedo=alb)





# =============================================================================
# 9.                  GENERIC CYLINDRICAL RASTER (ALBEDO / LDAM)
# =============================================================================

@dataclass(frozen=True, slots=True)
class PDS3RasterInfo:
    """Minimal metadata for a PDS3 cylindrical raster with linear scaling."""
    lines: int
    samples: int
    sample_type: str
    sample_bits: int
    unit: str

    scaling_factor: float
    offset: float
    missing_constant: float  # NaN if not specified

    map_projection_type: str
    map_resolution_ppd: float  # pixels per degree

    max_lat_deg: float
    min_lat_deg: float
    west_lon_deg: float
    east_lon_deg: float
    positive_lon_direction: str
    center_lon_deg: float
    center_lat_deg: float


@lru_cache(maxsize=128)
def parse_pds3_cyl_label(label_path: PathLike) -> PDS3RasterInfo:
    """
    Parse common fields for PDS3 cylindrical rasters with linear scaling:

        value = DN * SCALING_FACTOR + OFFSET

    Typical use: LOLA Albedo (LDAM_*) products.

    Raises
    ------
    FileNotFoundError
        If label does not exist.
    ValueError
        If required fields are missing.
    """
    p = Path(label_path)
    if not p.exists():
        raise FileNotFoundError(f"PDS3 label not found: {p}")

    raw = _read_text(p)
    txt = _strip_pds_comments(raw)

    # IMAGE object fields
    lines = _re_find_one(r"\bLINES\s*=\s*(\d+)", txt, int)
    samples = _re_find_one(r"\bLINE_SAMPLES\s*=\s*(\d+)", txt, int)
    sample_type = _re_find_one(r"\bSAMPLE_TYPE\s*=\s*([A-Z0-9_]+)", txt, str)
    sample_bits = _re_find_one(r"\bSAMPLE_BITS\s*=\s*(\d+)", txt, int)
    unit = _re_find_one(r"\bUNIT\s*=\s*([A-Z]+)", txt, str)

    scaling_factor = _re_find_one(r"\bSCALING_FACTOR\s*=\s*([0-9.+-Ee]+)", txt, float)
    offset = _re_find_one(r"\bOFFSET\s*=\s*([0-9.+-Ee]+)", txt, float)

    # Optional missing constant (often present in derived maps)
    # Robust numeric pattern (supports sign + scientific notation)
    missing_constant = _re_find_optional(
        r"\bMISSING_CONSTANT\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[Ee][+-]?\d+)?)",
        txt,
        float,
        default=float("nan"),
    )

    # Map projection block
    map_projection_type = _re_find_one(r"\bMAP_PROJECTION_TYPE\s*=\s*\"([^\"]+)\"", txt, str)
    map_resolution_ppd = _re_find_one(r"\bMAP_RESOLUTION\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)

    max_lat_deg = _re_find_one(r"\bMAXIMUM_LATITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    min_lat_deg = _re_find_one(r"\bMINIMUM_LATITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    west_lon_deg = _re_find_one(r"\bWESTERNMOST_LONGITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    east_lon_deg = _re_find_one(r"\bEASTERNMOST_LONGITUDE\s*=\s*([0-9.+-Ee]+)", txt, float)
    positive_lon_direction = _re_find_one(
        r"\bPOSITIVE_LONGITUDE_DIRECTION\s*=\s*\"([^\"]+)\"",
        txt,
        str,
    )

    center_lat_deg = _re_find_one(r"\bCENTER_LATITUDE\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)
    center_lon_deg = _re_find_one(r"\bCENTER_LONGITUDE\s*=\s*([0-9.+-Ee]+)\s*<", txt, float)

    return PDS3RasterInfo(
        lines=int(lines),
        samples=int(samples),
        sample_type=str(sample_type),
        sample_bits=int(sample_bits),
        unit=str(unit),
        scaling_factor=float(scaling_factor),
        offset=float(offset),
        missing_constant=float(missing_constant),
        map_projection_type=str(map_projection_type),
        map_resolution_ppd=float(map_resolution_ppd),
        max_lat_deg=float(max_lat_deg),
        min_lat_deg=float(min_lat_deg),
        west_lon_deg=float(west_lon_deg),
        east_lon_deg=float(east_lon_deg),
        positive_lon_direction=str(positive_lon_direction),
        center_lon_deg=float(center_lon_deg),
        center_lat_deg=float(center_lat_deg),
    )


def _iter_label_candidates(root: Path) -> list[Path]:
    """
    Return label-like files under a directory (non-recursive).

    Many of the repository datasets ship detached PDS3 labels as compound names
    like ``product.lbl.txt`` rather than plain ``product.lbl``. Matching only on
    the final suffix therefore misses perfectly valid labels and forces an
    unnecessary fallback to constant-radius/constant-albedo behavior.
    """

    out: list[Path] = []
    try:
        for child in root.iterdir():
            if not child.is_file():
                continue
            lower_name = child.name.lower()
            if any(lower_name.endswith(suffix) for suffix in _LABEL_NAME_SUFFIXES):
                out.append(child)
    except OSError:
        return []
    return sorted(out)


def find_ldem_product(root: PathLike, *, ppd: Optional[int] = None) -> Tuple[Path, Path]:
    """
    Locate a LOLA LDEM (topography) label + IMG pair.

    Selection policy
    ----------------
    - If `root` is a file, it is treated as the label.
    - If `ppd` is provided, the product whose MAP_RESOLUTION is *closest* to `ppd`
      is selected (ties resolved by higher resolution, then path sort).
    - If `ppd` is not provided, the highest-resolution valid product is selected.

    Returns
    -------
    (label_path, img_path)

    Raises
    ------
    FileNotFoundError
        If no valid product could be resolved.
    """
    p = Path(root)

    if p.is_file():
        img, _ = _resolve_img_from_label(p)
        return p, img

    if not p.is_dir():
        raise FileNotFoundError(f"Path is not a directory or file: {p}")

    labels = sorted(_iter_label_candidates(p))
    if not labels:
        raise FileNotFoundError(f"No .lbl/.txt labels found in: {p}")

    desired = float(ppd) if ppd is not None else None
    scored: list[tuple[float, float, Path]] = []
    for lbl in labels:
        try:
            info = parse_ldem_label(lbl)
            img, _ = _resolve_img_from_label(lbl)
        except Exception:
            continue

        res = float(info.map_resolution_ppd)
        # score = (distance to desired, -resolution)
        dist = abs(res - desired) if desired is not None else 0.0
        scored.append((dist, -res, lbl))

    if not scored:
        msg = f"No valid LDEM product found in: {p}"
        if ppd is not None:
            msg += f" (requested ~{ppd} ppd)"
        raise FileNotFoundError(msg)

    scored.sort()
    best_lbl = scored[0][2]
    best_img, _ = _resolve_img_from_label(best_lbl)
    return best_lbl, best_img




def find_lola_albedo_product(root: PathLike) -> Tuple[Path, Path]:
    """
    Locate a LOLA Albedo (LDAM) cylindrical label + IMG pair.

    Selection policy
    ----------------
    - If `root` is a file, it is treated as the label.
    - Otherwise, among parseable cylindrical rasters, the highest-resolution
      (largest MAP_RESOLUTION) product is selected.

    Returns
    -------
    (label_path, img_path)

    Raises
    ------
    FileNotFoundError
        If no parseable LDAM product is found.
    """
    p = Path(root)

    if p.is_file():
        img, _ = _resolve_img_from_label(p)
        return p, img

    if not p.is_dir():
        raise FileNotFoundError(f"Path is not a directory or file: {p}")

    labels = sorted(_iter_label_candidates(p))
    if not labels:
        raise FileNotFoundError(f"No .lbl/.txt labels found in: {p}")

    scored: list[tuple[float, Path]] = []
    for lbl in labels:
        try:
            info = parse_pds3_cyl_label(lbl)  # validate cylindrical raster
            img, _ = _resolve_img_from_label(lbl)
        except Exception:
            continue
        scored.append((-float(info.map_resolution_ppd), lbl))

    if not scored:
        raise FileNotFoundError(f"No valid LDAM/Albedo product found in: {p}")

    scored.sort()
    best_lbl = scored[0][1]
    best_img, _ = _resolve_img_from_label(best_lbl)
    return best_lbl, best_img




def _grid_albedo_payload(
    alb: Any,
    *,
    default_albedo: float,
) -> Dict[str, Any]:
    """
    Package an albedo grid into a plain dict for Numba-side RHS consumption.

    core.dynamics cannot hold arbitrary Python objects; it can hold NumPy arrays
    and POD-like metadata.

    Contract of returned dict:
      - if grid is missing/unusable -> {"albedo_const": <float>}
      - else includes:
          dn, n_lines, n_samples, res_deg, lon0_deg, lat0_deg,
          scale_factor, offset, missing_dn, flip_lat,
          lat_min_deg, lat_max_deg, albedo_const
    """
    if alb is None:
        return {"albedo_const": float(default_albedo)}

    # Must expose a numpy array "dn" and "info" metadata (PDS3 grids do)
    info = getattr(alb, "info", None)
    dn = getattr(alb, "dn", None)
    if info is None or dn is None:
        return {"albedo_const": float(default_albedo)}

    n_lines = int(getattr(info, "lines"))
    n_samples = int(getattr(info, "samples"))

    # Your grids expose ddeg = 1/ppd
    res_deg = float(getattr(alb, "ddeg", 1.0))

    # Use pixel-center origins to match bilinear sampling convention.
    lon_cent = getattr(alb, "lon_centers_deg", None)
    lat_cent = getattr(alb, "lat_centers_deg", None)
    if lon_cent is not None:
        lon0_deg = float(np.asarray(lon_cent, dtype=np.float64)[0])
    else:
        lon0_deg = float(getattr(info, "west_lon_deg", 0.0) + 0.5 * res_deg)

    if lat_cent is not None:
        lat0_deg = float(np.asarray(lat_cent, dtype=np.float64)[0])
    else:
        lat0_deg = float(getattr(info, "max_lat_deg", 90.0) - 0.5 * res_deg)

    scale_factor = float(getattr(info, "scaling_factor", 1.0))
    offset = float(getattr(info, "offset", 0.0))
    missing_dn = float(getattr(info, "missing_constant", float("nan")))
    flip_lat = 1 if bool(getattr(alb, "_flip_lat", False)) else 0

    # Clamp range consistent with sampling (cell centers)
    lat_min_deg = float(getattr(info, "min_lat_deg", -90.0)) + 0.5 * res_deg
    lat_max_deg = float(getattr(info, "max_lat_deg", 90.0)) - 0.5 * res_deg

    return {
        "dn": dn,
        "n_lines": n_lines,
        "n_samples": n_samples,
        "res_deg": res_deg,
        "lon0_deg": lon0_deg,
        "lat0_deg": lat0_deg,
        "scale_factor": scale_factor,
        "offset": offset,
        "missing_dn": missing_dn,
        "flip_lat": flip_lat,
        "lat_min_deg": lat_min_deg,
        "lat_max_deg": lat_max_deg,
        "albedo_const": float(default_albedo),
    }


class SurfaceProvider(Protocol):
    """
    Unified surface-query interface for the physics engine.

    Convention
    ----------
    Inputs are in DEGREES:
      - lat_deg in [-90, +90]
      - lon_deg in [0, 360) or [-180, 180] depending on your upstream

    Outputs:
      - radius_m: meters from body center
      - albedo  : dimensionless [0,1]
    """

    def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float: ...
    def albedo_deg(self, lat_deg: float, lon_deg: float) -> float: ...

    def grids(self) -> "SurfaceGrids":
        """Return underlying grid objects if available, otherwise empty."""
        return SurfaceGrids(None, None)

    def as_numba_dict(self) -> Dict[str, Any]:
        """Return plain dict payload for core.dynamics (Numba-side)."""
        return {"albedo_const": 0.12}


@dataclass(slots=True)
class FileBackedSurfaceProvider:
    """
    Disk-backed provider for PDS3 rasters (LDEM topography + LDAM albedo).

    Behavior
    --------
    - Loads what it can; missing datasets fall back to constants.
    - If ``strict_io=True``, failures to load *requested* datasets raise.
    - If ``warn_on_fallback=True``, non-strict failures emit warnings.

    Notes
    -----
    This class is intentionally lightweight. It keeps the loaded grids and exposes:
    - ``radius_m_deg(lat_deg, lon_deg)``
    - ``albedo_deg(lat_deg, lon_deg)``
    - ``as_numba_dict()`` for Numba-side RHS consumption.
    """

    ldem_root: Optional[PathLike] = None
    albedo_root: Optional[PathLike] = None
    mmap: bool = True
    flip_lat: bool = False
    ldem_ppd: Optional[int] = None

    default_radius_m: float = float(R_MOON_MEAN)
    default_albedo: float = 0.12

    strict_io: bool = False
    warn_on_fallback: bool = True

    # internal
    _grids: "SurfaceGrids" = None  # type: ignore
    _errors: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.default_radius_m = float(self.default_radius_m)
        self.default_albedo = float(self.default_albedo)

        topo = None
        alb = None

        # Load requested datasets independently so one failure doesn't discard the other.
        if self.ldem_root is not None:
            try:
                ldem_lbl, ldem_img = find_ldem_product(self.ldem_root, ppd=self.ldem_ppd)
                topo = TopographyGrid(ldem_lbl, ldem_img, mmap=bool(self.mmap), flip_lat=bool(self.flip_lat))
            except Exception as e:
                self._errors["ldem"] = f"{type(e).__name__}: {e}"
                if self.strict_io:
                    raise
                if self.warn_on_fallback:
                    warnings.warn(f"LDEM load failed, falling back to constant radius: {e}", RuntimeWarning)

        if self.albedo_root is not None:
            try:
                alb_lbl, alb_img = find_lola_albedo_product(self.albedo_root)
                alb = LOLAAlbedoGrid(alb_lbl, alb_img, mmap=bool(self.mmap), flip_lat=bool(self.flip_lat))
            except Exception as e:
                self._errors["albedo"] = f"{type(e).__name__}: {e}"
                if self.strict_io:
                    raise
                if self.warn_on_fallback:
                    warnings.warn(f"Albedo load failed, falling back to constant albedo: {e}", RuntimeWarning)

        self._grids = SurfaceGrids(topo=topo, albedo=alb)

    @property
    def errors(self) -> Dict[str, str]:
        """Human-readable load errors captured during initialization."""
        return dict(self._errors)

    def grids(self) -> "SurfaceGrids":
        return self._grids

    def as_numba_dict(self) -> Dict[str, Any]:
        return _grid_albedo_payload(self._grids.albedo, default_albedo=self.default_albedo)

    def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
        topo = self._grids.topo
        if topo is None:
            return float(self.default_radius_m)
        return float(topo.sample_bilinear(float(lat_deg), float(lon_deg), kind="radius_m"))

    def albedo_deg(self, lat_deg: float, lon_deg: float) -> float:
        alb = self._grids.albedo
        if alb is None:
            return float(self.default_albedo)

        val = float(alb.sample_bilinear(float(lat_deg), float(lon_deg)))
        # Defensive clamp (even if your grid does it)
        if math.isnan(val):
            return float(self.default_albedo)
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val



@dataclass(slots=True)
class InMemorySurfaceProvider:
    """
    Provider backed by injected provider objects (TopographyProvider/AlbedoProvider).

    This is great for:
      - unit tests
      - swapping in mock providers
      - using ConstantTopography / ConstantAlbedo
    """

    topo: Optional["TopographyProvider"] = None
    albedo: Optional["AlbedoProvider"] = None
    default_radius_m: float = float(R_MOON_MEAN)
    default_albedo: float = 0.12

    def __post_init__(self) -> None:
        self.default_radius_m = float(self.default_radius_m)
        self.default_albedo = float(self.default_albedo)

    def grids(self) -> "SurfaceGrids":
        # In-memory provider is not file-backed -> no grids
        return SurfaceGrids(None, None)

    def as_numba_dict(self) -> Dict[str, Any]:
        # Only expose grid payload if the injected albedo is grid-like
        return _grid_albedo_payload(self.albedo, default_albedo=self.default_albedo)

    def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
        if self.topo is None:
            return float(self.default_radius_m)
        return float(self.topo.radius_m(math.radians(float(lat_deg)), math.radians(float(lon_deg))))

    def albedo_deg(self, lat_deg: float, lon_deg: float) -> float:
        if self.albedo is None:
            return float(self.default_albedo)
        val = float(self.albedo.albedo(math.radians(float(lat_deg)), math.radians(float(lon_deg))))
        if math.isnan(val):
            return float(self.default_albedo)
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val





# =============================================================================
# Compatibility helper: pick the right PDS3 parser based on label contents
# =============================================================================

@lru_cache(maxsize=128)
def parse_pds3_label(lbl_path: Union[str, Path]) -> Union["PDS3MapInfo", "PDS3RasterInfo"]:
    """Heuristic parser selecting the appropriate PDS3 label parser.

    - LDEM-like topography labels commonly include ellipsoid radii fields
      (A_AXIS_RADIUS / B_AXIS_RADIUS / C_AXIS_RADIUS) or projection offsets.
    - Otherwise, treat as a generic cylindrical raster (LDAM etc.).
    """
    lbl_path = Path(lbl_path)
    if not lbl_path.exists():
        raise FileNotFoundError(lbl_path)

    txt = _strip_pds_comments(_read_text(lbl_path))

    is_ldem = (
        re.search(r"\bA_AXIS_RADIUS\b", txt) is not None
        or re.search(r"\bB_AXIS_RADIUS\b", txt) is not None
        or re.search(r"\bC_AXIS_RADIUS\b", txt) is not None
        or ("LINE_PROJECTION_OFFSET" in txt and "SAMPLE_PROJECTION_OFFSET" in txt)
        or re.search(r'PRODUCT_ID\s*=\s*"LDEM', txt, flags=re.IGNORECASE) is not None
    )

    return parse_ldem_label(lbl_path) if is_ldem else parse_pds3_cyl_label(lbl_path)

# =============================================================================
# Public API
# =============================================================================

__all__ = (
    # Provider abstractions
    "TopographyProvider",
    "AlbedoProvider",
    "ConstantTopography",
    "ConstantAlbedo",

    # Core grid types
    "TopographyGrid",
    "PDS3CylindricalGrid",
    "LOLAAlbedoGrid",
    "SurfaceGrids",

    # Loaders / resolvers
    "load_surface_grids",
    "find_ldem_product",
    "find_lola_albedo_product",

    # Facade providers
    "SurfaceProvider",
    "FileBackedSurfaceProvider",
    "InMemorySurfaceProvider",

    # PDS3 parsing utilities
    "PDS3ParseError",
    "PDS3RasterInfo",
    "PDS3MapInfo",
    "parse_ldem_label",
    "parse_pds3_cyl_label",
    "parse_pds3_label",
)
