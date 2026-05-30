# lunaris/loaders/io_gravity.py
"""
Gravity Model I/O Utilities.

This module provides robust loaders for spherical harmonic gravity models,
supporting both legacy binary formats and standard PDS ASCII tables.

Key Features:
  - **SHBDR Loader:** Efficient binary parsing for legacy formats.
  - **SHADR (PDS3) Loader:** Robust ASCII parsing for modern GRAIL/LRO datasets (.TAB, .SHA).
  - **Auto-Inference:** Automatically detects coefficient column order 
    ((degree, order) vs (order, degree)) by analyzing triangular index constraints.
  - **Validation:** Optional 'strict' mode to ensure data integrity, completeness,
    and correct normalization states.

The loaders return fully normalized coefficients (Cnm, Snm) and reference 
constants (R_ref, GM) consistently converted to SI units (meters).
"""



# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import os
import re
import struct
import logging

from typing import List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# 1.                          SHBDR PARSER (BINARY)
# =============================================================================

def load_shbdr(
    file_path: str,
    record_bytes: int = 512,
    names_record: int = 2,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """
    Parses legacy SHBDR-like binary gravity files.

    Format Layout
    -------------
    Header (56 bytes):
        <3d (R_ref, GM, GM_sigma)
        4i  (n_max, m_max, norm_state, num_params)
        2d  (ref_lon, ref_lat)

    Returns
    -------
    n_max : int
    R_ref : float [m]
    GM    : float [m^3/s^2]
    Cnm   : (n_max+1, n_max+1) array (padded)
    Snm   : (n_max+1, n_max+1) array (padded)
    """
    if record_bytes <= 0:
        raise ValueError(f"record_bytes must be positive. Got {record_bytes}.")
    if names_record <= 0:
        raise ValueError(f"names_record must be >= 1. Got {names_record}.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SHBDR file not found: {file_path}")

    with open(file_path, "rb") as f:
        header_data = f.read(56)
        if len(header_data) < 56:
            raise IOError("File too short (missing header).")

        R_ref, GM, GM_sigma, n_max, m_max, norm_state, num_params, ref_lon, ref_lat = struct.unpack(
            "<3d4i2d", header_data
        )

        if int(norm_state) != 1:
            raise ValueError(f"SHBDR norm_state must be 1 (Normalized). Got {norm_state}.")

        n_max = int(n_max)
        m_max = int(m_max)
        num_params = int(num_params)
        if n_max < 0 or m_max < 0 or num_params < 0:
            raise ValueError("Invalid header values (negative dimension/num_params).")

        # --- Names table ---
        names_offset = (names_record - 1) * record_bytes
        f.seek(names_offset)

        names_raw = f.read(num_params * 8)
        if len(names_raw) != num_params * 8:
            raise IOError("Failed to read parameter names table.")

        names = [
            names_raw[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip()
            for i in range(num_params)
        ]

        # --- Coefficients table ---
        names_records_count = (num_params * 8 + record_bytes - 1) // record_bytes
        coeff_record_start = names_record + names_records_count
        coeff_offset = (coeff_record_start - 1) * record_bytes

        f.seek(coeff_offset)
        coeff_raw = f.read(num_params * 8)
        if len(coeff_raw) != num_params * 8:
            raise IOError("Failed to read coefficients value table.")

        vals = np.frombuffer(coeff_raw, dtype="<f8", count=num_params)

    Cnm = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    Snm = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)

    for name, val in zip(names, vals):
        if len(name) < 7:
            continue

        coeff_type = name[0]  # 'C' or 'S'
        if coeff_type not in ("C", "S"):
            continue

        try:
            # Format: Cnnnmmm (e.g. C002000 -> n=2, m=0)
            n = int(name[1:4])
            m = int(name[4:7])
        except ValueError:
            continue

        if 0 <= n <= n_max and 0 <= m <= n and m <= n_max:
            if coeff_type == "C":
                Cnm[n, m] = float(val)
            else:
                Snm[n, m] = float(val)

    return n_max, float(R_ref), float(GM), Cnm, Snm



# =============================================================================
# 2.                         SHADR PARSER (ASCII / PDS) 
# =============================================================================

# Regex to find floating point numbers in loose ASCII text
# Matches: 123, -123.45, 1.23E-4, +5.0e+05
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
_NUM_DELIM_TRANS = str.maketrans({",": " ", "\t": " "})


def _parse_nums(line: str) -> List[float]:
    """
    Robust numeric tokenizer for ASCII lines.

    Strategy:
    1) Fast path: replace common delimiters -> split -> float(...)
    2) Fallback: regex scan (handles tightly packed fixed-width numbers)
    """
    s = line.translate(_NUM_DELIM_TRANS).strip()
    if not s:
        return []

    try:
        return [float(x) for x in s.split()]
    except ValueError:
        # Fallback for weird formatting (e.g. merged columns)
        return [float(m.group(0)) for m in _FLOAT_RE.finditer(line)]


def _infer_column_order(sample_lines: List[Tuple[int, int, float, float]]) -> bool:
    """
    Determines if the file columns are (Degree, Order) or (Order, Degree).

    Returns
    -------
    swap_indices : bool
        True  -> file uses (Order, Degree) and must be swapped to (n, m).
        False -> file uses (Degree, Order) already (n, m).
    """
    if not sample_lines:
        return False

    # Use "violation counting" instead of >= scoring:
    # Degree/order constraint: n >= m (triangular)
    viol_nm = sum(c1 < c2 for c1, c2, _, _ in sample_lines)  # interpret as (n,m)=(c1,c2)
    viol_mn = sum(c2 < c1 for c1, c2, _, _ in sample_lines)  # interpret as (n,m)=(c2,c1)

    # swap True if interpreting as (m,n) produces fewer violations
    return viol_mn < viol_nm


def load_shadr_ascii(
    file_path: str,
    degree_max: Optional[int] = None,
    coeff_start_line: int = 3,
    break_when_past_degree: bool = True,
    sample_size: int = 1000,
    *,
    strict: bool = True,
    require_normalization_state: Optional[int] = 1,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """
    Parses PDS3 ASCII SHADR gravity models (e.g., GRAIL .TAB files).

    Compared to a "best-effort" loader, this implementation can run in a STRICT mode that
    validates completeness and fails fast on malformed or partially-parsed files.

    Notes
    -----
    - Auto-detects (n,m) vs (m,n) column order using a small coefficient sample.
    - Units are converted from km -> m and km^3/s^2 -> m^3/s^2.
    - If `degree_max` is set, coefficients with n > n_use are ignored (optionally early-broken
      when the file is sorted by degree).

    Parameters
    ----------
    file_path : str
        Path to the .TAB (PDS3 SHADR ASCII) file.
    degree_max : int, optional
        If set, truncates to this degree for memory/perf.
    coeff_start_line : int
        1-based line index where coefficients start (typical PDS: 3).
    break_when_past_degree : bool
        Enables early break optimization for sorted files.
    sample_size : int
        Number of coefficient rows to sample before deciding column order.
    strict : bool
        If True, performs additional validations:
          - raises on any malformed coefficient line (after coeff_start_line),
          - raises on duplicates within the kept degree range,
          - enforces triangular indexing (m <= n) for kept coefficients,
          - validates that the number of stored coefficients matches expectation.
    require_normalization_state : int | None
        If not None and `strict` is True, require header normalization state to match this value.
        For GRAIL SHADR, fully-normalized models typically use state == 1.

    Returns
    -------
    n_use, R_ref[m], GM[m^3/s^2], Cnm, Snm
    """
    if coeff_start_line <= 0:
        raise ValueError(f"coeff_start_line must be >= 1. Got {coeff_start_line}.")
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive. Got {sample_size}.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SHADR file not found: {file_path}")

    header_nums: Optional[List[float]] = None
    R_ref_m = 0.0
    GM_m3s2 = 0.0
    n_use = 0
    file_n_max = 0
    file_m_max = 0
    norm_state: Optional[int] = None

    Cnm: Optional[np.ndarray] = None
    Snm: Optional[np.ndarray] = None

    swap: Optional[bool] = None
    sample_size_eff = int(sample_size)  # may be tightened after header is read
    # buffer: (raw1, raw2, C, S, line_no)
    buffered: List[Tuple[int, int, float, float, int]] = []

    # --- strict accounting ---
    stored_count = 0
    skipped_short = 0
    skipped_parse = 0
    skipped_due_to_degree = 0
    duplicate_count = 0
    blank_count = 0

    # For early-break logic (only if file appears sorted)
    last_n_seen = -1
    sorted_non_decreasing = True

    seen: Optional[np.ndarray] = None  # uint8 mask for duplicates / coverage (only if strict)

    def _maybe_fail(msg: str) -> None:
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    def _store_coeff(raw1: int, raw2: int, val_C: float, val_S: float, line_no: int) -> int:
        """
        Map indices, validate, store.
        Returns mapped n (even if ignored due to truncation) for sorting/break diagnostics.
        """
        nonlocal Cnm, Snm, stored_count, skipped_due_to_degree, duplicate_count, seen

        if swap:
            n, m = raw2, raw1
        else:
            n, m = raw1, raw2

        # Truncation: ignore degrees beyond n_use (this is expected when degree_max is set)
        if n > n_use:
            skipped_due_to_degree += 1
            return n

        # Hard validity checks inside the kept range
        if n < 0 or m < 0:
            _maybe_fail(f"Negative index at line {line_no}: n={n}, m={m}.")
            return n

        if m > n:
            # For SHADR models, coefficients are triangular (m <= n).
            # If this triggers after swap inference, the file is malformed or column order inference failed.
            _maybe_fail(
                f"Invalid triangular index at line {line_no}: (n={n}, m={m}) has m>n. "
                f"(swap={swap})"
            )
            return n

        if m > n_use:
            _maybe_fail(f"Index out of bounds at line {line_no}: n={n}, m={m}, n_use={n_use}.")
            return n

        assert Cnm is not None and Snm is not None
        Cnm[n, m] = val_C
        Snm[n, m] = val_S

        if strict:
            assert seen is not None
            if seen[n, m]:
                duplicate_count += 1
                raise ValueError(f"Duplicate coefficient entry for (n={n}, m={m}) at line {line_no}.")
            seen[n, m] = 1
            stored_count += 1

        return n

    open_errors = "strict" if strict else "ignore"
    with open(file_path, "r", encoding="utf-8", errors=open_errors) as f:
        for i_line, line in enumerate(f, 1):
            # --- Header parse: first non-empty line ---
            if header_nums is None:
                if not line.strip():
                    continue
                header_nums = _parse_nums(line)
                if len(header_nums) < 8:
                    raise ValueError(
                        "Invalid SHADR header. Expected at least 8 fields: "
                        "R_ref, GM, omega, deg, ord, state, tide, closed"
                    )

                R_ref_km = float(header_nums[0])
                GM_km3s2 = float(header_nums[1])
                file_n_max = int(header_nums[3])
                file_m_max = int(header_nums[4])
                try:
                    norm_state = int(header_nums[5])
                except Exception:
                    norm_state = None

                if strict and require_normalization_state is not None and norm_state is not None:
                    if norm_state != int(require_normalization_state):
                        raise ValueError(
                            f"Unsupported normalization_state={norm_state} in SHADR header. "
                            f"Expected {require_normalization_state}. If you know this model's "
                            f"normalization, set require_normalization_state=None or convert it."
                        )

                R_ref_m = R_ref_km * 1000.0
                GM_m3s2 = GM_km3s2 * 1.0e9

                n_cap = max(int(file_n_max), int(file_m_max))
                n_use = n_cap if degree_max is None else min(int(degree_max), n_cap)
                if n_use < 0:
                    raise ValueError(f"Computed n_use is negative ({n_use}). Check header/degree_max.")

                Cnm = np.zeros((n_use + 1, n_use + 1), dtype=np.float64)
                Snm = np.zeros((n_use + 1, n_use + 1), dtype=np.float64)

                if strict:
                    seen = np.zeros((n_use + 1, n_use + 1), dtype=np.uint8)

                # Bound sample size so we don't buffer far beyond the kept degree range when degree_max is small.
                # This keeps strict diagnostics cleaner and avoids large 'skipped_due_to_degree' counts.
                sample_size_eff = min(int(sample_size), max(64, 3 * (n_use + 1)))

                continue

            # --- Coefficients section ---
            # Some distributed/converted .TAB.txt files omit the blank "record 2" line, so coefficients may
            # start on line 2 rather than 3. Treat coeff_start_line as a hint and auto-detect if we see a
            # valid coefficient row earlier than the provided start.
            if i_line < coeff_start_line:
                if line.strip():
                    probe = _parse_nums(line)
                    if len(probe) >= 4:
                        a0, a1 = probe[0], probe[1]
                        # First two columns should be integer-like indices in [0, n_cap]
                        if abs(a0 - round(a0)) < 1e-9 and abs(a1 - round(a1)) < 1e-9:
                            ia0, ia1 = int(round(a0)), int(round(a1))
                            n_cap = max(int(file_n_max), int(file_m_max))
                            if 0 <= ia0 <= n_cap and 0 <= ia1 <= n_cap:
                                coeff_start_line = i_line
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            if not line.strip():
                blank_count += 1
                continue

            assert Cnm is not None and Snm is not None

            nums = _parse_nums(line)
            if len(nums) < 4:
                skipped_short += 1
                if strict:
                    raise ValueError(
                        f"Malformed coefficient line (expected >=4 numbers) at line {i_line}: {line.rstrip()[:200]}"
                    )
                continue

            try:
                raw1 = int(nums[0])
                raw2 = int(nums[1])
                val_C = float(nums[2])
                val_S = float(nums[3])
            except ValueError:
                skipped_parse += 1
                if strict:
                    raise ValueError(
                        f"Failed to parse coefficient indices/values at line {i_line}: {line.rstrip()[:200]}"
                    )
                continue

            # If column order not decided yet, buffer samples first
            if swap is None:
                buffered.append((raw1, raw2, val_C, val_S, i_line))
                if len(buffered) >= sample_size_eff:
                    swap = _infer_column_order([(a, b, c, d) for (a, b, c, d, _) in buffered])

                    # Flush buffered samples with decided mapping
                    for b1, b2, bC, bS, bline in buffered:
                        mapped_n = _store_coeff(b1, b2, bC, bS, bline)
                        if mapped_n < last_n_seen:
                            sorted_non_decreasing = False
                        if mapped_n > last_n_seen:
                            last_n_seen = mapped_n
                    buffered.clear()
                continue

            # Normal streaming store
            mapped_n = _store_coeff(raw1, raw2, val_C, val_S, i_line)
            if mapped_n < last_n_seen:
                sorted_non_decreasing = False
            if mapped_n > last_n_seen:
                last_n_seen = mapped_n

            if (
                break_when_past_degree
                and degree_max is not None
                and sorted_non_decreasing
                and mapped_n > n_use
            ):
                break

        # End of file: if swap never decided, decide now using whatever we have
        if Cnm is None or Snm is None:
            raise ValueError("Failed to parse SHADR header (no non-empty header line found).")

        if swap is None:
            swap = _infer_column_order([(a, b, c, d) for (a, b, c, d, _) in buffered])
            for b1, b2, bC, bS, bline in buffered:
                _store_coeff(b1, b2, bC, bS, bline)
            buffered.clear()

    # -------------------- STRICT VALIDATION --------------------
    if strict:
        assert seen is not None

        # Any malformed lines are fatal in strict mode (they would have raised already),
        # but keep the counters for error messages / diagnostics.
        if skipped_short or skipped_parse:
            raise ValueError(
                f"SHADR parse was not clean: skipped_short={skipped_short}, skipped_parse={skipped_parse}."
            )

        # Determine expected coefficient count in the kept range.
        # Some SHADR files omit the (0,0) coefficient row; accept both cases.
        expected_with_c00 = (n_use + 1) * (n_use + 2) // 2
        has_c00 = bool(seen[0, 0] == 1)
        expected = expected_with_c00 if has_c00 else (expected_with_c00 - 1)

        if stored_count != expected:
            # Count missing is cheap and gives a clear signal.
            missing = expected - stored_count

            # Provide a few concrete missing (n,m) pairs to make debugging actionable.
            missing_examples: List[Tuple[int, int]] = []
            if missing > 0:
                for nn in range(0, n_use + 1):
                    for mm in range(0, nn + 1):
                        if nn == 0 and mm == 0 and not has_c00:
                            continue
                        if seen[nn, mm] == 0:
                            missing_examples.append((nn, mm))
                            if len(missing_examples) >= 10:
                                break
                    if len(missing_examples) >= 10:
                        break

            raise ValueError(
                "SHADR coefficient table incomplete after parsing.\n"
                f"  file_path: {file_path}\n"
                f"  header: n_max={file_n_max}, m_max={file_m_max}, norm_state={norm_state}\n"
                f"  kept: n_use={n_use} (degree_max={degree_max})\n"
                f"  inferred column order: swap={swap}\n"
                f"  coeff_start_line={coeff_start_line}\n"
                f"  stored_count={stored_count}, expected={expected} (has_c00={has_c00}, missing={missing})\n"
                f"  first_missing={missing_examples}\n"
                f"  skipped_due_to_degree={skipped_due_to_degree} (ignored because n>n_use), "
                f"duplicate_count={duplicate_count}, blank_lines={blank_count}\n"
                "Common causes:\n"
                "  - coeff_start_line is off by 1 (many .TAB.txt exports start coefficients on line 2, not 3)\n"
                "  - the file was edited/re-saved with line-wrapping or truncation\n"
                "Fixes:\n"
                "  - use the original .TAB from the PDS bundle, or set coeff_start_line=2\n"
                "  - if you intentionally allow partial/irregular tables, set strict=False\n"
            )

    return int(n_use), float(R_ref_m), float(GM_m3s2), Cnm, Snm



# =============================================================================
# 3.                   UNIVERSAL LOADER DISPATCH (SSOT)
# =============================================================================

# ASCII table endings commonly used for PDS3 / GRAIL / LRO SHADR exports.
# Keep this local to the loader module (Separation of Concerns).
_ASCII_ENDINGS = (
    ".tab",
    ".sha",
    ".txt",
    ".tab.txt",
    ".sha.txt",
)


def _looks_like_ascii_model(file_path: str) -> bool:
    """Return True if *file_path* looks like a SHADR/PDS ASCII gravity table."""
    name = os.path.basename(str(file_path)).lower()
    return any(name.endswith(sfx) for sfx in _ASCII_ENDINGS)


def _slice_square(a: np.ndarray, n_use: int) -> np.ndarray:
    """Slice a 2D array to (n_use+1, n_use+1) and ensure float64 contiguous."""
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D coefficient array. Got ndim={a.ndim}.")
    n = int(n_use)
    if n < 0:
        raise ValueError(f"n_use must be >= 0. Got {n_use}.")
    # Robust against rectangular inputs
    hi0 = min(a.shape[0] - 1, n)
    hi1 = min(a.shape[1] - 1, n)
    hi = min(hi0, hi1)
    out = np.ascontiguousarray(a[: hi + 1, : hi + 1], dtype=np.float64)
    return out


def load_gravity_model(
    file_path: str,
    degree_max: Optional[int] = None,
    *,
    ascii_strict: bool = True,
    ascii_require_normalization_state: Optional[int] = 1,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """
    Universal entry point for loading gravity models.

    Dispatch
    --------
    - ASCII (PDS/GRAIL style): .tab/.sha/.txt (+ .tab.txt/.sha.txt) -> :func:`load_shadr_ascii`
    - Otherwise: legacy/binary SHBDR-like -> :func:`load_shbdr`

    Parameters
    ----------
    file_path:
        Path to the gravity model file.
    degree_max:
        If provided, coefficients are truncated to min(n_max, degree_max).
    ascii_strict:
        If True, SHADR ASCII parser runs in strict validation mode.
    ascii_require_normalization_state:
        If not None and *ascii_strict* is True, require SHADR header normalization state to match.

    Returns
    -------
    n_max : int
        Degree actually returned (after truncation if requested).
    R_ref : float
        Reference radius [m].
    GM : float
        Gravitational parameter [m^3/s^2].
    Cnm, Snm : np.ndarray
        Coefficient matrices in float64 contiguous layout, shape (n_max+1, n_max+1).
    """
    if degree_max is not None:
        degree_max = int(degree_max)
        if degree_max < 0:
            raise ValueError(f"degree_max must be >= 0. Got {degree_max}.")

    # 1) ASCII loader
    if _looks_like_ascii_model(file_path):
        n_use, R_ref_m, GM_m3s2, Cnm, Snm = load_shadr_ascii(
            str(file_path),
            degree_max=degree_max,
            strict=bool(ascii_strict),
            require_normalization_state=ascii_require_normalization_state,
        )
        # SHADR loader already allocates exact (n_use+1, n_use+1)
        Cnm = np.ascontiguousarray(Cnm, dtype=np.float64)
        Snm = np.ascontiguousarray(Snm, dtype=np.float64)
        return int(n_use), float(R_ref_m), float(GM_m3s2), Cnm, Snm

    # 2) Binary loader
    n_max, R_ref_m, GM_m3s2, Cnm, Snm = load_shbdr(str(file_path))

    # 3) Post-process truncation if requested
    n_use = int(n_max)
    if degree_max is not None:
        n_use = min(int(n_use), int(degree_max))

    if int(n_use) != int(n_max):
        Cnm = _slice_square(Cnm, int(n_use))
        Snm = _slice_square(Snm, int(n_use))
    else:
        Cnm = np.ascontiguousarray(Cnm, dtype=np.float64)
        Snm = np.ascontiguousarray(Snm, dtype=np.float64)

    return int(n_use), float(R_ref_m), float(GM_m3s2), Cnm, Snm
