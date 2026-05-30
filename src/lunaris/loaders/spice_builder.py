"""
SPICE Kernel Path Resolution and Frame-Kernel Discovery Helpers
==============================================================

This module contains the loader-side filesystem logic used before SPICE tables
are built:

1. Kernel path normalization
   - Expand `~`
   - Reject copy/paste path corruption such as embedded newlines/tabs
   - Accept common text-wrapped kernel variants (`.tls.txt`, `.tf.txt`, ...)

2. Lunar frame-kernel auto-discovery
   - When a high-fidelity Moon-fixed frame is requested and only a binary PCK
     (`.bpc`) is explicitly listed, attempt to locate a colocated lunar TF/FK.

Why this lives in `loaders`
---------------------------
These helpers are about filesystem conventions and asset discovery, not orbital
mechanics or runtime interpolation. Moving them out of `lunaris.physics.ephemeris`
reduces layering blur:

- `loaders` owns path resolution and disk-facing heuristics.
- `lunaris.physics.ephemeris` owns SPICE interaction and runtime table generation.

Design goals
------------
- Deterministic behavior: candidate ranking is explicit and stable.
- Compatibility-aware: optional `.txt` wrappers are handled centrally.
- Small public API: a few focused helpers that higher layers can reuse.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional


# =============================================================================
# 0.                           KERNEL SUFFIX POLICY
# =============================================================================

OPTIONAL_TEXT_SUFFIX = ".txt"

# "Base" extensions for SPICE kernel files, ignoring a trailing ".txt" wrapper.
SPICE_KERNEL_BASE_EXTS: frozenset[str] = frozenset(
    {".tls", ".tpc", ".tf", ".fk", ".bsp", ".bpc"}
)

SPICE_FRAME_BASE_EXTS: frozenset[str] = frozenset({".tf", ".fk"})
SPICE_BPC_BASE_EXTS: frozenset[str] = frozenset({".bpc"})

# High-fidelity lunar frames that usually require both a BPC and a TF/FK.
LUNAR_HIFI_FRAMES: frozenset[str] = frozenset({
    "MOON_PA",
    "MOON_ME",
    "MOON_PA_DE440",
    "MOON_ME_DE440_ME421",
})

BAD_PATH_CHARS: tuple[str, ...] = ("\n", "\r", "\t")


# =============================================================================
# 1.                          SMALL PATH NORMALIZERS
# =============================================================================

def strip_optional_txt(path: Path) -> Path:
    """
    Return `path` with a trailing `.txt` removed, if present.

    Examples
    --------
    `naif0012.tls.txt` -> `naif0012.tls`
    `de440.bsp`        -> `de440.bsp`
    """

    return path.with_suffix("") if path.suffix.lower() == OPTIONAL_TEXT_SUFFIX else path


def base_suffix(path: Path) -> str:
    """Return the kernel suffix after stripping an optional trailing `.txt`."""
    return strip_optional_txt(path).suffix.lower()


def has_base_suffix(path: Path, base_exts: frozenset[str]) -> bool:
    """True when `path` matches one of `base_exts`, allowing a trailing `.txt`."""
    return base_suffix(path) in base_exts


def toggle_optional_txt(path: Path) -> Path:
    """
    Toggle the optional `.txt` wrapper used by some packaged kernel sets.

    - `kernel.tls`     -> `kernel.tls.txt`
    - `kernel.tls.txt` -> `kernel.tls`
    """

    if path.suffix.lower() == OPTIONAL_TEXT_SUFFIX:
        return path.with_suffix("")
    return Path(str(path) + OPTIONAL_TEXT_SUFFIX)


def iter_optional_txt_variants(path: Path, *, base_exts: frozenset[str]) -> Iterator[Path]:
    """
    Yield the original path and, when appropriate, its `.txt`-toggled variant.

    This keeps the optional-wrapper policy centralized instead of scattering
    `foo.tls` vs `foo.tls.txt` fallback logic across the codebase.
    """

    yield path
    if has_base_suffix(path, base_exts):
        alt = toggle_optional_txt(path)
        if alt != path:
            yield alt


def is_kernel_with_base_ext(pathlike: str | Path, base_exts: frozenset[str]) -> bool:
    """Convenience wrapper around `has_base_suffix` for string/Path inputs."""
    path = pathlike if isinstance(pathlike, Path) else Path(str(pathlike))
    return has_base_suffix(path, base_exts)


def normalize_kernel_name(name: str) -> str:
    """
    Lowercase a filename and remove an optional trailing `.txt`.

    This produces a stable comparison key for candidate scoring.
    """

    lowered = str(name).lower()
    return lowered[:-len(OPTIONAL_TEXT_SUFFIX)] if lowered.endswith(OPTIONAL_TEXT_SUFFIX) else lowered


# =============================================================================
# 2.                        PUBLIC PATH RESOLUTION API
# =============================================================================

def resolve_kernel_paths(kernels: Sequence[str], *, auto_fix: bool = True) -> list[str]:
    """
    Resolve and validate SPICE kernel paths.

    Steps
    -----
    1. Expand `~`
    2. Reject control-character corruption (`\\n`, `\\r`, `\\t`)
    3. Accept exact on-disk matches
    4. Optionally retry the same path with the `.txt` wrapper toggled

    Raises
    ------
    ValueError
        If a path contains embedded control characters.
    FileNotFoundError
        If one or more kernels cannot be found.
    """

    resolved: list[str] = []
    missing: list[str] = []

    for raw in kernels:
        path = Path(raw).expanduser()
        raw_as_str = str(path)

        if any(ch in raw_as_str for ch in BAD_PATH_CHARS):
            raise ValueError(
                f"Kernel path contains control characters (newline/tab): {raw!r}\n"
                "Fix the string in your config (common copy/paste issue)."
            )

        if path.is_file():
            resolved.append(str(path.resolve()))
            continue

        if auto_fix:
            found: Optional[Path] = None
            for candidate in iter_optional_txt_variants(path, base_exts=SPICE_KERNEL_BASE_EXTS):
                if candidate.is_file():
                    found = candidate
                    break
            if found is not None:
                resolved.append(str(found.resolve()))
                continue

        missing.append(raw_as_str)

    if missing:
        message = "The following SPICE kernels could not be found:\n" + "\n".join(
            f" - {missing_path}" for missing_path in missing
        )
        raise FileNotFoundError(message)

    return resolved


def maybe_autoinclude_lunar_fk(kernels: Sequence[str], fixed_frame: str) -> list[str]:
    """
    Best-effort lunar frame-kernel auto-discovery.

    Use case
    --------
    Some kernel bundles explicitly list the lunar orientation BPC but omit the
    colocated TF/FK needed to define the requested Moon-fixed frame chain. When
    that pattern is detected, this helper searches nearby directories and injects
    the best candidate TF/FK before the binary kernels.

    Policy
    ------
    - Only applies to known high-fidelity lunar frames.
    - If a TF/FK is already present, the input list is returned unchanged.
    - Candidate scoring is deterministic and favors common DE440 lunar frame
      kernels before generic "moon*.tf/.fk" files.
    """

    if str(fixed_frame).strip().upper() not in LUNAR_HIFI_FRAMES:
        return list(kernels)

    if any(is_kernel_with_base_ext(kernel, SPICE_FRAME_BASE_EXTS) for kernel in kernels):
        return list(kernels)

    bpc_dirs: list[Path] = []
    other_dirs: list[Path] = []
    seen: set[Path] = set()

    for kernel in kernels:
        path = Path(str(kernel)).expanduser()
        directory = path.parent
        if directory in seen:
            continue
        seen.add(directory)
        if is_kernel_with_base_ext(path, SPICE_BPC_BASE_EXTS):
            bpc_dirs.append(directory)
        else:
            other_dirs.append(directory)

    search_dirs = bpc_dirs + other_dirs

    def score(name: str) -> tuple[int, str]:
        full = name.lower()
        base = normalize_kernel_name(full)

        if base == "moon_de440_220930.tf":
            return (0, full)
        if base.startswith("moon_de440_") and base.endswith(".tf"):
            return (1, full)
        if "de440" in base and base.endswith(".tf"):
            return (2, full)
        if "moon" in base and is_kernel_with_base_ext(full, SPICE_FRAME_BASE_EXTS):
            return (3, full)
        return (9, full)

    selected: Optional[Path] = None
    for directory in search_dirs:
        try:
            if not directory.is_dir():
                continue
            candidates = [
                path for path in directory.iterdir()
                if path.is_file()
                and ("moon" in path.name.lower())
                and is_kernel_with_base_ext(path, SPICE_FRAME_BASE_EXTS)
            ]
        except OSError:
            continue

        if not candidates:
            continue

        selected = min(candidates, key=lambda candidate: score(candidate.name))
        break

    if selected is None:
        return list(kernels)

    out = list(kernels)
    out.insert(0, str(selected))
    return out


__all__ = (
    "OPTIONAL_TEXT_SUFFIX",
    "SPICE_KERNEL_BASE_EXTS",
    "SPICE_FRAME_BASE_EXTS",
    "SPICE_BPC_BASE_EXTS",
    "LUNAR_HIFI_FRAMES",
    "resolve_kernel_paths",
    "maybe_autoinclude_lunar_fk",
)
