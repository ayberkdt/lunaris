"""
Loader-Side Path Discovery and Asset Resolution Helpers
======================================================

This module collects the small, reusable "filesystem intelligence" helpers that
multiple layers of the project depend on:

1. Repository data-root discovery
   - Detect likely LDEM, albedo, and SPICE kernel directories under the repo.
   - Prefer the repository's split data layout over legacy "reuse LDEM for albedo"
     behavior when a dedicated albedo directory exists.

2. Directory classification
   - Answer lightweight questions such as:
       * "Does this folder look like a surface-data directory?"
       * "Does this folder look like a SPICE kernel set?"

3. Report/plot asset discovery
   - Locate a lunar texture image from canonical asset folders, environment
     overrides, or explicit user-provided paths.

Why this module exists
----------------------
These routines are I/O-facing and repository-aware. Keeping them in `loaders`
instead of spreading them across `analysis`, `models`, and `ui_parts` helps the
rest of the codebase stay focused on its actual job:

- `analysis` should format and visualize data, not scan the filesystem.
- `ui_parts` should manage widgets and state, not own repository discovery rules.
- `models` should implement physics/runtime behavior, not asset lookup policy.

Design goals
------------
- Small public surface: the rest of the project should call a few stable helper
  functions instead of re-implementing path heuristics.
- Safe failure behavior: missing folders or unreadable paths should degrade to
  empty results instead of crashing unrelated workflows.
- Repository awareness: defaults should match this project's actual layout
  (`data/topografy_models`, `data/albedo_models`, `data/ephemeris_models`,
  `data/assets`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


# =============================================================================
# 0.                                 TYPES
# =============================================================================

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class DataRootHints:
    """
    Normalized repository data-root hints used by loader-side discovery helpers.

    Fields
    ------
    ldem_root
        Directory containing topography/LDEM products.
    albedo_root
        Directory containing albedo/reflectance products.
    kernel_dir
        Directory containing SPICE kernels.
    use_ldem_for_albedo
        Legacy compatibility flag indicating that albedo may be reusing the
        LDEM directory when no dedicated albedo root is available.
    """

    ldem_root: str = ""
    albedo_root: str = ""
    kernel_dir: str = ""
    use_ldem_for_albedo: bool = False


# =============================================================================
# 1.                          DIRECTORY NORMALIZATION
# =============================================================================

def resolve_existing_directory(path_value: PathLike | None) -> str:
    """
    Return a normalized absolute directory path, or an empty string.

    Rules
    -----
    - Empty / None input -> `""`
    - Non-existent path  -> `""`
    - Existing directory -> resolved absolute string path

    Rationale
    ---------
    UI/session code frequently stores directory paths as strings. Normalizing
    them at the loader boundary prevents each caller from re-implementing the
    same `expanduser()/resolve()/is_dir()` checks.
    """

    if not path_value:
        return ""
    try:
        resolved = Path(path_value).expanduser().resolve()
    except Exception:
        return ""
    return str(resolved) if resolved.is_dir() else ""


def same_resolved_directory(path_a: PathLike | None, path_b: PathLike | None) -> bool:
    """
    Return True when both inputs resolve to the same directory.

    Notes
    -----
    - Missing/empty inputs are treated as non-equal.
    - Resolution failures also return False.
    """

    if not path_a or not path_b:
        return False
    try:
        return Path(path_a).expanduser().resolve() == Path(path_b).expanduser().resolve()
    except Exception:
        return False


# =============================================================================
# 2.                         DIRECTORY CLASSIFICATION
# =============================================================================

def directory_looks_like_surface_data(directory: PathLike, *, keywords: Sequence[str]) -> bool:
    """
    Heuristically decide whether a directory resembles a surface-data folder.

    Positive evidence
    -----------------
    - Child filenames contain one of the provided keywords (`ldem`, `ldam`,
      `topo`, `surface`, `albedo`, ...).
    - Raster/label-like files are present (`.img`, `.lbl`, `.xml`, `.tif`, ...).

    This is intentionally lightweight and conservative. It is used only for
    auto-discovery and should not be interpreted as strict validation.
    """

    try:
        folder = Path(directory).expanduser().resolve()
    except Exception:
        return False

    if not folder.is_dir():
        return False

    file_suffixes = {".img", ".npy", ".npz", ".tif", ".tiff", ".lbl", ".xml", ".txt"}
    lowered_keywords = tuple(str(keyword).lower() for keyword in keywords)

    try:
        for child in folder.iterdir():
            child_name = child.name.lower()
            if any(keyword in child_name for keyword in lowered_keywords):
                return True
            if child.is_file() and any(suffix.lower() in file_suffixes for suffix in child.suffixes):
                return True
    except OSError:
        return False

    return False


def directory_looks_like_spice_kernels(directory: PathLike) -> bool:
    """
    Return True when a directory appears to contain SPICE kernels.

    Recognized suffixes include:
    - `.bsp`, `.bc`, `.bpc`
    - `.tf`, `.fk`
    - `.tls`, `.tpc`

    Optional text-wrapped forms such as `.tf.txt` are also accepted because the
    real repository data often uses those variants.
    """

    try:
        folder = Path(directory).expanduser().resolve()
    except Exception:
        return False

    if not folder.is_dir():
        return False

    kernel_suffixes = {".bsp", ".bc", ".bpc", ".tf", ".fk", ".tls", ".tpc"}

    try:
        for child in folder.iterdir():
            if child.is_file() and any(suffix.lower() in kernel_suffixes for suffix in child.suffixes):
                return True
    except OSError:
        return False

    return False


def first_matching_directory(
    candidates: Sequence[PathLike],
    predicate: Callable[[Path], bool],
) -> str:
    """
    Return the first candidate directory that satisfies `predicate`.

    The helper normalizes each candidate before invoking `predicate`, so callers
    can provide relative paths, `~`-expanded paths, or already-resolved `Path`
    objects without worrying about normalization details.
    """

    for candidate in candidates:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except Exception:
            continue
        if predicate(resolved):
            return str(resolved)
    return ""


# =============================================================================
# 3.                      REPOSITORY DATA-ROOT DISCOVERY
# =============================================================================

def prefer_dedicated_albedo_root(project_root: PathLike, hints: DataRootHints) -> DataRootHints:
    """
    Prefer a dedicated albedo directory over legacy LDEM/albedo coupling.

    Background
    ----------
    Older UI flows mirrored the LDEM path into the albedo field. That worked as
    a fallback, but this repository now stores albedo data in its own directory
    (`data/albedo_models`). When that directory exists, continuing to mirror the
    LDEM path causes avoidable loader warnings and prevents the intended raster
    dataset from being used.
    """

    try:
        root = Path(project_root).expanduser().resolve()
    except Exception:
        return hints

    ldem_root = resolve_existing_directory(hints.ldem_root) or hints.ldem_root
    albedo_root = resolve_existing_directory(hints.albedo_root) or hints.albedo_root
    use_ldem_for_albedo = bool(hints.use_ldem_for_albedo)

    dedicated_candidates = (
        root / "data" / "albedo_models",
        root / "data" / "albedo",
        root / "albedo",
    )
    dedicated_albedo = ""
    for candidate in dedicated_candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.is_dir():
            dedicated_albedo = str(resolved)
            break

    if albedo_root and ldem_root and not same_resolved_directory(albedo_root, ldem_root):
        return DataRootHints(
            ldem_root=ldem_root,
            albedo_root=albedo_root,
            kernel_dir=hints.kernel_dir,
            use_ldem_for_albedo=False,
        )

    if dedicated_albedo and ldem_root and not same_resolved_directory(dedicated_albedo, ldem_root):
        if use_ldem_for_albedo or not albedo_root or same_resolved_directory(albedo_root, ldem_root):
            return DataRootHints(
                ldem_root=ldem_root,
                albedo_root=dedicated_albedo,
                kernel_dir=hints.kernel_dir,
                use_ldem_for_albedo=False,
            )

    if use_ldem_for_albedo and ldem_root and not albedo_root:
        albedo_root = ldem_root

    return DataRootHints(
        ldem_root=ldem_root,
        albedo_root=albedo_root,
        kernel_dir=hints.kernel_dir,
        use_ldem_for_albedo=use_ldem_for_albedo,
    )


def autodetect_repository_data_roots(
    project_root: PathLike,
    current: Optional[DataRootHints] = None,
) -> tuple[DataRootHints, list[str]]:
    """
    Discover likely repository data directories for surface and SPICE assets.

    Search policy
    -------------
    1. Respect already-configured valid directories.
    2. Respect environment overrides when present.
    3. Search repository-default locations using project-specific naming
       conventions (`topografy_models`, `albedo_models`, `ephemeris_models`).
    4. Prefer a dedicated albedo directory over legacy "reuse LDEM" coupling.

    Returns
    -------
    (hints, messages)
        `hints` contains normalized directory strings.
        `messages` contains short human-readable notes suitable for UI logging.
    """

    state = current or DataRootHints()
    messages: list[str] = []

    root = Path(project_root).expanduser().resolve()
    data_root = root / "data"

    ldem_env = os.environ.get("LUNARSIM_LDEM_ROOT") or os.environ.get("LDEM_ROOT") or ""
    albedo_env = os.environ.get("LUNARSIM_ALBEDO_ROOT") or os.environ.get("ALBEDO_ROOT") or ""
    kernel_env = os.environ.get("LUNARSIM_KERNEL_DIR") or os.environ.get("SPICE_KERNELS") or ""

    ldem_root = resolve_existing_directory(state.ldem_root) or resolve_existing_directory(ldem_env)
    albedo_root = resolve_existing_directory(state.albedo_root) or resolve_existing_directory(albedo_env)
    kernel_dir = resolve_existing_directory(state.kernel_dir) or resolve_existing_directory(kernel_env)
    use_ldem_for_albedo = bool(state.use_ldem_for_albedo)

    if not ldem_root:
        ldem_root = first_matching_directory(
            [
                data_root / "topografy_models",
                data_root / "topography_models",
                data_root / "topography",
                data_root / "surface",
                data_root / "ldem",
                data_root / "LDEM",
                root / "ldem",
            ],
            lambda path: directory_looks_like_surface_data(path, keywords=("ldem", "topo", "surface")),
        )

    if not albedo_root:
        albedo_root = first_matching_directory(
            [
                data_root / "albedo_models",
                data_root / "albedo",
                root / "albedo",
            ],
            lambda path: directory_looks_like_surface_data(path, keywords=("albedo", "reflect", "ldam")),
        )

    if not albedo_root and ldem_root and use_ldem_for_albedo:
        albedo_root = ldem_root

    normalized = prefer_dedicated_albedo_root(
        root,
        DataRootHints(
            ldem_root=ldem_root,
            albedo_root=albedo_root,
            kernel_dir=kernel_dir,
            use_ldem_for_albedo=use_ldem_for_albedo,
        ),
    )

    if not kernel_dir:
        kernel_dir = first_matching_directory(
            [
                data_root / "ephemeris_models",
                data_root / "kernels",
                data_root / "spice_kernels",
                root / "kernels",
                root / "spice_kernels",
            ],
            directory_looks_like_spice_kernels,
        )

    hints = DataRootHints(
        ldem_root=normalized.ldem_root,
        albedo_root=normalized.albedo_root,
        kernel_dir=kernel_dir,
        use_ldem_for_albedo=normalized.use_ldem_for_albedo,
    )

    if hints.ldem_root:
        messages.append(f"[UI] LDEM auto-filled: {Path(hints.ldem_root).name}")
    if hints.albedo_root and not same_resolved_directory(hints.albedo_root, hints.ldem_root):
        messages.append(f"[UI] Albedo auto-filled: {Path(hints.albedo_root).name}")
    if hints.kernel_dir:
        messages.append(f"[UI] Kernels auto-filled: {Path(hints.kernel_dir).name}")

    return hints, messages


# =============================================================================
# 4.                        LUNAR MAP / ASSET DISCOVERY
# =============================================================================

_LUNAR_MAP_NAMES: tuple[str, ...] = (
    "lunar_surface.jpg",
    "lunar_surface.png",
    "lunar_map.jpg",
    "lunar_map.png",
    "lroc_color_2k.jpg",
    "lroc_color_2k.png",
    "moon_surface.jpg",
    "moon_surface.png",
    "moon_map.jpg",
    "moon_map.png",
)


def project_root_from_path(start_path: PathLike, *, max_levels: int = 6) -> Path:
    """
    Best-effort repository-root discovery relative to an arbitrary path.

    Heuristics
    ----------
    Walking upward from `start_path`, return the first directory that:
    - is itself named `LUNAR_SIMULATION`, or
    - contains `data/assets`

    If no strong signal is found within `max_levels`, the last visited path is
    returned. This keeps the helper deterministic and safe for callers that are
    only using it to build fallback search paths.
    """

    current = Path(start_path).expanduser().resolve()
    if current.is_file():
        current = current.parent

    last = current
    for _ in range(max(1, int(max_levels))):
        if current.name in ("LUNAR_SIMULATION", "ST_LRPS"):
            return current
        if (current / "data" / "assets").is_dir():
            return current
        last = current
        current = current.parent
    return last


def iter_lunar_map_candidates(
    explicit_path: Optional[str] = None,
    *,
    start_dir: PathLike | None = None,
) -> list[Path]:
    """
    Build candidate paths for lunar texture discovery.

    Search order
    ------------
    1. Explicit path passed by the caller
    2. `LUNARSIM_LUNAR_MAP` environment variable
    3. `LUNARSIM_ASSETS_DIR` environment variable + common filenames
    4. Canonical repository asset directory: `<PROJECT_ROOT>/data/assets`
    5. Local module-adjacent asset folders
    6. Current working directory fallbacks
    7. Final fallback: any image file found in the canonical asset directory
    """

    anchor = Path(start_dir).expanduser().resolve() if start_dir is not None else Path.cwd().resolve()
    project_root = project_root_from_path(anchor)
    canonical_assets = project_root / "data" / "assets"

    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_file = os.environ.get("LUNARSIM_LUNAR_MAP", "").strip()
    if env_file:
        candidates.append(Path(env_file).expanduser())

    env_assets_dir = os.environ.get("LUNARSIM_ASSETS_DIR", "").strip()
    env_dir = Path(env_assets_dir).expanduser() if env_assets_dir else None

    for name in _LUNAR_MAP_NAMES:
        if env_dir is not None:
            candidates.append(env_dir / name)
        candidates.append(canonical_assets / name)

    local_assets = anchor / "assets"
    for name in _LUNAR_MAP_NAMES:
        candidates.append(anchor / name)
        candidates.append(local_assets / name)

    cwd = Path.cwd()
    cwd_assets = cwd / "assets"
    for name in _LUNAR_MAP_NAMES:
        candidates.append(cwd / name)
        candidates.append(cwd_assets / name)

    seen: set[str] = set()
    normalized: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(resolved)

    try:
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            for candidate in sorted(canonical_assets.glob(pattern)):
                try:
                    resolved = candidate.expanduser().resolve()
                except Exception:
                    resolved = candidate.expanduser()
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(resolved)
    except Exception:
        pass

    return normalized


def find_lunar_map_path(
    explicit_path: Optional[str] = None,
    *,
    start_dir: PathLike | None = None,
) -> Optional[str]:
    """
    Return the first readable lunar texture path discovered on disk.

    The caller supplies `start_dir` so path search stays deterministic regardless
    of the current working directory used to launch the process.
    """

    for candidate in iter_lunar_map_candidates(explicit_path, start_dir=start_dir):
        try:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        except Exception:
            continue
    return None


__all__ = (
    "DataRootHints",
    "resolve_existing_directory",
    "same_resolved_directory",
    "directory_looks_like_surface_data",
    "directory_looks_like_spice_kernels",
    "first_matching_directory",
    "prefer_dedicated_albedo_root",
    "autodetect_repository_data_roots",
    "project_root_from_path",
    "iter_lunar_map_candidates",
    "find_lunar_map_path",
)
