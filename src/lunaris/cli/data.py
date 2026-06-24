"""``lunaris-data`` — manifest-driven external data acquisition for Lunaris.

A lightweight, headless, standard-library downloader/verifier for the large
external scientific data files Lunaris depends on (lunar gravity coefficients,
SPICE/ephemeris kernels, LOLA/LDEM topography, albedo grids) plus locally
generated ST-LRPS datasets.

The asset catalogue lives in ``data/data_sources.json``. Files are placed under
the resolved data root (``--data-dir`` > ``LUNARIS_DATA_DIR`` > repository
``data/``) in canonical subdirectories. The tool never writes into ``src/`` and
never requires GUI dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lunaris.common.paths import data_dir_from_root, find_project_root

try:  # tqdm is a core dependency; degrade gracefully if it is unavailable.
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is declared in core deps
    tqdm = None  # type: ignore[assignment]

SCHEMA_VERSION = 1

#: Canonical data subdirectories (also the expected on-disk layout under the root).
CANONICAL_SUBDIRS = (
    "gravity_models",
    "ephemeris_models",
    "topography_models",
    "albedo_models",
    "thermal_models",
    "assets",
    "datasets",
)

#: Logical groups a dataset entry may belong to.
GROUPS = ("gravity", "ephemeris", "topography", "albedo", "thermal", "assets", "datasets")

#: Named download presets — curated entry-name bundles for common onboarding scenarios.
DATA_PRESETS: dict[str, list[str]] = {
    "minimal": [
        "naif_lsk_naif0012",
        "naif_spk_de440",
        "naif_moon_pa_de440",
        "naif_moon_fk_de440",
        "naif_pck_gm_de440",
        "grail_gravity_jggrx",
        "lola_ldem_topography",
    ],
    "full-gravity": [
        "naif_lsk_naif0012",
        "naif_spk_de440",
        "naif_moon_pa_de440",
        "naif_moon_fk_de440",
        "naif_pck_gm_de440",
        "grail_gravity_jggrx",
        "lola_ldem_topography",
        "lola_ldam_albedo_10",
        "lola_ldam_albedo_8",
    ],
    "surface": [
        "lola_ldem_topography",
        "lola_ldam_albedo_10",
        "lola_ldam_albedo_8",
    ],
    "st-lrps-dev": [
        "naif_lsk_naif0012",
        "naif_spk_de440",
        "naif_moon_pa_de440",
        "naif_moon_fk_de440",
        "naif_pck_gm_de440",
        "grail_gravity_jggrx",
        "lola_ldem_topography",
        "lola_ldam_albedo_10",
    ],
}

_ALLOWED_SCHEMES = ("http", "https", "file")
_CHUNK = 1 << 16
_FAIL_STATUSES = ("missing", "hash_mismatch", "manual_missing")


# --------------------------------------------------------------------------- #
# Manifest + path resolution
# --------------------------------------------------------------------------- #
def default_manifest_path() -> Path:
    """Return the repository manifest path (``<project root>/data/data_sources.json``)."""
    root = find_project_root(Path(__file__).resolve())
    return root / "data" / "data_sources.json"


def find_manifest(explicit: str | None = None) -> Path:
    """Resolve the manifest path from ``--manifest`` or the repository default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_manifest_path()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the JSON manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Data manifest not found: {path}")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        raise ValueError(f"Malformed manifest (expected a 'datasets' list): {path}")
    return manifest


def resolve_data_root(cli_data_dir: str | None = None) -> Path:
    """Resolve the data root: ``--data-dir`` > ``LUNARIS_DATA_DIR`` > repo ``data/``."""
    if cli_data_dir:
        return Path(cli_data_dir).expanduser().resolve()
    root = find_project_root(Path(__file__).resolve())
    return data_dir_from_root(root)


def dataset_target_path(data_root: Path, entry: dict[str, Any]) -> Path:
    """Absolute path where ``entry`` is expected to live on disk."""
    subdir = entry.get("target_subdir") or ""
    return data_root / subdir / entry["filename"]


def dataset_candidate_paths(data_root: Path, entry: dict[str, Any]) -> list[Path]:
    """Return canonical and alias target paths for ``entry`` in priority order."""
    names = [str(entry["filename"])]
    for alias in entry.get("aliases") or ():
        alias_s = str(alias)
        if alias_s not in names:
            names.append(alias_s)
    subdir = entry.get("target_subdir") or ""
    return [data_root / subdir / name for name in names]


def resolve_dataset_path(data_root: Path, entry: dict[str, Any]) -> Path | None:
    """Return the first existing canonical/alias path for ``entry``."""
    for candidate in dataset_candidate_paths(data_root, entry):
        if candidate.exists():
            return candidate
    return None


def select_datasets(
    manifest: dict[str, Any],
    *,
    group: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter manifest datasets by group and/or name (no filter -> all)."""
    items = list(manifest.get("datasets", []))
    if group is not None:
        items = [d for d in items if d.get("group") == group]
    if name is not None:
        items = [d for d in items if d.get("name") == name]
    return items


def select_preset_datasets(
    manifest: dict[str, Any],
    preset: str,
) -> list[dict[str, Any]]:
    """Return the manifest entries named by *preset* (in preset order)."""
    names = DATA_PRESETS.get(preset, [])
    by_name = {d["name"]: d for d in manifest.get("datasets", []) if "name" in d}
    return [by_name[n] for n in names if n in by_name]


def select_for_download(
    manifest: dict[str, Any],
    *,
    all_groups: bool = False,
    group: str | None = None,
    name: str | None = None,
    preset: str | None = None,
    include_optional: bool = False,
) -> list[dict[str, Any]]:
    """Select datasets for the ``download`` command.

    ``--name`` selects that exact entry regardless of its required/optional
    status. ``--preset`` selects the named bundle (all entries in the preset
    are included regardless of their required/optional flag in the manifest).
    Otherwise an ``--all`` or ``--group`` selection is restricted to
    entries with ``required=true`` unless ``include_optional`` is set.
    """
    if name is not None:
        return select_datasets(manifest, name=name)
    if preset is not None:
        return select_preset_datasets(manifest, preset)
    base = select_datasets(manifest) if all_groups else select_datasets(manifest, group=group)
    if include_optional:
        return base
    return [d for d in base if d.get("required")]


# --------------------------------------------------------------------------- #
# Hashing + download
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path`` (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_to_file(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest`` with an optional tqdm progress bar."""
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Refusing URL with unsupported scheme {scheme!r}")
    request = urllib.request.Request(url, headers={"User-Agent": "lunaris-data"})
    # URLs come exclusively from the project-controlled manifest.
    with urllib.request.urlopen(request) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        with open(dest, "wb") as fh:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) \
                if (tqdm is not None and total > 0) else None
            try:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()


def download_entry(
    entry: dict[str, Any],
    data_root: Path,
    *,
    overwrite: bool = False,
    verify: bool = True,
    dry_run: bool = False,
) -> str:
    """Download a single manifest entry. Returns a status string."""
    name = entry.get("name", "<unnamed>")
    target = dataset_target_path(data_root, entry)
    url = entry.get("url")

    if not url:
        print(
            f"[manual] {name}: no download URL; place '{entry.get('filename')}' "
            f"under {target.parent} manually."
        )
        if entry.get("notes"):
            print(f"          {entry['notes']}")
        return "manual"

    if target.exists() and not overwrite:
        print(f"[skip] {name}: already present at {target} (use --overwrite to refetch)")
        return "present"

    if dry_run:
        print(f"[dry-run] {name}: would download {url} -> {target}")
        return "dry-run"

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        _stream_to_file(url, part)
    except (urllib.error.URLError, ValueError, OSError) as exc:
        if part.exists():
            part.unlink()
        print(f"[error] {name}: download failed: {exc}")
        return "error"

    expected = entry.get("sha256")
    if verify and expected:
        actual = sha256_file(part)
        if actual.lower() != str(expected).lower():
            part.unlink()
            print(f"[error] {name}: sha256 mismatch (expected {expected}, got {actual})")
            return "hash_mismatch"

    part.replace(target)  # atomic rename on the same filesystem
    print(f"[ok] {name}: installed -> {target}")
    return "ok"


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _companion_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize manifest companion-file declarations."""
    out: list[dict[str, Any]] = []
    for raw in entry.get("companion_files") or ():
        if isinstance(raw, str):
            out.append({"filename": raw, "aliases": (), "required": True})
        elif isinstance(raw, dict) and raw.get("filename"):
            spec = dict(raw)
            spec.setdefault("aliases", ())
            spec.setdefault("required", True)
            out.append(spec)
    return out


def _companion_target_paths(data_root: Path, entry: dict[str, Any], spec: dict[str, Any]) -> list[Path]:
    names = [str(spec["filename"])]
    for alias in spec.get("aliases") or ():
        alias_s = str(alias)
        if alias_s not in names:
            names.append(alias_s)
    subdir = spec.get("target_subdir") or entry.get("target_subdir") or ""
    return [data_root / subdir / name for name in names]


def verify_entry_detail(entry: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Return detailed verification status for a manifest entry.

    The top-level ``status`` preserves the historic ``verify_entry`` statuses.
    Alias and companion information is additive so callers can decide whether
    companion gaps are warnings or strict failures.
    """
    canonical = dataset_target_path(data_root, entry)
    matched = resolve_dataset_path(data_root, entry)
    if matched is None:
        status = "missing" if entry.get("url") else "manual_missing"
    else:
        expected = entry.get("sha256")
        if expected:
            status = "valid" if sha256_file(matched).lower() == str(expected).lower() else "hash_mismatch"
        else:
            status = "present"

    companions = []
    for spec in _companion_specs(entry):
        paths = _companion_target_paths(data_root, entry, spec)
        found = next((path for path in paths if path.exists()), None)
        if found is None:
            companion_status = "missing"
        else:
            expected = spec.get("sha256")
            if expected:
                companion_status = (
                    "valid" if sha256_file(found).lower() == str(expected).lower() else "hash_mismatch"
                )
            else:
                companion_status = "present"
        companions.append({
            "filename": spec["filename"],
            "required": bool(spec.get("required", True)),
            "target": paths[0],
            "matched_path": found,
            "status": companion_status,
        })

    return {
        "status": status,
        "target": canonical,
        "matched_path": matched,
        "alias_used": matched is not None and matched != canonical,
        "companions": companions,
    }


def verify_entry(entry: dict[str, Any], data_root: Path) -> str:
    """Return a verification status for a single entry.

    One of: ``valid``, ``present`` (no hash to check), ``missing``,
    ``manual_missing`` (absent and no URL), ``hash_mismatch``.
    """
    return str(verify_entry_detail(entry, data_root)["status"])


def _entry_required_for_mode(entry: dict[str, Any], *, strict: bool) -> bool:
    """Return whether an entry must exist for the current verification mode."""
    return bool(entry.get("required")) or (strict and bool(entry.get("strict_required")))


def _format_match_note(detail: dict[str, Any]) -> str:
    matched = detail.get("matched_path")
    if detail.get("alias_used") and matched is not None:
        return f" (via alias: {Path(matched).name})"
    return ""


def _required_missing_companions(detail: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        comp for comp in detail.get("companions", [])
        if bool(comp.get("required", True)) and comp.get("status") not in ("present", "valid")
    ]


def _find_entry_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in manifest.get("datasets", []):
        if entry.get("name") == name:
            return entry
    return None


def _runtime_ephemeris_check(manifest: dict[str, Any], data_root: Path, *, strict: bool) -> tuple[bool, list[str]]:
    """Build a small real SPICE table from manifest-resolved kernels.

    This check intentionally lives in the CLI layer and imports SPICE-facing
    code lazily. It verifies that the files the data tool sees can also be
    consumed by the ephemeris builder.
    """
    messages: list[str] = []
    kernel_names = [
        "naif_lsk_naif0012",
        "naif_pck_pck00011",
        "naif_moon_pa_de440",
        "naif_spk_de440",
    ]
    optional_kernel_names = [
        "naif_pck_gm_de440",
        "naif_moon_fk_de440",
    ]

    kernels: list[str] = []
    ok = True
    for name in kernel_names:
        entry = _find_entry_by_name(manifest, name)
        if entry is None:
            messages.append(f"[runtime] missing manifest entry: {name}")
            ok = False
            continue
        detail = verify_entry_detail(entry, data_root)
        matched = detail.get("matched_path")
        if detail["status"] in _FAIL_STATUSES or matched is None:
            messages.append(f"[runtime] missing required SPICE kernel for smoke check: {name}")
            ok = False
            continue
        kernels.append(str(matched))

    for name in optional_kernel_names:
        entry = _find_entry_by_name(manifest, name)
        if entry is None:
            continue
        detail = verify_entry_detail(entry, data_root)
        matched = detail.get("matched_path")
        if matched is not None and detail["status"] not in _FAIL_STATUSES:
            kernels.append(str(matched))

    if not ok:
        return False, messages

    try:
        import warnings

        import numpy as np

        from lunaris.physics.ephemeris import build_tables
    except Exception as exc:
        return False, [f"[runtime] failed to import ephemeris dependencies: {type(exc).__name__}: {exc}"]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tables = build_tables(
                start_utc="2026-01-01T00:00:00",
                duration_s=3600.0,
                output_dt_s=600.0,
                kernels=tuple(kernels),
                inertial_frame="J2000",
                fixed_frame="MOON_PA",
                observer="MOON",
                include_third_body=True,
                clear_kernels_after=True,
                clean_kernels_before=True,
            )

        q_norm = np.linalg.norm(tables.q_i2f_tab, axis=1)
        if tables.q_i2f_tab.shape[0] < 2 or not np.all(np.isfinite(q_norm)):
            return False, ["[runtime] q_i2f_tab is empty or non-finite."]
        if float(np.max(np.abs(q_norm - 1.0))) > 1.0e-6:
            return False, ["[runtime] q_i2f_tab quaternion norms are not unit within 1e-6."]
        if float(np.linalg.norm(tables.q_i2f_tab[-1] - tables.q_i2f_tab[0])) <= 1.0e-12:
            return False, ["[runtime] q_i2f_tab is effectively constant; expected real lunar attitude motion."]
        sun_norm = float(np.linalg.norm(tables.r_sun_tab_m[0]))
        earth_norm = float(np.linalg.norm(tables.r_earth_tab_m[0]))
        if not (sun_norm > 1.0e10 and earth_norm > 1.0e7):
            return False, [
                f"[runtime] third-body vectors look degenerate: |Sun|={sun_norm:.6g} m, |Earth|={earth_norm:.6g} m"
            ]

        gm_fallbacks = [
            str(w.message) for w in caught
            if "GM for" in str(w.message) and "fallback" in str(w.message).lower()
        ]
        if gm_fallbacks:
            messages.extend(f"[runtime] warning: {msg}" for msg in gm_fallbacks)
            if strict:
                messages.append("[runtime] strict mode requires GM from SPICE kernels; install gm_de440.tpc.")
                ok = False

        messages.append(
            f"[runtime] ephemeris smoke OK: q={tables.q_i2f_tab.shape}, "
            f"|Sun|={sun_norm:.6g} m, |Earth|={earth_norm:.6g} m"
        )
        return ok, messages
    except Exception as exc:
        return False, [f"[runtime] ephemeris smoke failed: {type(exc).__name__}: {exc}"]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(manifest: dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    preset = getattr(args, "preset", None)
    if preset is not None:
        datasets = select_preset_datasets(manifest, preset)
        print(f"manifest schema_version={manifest.get('schema_version')}   data root: {data_root}   preset: {preset}")
    else:
        datasets = select_datasets(manifest, group=getattr(args, "group", None))
        print(f"manifest schema_version={manifest.get('schema_version')}   data root: {data_root}")
    if not datasets:
        print("(no matching datasets)")
        return 0
    by_group: dict[str, list[dict[str, Any]]] = {}
    for d in datasets:
        by_group.setdefault(d.get("group", "other"), []).append(d)
    ordered = [g for g in GROUPS if g in by_group] + [g for g in by_group if g not in GROUPS]
    for group in ordered:
        print(f"\n[{group}]")
        for d in by_group[group]:
            req = "required" if d.get("required") else "optional"
            avail = "url" if d.get("url") else "manual"
            print(f"  - {str(d.get('name')):28s} {req:8s} {avail:6s} {d.get('filename')}")
    return 0


def cmd_download(manifest: dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    preset = getattr(args, "preset", None)
    if not (args.all or args.group or args.name or preset):
        print("error: choose what to download: --all, --group <g>, --name <n>, or --preset <p>", file=sys.stderr)
        return 2
    datasets = select_for_download(
        manifest,
        all_groups=args.all,
        group=args.group,
        name=args.name,
        preset=preset,
        include_optional=args.include_optional,
    )
    if not datasets:
        print("(no matching datasets — required entries only; "
              "pass --include-optional to include optional ones)")
        return 0
    statuses = [
        download_entry(
            entry,
            data_root,
            overwrite=args.overwrite,
            verify=not args.no_verify,
            dry_run=args.dry_run,
        )
        for entry in datasets
    ]
    failed = [s for s in statuses if s in ("error", "hash_mismatch")]
    print(
        f"\nDone: {len(statuses)} entries — {statuses.count('ok')} downloaded, "
        f"{statuses.count('present')} present, {statuses.count('manual')} manual, "
        f"{len(failed)} failed."
    )
    return 1 if failed else 0


def cmd_verify(manifest: dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    datasets = select_datasets(manifest, group=args.group)
    labels = {
        "valid": "OK (hash verified)",
        "present": "OK (present, no hash)",
        "missing": "MISSING",
        "manual_missing": "MISSING (manual placement required)",
        "hash_mismatch": "HASH MISMATCH",
    }
    strict = bool(getattr(args, "strict", False))
    failures = 0
    for entry in datasets:
        detail = verify_entry_detail(entry, data_root)
        status = str(detail["status"])
        required = _entry_required_for_mode(entry, strict=strict)
        target = detail["target"]
        print(f"[{'required' if required else 'optional':8s}] "
              f"{str(entry.get('name')):28s} {labels[status]:36s} {target}"
              f"{_format_match_note(detail)}")
        if status in _FAIL_STATUSES and required:
            failures += 1

        missing_companions = _required_missing_companions(detail)
        if detail.get("companions") and status not in _FAIL_STATUSES:
            if missing_companions:
                missing = ", ".join(
                    f"{comp['status']}: {comp['target']}" for comp in missing_companions
                )
                print(f"{'':13s}{'':28s} COMPANION INVALID/MISSING          {missing}")
                if strict or required:
                    failures += 1
            else:
                count = len(detail["companions"])
                print(f"{'':13s}{'':28s} companions OK ({count})")

    if getattr(args, "runtime", False):
        runtime_ok, runtime_messages = _runtime_ephemeris_check(manifest, data_root, strict=strict)
        for message in runtime_messages:
            print(message)
        if not runtime_ok:
            failures += 1

    if failures:
        print(f"\n{failures} required dataset(s) missing or invalid.", file=sys.stderr)
        return 1
    print("\nAll required datasets present and valid (optional items may be missing).")
    return 0


def cmd_path(manifest: dict[str, Any] | None, data_root: Path, args: argparse.Namespace) -> int:
    print(data_root)
    for sub in CANONICAL_SUBDIRS:
        print(data_root / sub)
    return 0


def cmd_inspect(manifest: dict[str, Any] | None, data_root: Path, args: argparse.Namespace) -> int:
    from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract

    contract = DatasetContract.from_hdf5(
        args.data,
        allow_legacy_dataset_contract=bool(args.allow_legacy),
        allow_missing_dataset_contract=bool(args.allow_legacy),
        allow_legacy_derivative_convention=bool(args.allow_legacy_derivative_convention),
    )
    payload = contract.to_dict()
    print(json.dumps({
        "dataset_id": payload.get("dataset_id"),
        "dataset_kind": payload.get("dataset_kind"),
        "target_mode": payload.get("target_mode"),
        "baseline_kind": payload.get("baseline_kind"),
        "degree_min": payload.get("degree_min"),
        "degree_max": payload.get("degree_max"),
        "n_samples": payload.get("n_samples"),
        "altitude_min_km": payload.get("altitude_min_km"),
        "altitude_max_km": payload.get("altitude_max_km"),
        "coordinate_frame": payload.get("coordinate_frame"),
        "units": payload.get("units"),
        "legacy_inferred": payload.get("legacy_inferred"),
    }, indent=2, sort_keys=True))
    return 0


def cmd_validate_dataset(manifest: dict[str, Any] | None, data_root: Path, args: argparse.Namespace) -> int:
    from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file

    report = validate_dataset_file(
        args.data,
        out_dir=args.out,
        n_check=int(args.n_check),
        seed=int(args.seed),
        allow_legacy_dataset_contract=bool(args.allow_legacy),
        allow_missing_dataset_contract=bool(args.allow_legacy),
        allow_legacy_derivative_convention=bool(args.allow_legacy_derivative_convention),
    )
    print(json.dumps({"passed": report["passed"], "errors": report["errors"], "warnings": report["warnings"]}, indent=2))
    return 0 if report["passed"] else 1


def cmd_report_dataset(manifest: dict[str, Any] | None, data_root: Path, args: argparse.Namespace) -> int:
    from lunaris.surrogate.st_lrps.data.quality_report import build_dataset_quality_report

    report = build_dataset_quality_report(
        args.data,
        out_dir=args.out,
        bins=int(args.bins),
        allow_legacy_dataset_contract=bool(args.allow_legacy),
    )
    print(json.dumps({
        "n_samples": report["n_samples"],
        "altitude_km": report["altitude_km"],
        "finite_fraction": report["finite_fraction"],
        "warnings": report["warnings"],
    }, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# Parser + entry point
# --------------------------------------------------------------------------- #
def cmd_presets(manifest: dict[str, Any] | None, data_root: Path, args: argparse.Namespace) -> int:
    """Print available named preset bundles."""
    for preset_name, names in DATA_PRESETS.items():
        print(f"{preset_name}:  {', '.join(names)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunaris-data",
        description="Manifest-driven external data acquisition for Lunaris "
        "(gravity, SPICE/ephemeris, topography, albedo, datasets).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Data root (overrides LUNARIS_DATA_DIR and the repo data/ fallback).",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Manifest path (default: <project root>/data/data_sources.json).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List catalogued datasets.")
    p_list.add_argument("--group", choices=GROUPS, default=None)
    p_list.add_argument("--preset", choices=list(DATA_PRESETS), default=None,
                        help="Filter list to a named preset bundle.")
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser(
        "download",
        help="Download datasets (required-only by default; --include-optional for the rest).",
        description="Download catalogued datasets. By default a --group or --all "
        "selection downloads only required entries; add --include-optional to also "
        "fetch optional ones. --name always downloads the exact entry named.",
    )
    p_dl.add_argument("--all", action="store_true",
                      help="Select required datasets across every group (add --include-optional for optional ones).")
    p_dl.add_argument("--group", choices=GROUPS, default=None,
                      help="Select a group; downloads only its required entries unless --include-optional is given.")
    p_dl.add_argument("--name", default=None,
                      help="Download a single dataset by exact name, regardless of required/optional status.")
    p_dl.add_argument("--preset", choices=list(DATA_PRESETS), default=None,
                      help="Download a named preset bundle (all entries regardless of required/optional flag).")
    p_dl.add_argument("--include-optional", action="store_true",
                      help="Also include optional (required=false) entries for --group/--all.")
    p_dl.add_argument("--dry-run", action="store_true", help="Show what would happen; download nothing.")
    p_dl.add_argument("--overwrite", action="store_true", help="Refetch even if the file is present.")
    p_dl.add_argument("--no-verify", action="store_true", help="Skip SHA-256 verification.")
    p_dl.set_defaults(func=cmd_download)

    p_verify = sub.add_parser("verify", help="Verify presence/integrity of datasets.")
    p_verify.add_argument("--group", choices=GROUPS, default=None)
    p_verify.add_argument(
        "--strict",
        action="store_true",
        help="Also require strict-required entries and present-entry companion files.",
    )
    p_verify.add_argument(
        "--runtime",
        action="store_true",
        help="Run a small real SPICE ephemeris smoke check using resolved manifest kernels.",
    )
    p_verify.set_defaults(func=cmd_verify)

    p_path = sub.add_parser("path", help="Print the resolved data root and subdirectories.")
    p_path.set_defaults(func=cmd_path)

    def _add_dataset_path_flags(p_ds: argparse.ArgumentParser) -> None:
        p_ds.add_argument("--data", required=True, help="ST-LRPS HDF5 dataset path.")
        p_ds.add_argument("--allow-legacy", action="store_true",
                          help="Allow reading old datasets whose DatasetContract is inferred from attrs.")
        p_ds.add_argument("--allow-legacy-derivative-convention", action="store_true",
                          help="Allow old derivative convention only for inspection.")

    p_inspect = sub.add_parser("inspect", help="Inspect an ST-LRPS dataset contract.")
    _add_dataset_path_flags(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_validate = sub.add_parser("validate", help="Validate an ST-LRPS HDF5 dataset.")
    _add_dataset_path_flags(p_validate)
    p_validate.add_argument("--out", default=None, help="Directory for dataset_validation_report.json.")
    p_validate.add_argument("--n-check", type=int, default=1024)
    p_validate.add_argument("--seed", type=int, default=0)
    p_validate.set_defaults(func=cmd_validate_dataset)

    p_report = sub.add_parser("report", help="Write ST-LRPS dataset quality JSON/Markdown reports.")
    _add_dataset_path_flags(p_report)
    p_report.add_argument("--out", required=True, help="Directory for dataset_quality_report.json and summary Markdown.")
    p_report.add_argument("--bins", type=int, default=20)
    p_report.set_defaults(func=cmd_report_dataset)

    p_presets = sub.add_parser("presets", help="List available named preset bundles.")
    p_presets.set_defaults(func=cmd_presets)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = resolve_data_root(args.data_dir)

    manifest: dict[str, Any] | None = None
    if args.command not in {"path", "inspect", "validate", "report", "presets"}:
        try:
            manifest = load_manifest(find_manifest(args.manifest))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return int(args.func(manifest, data_root, args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
