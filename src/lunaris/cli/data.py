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
from typing import Any, Dict, List, Optional
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
    "datasets",
)

#: Logical groups a dataset entry may belong to.
GROUPS = ("gravity", "ephemeris", "topography", "albedo", "datasets")

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


def find_manifest(explicit: Optional[str] = None) -> Path:
    """Resolve the manifest path from ``--manifest`` or the repository default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_manifest_path()


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the JSON manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Data manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        raise ValueError(f"Malformed manifest (expected a 'datasets' list): {path}")
    return manifest


def resolve_data_root(cli_data_dir: Optional[str] = None) -> Path:
    """Resolve the data root: ``--data-dir`` > ``LUNARIS_DATA_DIR`` > repo ``data/``."""
    if cli_data_dir:
        return Path(cli_data_dir).expanduser().resolve()
    root = find_project_root(Path(__file__).resolve())
    return data_dir_from_root(root)


def dataset_target_path(data_root: Path, entry: Dict[str, Any]) -> Path:
    """Absolute path where ``entry`` is expected to live on disk."""
    subdir = entry.get("target_subdir") or ""
    return data_root / subdir / entry["filename"]


def select_datasets(
    manifest: Dict[str, Any],
    *,
    group: Optional[str] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter manifest datasets by group and/or name (no filter -> all)."""
    items = list(manifest.get("datasets", []))
    if group is not None:
        items = [d for d in items if d.get("group") == group]
    if name is not None:
        items = [d for d in items if d.get("name") == name]
    return items


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
    entry: Dict[str, Any],
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
def verify_entry(entry: Dict[str, Any], data_root: Path) -> str:
    """Return a verification status for a single entry.

    One of: ``valid``, ``present`` (no hash to check), ``missing``,
    ``manual_missing`` (absent and no URL), ``hash_mismatch``.
    """
    target = dataset_target_path(data_root, entry)
    if not target.exists():
        return "missing" if entry.get("url") else "manual_missing"
    expected = entry.get("sha256")
    if expected:
        return "valid" if sha256_file(target).lower() == str(expected).lower() else "hash_mismatch"
    return "present"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(manifest: Dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    datasets = select_datasets(manifest, group=args.group)
    print(f"manifest schema_version={manifest.get('schema_version')}   data root: {data_root}")
    if not datasets:
        print("(no matching datasets)")
        return 0
    by_group: Dict[str, List[Dict[str, Any]]] = {}
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


def cmd_download(manifest: Dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    if not (args.all or args.group or args.name):
        print("error: choose what to download: --all, --group <g>, or --name <n>", file=sys.stderr)
        return 2
    datasets = (
        select_datasets(manifest)
        if args.all
        else select_datasets(manifest, group=args.group, name=args.name)
    )
    if not datasets:
        print("(no matching datasets)")
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


def cmd_verify(manifest: Dict[str, Any], data_root: Path, args: argparse.Namespace) -> int:
    datasets = select_datasets(manifest, group=args.group)
    labels = {
        "valid": "OK (hash verified)",
        "present": "OK (present, no hash)",
        "missing": "MISSING",
        "manual_missing": "MISSING (manual placement required)",
        "hash_mismatch": "HASH MISMATCH",
    }
    failures = 0
    for entry in datasets:
        status = verify_entry(entry, data_root)
        required = bool(entry.get("required"))
        target = dataset_target_path(data_root, entry)
        print(f"[{'required' if required else 'optional':8s}] "
              f"{str(entry.get('name')):28s} {labels[status]:36s} {target}")
        if status in _FAIL_STATUSES and required:
            failures += 1
    if failures:
        print(f"\n{failures} required dataset(s) missing or invalid.", file=sys.stderr)
        return 1
    print("\nAll required datasets present and valid (optional items may be missing).")
    return 0


def cmd_path(manifest: Optional[Dict[str, Any]], data_root: Path, args: argparse.Namespace) -> int:
    print(data_root)
    for sub in CANONICAL_SUBDIRS:
        print(data_root / sub)
    return 0


# --------------------------------------------------------------------------- #
# Parser + entry point
# --------------------------------------------------------------------------- #
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
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="Download datasets.")
    p_dl.add_argument("--all", action="store_true", help="Download every catalogued dataset.")
    p_dl.add_argument("--group", choices=GROUPS, default=None)
    p_dl.add_argument("--name", default=None, help="Download a single dataset by name.")
    p_dl.add_argument("--dry-run", action="store_true", help="Show what would happen; download nothing.")
    p_dl.add_argument("--overwrite", action="store_true", help="Refetch even if the file is present.")
    p_dl.add_argument("--no-verify", action="store_true", help="Skip SHA-256 verification.")
    p_dl.set_defaults(func=cmd_download)

    p_verify = sub.add_parser("verify", help="Verify presence/integrity of datasets.")
    p_verify.add_argument("--group", choices=GROUPS, default=None)
    p_verify.set_defaults(func=cmd_verify)

    p_path = sub.add_parser("path", help="Print the resolved data root and subdirectories.")
    p_path.set_defaults(func=cmd_path)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = resolve_data_root(args.data_dir)

    manifest: Optional[Dict[str, Any]] = None
    if args.command != "path":
        try:
            manifest = load_manifest(find_manifest(args.manifest))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return int(args.func(manifest, data_root, args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
