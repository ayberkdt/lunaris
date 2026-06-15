"""Audit pre-contract Monte Carlo / ST-LRPS results into a trust manifest.

Reviewer §10: results produced before the freeze-review fixes may rest on now-closed
contract/physics gaps, so they must be classified before being reused as evidence.
This module scans for Monte Carlo result archives and ST-LRPS run directories and
labels each one ``invalid`` / ``rerun_required`` / ``quarantined`` / ``trusted`` with
explicit, item-tagged reasons:

  * §3  impact_frame_available — Moon-fixed impact geography requires real ephemeris.
  * §6  impact_position_method — exact line-sphere crossing vs approximate/step endpoint.
  * §7  artifact / coefficient / kernel hash provenance.
  * v2  archive schema — pre-v2 archives predate the provenance manifest entirely.

Verdict ladder (most-to-least severe):

  * ``invalid``       — structurally not a usable result: a corrupt schema version,
                        or an archive that declares schema v2 yet is missing required
                        manifest fields (not produced by a complete MonteCarloEngine
                        run). Neither trustworthy nor safe to rerun-from.
  * ``rerun_required``— a loadable result whose impact geography/timing is
                        scientifically wrong (no Moon-fixed frame, or a non-exact
                        crossing); impact statistics must be regenerated.
  * ``quarantined``   — usable numbers but unverifiable: a provenance gap, or a
                        genuine pre-v2 / pre-contract archive. Held for review,
                        NOT auto-rerun (the numbers may still be fine).
  * ``trusted``       — complete v2 provenance and exact impact handling.

The classification functions (``classify_mc_archive`` / ``classify_st_lrps_run``) are
pure and unit-tested. ``audit_root`` / ``main`` handle discovery and I/O. Heavy
optional dependencies (h5py, the MC loader) are imported lazily so importing this
module — e.g. from a unit test — never requires them.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

# Hash-provenance keys written by MonteCarloEngine.run (§7). Presence of any one
# means the archive carries verifiable artifact/coefficient/kernel provenance.
_PROVENANCE_HASH_KEYS = (
    "kernel_source_sha256",
    "sh_coefficient_sha256",
    "st_lrps_checkpoint_sha256",
    "st_lrps_config_sha256",
)
_SHA256_HEX_LENGTH = 64

# Mirrors ``monte_carlo_engine.REQUIRED_ARCHIVE_V2_FIELDS`` so this module stays
# importable without the heavy engine deps. A v2 archive missing any of these
# declares the v2 contract but was not produced by a complete MonteCarloEngine
# run -> INVALID (the strict loader rejects the same archives).
_REQUIRED_V2_FIELDS = (
    "archive_schema_version",
    "n_samples",
    "seed",
    "duration_s",
    "output_dt_s",
    "backend",
    "requested_mc_backend",
    "actual_mc_backend",
    "mc_backend",
    "detect_impact",
    "compute_impact_statistics",
)

INVALID = "invalid"
TRUSTED = "trusted"
QUARANTINED = "quarantined"
RERUN = "rerun_required"


def _has_valid_provenance_hash(metadata: dict[str, Any]) -> bool:
    """Return whether the manifest contains at least one usable SHA-256 digest."""
    for key in _PROVENANCE_HASH_KEYS:
        value = metadata.get(key)
        if not isinstance(value, str):
            continue
        digest = value.strip().lower()
        is_hex = all(ch in "0123456789abcdef" for ch in digest)
        if len(digest) == _SHA256_HEX_LENGTH and is_hex:
            return True
    return False


def classify_mc_archive(metadata: dict[str, Any], *, has_impacts: bool) -> tuple[str, list[str]]:
    """Classify one Monte Carlo archive from its manifest metadata.

    ``has_impacts`` reflects whether any valid sample impacted (impact geography /
    timing only matters when there were impacts).
    """
    reasons: list[str] = []

    raw_version = metadata.get("archive_schema_version")
    if raw_version is None:
        # Genuine pre-v2 legacy archive: no provenance manifest at all. Hold for
        # review rather than auto-rerunning — the numbers may still be fine.
        return QUARANTINED, [
            "pre-v2 archive: no provenance manifest (§5); quarantined for review"
        ]
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        # A present-but-unparseable schema version is a corrupt manifest, not a
        # legacy archive: it is not a usable result.
        return INVALID, [
            f"archive_schema_version is not an integer ({raw_version!r}): corrupt manifest"
        ]

    if version < 2:
        return QUARANTINED, [
            f"pre-contract archive schema v{version}: no v2 provenance (§5); "
            "quarantined for review"
        ]

    missing = [f for f in _REQUIRED_V2_FIELDS if f not in metadata]
    if missing:
        # Declares the v2 contract but is incomplete -> not a real current run.
        return INVALID, [
            "declares schema v2 but is missing required manifest field(s): "
            + ", ".join(sorted(missing))
            + " (not a complete MonteCarloEngine run)"
        ]

    backend_diag = metadata.get("backend_diagnostics")
    backend_diag = backend_diag if isinstance(backend_diag, dict) else {}
    impact_method = backend_diag.get("impact_position_method")
    impact_frame = metadata.get("impact_frame_available")
    has_hashes = _has_valid_provenance_hash(metadata)

    geography_broken = False
    if has_impacts:
        if impact_frame is not True:
            reasons.append(
                "impacts present but Moon-fixed frame unavailable "
                "(§3: geography unreliable)"
            )
            geography_broken = True
        if impact_method != "line_sphere_quadratic":
            reasons.append(
                "impact crossing is not the exact line-sphere solution "
                "(§6: oblique impact time/position unreliable)"
            )
            geography_broken = True

    if not has_hashes:
        reasons.append("missing artifact/coefficient/kernel hash provenance (§7)")

    if not reasons:
        return TRUSTED, []
    # Impacted-result geography/timing defects make the impact statistics unusable
    # as evidence -> rerun. A pure provenance gap leaves the numbers usable but
    # unverifiable -> quarantine.
    if geography_broken:
        return RERUN, reasons
    return QUARANTINED, reasons


def classify_st_lrps_run(config: dict[str, Any], *, has_checkpoint: bool) -> tuple[str, list[str]]:
    """Classify one ST-LRPS run directory from its ``config.json`` content."""
    if not has_checkpoint:
        return RERUN, ["no checkpoint (ckpt_best.pt / ckpt_last.pt) present"]
    contract = config.get("artifact_contract")
    if not isinstance(contract, dict) or not contract:
        return RERUN, [
            "missing or invalid artifact_contract in config: "
            "not loadable by current runtime (pre-contract)"
        ]
    return TRUSTED, []


# ---------------------------------------------------------------------------
# Discovery / I/O
# ---------------------------------------------------------------------------

_MC_REQUIRED_KEYS = {"Y", "t", "impact_flags", "sc_samples"}


def _is_mc_hdf5(path: Path) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(str(path), "r") as f:
            return _MC_REQUIRED_KEYS.issubset(set(f.keys()))
    except Exception:
        return False


def _is_mc_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(str(path)) as zf:
            names = {Path(n).stem for n in zf.namelist()}
        return _MC_REQUIRED_KEYS.issubset(names)
    except Exception:
        return False


def _audit_mc_archive(path: Path) -> dict[str, Any]:
    from lunaris.core.monte_carlo_engine import load_mc_result

    result = load_mc_result(str(path), lazy=True, strict=False)
    valid = result.valid_sample_mask()
    has_impacts = bool((valid & (result.impact_mask > 0.5)).any())
    status, reasons = classify_mc_archive(result.diagnostics, has_impacts=has_impacts)
    return {
        "path": str(path),
        "kind": "mc_archive",
        "status": status,
        "reasons": reasons,
        "has_impacts": has_impacts,
        "archive_schema_version": result.diagnostics.get("archive_schema_version"),
    }


def _audit_st_lrps_run(run_dir: Path) -> dict[str, Any]:
    ckpt_dir = run_dir / "checkpoints"
    has_ckpt = (ckpt_dir / "ckpt_best.pt").is_file() or (ckpt_dir / "ckpt_last.pt").is_file()
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        config = {}
    status, reasons = classify_st_lrps_run(config, has_checkpoint=has_ckpt)
    return {"path": str(run_dir), "kind": "st_lrps_run", "status": status, "reasons": reasons}


def audit_root(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in (".h5", ".hdf5") and _is_mc_hdf5(path):
            entries.append(_audit_mc_archive(path))
        elif path.suffix.lower() == ".npz" and _is_mc_npz(path):
            entries.append(_audit_mc_archive(path))

    for cfg_path in sorted(root.rglob("config.json")):
        run_dir = cfg_path.parent
        if (run_dir / "checkpoints").is_dir():
            entries.append(_audit_st_lrps_run(run_dir))

    buckets: dict[str, list[dict[str, Any]]] = {
        INVALID: [], RERUN: [], QUARANTINED: [], TRUSTED: [],
    }
    for e in entries:
        buckets[e["status"]].append(e)

    return {
        "schema_version": 2,
        "root": str(root),
        "summary": {k: len(v) for k, v in buckets.items()},
        "invalid": buckets[INVALID],
        "rerun_required": buckets[RERUN],
        "quarantined": buckets[QUARANTINED],
        "trusted": buckets[TRUSTED],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit legacy MC/ST-LRPS results into a trust manifest.")
    ap.add_argument("--root", default="outputs", help="Directory to scan (default: outputs)")
    ap.add_argument(
        "--output",
        default="outputs/audit/legacy_results_manifest.json",
        help="Manifest output path",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[AUDIT][FATAL] root not found: {root}")
        return 1

    manifest = audit_root(root)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    s = manifest["summary"]
    print(f"[AUDIT] root={root}")
    print(
        f"[AUDIT] invalid={s[INVALID]}  rerun_required={s[RERUN]}  "
        f"quarantined={s[QUARANTINED]}  trusted={s[TRUSTED]}"
    )
    print(f"[AUDIT] manifest written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
