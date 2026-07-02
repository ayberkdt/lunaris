"""Provenance-stamped uncertainty-quantification report over an ensemble.

One call (or ``python -m lunaris.analysis.ensemble.uq_report``) turns a
batch-propagated :class:`~lunaris.common.batch_defs.MCRunResult` ensemble into a run
directory containing:

- ``uq_covariance.npz`` — epochs, mean state, 6×6 covariance history, RIC
  1-σ components, 3-σ ellipsoid axes/orientations, altitude statistics.
- ``uq_summary.csv``    — per-epoch scalar table for quick inspection.
- ``uq_manifest.json``  — canonical-JSON provenance: the scientific definition
  of the covariance, sample counts, source-archive hash and metadata, run
  configuration echo, git/environment snapshot, per-file SHA-256 hashes, and
  a deterministic content hash over the numerical arrays so a re-run with the
  same seed is provably identical.
- ``figures/``          — σ-envelopes, RIC history, eigenvalue spectrum,
  3-D ellipsoid tubes, altitude envelope (skipped, and recorded as skipped,
  when matplotlib is unavailable).

Scientific definition (stated verbatim in the manifest): the covariance is the
unbiased sample covariance (``ddof=1``) of the ensemble state in the
Moon-centred inertial integration frame at the shared output epochs, induced by
the declared initial-state / spacecraft-parameter dispersion under
deterministic dynamics. It contains no process noise and no measurement
updates, and it is **not** an orbit-determination covariance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import subprocess
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.batch_defs import MCRunResult
from lunaris.common.constants import R_MOON
from lunaris.common.hashing import canonical_json_sha256, canonical_json_text
from lunaris.common.provenance import sha256_file, utc_now_iso

from .statistics import (
    EnsembleStatistics,
    ErrorEllipsoids,
    RICUncertainty,
    compute_ensemble_statistics,
    compute_error_ellipsoids,
    compute_ric_uncertainty,
)

UQ_REPORT_SCHEMA_VERSION = 1

COVARIANCE_DEFINITION = (
    "Unbiased sample covariance (ddof=1) of the ensemble state in the "
    "Moon-centred inertial integration frame at the shared output epochs, "
    "induced by the declared initial-state/spacecraft-parameter dispersion "
    "under deterministic dynamics. No process noise, no measurement updates; "
    "this is not an orbit-determination covariance and does not represent "
    "navigation performance."
)

_NPZ_NAME = "uq_covariance.npz"
_CSV_NAME = "uq_summary.csv"
_MANIFEST_NAME = "uq_manifest.json"


def ensemble_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Deterministic SHA-256 over named float64 arrays (name, shape, raw bytes).

    Hashing the array *contents* (not the NPZ file) keeps the digest independent
    of zip timestamps, so identical seeds provably produce identical results.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float64))
        digest.update(f"{name}:{arr.shape}:".encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _collect_git_info() -> dict[str, Any]:
    """Minimal git snapshot (commit, dirty state) without heavy imports."""

    def run(args: list[str]) -> tuple[str | None, str | None]:
        try:
            proc = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            return None, str(exc)
        if proc.returncode != 0:
            return None, (proc.stderr or proc.stdout).strip() or f"git exited {proc.returncode}"
        return proc.stdout.strip(), None

    commit, commit_err = run(["rev-parse", "HEAD"])
    status, status_err = run(["status", "--porcelain"])
    return {
        "commit_sha": commit,
        "is_dirty": None if status is None else bool(status),
        "errors": {"commit": commit_err, "dirty": status_err},
    }


def _collect_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "collected_at_utc": utc_now_iso(),
    }


def _archive_record(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {"path": None, "sha256": None, "missing_reason": "source archive not configured"}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "sha256": None, "missing_reason": "source archive not found locally"}
    return {"path": str(p), "sha256": sha256_file(p), "missing_reason": None}


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of diagnostics/config values for canonical JSON."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<ndarray shape={value.shape}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_summary_csv(
    path: Path,
    ens: EnsembleStatistics,
    ric: RICUncertainty,
    ell: ErrorEllipsoids,
) -> None:
    P_pos = ens.pos_cov()
    P_vel = ens.vel_cov()
    sigma_pos_m = np.sqrt(np.maximum(np.trace(P_pos, axis1=1, axis2=2), 0.0))
    sigma_vel_ms = np.sqrt(np.maximum(np.trace(P_vel, axis1=1, axis2=2), 0.0))
    fieldnames = [
        "t_s",
        "sigma_pos_m",
        "sigma_vel_ms",
        "sigma_radial_m",
        "sigma_along_m",
        "sigma_cross_m",
        "ellipsoid_semi_axis_max_3sigma_m",
        "alt_mean_km",
        "alt_std_km",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(int(ens.t.shape[0])):
            writer.writerow(
                {
                    "t_s": float(ens.t[k]),
                    "sigma_pos_m": float(sigma_pos_m[k]),
                    "sigma_vel_ms": float(sigma_vel_ms[k]),
                    "sigma_radial_m": float(ric.sigma_ric_m[k, 0]),
                    "sigma_along_m": float(ric.sigma_ric_m[k, 1]),
                    "sigma_cross_m": float(ric.sigma_ric_m[k, 2]),
                    "ellipsoid_semi_axis_max_3sigma_m": float(np.max(ell.semi_axes[k])),
                    "alt_mean_km": float(ens.alt_mean[k]),
                    "alt_std_km": float(ens.alt_std[k]),
                }
            )


def _write_figures(
    out_dir: Path,
    result: MCRunResult,
    ens: EnsembleStatistics,
    ric: RICUncertainty,
    ell: ErrorEllipsoids,
) -> tuple[list[str], str | None]:
    """Render the standard UQ figures; return (written names, skip reason)."""
    try:
        from .plotting import (
            plot_altitude_envelope,
            plot_covariance_eigenvalues,
            plot_covariance_tubes_3d,
            plot_position_covariance_history,
            plot_ric_sigma_history,
        )
    except ImportError as exc:
        return [], f"matplotlib unavailable: {exc}"

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    builders = [
        ("position_covariance_history.png", lambda: plot_position_covariance_history(ens)),
        ("ric_sigma_history.png", lambda: plot_ric_sigma_history(ric)),
        ("covariance_eigenvalues.png", lambda: plot_covariance_eigenvalues(ens)),
        ("covariance_tubes_3d.png", lambda: plot_covariance_tubes_3d(result, ell)),
        ("altitude_envelope.png", lambda: plot_altitude_envelope(result, ens)),
    ]
    written: list[str] = []
    for name, build in builders:
        try:
            fig = build()
        except ImportError as exc:
            return [], f"matplotlib unavailable: {exc}"
        try:
            fig.savefig(figures_dir / name, dpi=150)
            written.append(f"figures/{name}")
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)
    return written, None


def build_uq_report(
    result: MCRunResult,
    out_dir: str | Path,
    *,
    run_config: Mapping[str, Any] | None = None,
    source_archive: str | Path | None = None,
    use_survived_only: bool = False,
    make_figures: bool = True,
    r_ref_m: float = R_MOON,
) -> dict[str, Any]:
    """Compute ensemble statistics and write the full UQ report bundle.

    Parameters
    ----------
    result : MCRunResult
        In-memory ensemble (fresh run or loaded archive).
    out_dir : path
        Report directory (created if needed).
    run_config : optional mapping
        Echo of the batch/ensemble configuration (seed, sampling method, sigmas,
        backend); recorded verbatim in the manifest for reproducibility.
    source_archive : optional path
        The HDF5/NPZ archive the ensemble came from; hashed into the manifest.
    use_survived_only : bool
        Exclude impacted samples from the statistics (recorded in the manifest).
    make_figures : bool
        Render the standard figure set; a missing matplotlib is recorded as a
        skip reason instead of failing the report.

    Returns
    -------
    dict
        The manifest payload that was written to ``uq_manifest.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ens = compute_ensemble_statistics(
        result, use_survived_only=use_survived_only, r_ref_m=float(r_ref_m)
    )
    ell = compute_error_ellipsoids(ens)
    ric = compute_ric_uncertainty(ens)

    arrays: dict[str, np.ndarray] = {
        "t_s": ens.t,
        "mean_state": ens.mean,
        "cov": ens.cov,
        "sigma_ric_m": ric.sigma_ric_m,
        "cov_ric": ric.cov_ric,
        "ellipsoid_semi_axes_3sigma_m": ell.semi_axes,
        "ellipsoid_eigvecs": ell.eigvecs,
        "alt_mean_km": ens.alt_mean,
        "alt_std_km": ens.alt_std,
    }
    content_hash = ensemble_content_sha256(arrays)

    npz_path = out / _NPZ_NAME
    np.savez_compressed(npz_path, **arrays)
    csv_path = out / _CSV_NAME
    _write_summary_csv(csv_path, ens, ric, ell)

    figure_files: list[str] = []
    figures_skipped_reason: str | None = None
    if make_figures:
        figure_files, figures_skipped_reason = _write_figures(out, result, ens, ric, ell)
        if figures_skipped_reason is not None:
            warnings.warn(
                f"UQ figures skipped: {figures_skipped_reason}", RuntimeWarning, stacklevel=2
            )

    files: dict[str, dict[str, Any]] = {
        _NPZ_NAME: {"sha256": sha256_file(npz_path)},
        _CSV_NAME: {"sha256": sha256_file(csv_path)},
    }
    for rel in figure_files:
        files[rel] = {"sha256": sha256_file(out / rel)}

    run_config_payload = _jsonable(dict(run_config)) if run_config is not None else None
    manifest: dict[str, Any] = {
        "schema_version": UQ_REPORT_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "covariance_definition": COVARIANCE_DEFINITION,
        "frame": "moon_centred_inertial",
        "units": {"position": "m", "velocity": "m/s", "time": "s", "altitude": "km"},
        "ensemble": {
            "n_samples": int(result.n_samples),
            "n_valid": int(result.n_valid),
            "n_survived": int(result.n_survived),
            "n_epochs": int(ens.t.shape[0]),
            "use_survived_only": bool(use_survived_only),
            "r_ref_m": float(r_ref_m),
        },
        "covariance_content_sha256": content_hash,
        "run_config": run_config_payload,
        "run_config_hash": (
            canonical_json_sha256(run_config_payload) if run_config_payload is not None else None
        ),
        "source_archive": _archive_record(
            source_archive if source_archive is not None else result.archive_path
        ),
        "archive_metadata": _jsonable(dict(result.diagnostics or {})),
        "files": files,
        "figures_skipped_reason": figures_skipped_reason,
        "repo": _collect_git_info(),
        "environment": _collect_environment(),
    }
    (out / _MANIFEST_NAME).write_text(canonical_json_text(manifest), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Post-hoc UQ report over an existing batch archive."""
    parser = argparse.ArgumentParser(
        description="Build a provenance-stamped ensemble UQ report from a batch archive (HDF5/NPZ)."
    )
    parser.add_argument("--archive", required=True, help="Path to the batch ensemble archive (.h5/.npz)")
    parser.add_argument("--out", required=True, help="Output report directory")
    parser.add_argument("--survived-only", action="store_true",
                        help="Exclude impacted samples from the statistics")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    args = parser.parse_args(argv)

    from lunaris.batch.storage import load_mc_result

    result = load_mc_result(str(args.archive), lazy=True, strict=False)
    manifest = build_uq_report(
        result,
        args.out,
        source_archive=args.archive,
        use_survived_only=bool(args.survived_only),
        make_figures=not bool(args.no_figures),
    )
    print(f"[UQ] report written: {Path(args.out) / _MANIFEST_NAME}", flush=True)
    print(f"[UQ] covariance content hash: {manifest['covariance_content_sha256']}", flush=True)
    return 0


__all__ = [
    "COVARIANCE_DEFINITION",
    "UQ_REPORT_SCHEMA_VERSION",
    "build_uq_report",
    "ensemble_content_sha256",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
